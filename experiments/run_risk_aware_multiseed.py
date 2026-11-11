"""Fair multi-seed training protocol for risk-aware voltage control.

The module keeps heavy power-system and deep-learning imports inside the
execution path so ``--dry-run`` can validate an experiment plan quickly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.fixed_test_protocol import (
    FixedTestResult,
    evaluate_policy_fixed_days,
    write_fixed_test_artifacts,
)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_ALGORITHMS = ["matd3", "maddpg", "mappo", "facmaddpg"]
SUMMARY_METRICS = [
    "percentage_of_v_out_of_control",
    "average_voltage_deviation",
    "q_loss",
    "total_line_loss",
    "totally_controllable_ratio",
]
BUDGET_DIAGNOSTIC_OUTPUT = ROOT / "exp" / "q1_budget_diagnostic_2000ep_case141"
SUBMISSION_OUTPUT_NAME = "q1_submission_v2_strict"


def validate_submission_preflight(args_cli: argparse.Namespace) -> None:
    """Reject noncanonical evaluation or unsafe formal output before training."""
    protocol = (
        args_cli.validation_start_day,
        args_cli.validation_days,
        args_cli.test_start_day,
        args_cli.test_days,
        args_cli.max_steps,
        args_cli.eval_start_hour,
    )
    if protocol != (1000, 30, 1030, 30, 240, 6):
        raise ValueError(
            "submission evaluation requires validation=1000/30, "
            "test=1030/30, max_steps=240, and eval_start_hour=6"
        )

    # The separately constrained validation-only budget diagnostic predates
    # the formal submission root and never writes final-test trajectories.
    if args_cli.validation_only:
        return

    output_root = Path(args_cli.save_path).resolve()
    if args_cli.dry_run:
        return
    if SUBMISSION_OUTPUT_NAME not in output_root.parts:
        raise ValueError(
            f"formal runs must use a path under {SUBMISSION_OUTPUT_NAME}"
        )
    if args_cli.aggregate_only:
        if not output_root.is_dir():
            raise FileNotFoundError(
                f"aggregate-only root does not exist: {output_root}"
            )
        return
    if args_cli.skip_existing:
        raise ValueError("--skip-existing is incompatible with a fresh formal run root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"formal run requires a fresh empty output root: {output_root}"
        )


def validate_validation_only_args(args_cli: argparse.Namespace) -> None:
    """Fail before any filesystem write unless the budget run is canonical."""
    if not args_cli.validation_only:
        return
    if args_cli.aggregate_only or args_cli.skip_existing:
        raise ValueError(
            "--validation-only is incompatible with --aggregate-only and --skip-existing"
        )
    actual_output = Path(args_cli.save_path).resolve()
    canonical = BUDGET_DIAGNOSTIC_OUTPUT.resolve()
    configuration = (
        args_cli.scenario,
        args_cli.algorithms,
        args_cli.seeds,
        args_cli.train_episodes,
        args_cli.eval_freq,
        args_cli.pv_scale,
        args_cli.voltage_weight,
        args_cli.violation_weight,
        args_cli.q_weight,
        args_cli.validation_start_day,
        args_cli.validation_days,
        args_cli.test_start_day,
        args_cli.test_days,
        args_cli.max_steps,
        args_cli.eval_start_hour,
    )
    expected = (
        "case141_3min_final",
        ["mappo"],
        [1],
        2000,
        50,
        1.5,
        1.0,
        0.0,
        0.1,
        1000,
        30,
        1030,
        30,
        240,
        6,
    )
    if actual_output != canonical or configuration != expected:
        raise ValueError(
            "--validation-only requires the canonical budget diagnostic: "
            "save-path=exp/q1_budget_diagnostic_2000ep_case141, "
            "scenario=case141_3min_final, algorithms=mappo, seed=1, "
            "train-episodes=2000, eval-freq=50, pv-scale=1.5, "
            "reward-weights=1/0/0.1, validation=1000/30, test=1030/30, "
            "max-steps=240, eval-start-hour=6"
        )


def make_validation_record(
    episode: int, validation_metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "episode": int(episode),
        "validation_metrics": {
            key: float(value) for key, value in validation_metrics.items()
        },
    }


def write_validation_history_atomic(
    history_path: Path, history: list[dict[str, Any]]
) -> None:
    temporary = history_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, history_path)


def build_day_split(
    *,
    total_days: int,
    validation_start_day: int,
    validation_days: int,
    test_start_day: int,
    test_days: int,
    guard_days: int,
) -> dict[str, list[int]]:
    if (
        validation_start_day,
        validation_days,
        test_start_day,
        test_days,
    ) != (1000, 30, 1030, 30):
        raise ValueError(
            "submission boundaries must be validation_start_day=1000, "
            "validation_days=30, test_start_day=1030, and test_days=30"
        )
    if total_days <= 0:
        raise ValueError("total_days must be positive")
    if validation_days <= 0:
        raise ValueError("validation_days must be positive")
    if test_days <= 0:
        raise ValueError("test_days must be positive")
    if guard_days != 1:
        raise ValueError("guard_days must be exactly 1")
    if validation_start_day < guard_days:
        raise ValueError("validation_start_day leaves no training days before guard_days")

    validation_stop = validation_start_day + validation_days
    test_stop = test_start_day + test_days
    if test_start_day < validation_stop:
        raise ValueError("test_start_day creates validation/test overlap")
    if test_start_day != validation_stop:
        raise ValueError("test_start_day must immediately follow validation_days")
    if test_stop > total_days:
        raise ValueError("test_days extends beyond total_days")

    training_stop = validation_start_day - guard_days
    training = list(range(0, training_stop))
    guard = list(range(training_stop, validation_start_day))
    validation = list(range(validation_start_day, validation_stop))
    test = list(range(test_start_day, test_stop))
    unused = list(range(test_stop, total_days))
    if not training:
        raise ValueError("training split is empty")
    return {
        "training_days": training,
        "guard_days": guard,
        "validation_days": validation,
        "test_days": test,
        "unused_days": unused,
    }


def build_run_plan(
    *,
    scenario: str,
    algorithms: list[str],
    seeds: list[int],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    return [
        {
            "scenario": scenario,
            "algorithm": algorithm,
            "seed": int(seed),
            "output_dir": str(
                root / scenario / algorithm / f"seed_{int(seed)}"
            ).replace("\\", "/"),
        }
        for algorithm in algorithms
        for seed in seeds
    ]


def merge_existing_plan(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Append compatible extension runs without losing prior provenance.

    A multi-algorithm study can be completed in stages.  This helper permits
    adding algorithms to an existing output root only when the data split,
    training protocol, and reward weights are identical to those of the
    original study.  It also keeps every prior run entry in ``run_plan.json``.
    """
    if not plan_path.exists():
        return plan

    existing = json.loads(plan_path.read_text(encoding="utf-8"))
    for key in ("total_days", "split", "training_protocol", "reward_weights"):
        if existing.get(key) != plan.get(key):
            raise ValueError(
                f"existing run plan is incompatible on {key}; "
                "use a new output path for a different protocol"
            )

    merged_runs = {
        (row["scenario"], row["algorithm"], int(row["seed"])): row
        for row in existing.get("runs", [])
    }
    for row in plan["runs"]:
        merged_runs[(row["scenario"], row["algorithm"], int(row["seed"]))] = row
    plan["runs"] = list(merged_runs.values())
    return plan


def _read_last_nonempty_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell() - 1
        while position > 0:
            handle.seek(position)
            if handle.read(1) == b"\n":
                line = handle.readline().decode("utf-8").strip()
                if line:
                    return line
            position -= 1
    raise ValueError(f"no data row found in {path}")


def infer_total_days(scenario: str) -> int:
    path = ROOT / "environments" / "var_voltage_control" / "data" / scenario / "pv_active.csv"
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        first = handle.readline().split(",", 1)[0]
    last = _read_last_nonempty_line(path).split(",", 1)[0]
    start = datetime.fromisoformat(first)
    end = datetime.fromisoformat(last)
    return (end.date() - start.date()).days + 1


@contextlib.contextmanager
def preserve_module_training_mode(module):
    was_training = bool(module.training)
    try:
        module.eval()
        yield module
    finally:
        module.train(was_training)


def _evaluate_fixed_days(
    env,
    behaviour_net,
    args,
    days: list[int],
    max_steps: int,
    start_hour: int,
) -> dict[str, float]:
    # Compatibility surface for the one-step Task 1 mode-restoration test.
    # Production execution calls ``_evaluate_fixed_result`` below, which is
    # always strict and cannot select this reduced shape from the CLI.
    return evaluate_policy_fixed_days(
        env=env,
        behaviour_net=behaviour_net,
        args=args,
        days=days,
        max_steps=max_steps,
        start_hour=start_hour,
        _strict_shape=False,
    ).summary


def _evaluate_fixed_result(
    env,
    behaviour_net,
    args,
    days: list[int],
    max_steps: int,
    start_hour: int,
) -> FixedTestResult:
    return evaluate_policy_fixed_days(
        env=env,
        behaviour_net=behaviour_net,
        args=args,
        days=days,
        max_steps=max_steps,
        start_hour=start_hour,
    )


def consume_fixed_test_result(
    result: FixedTestResult,
    *,
    output_dir: str | Path,
    stem: str,
    retain_artifacts: bool,
) -> dict[str, Any]:
    """Return ranking metrics and optionally persist the full trajectory."""
    if retain_artifacts:
        write_fixed_test_artifacts(result, output_dir, stem)
    return {
        key: float(value)
        for key, value in result.summary.items()
        if isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
    }


FIXED_TEST_ARTIFACT_NAMES = (
    "fixed_test_step_metrics.csv",
    "fixed_test_daily_metrics.csv",
    "fixed_test_node_voltages.npz",
    "fixed_test_trajectory_schema.json",
    "fixed_test_summary.json",
)


def _require_fixed_test_artifacts(output_dir: Path) -> None:
    missing = [
        str(output_dir / name)
        for name in FIXED_TEST_ARTIFACT_NAMES
        if not (output_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"required fixed-test artifacts are missing: {missing}")


def _run_plan_content_for_provenance(
    run: dict[str, Any],
    split: dict[str, list[int]],
    args_cli: argparse.Namespace,
    output_dir: Path,
) -> Any:
    if "run_plan_content" in run:
        return run["run_plan_content"]
    if "run_plan_path" in run:
        return Path(run["run_plan_path"])
    for candidate_root in (output_dir, *output_dir.parents):
        if candidate_root.name == SUBMISSION_OUTPUT_NAME:
            plan_path = candidate_root / "run_plan.json"
            if plan_path.is_file():
                return plan_path
            break
    return {
        "run": {
            key: value
            for key, value in run.items()
            if key
            not in {"run_plan_content", "run_plan_path", "_start_git_snapshot"}
        },
        "split": split,
        "training_protocol": {
            "train_episodes": int(args_cli.train_episodes),
            "eval_freq": int(args_cli.eval_freq),
            "max_steps": int(args_cli.max_steps),
            "eval_start_hour": int(args_cli.eval_start_hour),
        },
    }


def _provenance_data_paths(
    run: dict[str, Any], env_config: dict[str, Any]
) -> list[Path]:
    if "data_paths" in run:
        return [Path(path) for path in run["data_paths"]]
    configured = env_config.get("data_path")
    if not configured:
        return []
    root = Path(configured)
    if not root.is_absolute():
        root = ROOT / root
    if root.is_dir():
        return sorted(path for path in root.rglob("*") if path.is_file())
    return [root]


def _provenance_config_paths(
    run: dict[str, Any], algorithm: str
) -> list[Path]:
    if "config_paths" in run:
        return [Path(path) for path in run["config_paths"]]
    return [
        ROOT / "args" / "default.yaml",
        ROOT / "args" / "env_args" / "var_voltage_control.yaml",
        ROOT / "args" / "alg_args" / f"{algorithm}.yaml",
    ]


def _write_run_json_atomic(value: Any, output_path: Path) -> None:
    """Write intermediate run JSON without consuming the final marker writer."""
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_result_identity(
    run: dict[str, Any], algorithm: str, seed: int
) -> MappingProxyType:
    """Return the registered identity shared by every completed result payload."""
    run_id = str(run.get("run_id", f"{run['scenario']}__{algorithm}__seed{seed}"))
    return MappingProxyType(
        {
            "status": "completed",
            "run_id": run_id,
            "variant_id": run.get("variant_id"),
            "scenario": run["scenario"],
            "algorithm": algorithm,
            "seed": int(seed),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_completion_manifest(
    *,
    output_dir: Path,
    run_id: str,
    plan_sha256: str | None,
    evaluate_terminal: bool,
) -> Path:
    relative_artifacts = [
        "best_model.pt",
        "final_model.pt",
        "validation_history.json",
        "seed_result_best.json",
        "seed_result_final.json",
        "provenance.json",
        *(f"test_best/{name}" for name in FIXED_TEST_ARTIFACT_NAMES),
    ]
    if evaluate_terminal:
        relative_artifacts.extend(
            f"test_final/{name}" for name in FIXED_TEST_ARTIFACT_NAMES
        )
    artifacts: dict[str, str] = {}
    for relative in relative_artifacts:
        path = output_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"completion artifact is missing: {path}")
        artifacts[relative] = _sha256_file(path)
    manifest_path = output_dir / "completion_manifest.json"
    _write_run_json_atomic(
        {
            "schema_version": 2,
            "status": "completed",
            "run_id": run_id,
            "plan_sha256": plan_sha256,
            "artifacts": artifacts,
        },
        manifest_path,
    )
    return manifest_path


class _RunResources:
    def __init__(self) -> None:
        self.logger: Any | None = None
        self.environments: list[Any] = []
        self.closed = False

    def add_environment(self, environment: Any) -> Any:
        self.environments.append(environment)
        return environment

    def set_logger(self, logger: Any) -> Any:
        self.logger = logger
        return logger

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        first_error: Exception | None = None
        resources = ([self.logger] if self.logger is not None else []) + self.environments
        for resource in resources:
            if not hasattr(resource, "close"):
                continue
            try:
                resource.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _run_single_impl(
    run: dict[str, Any],
    split: dict[str, list[int]],
    args_cli: argparse.Namespace,
    resources: _RunResources,
) -> dict[str, Any]:
    import torch as th
    from tensorboardX import SummaryWriter

    from experiments.risk_aware_checkpointing import CheckpointManager
    from experiments.submission_provenance import (
        build_provenance,
        capture_git_snapshot,
        write_provenance_atomic,
    )
    from experiments.run_q1_ml_compare import (
        load_alg_config,
        load_env_config,
        make_args,
        make_env,
    )
    from models.model_registry import Model
    from utilities.trainer import PGTrainer

    algorithm = run.get("solver", run.get("algorithm"))
    if not algorithm:
        raise ValueError("run must define solver or algorithm")
    seed = int(run["seed"])
    output_dir = Path(run["output_dir"])
    start_git_snapshot = run.get("_start_git_snapshot")
    if start_git_snapshot is None:
        start_git_snapshot = capture_git_snapshot(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_result_path = output_dir / "seed_result.json"
    seed_result_path.unlink(missing_ok=True)
    for stale_name in (
        "completion_manifest.json",
        "seed_result_best.json",
        "seed_result_final.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    training_protocol = run.get("training")

    def phase_environment_config(
        phase: str, *, fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        protocol = run.get(phase)
        if not isinstance(protocol, dict):
            if fallback is None:
                pv_scale = float(args_cli.pv_scale)
                config = load_env_config(
                    run["scenario"], pv_scale, mode="distributed"
                )
            else:
                config = dict(fallback)
            protocol = {}
        else:
            config = load_env_config(
                run["scenario"], float(protocol["pv_scale"]), mode="distributed"
            )
        for key in (
            "pv_scale",
            "action_mapping",
            "action_scale",
            "action_bias",
            "voltage_weight",
            "violation_weight",
            "q_weight",
            "line_weight",
            "reset_action",
            "add_noise",
        ):
            if key in protocol:
                config[key] = protocol[key]
        config.update(
            {
                "seed": seed,
                "start_day_candidates": split[f"{phase}_days"],
            }
        )
        return config

    env_config = phase_environment_config("training")
    if not isinstance(training_protocol, dict):
        env_config.update(
            {
                "violation_weight": args_cli.violation_weight,
                "voltage_weight": args_cli.voltage_weight,
                "q_weight": args_cli.q_weight,
            }
        )
    env = resources.add_environment(make_env(env_config))
    algorithm_config = load_alg_config(algorithm, env_config)
    train_args = make_args(
        env_config,
        algorithm_config,
        env,
        {
            "train_episodes_num": args_cli.train_episodes,
            "eval_freq": args_cli.train_episodes + 1,
            "max_steps": args_cli.max_steps,
            "cuda": th.cuda.is_available(),
        },
    )
    logger = resources.set_logger(SummaryWriter(str(output_dir / "tensorboard")))
    trainer = PGTrainer(train_args, Model[algorithm], env, logger)
    manager = None if args_cli.validation_only else CheckpointManager(output_dir)
    validation_history: list[dict[str, Any]] = []

    validation_config = phase_environment_config("validation", fallback=env_config)
    validation_env = resources.add_environment(make_env(validation_config))

    for episode_index in range(args_cli.train_episodes):
        stat: dict[str, float] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.behaviour_net.train()
            trainer.behaviour_net.train_process(stat, trainer)
        trainer.logging(stat)
        episode = episode_index + 1
        if episode % args_cli.eval_freq == 0 or episode == args_cli.train_episodes:
            validation_result = _evaluate_fixed_result(
                validation_env,
                trainer.behaviour_net,
                train_args,
                split["validation_days"],
                args_cli.max_steps,
                args_cli.eval_start_hour,
            )
            validation_metrics = consume_fixed_test_result(
                validation_result,
                output_dir=(
                    output_dir
                    / "validation_trajectories"
                    / f"episode_{episode:04d}"
                ),
                stem="fixed_test",
                retain_artifacts=bool(
                    getattr(args_cli, "retain_validation_trajectories", False)
                ),
            )
            if manager is not None:
                # The checkpoint selector accepts its three ranking endpoints;
                # the full trajectory summary also contains non-ranking fields.
                selection_metrics = {key: validation_metrics[key] for key in (
                    "percentage_of_v_out_of_control", "average_voltage_deviation", "q_loss"
                )}
                manager.consider(
                    trainer.behaviour_net.state_dict(), selection_metrics, episode
                )
            else:
                validation_history.append(
                    make_validation_record(episode, validation_metrics)
                )
                write_validation_history_atomic(
                    output_dir / "validation_history.json", validation_history
                )

    if args_cli.validation_only:
        return {
            "scenario": run["scenario"],
            "algorithm": algorithm,
            "seed": seed,
            "validation_only": True,
            "validation_history": str(output_dir / "validation_history.json").replace("\\", "/"),
        }
    assert manager is not None
    final_path = manager.save_final(
        trainer.behaviour_net.state_dict(), args_cli.train_episodes
    )
    best_path = manager.finalize()
    best = th.load(
        best_path,
        map_location="cuda"
        if train_args.cuda and th.cuda.is_available()
        else "cpu",
    )
    trainer.behaviour_net.load_state_dict(best["model_state_dict"])

    test_config = phase_environment_config("test", fallback=env_config)
    test_env = resources.add_environment(make_env(test_config))
    test_result = _evaluate_fixed_result(
        test_env,
        trainer.behaviour_net,
        train_args,
        split["test_days"],
        args_cli.max_steps,
        args_cli.eval_start_hour,
    )
    test_metrics = consume_fixed_test_result(
        test_result,
        output_dir=output_dir / "test_best",
        stem="fixed_test",
        retain_artifacts=True,
    )
    _require_fixed_test_artifacts(output_dir / "test_best")
    evaluate_terminal = run.get("evaluate_terminal") is True
    terminal_test_metrics = None
    if evaluate_terminal:
        final = th.load(
            final_path,
            map_location="cuda"
            if train_args.cuda and th.cuda.is_available()
            else "cpu",
        )
        trainer.behaviour_net.load_state_dict(final["model_state_dict"])
        terminal_result = _evaluate_fixed_result(
            test_env,
            trainer.behaviour_net,
            train_args,
            split["test_days"],
            args_cli.max_steps,
            args_cli.eval_start_hour,
        )
        terminal_test_metrics = consume_fixed_test_result(
            terminal_result,
            output_dir=output_dir / "test_final",
            stem="fixed_test",
            retain_artifacts=True,
        )
        _require_fixed_test_artifacts(output_dir / "test_final")
    identity = _immutable_result_identity(run, algorithm, seed)
    result = {
        **identity,
        "selected_episode": int(best["episode"]),
        "terminal_episode": int(args_cli.train_episodes),
        "evaluate_terminal": evaluate_terminal,
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
        "reward_weights": {
            "voltage_weight": args_cli.voltage_weight,
            "violation_weight": args_cli.violation_weight,
            "q_weight": args_cli.q_weight,
        },
    }
    if terminal_test_metrics is not None:
        result["terminal_test_metrics"] = terminal_test_metrics
    best_result = {
        **identity,
        "checkpoint": "best",
        "evaluated": True,
        "episode": int(best["episode"]),
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
    }
    final_result = {
        **identity,
        "checkpoint": "final",
        "evaluated": terminal_test_metrics is not None,
        "episode": int(args_cli.train_episodes),
        "validation_metrics": None,
        "test_metrics": terminal_test_metrics,
    }
    if terminal_test_metrics is None:
        final_result["evaluation_note"] = (
            "terminal fixed testing was not registered for this run"
        )
    _write_run_json_atomic(best_result, output_dir / "seed_result_best.json")
    _write_run_json_atomic(final_result, output_dir / "seed_result_final.json")
    manifest = build_provenance(
        ROOT,
        data_paths=_provenance_data_paths(run, env_config),
        config_paths=_provenance_config_paths(run, algorithm),
        weight_paths={"best": best_path, "final": final_path},
        run_dir=output_dir,
        run_plan_content=_run_plan_content_for_provenance(
            run, split, args_cli, output_dir
        ),
        best_episode=int(best["episode"]),
        final_episode=int(args_cli.train_episodes),
        git_revision=start_git_snapshot["revision"],
        git_dirty=start_git_snapshot["dirty"],
    )
    write_provenance_atomic(manifest, output_dir / "provenance.json")
    return result


def _run_single(
    run: dict[str, Any],
    split: dict[str, list[int]],
    args_cli: argparse.Namespace,
) -> dict[str, Any]:
    """Run one seed and close every acquired resource before completion markers."""
    output_dir = Path(run["output_dir"])
    resources = _RunResources()
    result: dict[str, Any] | None = None
    try:
        result = _run_single_impl(run, split, args_cli, resources)
    finally:
        try:
            resources.close()
        finally:
            if result is None:
                (output_dir / "completion_manifest.json").unlink(missing_ok=True)
                (output_dir / "seed_result.json").unlink(missing_ok=True)

    if result.get("validation_only") is True:
        return result

    completed = False
    try:
        _write_completion_manifest(
            output_dir=output_dir,
            run_id=result["run_id"],
            plan_sha256=run.get("plan_sha256"),
            evaluate_terminal=result["evaluate_terminal"],
        )
        from experiments.submission_provenance import write_json_atomic

        write_json_atomic(result, output_dir / "seed_result.json")
        completed = True
    finally:
        if not completed:
            (output_dir / "completion_manifest.json").unlink(missing_ok=True)
            (output_dir / "seed_result.json").unlink(missing_ok=True)
    return result


def _save_summary(
    results: list[dict[str, Any]],
    output_root: Path,
) -> None:
    from experiments.risk_aware_protocol import aggregate_seed_results

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        flattened = {"seed": result["seed"]}
        flattened.update(result["test_metrics"])
        grouped.setdefault(result["algorithm"], []).append(flattened)
    summary = {
        algorithm: aggregate_seed_results(
            records,
            [key for key in SUMMARY_METRICS if all(key in row for row in records)],
        )
        for algorithm, records in grouped.items()
    }
    (output_root / "multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "multiseed_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "metric", "mean", "std"])
        for algorithm, data in summary.items():
            for metric, values in data["metrics"].items():
                writer.writerow(
                    [algorithm, metric, values["mean"], values["std"]]
                )


def _collect_results_from_disk(output_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("**/seed_result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def aggregate_from_disk(output_root: Path) -> dict[str, Any]:
    results = _collect_results_from_disk(output_root)
    if not results:
        raise ValueError(f"no seed_result.json found under {output_root}")
    _save_summary(results, output_root)
    return json.loads((output_root / "multiseed_summary.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="case33_3min_final")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--save-path", default="exp/q1_submission_v2_strict")
    parser.add_argument("--pv-scale", type=float, default=1.5)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--eval-start-hour", type=int, default=6)
    parser.add_argument("--validation-start-day", type=int, default=1000)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-start-day", type=int, default=1030)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--voltage-weight", type=float, default=1.0)
    parser.add_argument("--violation-weight", type=float, default=100.0)
    parser.add_argument("--q-weight", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--retain-validation-trajectories", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()
    validate_validation_only_args(args_cli)
    validate_submission_preflight(args_cli)
    unknown = sorted(set(args_cli.algorithms) - set(DEFAULT_ALGORITHMS))
    if unknown:
        raise ValueError(f"unsupported algorithms: {unknown}")
    total_days = infer_total_days(args_cli.scenario)
    split = build_day_split(
        total_days=total_days,
        validation_start_day=args_cli.validation_start_day,
        validation_days=args_cli.validation_days,
        test_start_day=args_cli.test_start_day,
        test_days=args_cli.test_days,
        guard_days=1,
    )
    output_root = Path(args_cli.save_path)
    runs = build_run_plan(
        scenario=args_cli.scenario,
        algorithms=args_cli.algorithms,
        seeds=args_cli.seeds,
        output_root=output_root,
    )
    training_protocol = {
        "train_episodes": args_cli.train_episodes,
        "eval_freq": args_cli.eval_freq,
        "max_steps": args_cli.max_steps,
        "eval_start_hour": args_cli.eval_start_hour,
    }
    if args_cli.validation_only:
        training_protocol.update(
            {
                "validation_episodes": list(
                    range(args_cli.eval_freq, args_cli.train_episodes + 1, args_cli.eval_freq)
                )
                if args_cli.train_episodes % args_cli.eval_freq == 0
                else list(range(args_cli.eval_freq, args_cli.train_episodes + 1, args_cli.eval_freq))
                + [args_cli.train_episodes],
                "initialization": "seed initialization only; no checkpoint resume",
                "prohibited_checkpoint_inputs": ["best_model.pt"],
                "validation_only": True,
            }
        )
    else:
        training_protocol.update(
            {
                "checkpoint_selection": "minimum validation violation rate with lexicographic tie-breaks",
                "final_report": "single blind evaluation of the selected checkpoint on fixed test days",
            }
        )
    plan = {
        "total_days": total_days,
        "split": split,
        "training_protocol": training_protocol,
        "reward_weights": {
            "voltage_weight": args_cli.voltage_weight,
            "violation_weight": args_cli.violation_weight,
            "q_weight": args_cli.q_weight,
        },
        "runs": runs,
    }
    if args_cli.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    from experiments.submission_provenance import capture_git_snapshot

    start_git_snapshot = capture_git_snapshot(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "run_plan.json"
    plan = merge_existing_plan(plan_path, plan)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args_cli.aggregate_only:
        summary = aggregate_from_disk(output_root)
        print(f"Aggregated {sum(len(v['records']) for v in summary.values())} seed records")
        return

    results: list[dict[str, Any]] = []
    for run in runs:
        result_path = Path(run["output_dir"]) / "seed_result.json"
        if args_cli.skip_existing and result_path.exists():
            print(f"Skip existing: {result_path}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        execution_run = dict(run)
        execution_run["_start_git_snapshot"] = start_git_snapshot
        results.append(_run_single(execution_run, split, args_cli))
    if args_cli.validation_only:
        print(f"Completed {len(results)} validation-only runs: {output_root}")
        return
    all_results = _collect_results_from_disk(output_root)
    _save_summary(all_results, output_root)
    print(
        f"Completed {len(results)} requested runs; "
        f"aggregated {len(all_results)} seed records: {output_root}"
    )


if __name__ == "__main__":
    main()
