from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "exp" / "q1_submission_v2_smoke"
FIXED_FILES = (
    "fixed_test_step_metrics.csv",
    "fixed_test_daily_metrics.csv",
    "fixed_test_node_voltages.npz",
    "fixed_test_trajectory_schema.json",
    "fixed_test_summary.json",
)
SOLVERS = ("matd3", "maddpg", "mappo", "facmaddpg")
MAPPINGS = ("capacity_adaptive", "static_projected")
ACTION_DIAGNOSTICS = (
    "action_projection_ratio",
    "reactive_capacity_utilization",
    "reactive_saturation_ratio",
)


@contextlib.contextmanager
def _repo_import_path():
    repo_text = str(REPO_ROOT)
    added = repo_text not in sys.path
    if added:
        sys.path.insert(0, repo_text)
    try:
        yield
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(repo_text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "test_file_sha256": _file_sha256(Path(__file__).resolve()),
        "git_revision": revision,
        "git_dirty": bool(status.strip()),
    }


def _assert_current_source_identity(evidence: dict) -> None:
    identity = evidence["source_identity"]
    assert identity["test_file_sha256"] == _file_sha256(Path(__file__).resolve())
    assert len(identity["git_revision"]) == 40
    assert all(
        character in "0123456789abcdef" for character in identity["git_revision"]
    )
    assert isinstance(identity["git_dirty"], bool)


@contextlib.contextmanager
def _capture_first_runpp_q(pandapower_module):
    original_runpp = pandapower_module.runpp
    observation: dict[str, list[float]] = {}

    def tracked_runpp(network, *args, **kwargs):
        if "q_mvar" not in observation:
            observation["q_mvar"] = (
                network.sgen["q_mvar"].to_numpy(dtype=float, copy=True).tolist()
            )
        return original_runpp(network, *args, **kwargs)

    pandapower_module.runpp = tracked_runpp
    try:
        yield observation
    finally:
        pandapower_module.runpp = original_runpp


def _session_root(label: str) -> Path:
    root = SMOKE_ROOT / f"{label[:8]}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _child_command(*arguments: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *arguments]


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    temporary = SMOKE_ROOT / "child-temp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TEMP"] = str(temporary)
    environment["TMP"] = str(temporary)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_child(arguments: list[str], *, timeout: int) -> dict:
    completed = subprocess.run(
        _child_command(*arguments),
        cwd=REPO_ROOT,
        env=_child_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "child smoke failed\n"
            f"command: {completed.args!r}\n"
            f"stdout:\n{completed.stdout[-8000:]}\n"
            f"stderr:\n{completed.stderr[-8000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stdout
    return json.loads(lines[-1])


def _small_fixed_result(violation: float, *, points: int = 12):
    from experiments.fixed_test_protocol import FixedTestResult

    timestamps = np.asarray(
        [
            np.datetime64("2022-10-28T06:00:00", "ns") + np.timedelta64(3 * step, "m")
            for step in range(points)
        ],
        dtype="datetime64[ns]",
    )
    rows = [
        {
            "day_index": 1030,
            "day_position": 0,
            "step_index": step,
            "timestamp": np.datetime_as_string(timestamp, unit="s"),
            "executed": True,
            "vrate": float(violation),
            "cr": 1.0 if violation == 0.0 else 0.0,
            "destroy": 0.0,
            "percentage_of_v_out_of_control": float(violation),
            "totally_controllable_ratio": 1.0 if violation == 0.0 else 0.0,
            "average_voltage_deviation": 0.01,
            "q_loss": 0.2,
            "total_line_loss": 0.03,
            "reward": -0.03,
            "inverter_p_mw": json.dumps([0.6], separators=(",", ":")),
            "inverter_q_mvar": json.dumps([0.2], separators=(",", ":")),
            "inverter_s_max_mva": json.dumps([1.0], separators=(",", ":")),
        }
        for step, timestamp in enumerate(timestamps)
    ]
    daily = {
        "day_index": 1030,
        "day_position": 0,
        "date": "2022-10-28",
        "failure_step": None,
        "planned_steps": points,
        "executed_steps": points,
        "failed_steps": 0,
        "vrate": float(violation),
        "cr": 1.0 if violation == 0.0 else 0.0,
    }
    summary = {
        **daily,
        "planned_days": 1,
        "completed_days": 1,
        "failed_days": 0,
        "start_day": 1030,
        "end_day": 1030,
        "points_per_day": points,
        "start_hour": 6,
        "cadence_minutes": 3,
        "bus_order": [0],
        "percentage_of_v_out_of_control": float(violation),
        "average_voltage_deviation": 0.01,
        "q_loss": 0.2,
        "total_line_loss": 0.03,
        "metric_denominators": {
            "percentage_of_v_out_of_control": points,
            "average_voltage_deviation": points,
            "q_loss": points,
            "total_line_loss": points,
        },
    }
    return FixedTestResult(
        summary=summary,
        step_rows=rows,
        daily_rows=[daily],
        node_voltages=np.ones((points, 1), dtype=float),
        timestamps=timestamps,
        executed_mask=np.ones(points, dtype=bool),
    )


def _install_fast_runner_fakes() -> None:
    import torch

    class FakeWriter:
        def __init__(self, _path):
            self.closed = False

        def add_scalar(self, *_args, **_kwargs):
            return None

        def close(self):
            self.closed = True

    class FakeEnvironment:
        def close(self):
            return None

    class FakeBehaviourNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.episode = 0

        def train_process(self, stat, _trainer):
            self.episode += 1
            with torch.no_grad():
                self.weight.fill_(self.episode)
            stat["mean_train_policy_loss"] = 1.0 / self.episode

    class FakeTrainer:
        def __init__(self, args, _model, _env, _logger):
            self.args = args
            self.behaviour_net = FakeBehaviourNet()

        def logging(self, _stat):
            return None

    tensorboard_module = types.ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = FakeWriter
    compare_module = types.ModuleType("experiments.run_q1_ml_compare")
    compare_module.load_env_config = lambda *_args, **_kwargs: {}
    compare_module.load_alg_config = lambda *_args, **_kwargs: {}
    compare_module.make_args = lambda *_args, **_kwargs: SimpleNamespace(cuda=False)
    compare_module.make_env = lambda _config: FakeEnvironment()
    registry_module = types.ModuleType("models.model_registry")
    registry_module.Model = {solver: object for solver in SOLVERS}
    trainer_module = types.ModuleType("utilities.trainer")
    trainer_module.PGTrainer = FakeTrainer
    sys.modules["tensorboardX"] = tensorboard_module
    sys.modules["experiments.run_q1_ml_compare"] = compare_module
    sys.modules["models.model_registry"] = registry_module
    sys.modules["utilities.trainer"] = trainer_module


def _child_fast(output_dir: Path, *, seed: int, evaluate_terminal: bool) -> dict:
    import torch

    _install_fast_runner_fakes()
    import experiments.run_risk_aware_multiseed as runner
    from experiments.ir_crpc_submission_v2 import build_submission_runs

    output_dir.mkdir(parents=True, exist_ok=False)
    data_marker = output_dir / "smoke_data.txt"
    data_marker.write_text("test-only abbreviated flow\n", encoding="utf-8")
    source = next(
        row
        for row in build_submission_runs(SMOKE_ROOT)
        if row["variant_id"] == "full_matd3"
        and row["scenario"] == "case33_3min_final"
        and row["seed"] == seed
    )
    run = {
        "scenario": source["scenario"],
        "algorithm": source["solver"],
        "seed": seed,
        "output_dir": str(output_dir),
        "evaluate_terminal": bool(evaluate_terminal),
        "data_paths": [str(data_marker)],
        "config_paths": [str(REPO_ROOT / "args" / "alg_args" / "matd3.yaml")],
        "run_plan_content": {
            "smoke": True,
            "source_run_id": source["run_id"],
            "points": 12,
        },
        "_start_git_snapshot": {"revision": "task8-smoke", "dirty": False},
    }
    validation_calls = 0

    def abbreviated_evaluation(_env, _net, _args, days, max_steps, start_hour):
        nonlocal validation_calls
        assert max_steps == 12
        assert start_hour == 6
        if list(days) == [1000]:
            validation_calls += 1
            violation = 0.1 if validation_calls == 1 else 0.2
        else:
            assert list(days) == [1030]
            violation = 0.05
        return _small_fixed_result(violation, points=12)

    runner._evaluate_fixed_result = abbreviated_evaluation
    args_cli = SimpleNamespace(
        pv_scale=1.5,
        voltage_weight=1.0,
        violation_weight=0.0,
        q_weight=0.1,
        train_episodes=2,
        eval_freq=1,
        max_steps=12,
        eval_start_hour=6,
        validation_only=False,
        retain_validation_trajectories=False,
    )
    result = runner._run_single(
        run,
        {
            "training_days": [0],
            "validation_days": [1000],
            "test_days": [1030],
        },
        args_cli,
    )
    best = torch.load(output_dir / "best_model.pt", map_location="cpu")
    final = torch.load(output_dir / "final_model.pt", map_location="cpu")
    expected = [
        output_dir / "best_model.pt",
        output_dir / "final_model.pt",
        output_dir / "validation_history.json",
        output_dir / "provenance.json",
        *(output_dir / "test_best" / name for name in FIXED_FILES),
    ]
    if evaluate_terminal:
        expected.extend(output_dir / "test_final" / name for name in FIXED_FILES)
    assert all(path.is_file() for path in expected)
    assert (output_dir / "test_final").is_dir() is bool(evaluate_terminal)
    marker = output_dir / "seed_result.json"
    assert marker.is_file()
    assert marker.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in expected
    )
    return {
        "pid": os.getpid(),
        "source_identity": _source_identity(),
        "output_dir": str(output_dir.resolve()),
        "selected_episode": int(best["episode"]),
        "terminal_episode": int(final["episode"]),
        "validation_records": len(
            json.loads((output_dir / "validation_history.json").read_text("utf-8"))
        ),
        "test_summary": json.loads(
            (output_dir / "test_best" / "fixed_test_summary.json").read_text(
                encoding="utf-8"
            )
        ),
        "evaluate_terminal": bool(result["evaluate_terminal"]),
        "seed_result_last": True,
    }


def _instrument_updates(trainer) -> dict[str, int]:
    counts = {"policy": 0, "value": 0, "mixer": 0}
    for label, attribute in (
        ("policy", "policy_replay_process"),
        ("value", "value_replay_process"),
        ("mixer", "mixer_replay_process"),
    ):
        original = getattr(trainer, attribute)

        def wrapped(*args, _label=label, _original=original, **kwargs):
            counts[_label] += 1
            return _original(*args, **kwargs)

        setattr(trainer, attribute, wrapped)
    return counts


def _finite_loss_dict(stat: dict) -> dict[str, float]:
    losses = {
        key: float(value)
        for key, value in stat.items()
        if "loss" in key and isinstance(value, (int, float, np.number))
    }
    assert losses
    assert all(math.isfinite(value) for value in losses.values())
    return losses


def _child_real_solver(output_dir: Path, *, solver: str, mapping: str) -> dict:
    import torch
    from tensorboardX import SummaryWriter

    from experiments.fixed_test_protocol import (
        evaluate_controller_fixed_days,
        evaluate_policy_fixed_days,
        write_fixed_test_artifacts,
    )
    from experiments.risk_aware_checkpointing import CheckpointManager
    from experiments.run_q1_ml_compare import (
        load_alg_config,
        load_env_config,
        make_args,
        make_env,
    )
    from experiments.submission_provenance import (
        build_provenance,
        write_provenance_atomic,
    )
    from models.model_registry import Model
    from utilities.trainer import PGTrainer

    assert solver in SOLVERS
    assert mapping in MAPPINGS
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    output_dir.mkdir(parents=True, exist_ok=False)
    env_config = load_env_config("case33_3min_final", 1.5, mode="distributed")
    env_config.update(
        {
            "action_mapping": mapping,
            "episode_limit": 2,
            "reset_action": False,
            "seed": 0,
            "start_day_candidates": [0],
            "voltage_weight": 1.0,
            "violation_weight": 0.0,
            "q_weight": 0.1,
        }
    )
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_env(env_config)
    env._select_start_day = lambda: 0
    env._select_start_hour = lambda: 12
    env._select_start_interval = lambda: 0
    algorithm_config = load_alg_config(solver, env_config)
    train_args = make_args(
        env_config,
        algorithm_config,
        env,
        {
            "train_episodes_num": 1,
            "max_steps": 2,
            "cuda": False,
            "batch_size": 2,
            "replay_warmup": 0,
            "behaviour_update_freq": 1,
            "target_update_freq": 10000,
            "policy_update_epochs": 1,
            "value_update_epochs": 1,
            "mixer_update_epochs": 1,
            "fixed_policy_std": 0.001,
            "init_std": 0.001,
        },
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    trainer = PGTrainer(train_args, Model[solver], env, writer)
    update_counts = _instrument_updates(trainer)
    stat: dict[str, float] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.behaviour_net.train()
        trainer.behaviour_net.train_process(stat, trainer)
    assert trainer.steps == 2
    assert update_counts["policy"] == 1
    assert update_counts["value"] == 1
    assert update_counts["mixer"] == (1 if solver == "facmaddpg" else 0)
    losses = _finite_loss_dict(stat)

    manager = CheckpointManager(output_dir)
    validation = evaluate_policy_fixed_days(
        env=env,
        behaviour_net=trainer.behaviour_net,
        args=train_args,
        days=[1000],
        max_steps=2,
        start_hour=6,
        _strict_shape=False,
    )
    assert validation.summary["failed_steps"] == 0
    validation_metrics = {
        key: float(value)
        for key, value in validation.summary.items()
        if isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
    }
    manager.consider(trainer.behaviour_net.state_dict(), validation_metrics, 1)
    final_path = manager.save_final(trainer.behaviour_net.state_dict(), 1)
    best_path = manager.finalize()
    best = torch.load(best_path, map_location="cpu")
    trainer.behaviour_net.load_state_dict(best["model_state_dict"])
    fixed = evaluate_policy_fixed_days(
        env=env,
        behaviour_net=trainer.behaviour_net,
        args=train_args,
        days=[1030],
        max_steps=2,
        start_hour=6,
        _strict_shape=False,
    )
    assert fixed.summary["failed_steps"] == 0
    test_dir = output_dir / "test_best"
    write_fixed_test_artifacts(fixed, test_dir, "fixed_test")
    assert all((test_dir / name).is_file() for name in FIXED_FILES)

    projection = None
    if mapping == "static_projected":
        action = np.full(env.get_num_of_agents(), env_config["action_scale"])
        action_snapshot: dict[str, object] = {}

        def high_pv_controller(current_env):
            current_env.powergrid.sgen["p_mw"] = 0.9 * current_env.s_max
            converged = current_env._take_action(action)
            p = current_env.powergrid.sgen["p_mw"].to_numpy(dtype=float, copy=True)
            q = current_env.powergrid.sgen["q_mvar"].to_numpy(dtype=float, copy=True)
            capacity = np.sqrt(np.maximum(current_env.s_max**2 - p**2, 0.0))
            diagnostics = {
                key: float(current_env._last_action_diagnostics[key])
                for key in ACTION_DIAGNOSTICS
            }
            assert np.all(np.abs(q) <= capacity + 1e-10)
            assert all(math.isfinite(value) for value in diagnostics.values())
            action_snapshot.update(
                {
                    "p_mw": p.tolist(),
                    "q_mvar": q.tolist(),
                    "capacity_mvar": capacity.tolist(),
                    "diagnostics": diagnostics,
                }
            )
            return converged

        projected = evaluate_controller_fixed_days(
            env=env,
            controller=high_pv_controller,
            days=[1030],
            max_steps=1,
            start_hour=6,
            _strict_shape=False,
        )
        projection_dir = output_dir / "projection_fixture"
        write_fixed_test_artifacts(projected, projection_dir, "fixed_test")
        row = projected.step_rows[0]
        assert bool(row["executed"])
        assert float(row["action_projection_ratio"]) > 0.0
        assert action_snapshot
        with (projection_dir / "fixed_test_step_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            persisted = next(csv.DictReader(handle))
        persisted_diagnostics = {}
        for key in ACTION_DIAGNOSTICS:
            assert key in persisted
            value = float(persisted[key])
            assert math.isfinite(value)
            assert value == pytest.approx(action_snapshot["diagnostics"][key])
            persisted_diagnostics[key] = value
        assert persisted_diagnostics["action_projection_ratio"] > 0.0
        projection = {
            "ratio": persisted_diagnostics["action_projection_ratio"],
            "within_capacity": True,
            "captured_before_advance": True,
            "p_mw": action_snapshot["p_mw"],
            "q_mvar": action_snapshot["q_mvar"],
            "capacity_mvar": action_snapshot["capacity_mvar"],
            "diagnostics": persisted_diagnostics,
        }

    manifest = build_provenance(
        REPO_ROOT,
        data_paths=[
            REPO_ROOT
            / "environments"
            / "var_voltage_control"
            / "data"
            / "case33_3min_final"
            / "model.p"
        ],
        config_paths=[
            REPO_ROOT / "args" / "default.yaml",
            REPO_ROOT / "args" / "env_args" / "var_voltage_control.yaml",
            REPO_ROOT / "args" / "alg_args" / f"{solver}.yaml",
        ],
        weight_paths={"best": best_path, "final": final_path},
        run_dir=output_dir,
        run_plan_content={
            "task": 8,
            "solver": solver,
            "mapping": mapping,
            "real_training_updates": 1,
        },
        best_episode=1,
        final_episode=1,
    )
    write_provenance_atomic(manifest, output_dir / "provenance.json")
    writer.close()
    env.close()
    evidence = {
        "pid": os.getpid(),
        "source_identity": _source_identity(),
        "solver": solver,
        "mapping": mapping,
        "elapsed_seconds": time.perf_counter() - started,
        "trainer_steps": trainer.steps,
        "update_counts": update_counts,
        "losses": losses,
        "validation_summary": validation.summary,
        "fixed_summary": fixed.summary,
        "best_checkpoint_episode": int(best["episode"]),
        "checkpoint_readable": True,
        "projection": projection,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "smoke_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return evidence


def _child_full_day(output_dir: Path) -> dict:
    from environments.var_voltage_control import voltage_control_env
    from experiments.fixed_test_protocol import (
        evaluate_controller_fixed_days,
        write_fixed_test_artifacts,
    )
    from experiments.run_q1_ml_compare import load_env_config, make_env

    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_env_config("case33_3min_final", 1.5, mode="distributed")
    config.update(
        {
            "action_mapping": "capacity_adaptive",
            "episode_limit": 240,
            "reset_action": False,
            "seed": 0,
            "start_day_candidates": [1030],
            "voltage_weight": 1.0,
            "violation_weight": 0.0,
            "q_weight": 0.1,
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_env(config)
    original_set = env._set_demand_and_pv
    noise_flags: list[bool] = []

    def tracked_set(*args, **kwargs):
        noise_flags.append(bool(kwargs.get("add_noise", True)))
        return original_set(*args, **kwargs)

    env._set_demand_and_pv = tracked_set
    initial_q: list[float] | None = None
    progress_path = output_dir / "progress.log"
    transitions = 0

    def zero_controller(current_env):
        nonlocal initial_q, transitions
        if initial_q is None:
            initial_q = (
                current_env.powergrid.sgen["q_mvar"]
                .to_numpy(dtype=float, copy=True)
                .tolist()
            )
        converged = current_env._take_action(
            np.zeros(current_env.get_num_of_agents(), dtype=float)
        )
        transitions += 1
        if transitions % 30 == 0 or transitions == 240:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{transitions}/240\n")
        return converged

    started = time.perf_counter()
    with _capture_first_runpp_q(voltage_control_env.pp) as first_powerflow:
        result = evaluate_controller_fixed_days(
            env=env,
            controller=zero_controller,
            days=[1030],
            max_steps=240,
            start_hour=6,
            _strict_shape=False,
        )
    artifact_dir = output_dir / "full_day"
    write_fixed_test_artifacts(result, artifact_dir, "fixed_test")
    with (artifact_dir / "fixed_test_step_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(
        (artifact_dir / "fixed_test_summary.json").read_text(encoding="utf-8")
    )
    timestamps = [row["timestamp"] for row in rows]
    assert len(rows) == 240
    assert len(set(timestamps)) == 240
    assert timestamps[0].endswith("T06:00:00")
    assert timestamps[-1].endswith("T17:57:00")
    assert all(row["executed"].lower() == "true" for row in rows)
    assert summary["failed_steps"] == 0
    assert summary["completed_days"] == 1
    assert "q_mvar" in first_powerflow
    assert np.allclose(first_powerflow["q_mvar"], 0.0)
    assert initial_q is not None and np.allclose(initial_q, 0.0)
    assert noise_flags and not any(noise_flags)
    recomputed = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("vrate", "cr", "destroy", "q_loss", "total_line_loss")
    }
    for key, value in recomputed.items():
        summary_key = {
            "vrate": "vrate",
            "cr": "cr",
            "destroy": "destroy",
            "q_loss": "q_loss",
            "total_line_loss": "total_line_loss",
        }[key]
        assert float(summary[summary_key]) == pytest.approx(value, abs=1e-12)
    env.close()
    evidence = {
        "pid": os.getpid(),
        "source_identity": _source_identity(),
        "elapsed_seconds": time.perf_counter() - started,
        "row_count": len(rows),
        "unique_timestamps": len(set(timestamps)),
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "initial_q_zero": True,
        "first_powerflow_q_zero": True,
        "noise_flags_all_false": True,
        "summary_recomputed": recomputed,
        "summary_matches": True,
        "failed_steps": int(summary["failed_steps"]),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "full_day_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evidence


def test_fast_plan_to_result_subprocess_emits_complete_artifact_contract():
    session = _session_root("fast-reference")
    root = session / "q1_submission_v2_strict"
    output = root / "full_matd3" / "case33_3min_final" / "seed_0"
    evidence = _run_child(
        [
            "--child-fast",
            "--output",
            str(output),
            "--seed",
            "0",
            "--evaluate-terminal",
        ],
        timeout=120,
    )
    assert evidence["selected_episode"] == 1
    assert evidence["terminal_episode"] == 2
    assert evidence["validation_records"] == 2
    assert evidence["test_summary"]["planned_steps"] == 12
    assert evidence["evaluate_terminal"] is True
    assert evidence["seed_result_last"] is True
    _assert_current_source_identity(evidence)
    (session / "fast_reference_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_three_process_fast_launcher_is_isolated_and_deterministic():
    session = _session_root("fast-multiprocess")
    root = session / "q1_submission_v2_strict"
    reference_output = root / "reference" / "seed_0"
    reference = _run_child(
        ["--child-fast", "--output", str(reference_output), "--seed", "0"],
        timeout=120,
    )
    evidence = []
    processes = []
    try:
        for seed in range(3):
            output = root / "workers" / f"seed_{seed}"
            processes.append(
                (
                    output,
                    subprocess.Popen(
                        _child_command(
                            "--child-fast",
                            "--output",
                            str(output),
                            "--seed",
                            str(seed),
                        ),
                        cwd=REPO_ROOT,
                        env=_child_environment(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ),
                )
            )
        for output, process in processes:
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, f"{stdout}\n{stderr}"
            rows = [line for line in stdout.splitlines() if line.strip()]
            record = json.loads(rows[-1])
            assert Path(record["output_dir"]) == output.resolve()
            evidence.append(record)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    assert len({record["pid"] for record in evidence}) == 3
    assert len({record["output_dir"] for record in evidence}) == 3
    assert all(
        record["test_summary"] == reference["test_summary"] for record in evidence
    )
    assert all(record["evaluate_terminal"] is False for record in evidence)
    assert all(
        not (Path(record["output_dir"]) / "test_final").exists() for record in evidence
    )
    _assert_current_source_identity(reference)
    for record in evidence:
        _assert_current_source_identity(record)
    (session / "multiprocess_evidence.json").write_text(
        json.dumps(
            {"reference": reference, "workers": evidence},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("mapping", MAPPINGS)
def test_real_solver_performs_one_update_and_short_fixed_test(solver, mapping):
    root = _session_root(f"real-{solver}-{mapping}") / "q1_submission_v2_strict"
    output = root / mapping / solver
    evidence = _run_child(
        [
            "--child-real-solver",
            "--output",
            str(output),
            "--solver",
            solver,
            "--mapping",
            mapping,
        ],
        timeout=600,
    )
    assert evidence["solver"] == solver
    assert evidence["mapping"] == mapping
    assert evidence["trainer_steps"] == 2
    assert evidence["update_counts"]["policy"] == 1
    assert evidence["update_counts"]["value"] == 1
    assert evidence["checkpoint_readable"] is True
    assert evidence["fixed_summary"]["failed_steps"] == 0
    assert all(math.isfinite(value) for value in evidence["losses"].values())
    _assert_current_source_identity(evidence)
    assert (output / "provenance.json").is_file()
    assert all((output / "test_best" / name).is_file() for name in FIXED_FILES)
    if mapping == "static_projected":
        assert evidence["projection"]["ratio"] > 0.0
        assert evidence["projection"]["within_capacity"] is True
        assert evidence["projection"]["captured_before_advance"] is True
        diagnostics = evidence["projection"]["diagnostics"]
        assert set(diagnostics) == {
            "action_projection_ratio",
            "reactive_capacity_utilization",
            "reactive_saturation_ratio",
        }
        assert all(math.isfinite(value) for value in diagnostics.values())


@pytest.mark.skipif(
    os.environ.get("IR_CRPC_RUN_FULL_DAY") != "1",
    reason="set IR_CRPC_RUN_FULL_DAY=1 for the bounded 240-point real-day smoke",
)
def test_full_real_case33_day_has_exact_noise_free_240_point_contract():
    root = _session_root("real-full-day") / "q1_submission_v2_strict"
    output = root / "case33_day1030"
    evidence = _run_child(["--child-full-day", "--output", str(output)], timeout=900)
    assert evidence["row_count"] == 240
    assert evidence["unique_timestamps"] == 240
    assert evidence["first_timestamp"].endswith("T06:00:00")
    assert evidence["last_timestamp"].endswith("T17:57:00")
    assert evidence["initial_q_zero"] is True
    assert evidence["first_powerflow_q_zero"] is True
    assert evidence["noise_flags_all_false"] is True
    assert evidence["summary_matches"] is True
    assert evidence["failed_steps"] == 0
    assert (output / "progress.log").read_text(encoding="utf-8").splitlines()[
        -1
    ] == "240/240"


def test_first_powerflow_capture_observes_boundary_and_restores_runpp():
    calls: list[list[float]] = []

    def original_runpp(network):
        calls.append(network.sgen["q_mvar"].to_numpy(dtype=float).tolist())
        return True

    pandapower_module = SimpleNamespace(runpp=original_runpp)
    first_series = SimpleNamespace(
        to_numpy=lambda **_kwargs: np.asarray([0.0, 0.0], dtype=float)
    )
    second_series = SimpleNamespace(
        to_numpy=lambda **_kwargs: np.asarray([1.0, 1.0], dtype=float)
    )
    network = SimpleNamespace(sgen={"q_mvar": first_series})

    with _capture_first_runpp_q(pandapower_module) as observation:
        pandapower_module.runpp(network)
        network.sgen["q_mvar"] = second_series
        pandapower_module.runpp(network)

    assert observation == {"q_mvar": [0.0, 0.0]}
    assert calls == [[0.0, 0.0], [1.0, 1.0]]
    assert pandapower_module.runpp is original_runpp


def _parse_child_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--child-fast", action="store_true")
    mode.add_argument("--child-real-solver", action="store_true")
    mode.add_argument("--child-full-day", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-terminal", action="store_true")
    parser.add_argument("--solver", choices=SOLVERS)
    parser.add_argument("--mapping", choices=MAPPINGS)
    return parser.parse_args()


def _child_main() -> int:
    args = _parse_child_args()
    resolved = args.output.resolve()
    if not resolved.is_relative_to(SMOKE_ROOT.resolve()):
        raise ValueError("Task 8 child output must remain below the smoke root")
    if "q1_submission_v2_strict" not in resolved.parts:
        raise ValueError(
            "child output must use a nested q1_submission_v2_strict component"
        )
    with _repo_import_path():
        if args.child_fast:
            evidence = _child_fast(
                resolved,
                seed=args.seed,
                evaluate_terminal=args.evaluate_terminal,
            )
        elif args.child_real_solver:
            if args.solver is None or args.mapping is None:
                raise ValueError("real solver child requires --solver and --mapping")
            evidence = _child_real_solver(
                resolved, solver=args.solver, mapping=args.mapping
            )
        else:
            evidence = _child_full_day(resolved)
    print(json.dumps(evidence, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main())
