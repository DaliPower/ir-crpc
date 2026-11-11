from __future__ import annotations

import hashlib
import json
import sys
import types
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.ir_crpc_submission_v2 import (
    FIXED_TEST_FILES,
    FORMAL_OUTPUT_RELATIVE,
    build_execution_preflight,
    build_registered_run_context,
    build_submission_plan,
    execute_submission_plan,
    main,
)


def _write_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_completed_run(payload: dict) -> dict:
    run = payload["run"]
    run_dir = Path(run["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "best_model.pt",
        "final_model.pt",
        "validation_history.json",
        "seed_result_best.json",
        "seed_result_final.json",
        "provenance.json",
        *(f"test_best/{name}" for name in FIXED_TEST_FILES),
    ]
    if run["evaluate_terminal"]:
        required.extend(f"test_final/{name}" for name in FIXED_TEST_FILES)
    artifacts = {}
    for relative in required:
        artifact = run_dir / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f'{run["run_id"]}:{relative}'.encode())
        artifacts[relative] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "run_id": run["run_id"],
        "plan_sha256": payload["plan_sha256"],
        "artifacts": artifacts,
    }
    manifest_path = run_dir / "completion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (run_dir / "seed_result.json").write_text(
        json.dumps({"run_id": run["run_id"], "status": "completed"}),
        encoding="utf-8",
    )
    return {"run_id": run["run_id"], "status": "completed"}


def _fake_git_snapshot(_repo: Path) -> dict:
    return {"revision": "frozen123", "dirty": False}


def _fake_preflight(_repo: Path, plan: dict) -> dict:
    return {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "git": {"revision": "frozen123", "dirty_paths": []},
        "configuration_files": [],
        "dataset_files": [],
        "runtime_files": [],
    }


class _ImmediateExecutor:
    max_workers_seen: list[int] = []
    instances = []

    def __init__(self, *, max_workers: int):
        self.max_workers_seen.append(max_workers)
        self.submit_count = 0
        self.terminated = False
        self.futures: list[Future] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, fn, payload):
        self.submit_count += 1
        future = Future()
        try:
            future.set_result(fn(payload))
        except BaseException as exc:  # match ProcessPoolExecutor propagation
            future.set_exception(exc)
        self.futures.append(future)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))
        return None

    def terminate(self):
        self.terminated = True


def test_execute_cli_rejects_tampered_registry_plan_before_any_root_write(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan_path = repo / "reviewed.json"
    plan = build_submission_plan(root)
    plan["runs"][0]["training"]["pv_scale"] = 9.0
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_plan(plan_path, plan)

    with pytest.raises(ValueError, match="registered plan"):
        main(
            [
                "--execute",
                "--plan",
                str(plan_path),
                "--output-root",
                str(root),
                "--workers",
                "3",
            ],
            repo_root=repo,
        )

    assert not root.exists()


def test_cli_requires_explicit_execute_mode_before_any_root_write(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan_path = repo / "reviewed.json"
    _write_plan(plan_path, build_submission_plan(root))

    with pytest.raises(ValueError, match="--execute"):
        main(
            ["--plan", str(plan_path), "--output-root", str(root)],
            repo_root=repo,
        )

    assert not root.exists()


def test_registered_context_keeps_training_and_evaluation_protocols_independent(tmp_path):
    root = tmp_path / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    nominal = next(r for r in plan["runs"] if r["variant_id"] == "nominal_train_matd3")
    static = next(r for r in plan["runs"] if r["variant_id"] == "static_projected_matd3")

    nominal_context = build_registered_run_context(nominal, root, plan)
    static_context = build_registered_run_context(static, root, plan)

    assert nominal_context["run"]["algorithm"] == "matd3"
    assert nominal_context["run"]["training"]["pv_scale"] == 1.0
    assert nominal_context["run"]["validation"]["pv_scale"] == 1.5
    assert nominal_context["run"]["test"]["pv_scale"] == 1.5
    assert nominal_context["args"]["pv_scale"] == 1.0
    assert nominal_context["split"]["validation_days"] == list(range(1000, 1030))
    assert nominal_context["split"]["test_days"] == list(range(1030, 1060))
    assert static_context["run"]["training"]["action_mapping"] == "static_projected"
    assert static_context["run"]["validation"]["action_mapping"] == "static_projected"


def test_dispatch_initializes_frozen_root_and_honors_worker_bound(tmp_path):
    _ImmediateExecutor.max_workers_seen.clear()
    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)

    snapshots = []

    def capture_before_write(_repo):
        snapshots.append(not root.exists())
        return _fake_git_snapshot(_repo)

    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=3,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=capture_before_write,
    )

    assert result == {"completed": 36, "failed": 0, "skipped": 0}
    assert _ImmediateExecutor.max_workers_seen == [1] * 36
    assert all(instance.submit_count == 1 for instance in _ImmediateExecutor.instances)
    assert snapshots == [True]
    assert json.loads((root / "run_plan.json").read_text("utf-8")) == plan
    assert (root / "run_plan.sha256").read_text("ascii").strip() == plan["plan_sha256"]
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert len(metadata["completed_runs"]) == 36
    assert metadata["failed_runs"] == []
    assert metadata["git_snapshot"] == _fake_git_snapshot(repo)
    assert metadata["preflight_sha256"] == json.loads(
        (root / "execution_preflight.json").read_text("utf-8")
    )["preflight_sha256"]
    assert len({item["run_id"] for item in metadata["completed_runs"]}) == 36


def test_dispatch_failure_is_auditable_but_never_marked_complete(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    failed_id = plan["runs"][4]["run_id"]

    def fail_one(payload):
        if payload["run"]["run_id"] == failed_id:
            raise RuntimeError("synthetic worker failure")
        return _write_completed_run(payload)

    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=False,
        repo_root=repo,
        worker_fn=fail_one,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )

    assert result == {"completed": 35, "failed": 1, "skipped": 0}
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert failed_id not in {item["run_id"] for item in metadata["completed_runs"]}
    assert [item["run_id"] for item in metadata["failed_runs"]] == [failed_id]
    run = next(item for item in plan["runs"] if item["run_id"] == failed_id)
    run_dir = root / run["output_dir"]
    assert "synthetic worker failure" in (run_dir / "error.json").read_text("utf-8")
    assert not (run_dir / "completion_manifest.json").exists()
    assert not (run_dir / "seed_result.json").exists()


def test_dispatch_rejects_a_noncompleted_final_seed_marker(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    invalid_id = plan["runs"][0]["run_id"]

    def invalid_marker(payload):
        result = _write_completed_run(payload)
        if payload["run"]["run_id"] == invalid_id:
            marker = Path(payload["run"]["output_dir"]) / "seed_result.json"
            marker.write_text(
                json.dumps({"run_id": invalid_id, "status": "failed"}),
                encoding="utf-8",
            )
        return result

    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=invalid_marker,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )

    assert result == {"completed": 35, "failed": 1, "skipped": 0}
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert invalid_id not in {entry["run_id"] for entry in metadata["completed_runs"]}


def test_resume_skips_only_manifest_verified_completed_runs(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    first = plan["runs"][0]["run_id"]
    calls: list[str] = []

    def recording_worker(payload):
        calls.append(payload["run"]["run_id"])
        return _write_completed_run(payload)

    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=recording_worker,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )
    calls.clear()
    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=3,
        resume=True,
        repo_root=repo,
        worker_fn=recording_worker,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )

    assert result == {"completed": 0, "failed": 0, "skipped": 36}
    assert calls == []
    assert first not in calls
    assert not (root.parent / f".{root.name}.execution-lock").exists()


def test_existing_execution_lock_rejects_resume_without_formal_state_write(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    metadata_path = root / "resume_metadata.json"
    metadata_before = metadata_path.read_bytes()
    lock = root.parent / f".{root.name}.execution-lock"
    lock.mkdir()

    with pytest.raises(FileExistsError, match="already active"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=True,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )

    assert metadata_path.read_bytes() == metadata_before
    assert lock.is_dir()


def test_existing_execution_lock_rejects_fresh_without_root_write(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    lock = root.parent / f".{root.name}.execution-lock"
    lock.parent.mkdir(parents=True)
    lock.mkdir()

    with pytest.raises(FileExistsError, match="already active"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=False,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )

    assert not root.exists()
    assert lock.is_dir()


def test_formal_resume_rejects_a_tampered_plan_hash_file(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )
    (root / "run_plan.sha256").write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="plan hash file"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=True,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
        )


def test_failed_partial_run_is_not_pseudo_complete_and_is_archived_on_resume(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    failed_run = plan["runs"][0]

    def fail_after_partial(payload):
        if payload["run"]["run_id"] != failed_run["run_id"]:
            return _write_completed_run(payload)
        run_dir = Path(payload["run"]["output_dir"])
        (run_dir / "test_best").mkdir(parents=True)
        (run_dir / "test_best" / "partial.csv").write_text("partial")
        (run_dir / "completion_manifest.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        raise RuntimeError("late failure")

    first = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=False,
        repo_root=repo,
        worker_fn=fail_after_partial,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
    )
    failed_dir = root / failed_run["output_dir"]
    assert first["failed"] == 1
    assert not (failed_dir / "completion_manifest.json").exists()

    def require_clean_retry(payload):
        if payload["run"]["run_id"] == failed_run["run_id"]:
            assert not Path(payload["run"]["output_dir"]).exists()
        return _write_completed_run(payload)

    resumed = execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=True,
        repo_root=repo,
        worker_fn=require_clean_retry,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=lambda _repo: (_ for _ in ()).throw(
            AssertionError("resume must reuse the frozen Git snapshot")
        ),
    )
    assert resumed == {"completed": 1, "failed": 0, "skipped": 35}
    archives = list((root / "_failed_attempts").glob("*/attempt_*"))
    assert len(archives) == 1
    assert (archives[0] / "error.json").is_file()
    assert (archives[0] / "test_best" / "partial.csv").is_file()
    archive_metadata = json.loads(
        (archives[0] / "archive_metadata.json").read_text("utf-8")
    )
    assert archive_metadata["run_id"] == failed_run["run_id"]


@pytest.mark.parametrize("workers", [0, 4])
def test_dispatch_rejects_worker_counts_outside_formal_bound_before_writes(
    tmp_path, workers
):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)

    with pytest.raises(ValueError, match="workers"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=workers,
            resume=False,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
        )

    assert not root.exists()


def _run_lightweight_registered_variant(
    tmp_path,
    monkeypatch,
    *,
    variant_id,
    fail_validation=False,
    external_close_events=None,
):
    import experiments.run_risk_aware_multiseed as runner
    import experiments.submission_provenance as provenance

    seen_configs = []
    close_events = (
        external_close_events if external_close_events is not None else []
    )

    class FakeWriter:
        def __init__(self, _path):
            pass

        def close(self):
            close_events.append("logger")

    class FakeEnvironment:
        def __init__(self, config):
            self.config = dict(config)

        def close(self):
            close_events.append(self.config["pv_scale"])

    class FakeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def train_process(self, stat, _trainer):
            with torch.no_grad():
                self.weight.add_(1)
            stat["finite_loss"] = 1.0

    class FakeTrainer:
        def __init__(self, args, _model, _env, _logger):
            self.behaviour_net = FakeNet()

        def logging(self, _stat):
            pass

    compare = types.ModuleType("experiments.run_q1_ml_compare")
    compare.load_env_config = lambda _scenario, pv, mode="distributed": {
        "pv_scale": float(pv),
        "mode": mode,
    }
    compare.load_alg_config = lambda _algorithm, _env: {}
    compare.make_args = lambda *_args, **_kwargs: SimpleNamespace(cuda=False)

    def make_env(config):
        seen_configs.append(dict(config))
        return FakeEnvironment(config)

    compare.make_env = make_env
    tensorboard = types.ModuleType("tensorboardX")
    tensorboard.SummaryWriter = FakeWriter
    registry = types.ModuleType("models.model_registry")
    registry.Model = {
        "matd3": object,
        "maddpg": object,
        "mappo": object,
        "facmaddpg": object,
    }
    trainer = types.ModuleType("utilities.trainer")
    trainer.PGTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "experiments.run_q1_ml_compare", compare)
    monkeypatch.setitem(sys.modules, "tensorboardX", tensorboard)
    monkeypatch.setitem(sys.modules, "models.model_registry", registry)
    monkeypatch.setitem(sys.modules, "utilities.trainer", trainer)

    validation_values = iter([0.2, 0.1])

    def fake_evaluate(env, _net, _args, days, _max_steps, _start_hour):
        if list(days) == list(range(1000, 1030)):
            if fail_validation:
                raise RuntimeError("validation exploded")
            violation = next(validation_values)
        else:
            violation = env.config["pv_scale"] / 100.0
        return SimpleNamespace(
            summary={
                "percentage_of_v_out_of_control": violation,
                "average_voltage_deviation": 0.01,
                "q_loss": 0.02,
            }
        )

    def fake_consume(result, *, output_dir, stem, retain_artifacts):
        if retain_artifacts:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            for name in FIXED_TEST_FILES:
                (output_dir / name).write_bytes(b"fixed")
        return dict(result.summary)

    monkeypatch.setattr(runner, "_evaluate_fixed_result", fake_evaluate)
    monkeypatch.setattr(runner, "consume_fixed_test_result", fake_consume)
    monkeypatch.setattr(provenance, "capture_git_snapshot", lambda _root: {
        "revision": "abc123",
        "dirty": False,
    })
    monkeypatch.setattr(provenance, "build_provenance", lambda *_a, **_k: {
        "schema_version": 1
    })
    monkeypatch.setattr(
        provenance,
        "write_provenance_atomic",
        lambda value, path: Path(path).write_text(json.dumps(value), encoding="utf-8"),
    )

    root = tmp_path / "r"
    plan = build_submission_plan(root)
    registered = dict(next(
        row for row in plan["runs"] if row["variant_id"] == variant_id
    ))
    # Keep this mocked unit run below legacy Windows MAX_PATH; dispatcher tests
    # above exercise the canonical relative output separately.
    registered["output_dir"] = "o"
    context = build_registered_run_context(registered, root, plan)
    context["args"].update({"train_episodes": 2, "eval_freq": 1})
    result = runner._run_single(
        context["run"], context["split"], SimpleNamespace(**context["args"])
    )
    return result, Path(context["run"]["output_dir"]), seen_configs, close_events


def test_registered_runner_constructs_independent_phase_environments(
    tmp_path, monkeypatch
):
    _, _, configs, _ = _run_lightweight_registered_variant(
        tmp_path, monkeypatch, variant_id="nominal_train_matd3"
    )

    assert [config["pv_scale"] for config in configs] == [1.0, 1.5, 1.5]
    assert all(config["action_mapping"] == "capacity_adaptive" for config in configs)
    assert configs[0]["add_noise"] is True
    assert configs[1]["add_noise"] is False
    assert configs[2]["add_noise"] is False
    assert configs[1]["reset_action"] is False
    assert configs[2]["reset_action"] is False


def test_registered_runner_writes_complete_manifest_before_final_marker(
    tmp_path, monkeypatch
):
    result, run_dir, _, close_events = _run_lightweight_registered_variant(
        tmp_path, monkeypatch, variant_id="static_projected_matd3"
    )

    assert result["status"] == "completed"
    assert (run_dir / "seed_result.json").is_file()
    assert (run_dir / "seed_result_best.json").is_file()
    final_result = json.loads((run_dir / "seed_result_final.json").read_text("utf-8"))
    assert final_result["evaluated"] is False
    assert final_result["test_metrics"] is None
    manifest = json.loads((run_dir / "completion_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["plan_sha256"] == build_submission_plan(
        tmp_path / FORMAL_OUTPUT_RELATIVE
    )["plan_sha256"]
    for relative in (
        "best_model.pt",
        "final_model.pt",
        "validation_history.json",
        "seed_result_best.json",
        "seed_result_final.json",
        "provenance.json",
        *(f"test_best/{name}" for name in FIXED_TEST_FILES),
    ):
        artifact = run_dir / relative
        assert manifest["artifacts"][relative] == hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    assert close_events == ["logger", 1.5, 1.5, 1.5]


@pytest.mark.parametrize(
    "variant_id", ["full_matd3", "full_maddpg"], ids=["terminal", "non-terminal"]
)
def test_registered_runner_persists_registered_identity_in_every_result_payload(
    tmp_path, monkeypatch, variant_id
):
    result, run_dir, _, _ = _run_lightweight_registered_variant(
        tmp_path, monkeypatch, variant_id=variant_id
    )

    expected = {
        "status": "completed",
        "run_id": result["run_id"],
        "variant_id": result["variant_id"],
        "scenario": result["scenario"],
        "algorithm": result["algorithm"],
        "seed": result["seed"],
    }
    for filename in (
        "seed_result_best.json",
        "seed_result_final.json",
        "seed_result.json",
    ):
        payload = json.loads((run_dir / filename).read_text("utf-8"))
        assert {key: payload.get(key) for key in expected} == expected


def test_registered_runner_closes_created_resources_on_validation_failure(
    tmp_path, monkeypatch
):
    close_events = []
    with pytest.raises(RuntimeError, match="validation exploded"):
        _run_lightweight_registered_variant(
            tmp_path,
            monkeypatch,
            variant_id="nominal_train_matd3",
            fail_validation=True,
            external_close_events=close_events,
        )

    assert close_events == ["logger", 1.0, 1.5]
    run_dir = tmp_path / "r" / "o"
    assert not (run_dir / "completion_manifest.json").exists()
    assert not (run_dir / "seed_result.json").exists()


def test_execute_requires_an_explicit_workers_argument_before_writes(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan_path = repo / "reviewed.json"
    _write_plan(plan_path, build_submission_plan(root))

    with pytest.raises(ValueError, match="--workers"):
        main(
            [
                "--execute",
                "--plan",
                str(plan_path),
                "--output-root",
                str(root),
            ],
            repo_root=repo,
        )
    assert not root.exists()


def test_dry_run_rejects_execution_options_before_creating_plan_out(tmp_path):
    repo = tmp_path / "repo"
    plan_out = tmp_path / "reviewed.json"

    with pytest.raises(ValueError, match="dry-run"):
        main(
            ["--dry-run", "--plan-out", str(plan_out), "--resume"],
            repo_root=repo,
        )
    assert not plan_out.exists()


def test_fresh_execution_rejects_even_an_existing_empty_formal_root(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    root.mkdir(parents=True)
    plan = build_submission_plan(root)

    with pytest.raises(FileExistsError, match="must not exist"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=False,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )


def test_formal_root_rejects_a_reparse_point_that_resolves_outside_repo(
    tmp_path, monkeypatch
):
    from experiments.ir_crpc_submission_v2 import validate_formal_output_root

    repo = tmp_path / "repo"
    logical_exp = repo / "exp"
    root = repo / FORMAL_OUTPUT_RELATIVE
    outside = tmp_path / "outside"
    real_resolve = Path.resolve

    def resolve_reparse(path, *args, **kwargs):
        try:
            relative = path.absolute().relative_to(logical_exp.absolute())
        except ValueError:
            return real_resolve(path, *args, **kwargs)
        return real_resolve(outside / relative, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_reparse)
    with pytest.raises(ValueError, match="physical|repository"):
        validate_formal_output_root(root, repo_root=repo)
    assert not root.exists()


def test_tiny_execution_preflight_hashes_every_declared_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "config.yaml"
    data = repo / "data.csv"
    runtime = repo / "runner.py"
    config.write_bytes(b"config")
    data.write_bytes(b"dataset")
    runtime.write_bytes(b"runtime")
    plan = build_submission_plan(repo / FORMAL_OUTPUT_RELATIVE)

    preflight = build_execution_preflight(
        repo,
        plan,
        git_probe=lambda _repo: {"revision": "abc", "dirty_paths": []},
        configuration_paths=[config],
        dataset_paths=[data],
        runtime_paths=[runtime],
    )

    assert preflight["git"] == {"revision": "abc", "dirty_paths": []}
    for group, path, content in (
        ("configuration_files", config, b"config"),
        ("dataset_files", data, b"dataset"),
        ("runtime_files", runtime, b"runtime"),
    ):
        assert preflight[group] == [
            {
                "path": path.relative_to(repo).as_posix(),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ]
    assert len(preflight["preflight_sha256"]) == 64


def test_default_execution_preflight_freezes_traditional_runtime_sources():
    """Catch an execution preflight that can launch unfrozen references."""
    repository = Path(__file__).resolve().parents[1]
    plan = build_submission_plan(repository / FORMAL_OUTPUT_RELATIVE)

    preflight = build_execution_preflight(
        repository,
        plan,
        git_probe=lambda _repo: {"revision": "abc", "dirty_paths": []},
    )

    runtime = {item["path"]: item for item in preflight["runtime_files"]}
    for relative in (
        "experiments/fixed_test_protocol.py",
        "environments/var_voltage_control/voltage_control_env.py",
        "experiments/run_q1_fixed_test_traditional_baselines.py",
        "experiments/traditional_action_limits.py",
    ):
        source = repository / relative
        assert runtime[relative] == {
            "path": relative,
            "size": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }


def test_resume_rejects_schema_or_preflight_changes(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["schema_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="resume metadata schema"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=True,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )


def test_resume_rejects_non_v2_completion_manifest(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    run = plan["runs"][0]
    manifest_path = root / run["output_dir"] / "completion_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["completed_runs"][0]["completion_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest schema"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=True,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )


def test_live_completion_rejects_non_v2_manifest(tmp_path):
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    invalid_id = plan["runs"][0]["run_id"]

    def old_manifest(payload):
        result = _write_completed_run(payload)
        if payload["run"]["run_id"] == invalid_id:
            path = Path(payload["run"]["output_dir"]) / "completion_manifest.json"
            manifest = json.loads(path.read_text("utf-8"))
            manifest["schema_version"] = 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=old_manifest,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    assert result["failed"] == 1
    assert invalid_id not in {
        entry["run_id"]
        for entry in json.loads(
            (root / "resume_metadata.json").read_text("utf-8")
        )["completed_runs"]
    }


def test_preflight_change_or_dirty_state_is_rejected_before_formal_writes(tmp_path):
    dirty_repo = tmp_path / "dirty_repo"
    dirty_root = dirty_repo / FORMAL_OUTPUT_RELATIVE
    dirty_plan = build_submission_plan(dirty_root)

    def dirty_preflight(_repo, plan):
        payload = _fake_preflight(_repo, plan)
        payload["git"]["dirty_paths"] = ["experiments/changed.py"]
        return payload

    with pytest.raises(ValueError, match="clean Git worktree"):
        execute_submission_plan(
            dirty_plan,
            output_root=dirty_root,
            workers=1,
            resume=False,
            repo_root=dirty_repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=dirty_preflight,
        )
    assert not dirty_root.exists()

    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    execute_submission_plan(
        plan,
        output_root=root,
        workers=1,
        resume=False,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )

    def changed_preflight(_repo, requested):
        payload = _fake_preflight(_repo, requested)
        payload["git"]["revision"] = "changed456"
        return payload

    with pytest.raises(ValueError, match="preflight has changed"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=True,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=changed_preflight,
        )


def test_failed_atomic_publish_removes_staging_and_never_exposes_root(
    tmp_path, monkeypatch
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    real_replace = dispatcher.os.replace

    def fail_publication(source, destination):
        if Path(destination) == root:
            raise FileExistsError("simulated initialization race")
        return real_replace(source, destination)

    monkeypatch.setattr(dispatcher.os, "replace", fail_publication)
    with pytest.raises(FileExistsError, match="race"):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=False,
            repo_root=repo,
            worker_fn=_write_completed_run,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )

    assert not root.exists()
    assert list(root.parent.glob(f".{root.name}.staging-*")) == []
    assert not (root.parent / f".{root.name}.init-lock").exists()


@pytest.mark.parametrize("escaped_component", ["run_dir", "failed_attempts"])
def test_partial_archive_rejects_run_or_archive_reparse_escape_before_move(
    tmp_path, monkeypatch, escaped_component
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    root = (tmp_path / "formal").resolve()
    run = {
        "run_id": "full_matd3__case33_3min_final__seed0",
        "output_dir": "full_matd3/case33_3min_final/seed_0",
    }
    logical_run = root / run["output_dir"]
    logical_run.mkdir(parents=True)
    (logical_run / "partial.txt").write_text("partial")
    logical_archive = root / "_failed_attempts"
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    real_resolve = Path.resolve

    def escaped_resolve(path, *args, **kwargs):
        absolute = path.absolute()
        if escaped_component == "run_dir" and absolute == logical_run.absolute():
            return outside / "escaped_run"
        if escaped_component == "failed_attempts":
            try:
                relative = absolute.relative_to(logical_archive.absolute())
            except ValueError:
                pass
            else:
                return outside / "escaped_archive" / relative
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    monkeypatch.setattr(
        dispatcher.os,
        "replace",
        lambda *_args: pytest.fail("escape must be rejected before move"),
    )
    with pytest.raises(ValueError, match="formal root"):
        dispatcher._archive_incomplete_run(root, run)
    assert (logical_run / "partial.txt").is_file()


def test_cuda_oom_stops_new_submissions_and_returns_resume_instruction(tmp_path):
    _ImmediateExecutor.instances.clear()
    _ImmediateExecutor.max_workers_seen.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    first_id = plan["runs"][0]["run_id"]

    def oom_first(payload):
        if payload["run"]["run_id"] == first_id:
            raise RuntimeError("CUDA out of memory")
        return _write_completed_run(payload)

    result = execute_submission_plan(
        plan,
        output_root=root,
        workers=3,
        resume=False,
        repo_root=repo,
        worker_fn=oom_first,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )

    assert result["state"] == "paused_cuda_oom"
    assert result["failed"] == 1
    assert result["completed"] == 2
    assert result["pending"] == 34
    assert "--resume --workers 2" in result["resume_hint"]
    assert len(_ImmediateExecutor.instances) == 3
    assert all(instance.submit_count == 1 for instance in _ImmediateExecutor.instances)
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert metadata["execution_state"] == "paused_cuda_oom"
    assert len(metadata["pending_run_ids"]) == 34


def test_keyboard_interrupt_terminates_active_worker_and_records_state(tmp_path):
    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)

    def interrupted(_payload):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=2,
            resume=False,
            repo_root=repo,
            worker_fn=interrupted,
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
        )

    assert all(instance.terminated for instance in _ImmediateExecutor.instances)
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert metadata["execution_state"] == "interrupted"
    assert len(metadata["pending_run_ids"]) == 36
    assert len(set(metadata["pending_run_ids"])) == 36
    assert not (root.parent / f".{root.name}.execution-lock").exists()

    resumed = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=True,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    assert resumed == {"completed": 36, "failed": 0, "skipped": 0}


@pytest.mark.parametrize(
    "interrupt_phase", ["after_executor_created", "after_submit"]
)
def test_submission_interrupt_is_transactional_and_preserves_all_pending_runs(
    tmp_path, interrupt_phase
):
    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    hook_calls: list[str] = []

    def interrupt_once(phase, _run, _executor, _future):
        hook_calls.append(phase)
        if phase == interrupt_phase:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute_submission_plan(
            plan,
            output_root=root,
            workers=1,
            resume=False,
            repo_root=repo,
            worker_fn=lambda _payload: {"status": "unused"},
            executor_factory=_ImmediateExecutor,
            git_snapshot_fn=_fake_git_snapshot,
            preflight_fn=_fake_preflight,
            submission_hook=interrupt_once,
        )

    assert interrupt_phase in hook_calls
    assert len(_ImmediateExecutor.instances) == 1
    assert _ImmediateExecutor.instances[0].terminated is True
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    expected_ids = [run["run_id"] for run in plan["runs"]]
    assert metadata["execution_state"] == "interrupted"
    assert metadata["pending_run_ids"] == expected_ids
    assert metadata["completed_runs"] == []
    assert not (root.parent / f".{root.name}.execution-lock").exists()


def _first_submitted_future_only(futures, *, return_when):
    del return_when
    submitted = _ImmediateExecutor.instances[0].futures[0]
    candidates = set(futures)
    assert submitted in candidates
    return {submitted}, candidates - {submitted}


def _executor_has_no_live_work(executor: _ImmediateExecutor) -> bool:
    return executor.terminated or any(wait for wait, _cancel in executor.shutdown_calls)


def test_record_failure_write_error_reaps_other_active_worker_before_unlock_and_resume(
    tmp_path, monkeypatch
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    first_id = plan["runs"][0]["run_id"]
    real_write_json = dispatcher._write_json_atomic

    def fail_error_record(path, payload):
        if Path(path).name == "error.json":
            raise OSError("synthetic error-record write failure")
        return real_write_json(path, payload)

    def worker(payload):
        if payload["run"]["run_id"] == first_id:
            raise RuntimeError("synthetic worker failure")
        return _write_completed_run(payload)

    with monkeypatch.context() as fault:
        fault.setattr(dispatcher, "wait", _first_submitted_future_only)
        fault.setattr(dispatcher, "_write_json_atomic", fail_error_record)
        with pytest.raises(OSError, match="error-record"):
            execute_submission_plan(
                plan,
                output_root=root,
                workers=2,
                resume=False,
                repo_root=repo,
                worker_fn=worker,
                executor_factory=_ImmediateExecutor,
                git_snapshot_fn=_fake_git_snapshot,
                preflight_fn=_fake_preflight,
            )

    aborted_executors = list(_ImmediateExecutor.instances)
    assert len(aborted_executors) == 2
    assert all(_executor_has_no_live_work(item) for item in aborted_executors)
    assert not (root.parent / f".{root.name}.execution-lock").exists()
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert metadata["execution_state"] == "aborted_exception"
    assert metadata["pending_run_ids"] == [run["run_id"] for run in plan["runs"]]

    resumed = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=True,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    assert resumed == {"completed": 36, "failed": 0, "skipped": 0}


def test_metadata_write_error_still_reaps_active_worker_and_preserves_resume_boundary(
    tmp_path, monkeypatch
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    real_write_json = dispatcher._write_json_atomic

    def fail_live_metadata(path, payload):
        if Path(path) == root / "resume_metadata.json":
            raise OSError("synthetic resume-metadata write failure")
        return real_write_json(path, payload)

    with monkeypatch.context() as fault:
        fault.setattr(dispatcher, "wait", _first_submitted_future_only)
        fault.setattr(dispatcher, "_write_json_atomic", fail_live_metadata)
        with pytest.raises(OSError, match="resume-metadata") as raised:
            execute_submission_plan(
                plan,
                output_root=root,
                workers=2,
                resume=False,
                repo_root=repo,
                worker_fn=_write_completed_run,
                executor_factory=_ImmediateExecutor,
                git_snapshot_fn=_fake_git_snapshot,
                preflight_fn=_fake_preflight,
            )

    aborted_executors = list(_ImmediateExecutor.instances)
    assert len(aborted_executors) == 2
    assert all(_executor_has_no_live_work(item) for item in aborted_executors)
    assert "resume-metadata" in " ".join(
        getattr(raised.value, "dispatcher_cleanup_errors", ())
    )
    assert not (root.parent / f".{root.name}.execution-lock").exists()
    stored = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert stored["pending_run_ids"] == [run["run_id"] for run in plan["runs"]]

    resumed = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=True,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    assert resumed == {"completed": 36, "failed": 0, "skipped": 0}


def test_system_exit_reaps_all_active_workers_before_unlock_and_can_resume(
    tmp_path, monkeypatch
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    _ImmediateExecutor.instances.clear()
    repo = tmp_path / "repo"
    root = repo / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    first_id = plan["runs"][0]["run_id"]

    def worker(payload):
        if payload["run"]["run_id"] == first_id:
            raise SystemExit("synthetic system exit")
        return _write_completed_run(payload)

    with monkeypatch.context() as fault:
        fault.setattr(dispatcher, "wait", _first_submitted_future_only)
        with pytest.raises(SystemExit, match="synthetic system exit"):
            execute_submission_plan(
                plan,
                output_root=root,
                workers=2,
                resume=False,
                repo_root=repo,
                worker_fn=worker,
                executor_factory=_ImmediateExecutor,
                git_snapshot_fn=_fake_git_snapshot,
                preflight_fn=_fake_preflight,
            )

    aborted_executors = list(_ImmediateExecutor.instances)
    assert len(aborted_executors) == 2
    assert all(_executor_has_no_live_work(item) for item in aborted_executors)
    assert not (root.parent / f".{root.name}.execution-lock").exists()
    metadata = json.loads((root / "resume_metadata.json").read_text("utf-8"))
    assert metadata["execution_state"] == "aborted_exception"
    assert metadata["pending_run_ids"] == [run["run_id"] for run in plan["runs"]]

    resumed = execute_submission_plan(
        plan,
        output_root=root,
        workers=2,
        resume=True,
        repo_root=repo,
        worker_fn=_write_completed_run,
        executor_factory=_ImmediateExecutor,
        git_snapshot_fn=_fake_git_snapshot,
        preflight_fn=_fake_preflight,
    )
    assert resumed == {"completed": 36, "failed": 0, "skipped": 0}


def test_forced_process_cleanup_uses_terminate_join_kill_join_order():
    import experiments.ir_crpc_submission_v2 as dispatcher

    events: list[str] = []

    class Process:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("terminate")

        def join(self, timeout):
            assert timeout == 2
            events.append("join")

        def kill(self):
            events.append("kill")
            self.alive = False

    class Executor:
        def __init__(self):
            self.process = Process()
            self._processes = {1: self.process}

        def shutdown(self, wait, cancel_futures):
            assert wait is False
            assert cancel_futures is True
            events.append("shutdown")

    executor = Executor()
    dispatcher._shutdown_isolated_executor(executor, terminate=True)

    assert events == ["terminate", "join", "kill", "join", "shutdown"]
    assert executor.process.is_alive() is False


def test_unreaped_process_failure_is_fatal_and_retains_execution_lock(tmp_path):
    import experiments.ir_crpc_submission_v2 as dispatcher

    class Process:
        def is_alive(self):
            return True

        def terminate(self):
            return None

        def join(self, timeout):
            assert timeout == 2

        def kill(self):
            return None

    class Executor:
        _processes = {1: Process()}

        def shutdown(self, wait, cancel_futures):
            return None

    root = tmp_path / "exp" / "q1_submission_v2_strict"
    lock = root.parent / f".{root.name}.execution-lock"

    with pytest.raises(RuntimeError, match="fatal.*alive") as raised:
        with dispatcher._formal_execution_lock(root):
            dispatcher._shutdown_isolated_executor(Executor(), terminate=True)

    assert getattr(raised.value, "retain_execution_lock", False) is True
    assert lock.is_dir()


def test_lock_release_error_is_attached_without_overriding_original_exception(
    tmp_path, monkeypatch
):
    import experiments.ir_crpc_submission_v2 as dispatcher

    root = tmp_path / "exp" / "q1_submission_v2_strict"
    lock = root.parent / f".{root.name}.execution-lock"
    real_rmdir = Path.rmdir

    def fail_lock_release(path):
        if path == lock:
            raise OSError("synthetic lock release failure")
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_lock_release)
    with pytest.raises(SystemExit, match="original system exit") as raised:
        with dispatcher._formal_execution_lock(root):
            raise SystemExit("original system exit")

    assert "lock release" in " ".join(
        getattr(raised.value, "dispatcher_cleanup_errors", ())
    )
    assert lock.is_dir()
