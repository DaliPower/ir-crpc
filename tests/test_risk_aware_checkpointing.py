from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiments.risk_aware_checkpointing as checkpointing_module
from experiments.risk_aware_checkpointing import CheckpointManager


def metrics(violation, deviation=0.01, q_loss=0.2):
    return {
        "percentage_of_v_out_of_control": violation,
        "average_voltage_deviation": deviation,
        "q_loss": q_loss,
    }


def test_manager_replaces_best_model_only_when_validation_rank_improves(tmp_path):
    manager = CheckpointManager(tmp_path)

    assert manager.consider({"weight": torch.tensor([1.0])}, metrics(0.02), 50)
    assert not manager.consider({"weight": torch.tensor([2.0])}, metrics(0.03), 100)
    assert manager.consider({"weight": torch.tensor([3.0])}, metrics(0.01), 150)

    checkpoint = torch.load(manager.finalize(), map_location="cpu")
    assert checkpoint["episode"] == 150
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(3.0)


def test_manager_records_every_validation_decision(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.consider({"weight": torch.tensor([1.0])}, metrics(0.02), 50)
    manager.consider({"weight": torch.tensor([2.0])}, metrics(0.03), 100)

    history = json.loads((tmp_path / "validation_history.json").read_text("utf-8"))

    assert [row["selected"] for row in history] == [True, False]
    assert [row["episode"] for row in history] == [50, 100]
    assert history[0]["validation_metrics"]["percentage_of_v_out_of_control"] == 0.02


def test_finalize_requires_at_least_one_validation_record(tmp_path):
    manager = CheckpointManager(tmp_path)

    with pytest.raises(RuntimeError, match="No validation checkpoint"):
        manager.finalize()


def test_manager_saves_best_and_terminal_checkpoints_independently(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.consider({"weight": torch.tensor([1.0])}, metrics(0.01), 50)

    final_path = manager.save_final({"weight": torch.tensor([2.0])}, 1000)

    best = torch.load(manager.finalize(), map_location="cpu")
    final = torch.load(final_path, map_location="cpu")
    assert best["episode"] == 50
    assert best["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert final["episode"] == 1000
    assert final["model_state_dict"]["weight"].item() == pytest.approx(2.0)
    assert "validation_metrics" not in final


def test_final_checkpoint_overwrite_is_atomic_and_leaves_no_temporary_file(tmp_path):
    manager = CheckpointManager(tmp_path)
    first_path = manager.save_final({"weight": torch.tensor([1.0])}, 500)

    second_path = manager.save_final({"weight": torch.tensor([2.0])}, 1000)

    assert first_path == second_path == tmp_path / "final_model.pt"
    final = torch.load(second_path, map_location="cpu")
    assert final["episode"] == 1000
    assert final["model_state_dict"]["weight"].item() == pytest.approx(2.0)
    assert not list(tmp_path.glob("final_model.pt.tmp*"))


def test_failed_final_checkpoint_promotion_preserves_old_file_and_cleans_tmp(
    tmp_path, monkeypatch
):
    manager = CheckpointManager(tmp_path)
    final_path = manager.save_final({"weight": torch.tensor([1.0])}, 500)

    def fail_replace(_source, _destination):
        raise OSError("forced final checkpoint promotion failure")

    monkeypatch.setattr(checkpointing_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced final checkpoint promotion failure"):
        manager.save_final({"weight": torch.tensor([2.0])}, 1000)

    retained = torch.load(final_path, map_location="cpu")
    assert retained["episode"] == 500
    assert retained["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert not list(tmp_path.glob("final_model.pt.tmp*"))


def _install_runner_fakes(monkeypatch, events=None):
    events = events if events is not None else []

    class FakeWriter:
        def __init__(self, _path):
            self.closed = False

        def close(self):
            self.closed = True
            events.append("logger_close")

    class FakeEnvironment:
        def close(self):
            events.append("environment_close")

    class FakeBehaviourNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.episode = 0

        def train_process(self, stat, _trainer):
            self.episode += 1
            with torch.no_grad():
                self.weight.fill_(self.episode)
            stat["episode"] = float(self.episode)

    class FakeTrainer:
        def __init__(self, args, _model, _env, _logger):
            self.args = args
            self.behaviour_net = FakeBehaviourNet()

        def logging(self, _stat):
            pass

    tensorboard_module = types.ModuleType("tensorboardX")
    tensorboard_module.SummaryWriter = FakeWriter
    compare_module = types.ModuleType("experiments.run_q1_ml_compare")
    compare_module.load_env_config = lambda *_args, **_kwargs: {}
    compare_module.load_alg_config = lambda *_args, **_kwargs: {}
    compare_module.make_args = lambda *_args, **_kwargs: SimpleNamespace(cuda=False)
    compare_module.make_env = lambda _config: FakeEnvironment()
    registry_module = types.ModuleType("models.model_registry")
    registry_module.Model = {"matd3": object, "maddpg": object}
    trainer_module = types.ModuleType("utilities.trainer")
    trainer_module.PGTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "tensorboardX", tensorboard_module)
    monkeypatch.setitem(sys.modules, "experiments.run_q1_ml_compare", compare_module)
    monkeypatch.setitem(sys.modules, "models.model_registry", registry_module)
    monkeypatch.setitem(sys.modules, "utilities.trainer", trainer_module)


def _lightweight_runner_call(
    tmp_path,
    monkeypatch,
    *,
    output_dir=None,
    run_overrides=None,
    events=None,
):
    import experiments.run_risk_aware_multiseed as runner

    _install_runner_fakes(monkeypatch, events=events)
    validation_values = iter([0.1, 0.2])

    def fake_evaluate(_env, net, _args, days, _max_steps, _start_hour):
        if list(days) == list(range(1000, 1030)):
            value = next(validation_values)
        else:
            value = float(net.weight.detach().cpu().item()) / 100.0
        return SimpleNamespace(
            summary={
                "percentage_of_v_out_of_control": value,
                "average_voltage_deviation": 0.01,
                "q_loss": 0.2,
            }
        )

    artifact_names = (
        "fixed_test_step_metrics.csv",
        "fixed_test_daily_metrics.csv",
        "fixed_test_node_voltages.npz",
        "fixed_test_trajectory_schema.json",
        "fixed_test_summary.json",
    )

    def fake_consume(result, *, output_dir, stem, retain_artifacts):
        assert stem == "fixed_test"
        if retain_artifacts:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            for name in artifact_names:
                (output_dir / name).write_bytes(b"artifact")
        return dict(result.summary)

    monkeypatch.setattr(runner, "_evaluate_fixed_result", fake_evaluate)
    monkeypatch.setattr(runner, "consume_fixed_test_result", fake_consume)
    output_dir = output_dir or (
        tmp_path
        / "exp"
        / "q1_submission_v2_strict"
        / "case33_3min_final"
        / "matd3"
        / "seed_0"
    )
    run = {
        "scenario": "case33_3min_final",
        "algorithm": "matd3",
        "seed": 0,
        "output_dir": str(output_dir),
        "evaluate_terminal": False,
    }
    run.update(run_overrides or {})
    args_cli = SimpleNamespace(
        pv_scale=1.5,
        voltage_weight=1.0,
        violation_weight=0.0,
        q_weight=0.1,
        train_episodes=2,
        eval_freq=1,
        max_steps=240,
        eval_start_hour=6,
        validation_only=False,
        retain_validation_trajectories=False,
    )
    split = {
        "training_days": list(range(999)),
        "validation_days": list(range(1000, 1030)),
        "test_days": list(range(1030, 1060)),
    }
    return runner._run_single(run, split, args_cli), Path(output_dir)


@pytest.mark.parametrize(
    ("algorithm", "evaluate_terminal", "expected_test_weights"),
    [
        ("matd3", False, [1.0]),
        ("maddpg", True, [1.0, 2.0]),
    ],
)
def test_runner_saves_terminal_before_loading_best_and_obeys_explicit_flag(
    tmp_path,
    monkeypatch,
    algorithm,
    evaluate_terminal,
    expected_test_weights,
):
    import experiments.run_risk_aware_multiseed as runner

    _install_runner_fakes(monkeypatch)
    observed_weights = []
    validation_values = iter([0.1, 0.2])

    def fake_evaluate(_env, net, _args, days, _max_steps, _start_hour):
        weight = float(net.weight.detach().cpu().item())
        if list(days) == list(range(1000, 1030)):
            value = next(validation_values)
        else:
            observed_weights.append(weight)
            value = weight / 100.0
        return SimpleNamespace(
            summary={
                "percentage_of_v_out_of_control": value,
                "average_voltage_deviation": 0.01,
                "q_loss": 0.2,
            }
        )

    artifact_names = (
        "fixed_test_step_metrics.csv",
        "fixed_test_daily_metrics.csv",
        "fixed_test_node_voltages.npz",
        "fixed_test_trajectory_schema.json",
        "fixed_test_summary.json",
    )

    def fake_consume(result, *, output_dir, stem, retain_artifacts):
        assert stem == "fixed_test"
        if retain_artifacts:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            for name in artifact_names:
                (output_dir / name).write_bytes(b"artifact")
        return dict(result.summary)

    monkeypatch.setattr(runner, "_evaluate_fixed_result", fake_evaluate)
    monkeypatch.setattr(runner, "consume_fixed_test_result", fake_consume)
    output_dir = (
        tmp_path
        / "exp"
        / "q1_submission_v2_strict"
        / "case33_3min_final"
        / algorithm
        / "seed_0"
    )
    args_cli = SimpleNamespace(
        pv_scale=1.5,
        voltage_weight=1.0,
        violation_weight=0.0,
        q_weight=0.1,
        train_episodes=2,
        eval_freq=1,
        max_steps=240,
        eval_start_hour=6,
        validation_only=False,
        retain_validation_trajectories=False,
    )
    split = {
        "training_days": list(range(999)),
        "validation_days": list(range(1000, 1030)),
        "test_days": list(range(1030, 1060)),
    }

    result = runner._run_single(
        {
            "scenario": "case33_3min_final",
            "algorithm": algorithm,
            "seed": 0,
            "output_dir": str(output_dir),
            "evaluate_terminal": evaluate_terminal,
        },
        split,
        args_cli,
    )

    best = torch.load(output_dir / "best_model.pt", map_location="cpu")
    final = torch.load(output_dir / "final_model.pt", map_location="cpu")
    assert best["episode"] == 1
    assert best["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert final["episode"] == 2
    assert final["model_state_dict"]["weight"].item() == pytest.approx(2.0)
    assert observed_weights == expected_test_weights
    assert (output_dir / "test_best").is_dir()
    assert (output_dir / "test_final").is_dir() is evaluate_terminal
    assert result["evaluate_terminal"] is evaluate_terminal
    assert (output_dir / "provenance.json").is_file()


def test_provenance_failure_removes_stale_seed_completion_marker(
    tmp_path, monkeypatch
):
    import experiments.submission_provenance as provenance

    output_dir = (
        tmp_path
        / "exp"
        / "q1_submission_v2_strict"
        / "case33_3min_final"
        / "matd3"
        / "seed_0"
    )
    output_dir.mkdir(parents=True)
    marker = output_dir / "seed_result.json"
    marker.write_text('{"stale": true}', encoding="utf-8")

    def fail_provenance(_manifest, _output_path):
        raise OSError("forced provenance failure")

    monkeypatch.setattr(provenance, "write_provenance_atomic", fail_provenance)
    with pytest.raises(OSError, match="forced provenance failure"):
        _lightweight_runner_call(
            tmp_path, monkeypatch, output_dir=output_dir
        )

    assert not marker.exists()


def test_runner_captures_git_before_output_and_passes_start_snapshot(
    tmp_path, monkeypatch
):
    import experiments.submission_provenance as provenance

    output_dir = (
        tmp_path
        / "exp"
        / "q1_submission_v2_strict"
        / "case33_3min_final"
        / "matd3"
        / "seed_0"
    )
    observed = {}
    real_build = provenance.build_provenance

    def fake_capture(_repo_root):
        observed["captured_before_output"] = not output_dir.exists()
        return {"revision": "start123", "dirty": False}

    def recording_build(*args, **kwargs):
        observed["git_revision"] = kwargs.get("git_revision")
        observed["git_dirty"] = kwargs.get("git_dirty")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(provenance, "capture_git_snapshot", fake_capture, raising=False)
    monkeypatch.setattr(provenance, "build_provenance", recording_build)

    _, completed_dir = _lightweight_runner_call(
        tmp_path, monkeypatch, output_dir=output_dir
    )

    assert observed == {
        "captured_before_output": True,
        "git_revision": "start123",
        "git_dirty": False,
    }
    manifest = json.loads((completed_dir / "provenance.json").read_text("utf-8"))
    assert manifest["git"] == {"revision": "start123", "dirty": False}
    assert (completed_dir / "seed_result.json").is_file()


def test_seed_result_is_the_last_atomic_completion_action(tmp_path, monkeypatch):
    import experiments.submission_provenance as provenance

    events = []
    real_json_writer = provenance.write_json_atomic

    def provenance_writer(manifest, output_path):
        events.append("provenance")
        Path(output_path).write_text(json.dumps(manifest), encoding="utf-8")
        return Path(output_path)

    def completion_writer(result, output_path):
        events.append("seed_result")
        return real_json_writer(result, output_path)

    monkeypatch.setattr(provenance, "write_provenance_atomic", provenance_writer)
    monkeypatch.setattr(provenance, "write_json_atomic", completion_writer)

    _, output_dir = _lightweight_runner_call(
        tmp_path, monkeypatch, events=events
    )

    assert events[-1] == "seed_result"
    assert events.index("provenance") < events.index("seed_result")
    assert events.index("logger_close") < events.index("seed_result")
    assert max(
        index for index, event in enumerate(events) if event == "environment_close"
    ) < events.index("seed_result")
    assert (output_dir / "seed_result.json").is_file()
