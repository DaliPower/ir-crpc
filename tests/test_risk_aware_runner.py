from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from experiments.run_risk_aware_multiseed import (
    _evaluate_fixed_days,
    build_day_split,
    build_run_plan,
    merge_existing_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakePolicy:
    def init_hidden(self):
        return None


class _FakeBehaviourNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(1)
        self.policy_dicts = [_FakePolicy()]

    def get_actions(self, *args, **kwargs):
        self.batch_norm(torch.zeros((2, 1)))
        action = torch.zeros((1, 1, 1))
        return action, None, None, None, None


class _FakeEvalEnv:
    def __init__(self):
        self.raise_on_step = False

    def manual_reset(self, day, start_hour, value):
        return [[0.0]], None

    def get_avail_actions(self):
        return [[1.0]]

    def step(self, actual, add_noise=False):
        if self.raise_on_step:
            raise RuntimeError("forced evaluation failure")
        return 0.0, True, {}

    def get_obs(self):
        return [[0.0]]


@pytest.fixture
def fake_eval_inputs(monkeypatch):
    aggregate_module = types.ModuleType("experiments.run_q1_ml_compare")
    aggregate_module.aggregate_info_steps = lambda rows: {
        "mean_reward": sum(row["reward"] for row in rows) / len(rows)
    }
    monkeypatch.setitem(
        sys.modules, "experiments.run_q1_ml_compare", aggregate_module
    )
    env = _FakeEvalEnv()
    net = _FakeBehaviourNet()
    args = type(
        "FakeArgs",
        (),
        {
            "cuda": False,
            "agent_num": 1,
            "obs_size": 1,
            "continuous": True,
            "action_bias": 0.0,
            "action_scale": 1.0,
        },
    )()
    return net, {
        "env": env,
        "behaviour_net": net,
        "args": args,
        "days": [0],
        "max_steps": 1,
        "start_hour": 6,
    }


def test_fixed_evaluation_restores_training_mode_after_success(fake_eval_inputs):
    net, kwargs = fake_eval_inputs
    net.train()
    _evaluate_fixed_days(**kwargs)
    assert net.training is True


def test_fixed_evaluation_keeps_batch_norm_ready_for_training(fake_eval_inputs):
    net, kwargs = fake_eval_inputs
    net.train()
    batches_before_validation = net.batch_norm.num_batches_tracked.item()

    _evaluate_fixed_days(**kwargs)
    net.get_actions()

    assert net.batch_norm.num_batches_tracked.item() == batches_before_validation + 1


def test_fixed_evaluation_restores_training_mode_after_exception(fake_eval_inputs):
    net, kwargs = fake_eval_inputs
    kwargs["env"].raise_on_step = True
    net.train()
    with pytest.raises(RuntimeError, match="forced evaluation failure"):
        _evaluate_fixed_days(**kwargs)
    assert net.training is True


def test_day_split_excludes_guard_validation_and_test_days_from_training():
    split = build_day_split(
        total_days=1097,
        validation_start_day=1000,
        validation_days=30,
        test_start_day=1030,
        test_days=30,
        guard_days=1,
    )

    assert split["guard_days"] == [999]
    assert split["validation_days"] == list(range(1000, 1030))
    assert split["test_days"] == list(range(1030, 1060))
    assert split["unused_days"] == list(range(1060, 1097))
    assert split["training_days"] == list(range(999))
    assert not set(split["training_days"]) & set(split["validation_days"])
    assert not set(split["training_days"]) & set(split["test_days"])


def test_day_split_rejects_noncanonical_test_start_day():
    with pytest.raises(ValueError, match="submission boundaries"):
        build_day_split(
            total_days=1097,
            validation_start_day=1000,
            validation_days=30,
            test_start_day=1029,
            test_days=30,
            guard_days=1,
        )


def test_run_plan_isolates_every_algorithm_and_seed(tmp_path):
    plan = build_run_plan(
        scenario="case33_3min_final",
        algorithms=["matd3", "maddpg"],
        seeds=[0, 1],
        output_root=tmp_path,
    )

    paths = [row["output_dir"] for row in plan]
    assert len(paths) == len(set(paths)) == 4
    assert paths[0].endswith("case33_3min_final/matd3/seed_0")
    assert paths[-1].endswith("case33_3min_final/maddpg/seed_1")


def test_extension_plan_merges_only_when_protocol_is_identical(tmp_path):
    path = tmp_path / "run_plan.json"
    base = {
        "total_days": 1100,
        "split": {"training_days": [0], "validation_days": [1], "test_days": [2]},
        "training_protocol": {"train_episodes": 1000},
        "reward_weights": {"q_weight": 0.1},
        "runs": build_run_plan(
            scenario="case33_3min_final",
            algorithms=["matd3", "maddpg"],
            seeds=[0, 1, 2],
            output_root=tmp_path,
        ),
    }
    path.write_text(json.dumps(base), encoding="utf-8")
    extension = {
        **base,
        "runs": build_run_plan(
            scenario="case33_3min_final",
            algorithms=["mappo", "facmaddpg"],
            seeds=[0, 1, 2],
            output_root=tmp_path,
        ),
    }

    merged = merge_existing_plan(path, extension)

    assert len(merged["runs"]) == 12
    assert {row["algorithm"] for row in merged["runs"]} == {
        "matd3",
        "maddpg",
        "mappo",
        "facmaddpg",
    }


def test_dry_run_writes_five_seed_plan_without_loading_training_stack(tmp_path):
    output_root = tmp_path / "q1_submission_v2_strict"
    command = [
        sys.executable,
        "experiments/run_risk_aware_multiseed.py",
        "--scenario",
        "case33_3min_final",
        "--algorithms",
        "matd3",
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--save-path",
        str(output_root),
        "--dry-run",
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert len(plan["runs"]) == 5
    assert plan["reward_weights"]["violation_weight"] == 100.0
    assert not output_root.exists()


def test_dry_run_records_full_training_and_evaluation_protocol(tmp_path):
    output_root = tmp_path / "q1_submission_v2_strict"
    command = [
        sys.executable,
        "experiments/run_risk_aware_multiseed.py",
        "--scenario",
        "case33_3min_final",
        "--algorithms",
        "matd3",
        "maddpg",
        "--seeds",
        "0",
        "1",
        "2",
        "--save-path",
        str(output_root),
        "--train-episodes",
        "1000",
        "--eval-freq",
        "50",
        "--validation-days",
        "30",
        "--test-start-day",
        "1030",
        "--test-days",
        "30",
        "--dry-run",
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["training_protocol"] == {
        "train_episodes": 1000,
        "eval_freq": 50,
        "max_steps": 240,
        "eval_start_hour": 6,
        "checkpoint_selection": "minimum validation violation rate with lexicographic tie-breaks",
        "final_report": "single blind evaluation of the selected checkpoint on fixed test days",
    }
    assert plan["split"]["validation_days"] == list(range(1000, 1030))
    assert plan["split"]["test_days"] == list(range(1030, 1060))
    assert plan["split"]["training_days"] == list(range(999))
    assert plan["split"]["guard_days"] == [999]
    assert plan["split"]["unused_days"] == list(range(1060, 1097))
    assert not output_root.exists()
