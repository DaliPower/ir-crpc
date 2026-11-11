from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.fixed_test_protocol as fixed_protocol
from experiments.fixed_test_protocol import (
    evaluate_controller_fixed_days,
    evaluate_policy_fixed_days,
    write_fixed_test_artifacts,
)
from experiments.run_risk_aware_multiseed import (
    _evaluate_fixed_result,
    _run_single,
    consume_fixed_test_result,
    parse_args,
    validate_submission_preflight,
)

TEST_DAYS = list(range(1030, 1060))
ROOT = Path(__file__).resolve().parents[1]


def test_force_zero_reset_action_replaces_immutable_generic_dict_temporarily():
    generic_dict = namedtuple("GenericDict", ("reset_action", "seed"))
    original_args = generic_dict(reset_action=True, seed=7)
    env = SimpleNamespace(args=original_args)

    with fixed_protocol._force_zero_reset_action(env):
        assert env.args is not original_args
        assert env.args.reset_action is False
        assert env.args.seed == 7

    assert env.args is original_args
    assert env.args == generic_dict(reset_action=True, seed=7)


def test_force_zero_reset_action_restores_immutable_args_after_exception():
    generic_dict = namedtuple("GenericDict", ("reset_action", "seed"))
    original_args = generic_dict(reset_action=True, seed=11)
    env = SimpleNamespace(args=original_args)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with fixed_protocol._force_zero_reset_action(env):
            assert env.args.reset_action is False
            raise RuntimeError("evaluation failed")

    assert env.args is original_args
    assert env.args == generic_dict(reset_action=True, seed=11)


@pytest.mark.parametrize(
    "original_args",
    [
        pytest.param(SimpleNamespace(reset_action=True, seed=13), id="namespace"),
        pytest.param({"reset_action": True, "seed": 13}, id="dictionary"),
    ],
)
def test_force_zero_reset_action_restores_mutable_args_exactly(original_args):
    env = SimpleNamespace(args=original_args)

    with fixed_protocol._force_zero_reset_action(env):
        assert env.args is original_args
        if isinstance(env.args, dict):
            assert env.args["reset_action"] is False
        else:
            assert env.args.reset_action is False

    assert env.args is original_args
    if isinstance(env.args, dict):
        assert env.args == {"reset_action": True, "seed": 13}
    else:
        assert vars(env.args) == {"reset_action": True, "seed": 13}


@pytest.mark.parametrize(
    "original_args",
    [
        pytest.param(SimpleNamespace(seed=17), id="namespace"),
        pytest.param({"seed": 17}, id="dictionary"),
    ],
)
def test_force_zero_reset_action_removes_temporary_mutable_value_after_exception(
    original_args,
):
    env = SimpleNamespace(args=original_args)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with fixed_protocol._force_zero_reset_action(env):
            if isinstance(env.args, dict):
                assert env.args["reset_action"] is False
            else:
                assert env.args.reset_action is False
            raise RuntimeError("evaluation failed")

    assert env.args is original_args
    if isinstance(env.args, dict):
        assert env.args == {"seed": 17}
    else:
        assert vars(env.args) == {"seed": 17}


class _FakePolicy:
    def init_hidden(self):
        return None


class _FakeBehaviourNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_dicts = [_FakePolicy()]

    def get_actions(self, *args, **kwargs):
        action = torch.zeros((1, 1, 1))
        return action, None, None, None, None


class _FakeFixedDayEnv:
    def __init__(self, failed_transition: tuple[int, int] | None = None):
        self.args = SimpleNamespace(reset_action=True)
        self.episode_limit = 240
        self.failed_transition = failed_transition
        self.steps = 0
        self.current_day = TEST_DAYS[0]
        self.reset_action_values: list[bool] = []
        self.noise_values: list[bool] = []
        self.powergrid = SimpleNamespace(
            res_bus=pd.DataFrame(
                {"vm_pu": [1.0, 1.01]}, index=pd.Index([0, 1], name="bus")
            )
        )

    def manual_reset(self, day, start_hour, interval):
        self.current_day = int(day)
        self.start_hour = int(start_hour)
        self.steps = 0
        self.reset_action_values.append(bool(self.args.reset_action))
        self._set_voltage()
        return [[0.0]], None

    def current_timestamp(self):
        return (
            pd.Timestamp("2020-01-01")
            + pd.Timedelta(days=self.current_day)
            + pd.Timedelta(hours=self.start_hour)
            + pd.Timedelta(minutes=3 * self.steps)
        )

    def get_avail_actions(self):
        return [[1.0]]

    def get_obs(self):
        return [[float(self.steps)]]

    def _set_voltage(self):
        self.powergrid.res_bus.loc[:, "vm_pu"] = [
            1.0 + self.steps * 1e-6,
            1.01 + self.steps * 1e-6,
        ]

    def _is_failed_transition(self):
        return self.failed_transition == (self.current_day, self.steps)

    def _info(self):
        return {
            "percentage_of_v_out_of_control": 0.0,
            "totally_controllable_ratio": 1.0,
            "average_voltage_deviation": 0.005,
            "average_voltage": 1.005,
            "max_voltage_rise_deviation": 0.0,
            "max_voltage_drop_deviation": 0.0,
            "q_loss": 0.2,
            "total_line_loss": 0.03,
            "action_projection_ratio": 0.25,
            "inverter_p_mw": np.asarray([0.6]),
            "inverter_q_mvar": np.asarray([0.2]),
            "inverter_s_max_mva": np.asarray([1.0]),
            "destroy": 0.0,
        }

    def step(self, action, add_noise=True):
        self.noise_values.append(bool(add_noise))
        if self._is_failed_transition():
            info = self._info()
            info["destroy"] = 1.0
            info["totally_controllable_ratio"] = 0.0
            return -200.0, True, info
        reward = -0.1
        info = self._info()
        self.steps += 1
        done = self.steps >= self.episode_limit
        if not done:
            self._set_voltage()
        return reward, done, info

    def _calc_reward(self):
        return -0.1, self._info()

    def _set_demand_and_pv(self, add_noise=True):
        self.noise_values.append(bool(add_noise))
        self._set_voltage()


_MISSING = object()


class _MalformedSafetyEnv(_FakeFixedDayEnv):
    def __init__(self, key, value):
        super().__init__()
        self.invalid_key = key
        self.invalid_value = value

    def _info(self):
        info = super()._info()
        if self.current_day == 1030 and self.steps == 2:
            if self.invalid_value is _MISSING:
                info.pop(self.invalid_key)
            else:
                info[self.invalid_key] = self.invalid_value
        return info


@pytest.fixture(scope="module")
def policy_args():
    return SimpleNamespace(
        cuda=False,
        agent_num=1,
        obs_size=1,
        continuous=True,
        action_bias=0.0,
        action_scale=1.0,
    )


def _valid_inverter_info(**overrides):
    info = {
        "percentage_of_v_out_of_control": 0.0,
        "totally_controllable_ratio": 1.0,
        "destroy": 0.0,
        "inverter_p_mw": np.asarray([0.6, 0.8]),
        "inverter_q_mvar": np.asarray([0.2, -0.3]),
        "inverter_s_max_mva": np.asarray([1.0, 1.0]),
    }
    info.update(overrides)
    return info


def _executed_physical_row(info=None):
    return fixed_protocol._executed_row(
        day=1030,
        day_position=0,
        step=0,
        timestamp=np.datetime64("2022-10-28T06:00:00", "ns"),
        reward=-0.1,
        info=_valid_inverter_info() if info is None else info,
    )


def test_calc_reward_captures_inverter_vectors_before_next_pv_state():
    from environments.var_voltage_control.voltage_control_env import VoltageControl

    env = VoltageControl.__new__(VoltageControl)
    env.powergrid = SimpleNamespace(
        sgen=pd.DataFrame({"p_mw": [0.6, 0.8], "q_mvar": [0.2, -0.3]}),
        res_sgen=pd.DataFrame({"q_mvar": [0.2, -0.3]}),
        res_bus=pd.DataFrame({"vm_pu": [1.0, 1.01]}),
        res_line=pd.DataFrame({"pl_mw": [0.01]}),
    )
    env.s_max = np.asarray([1.0, 1.0])
    env.v_lower = 0.95
    env.v_upper = 1.05
    env.voltage_weight = 1.0
    env.violation_weight = 0.0
    env.q_weight = 0.1
    env.line_weight = None
    env._last_action_diagnostics = None

    _, info = env._calc_reward()
    env.powergrid.sgen.loc[:, "p_mw"] = [0.1, 0.2]
    env.powergrid.sgen.loc[:, "q_mvar"] = [0.0, 0.0]
    env.s_max[:] = [2.0, 2.0]

    np.testing.assert_array_equal(info["inverter_p_mw"], [0.6, 0.8])
    np.testing.assert_array_equal(info["inverter_q_mvar"], [0.2, -0.3])
    np.testing.assert_array_equal(info["inverter_s_max_mva"], [1.0, 1.0])


@pytest.mark.parametrize(
    "info",
    [
        _valid_inverter_info(inverter_p_mw=np.asarray([[0.6, 0.8]])),
        _valid_inverter_info(inverter_q_mvar=np.asarray([0.2])),
        _valid_inverter_info(inverter_p_mw=np.asarray([0.6, np.nan])),
        _valid_inverter_info(inverter_s_max_mva=np.asarray([1.0, -1.0])),
    ],
    ids=["two-dimensional", "length-mismatch", "nonfinite", "negative-s-max"],
)
def test_executed_row_rejects_malformed_inverter_vectors(info):
    with pytest.raises(ValueError, match="inverter"):
        _executed_physical_row(info)


def test_executed_row_rejects_reactive_power_outside_inverter_capability():
    info = _valid_inverter_info(inverter_q_mvar=np.asarray([0.81, -0.3]))

    with pytest.raises(ValueError, match="capability"):
        _executed_physical_row(info)


def test_executed_row_enforces_active_power_nameplate_limit_with_tolerance():
    outside_tolerance = _valid_inverter_info(
        inverter_p_mw=np.asarray([1.0000000002, 0.8]),
        inverter_q_mvar=np.asarray([0.0, -0.3]),
    )

    with pytest.raises(ValueError, match="active power.*nameplate"):
        _executed_physical_row(outside_tolerance)

    within_tolerance = _valid_inverter_info(
        inverter_p_mw=np.asarray([1.00000000005, 0.8]),
        inverter_q_mvar=np.asarray([0.0, -0.3]),
    )
    accepted = _executed_physical_row(within_tolerance)
    np.testing.assert_array_equal(accepted["inverter_p_mw"], [1.00000000005, 0.8])


def test_step_csv_roundtrips_inverter_vectors_and_leaves_failure_rows_blank():
    executed = _executed_physical_row()
    failed = fixed_protocol._failure_row(
        day=1030,
        day_position=0,
        step=1,
        timestamp=np.datetime64("2022-10-28T06:03:00", "ns"),
    )

    rows = list(
        csv.DictReader(fixed_protocol._csv_text([executed, failed]).splitlines())
    )

    assert rows[0]["executed"] == "true"
    assert rows[1]["executed"] == "false"
    assert json.loads(rows[0]["inverter_p_mw"]) == [0.6, 0.8]
    assert json.loads(rows[0]["inverter_q_mvar"]) == [0.2, -0.3]
    assert json.loads(rows[0]["inverter_s_max_mva"]) == [1.0, 1.0]
    assert rows[1]["inverter_p_mw"] == ""
    assert rows[1]["inverter_q_mvar"] == ""
    assert rows[1]["inverter_s_max_mva"] == ""
    denominators = fixed_protocol._summarize_rows([executed])["metric_denominators"]
    assert "inverter_p_mw" not in denominators
    assert "inverter_q_mvar" not in denominators
    assert "inverter_s_max_mva" not in denominators


@pytest.fixture(scope="module")
def complete_policy_result(policy_args):
    env = _FakeFixedDayEnv()
    result = evaluate_policy_fixed_days(
        env=env,
        behaviour_net=_FakeBehaviourNet(),
        args=policy_args,
        days=TEST_DAYS,
        max_steps=240,
        start_hour=6,
    )
    return env, result


def _formal_cli_args(tmp_path, **overrides):
    values = {
        "validation_only": False,
        "aggregate_only": False,
        "skip_existing": False,
        "dry_run": False,
        "save_path": str(tmp_path / "exp" / "q1_submission_v2_strict"),
        "max_steps": 240,
        "eval_start_hour": 6,
        "validation_start_day": 1000,
        "validation_days": 30,
        "test_start_day": 1030,
        "test_days": 30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_production_evaluator_never_uses_the_small_test_shape(policy_args):
    with pytest.raises(ValueError, match="30 days x 240"):
        _evaluate_fixed_result(
            _FakeFixedDayEnv(),
            _FakeBehaviourNet(),
            policy_args,
            [0],
            1,
            6,
        )


@pytest.mark.parametrize(
    ("days", "start_hour", "message"),
    [
        (list(range(30)), 6, "validation or test date window"),
        (TEST_DAYS, 7, "start at 06:00"),
    ],
)
def test_strict_evaluator_rejects_noncanonical_window_or_hour(
    policy_args, days, start_hour, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_policy_fixed_days(
            env=_FakeFixedDayEnv(),
            behaviour_net=_FakeBehaviourNet(),
            args=policy_args,
            days=days,
            max_steps=240,
            start_hour=start_hour,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_steps": 239}, "max_steps=240"),
        ({"eval_start_hour": 7}, "eval_start_hour=6"),
        ({"validation_start_day": 999}, "validation=1000/30"),
        ({"validation_days": 29}, "validation=1000/30"),
        ({"test_start_day": 1029}, "test=1030/30"),
        ({"test_days": 29}, "test=1030/30"),
    ],
)
def test_submission_preflight_rejects_noncanonical_evaluation_before_writes(
    tmp_path, override, message
):
    args = _formal_cli_args(tmp_path, **override)

    with pytest.raises(ValueError, match=message):
        validate_submission_preflight(args)

    assert not Path(args.save_path).exists()


def test_submission_preflight_requires_a_fresh_strict_root(tmp_path):
    legacy = _formal_cli_args(
        tmp_path, save_path=str(tmp_path / "exp" / "q1_risk_aware")
    )
    with pytest.raises(ValueError, match="q1_submission_v2_strict"):
        validate_submission_preflight(legacy)

    occupied = Path(_formal_cli_args(tmp_path).save_path)
    occupied.mkdir(parents=True)
    (occupied / "existing.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh"):
        validate_submission_preflight(_formal_cli_args(tmp_path))


def test_default_cli_output_is_the_submission_v2_root(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_risk_aware_multiseed.py"])

    args = parse_args()

    assert args.save_path == "exp/q1_submission_v2_strict"


def test_cli_rejects_small_evaluation_shape_before_creating_output(tmp_path):
    output_root = tmp_path / "exp" / "q1_submission_v2_strict"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/run_risk_aware_multiseed.py",
            "--max-steps",
            "1",
            "--save-path",
            str(output_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode != 0
    assert "max_steps=240" in completed.stderr
    assert not output_root.exists()


def test_dry_run_prints_plan_json_without_creating_formal_output(tmp_path):
    output_root = tmp_path / "exp" / "q1_submission_v2_strict"
    completed = subprocess.run(
        [
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
            "--save-path",
            str(output_root),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert len(plan["runs"]) == 3
    assert plan["split"]["validation_days"] == list(range(1000, 1030))
    assert plan["split"]["test_days"] == TEST_DAYS
    assert not output_root.exists()


def test_complete_policy_evaluation_has_equal_7200_step_denominator(
    complete_policy_result,
):
    env, result = complete_policy_result

    assert len(result.step_rows) == 7200
    assert len(result.daily_rows) == 30
    assert result.node_voltages.shape == (7200, 2)
    assert result.timestamps.shape == (7200,)
    assert result.executed_mask.shape == (7200,)
    assert result.executed_mask.all()
    assert result.summary["planned_steps"] == 7200
    assert result.summary["executed_steps"] == 7200
    assert result.summary["failed_steps"] == 0
    assert result.summary["metric_denominators"]["vrate"] == 7200
    assert result.summary["metric_denominators"]["q_loss"] == 7200
    assert set(env.reset_action_values) == {False}
    assert set(env.noise_values) == {False}
    assert env.args.reset_action is True


def test_policy_failure_fills_current_and_remaining_day_conservatively(policy_args):
    env = _FakeFixedDayEnv(failed_transition=(1032, 17))

    result = evaluate_policy_fixed_days(
        env=env,
        behaviour_net=_FakeBehaviourNet(),
        args=policy_args,
        days=TEST_DAYS,
        max_steps=240,
        start_hour=6,
    )

    failed = [row for row in result.step_rows if not row["executed"]]
    assert len(failed) == 240 - 17
    assert {row["vrate"] for row in failed} == {1.0}
    assert {row["cr"] for row in failed} == {0.0}
    assert {row["destroy"] for row in failed} == {1.0}
    assert all(np.isnan(row["q_loss"]) for row in failed)
    assert all(np.isnan(row["total_line_loss"]) for row in failed)
    assert all(np.isnan(row["average_voltage"]) for row in failed)
    assert all(np.isnan(row["action_projection_ratio"]) for row in failed)
    assert all(
        all(
            key not in row or row[key] in (None, "")
            for key in (
                "inverter_p_mw",
                "inverter_q_mvar",
                "inverter_s_max_mva",
            )
        )
        for row in failed
    )
    assert np.isnan(result.node_voltages[~result.executed_mask]).all()
    assert result.summary["planned_steps"] == 7200
    assert result.summary["executed_steps"] == 7200 - len(failed)
    assert result.summary["failed_steps"] == len(failed)
    assert result.summary["metric_denominators"]["vrate"] == 7200
    assert result.summary["metric_denominators"]["q_loss"] == 7200 - len(failed)
    failed_day = result.daily_rows[2]
    assert failed_day["planned_steps"] == 240
    assert failed_day["executed_steps"] == 17
    assert failed_day["failure_step"] == 17
    assert failed_day["vrate"] == pytest.approx((240 - 17) / 240)
    assert failed_day["cr"] == pytest.approx(17 / 240)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("percentage_of_v_out_of_control", _MISSING),
        ("totally_controllable_ratio", np.nan),
        ("destroy", np.inf),
    ],
)
def test_missing_or_nonfinite_safety_metric_conservatively_fails_the_day(
    policy_args, key, value
):
    result = evaluate_policy_fixed_days(
        env=_MalformedSafetyEnv(key, value),
        behaviour_net=_FakeBehaviourNet(),
        args=policy_args,
        days=TEST_DAYS,
        max_steps=240,
        start_hour=6,
    )

    first_day = result.daily_rows[0]
    assert first_day["executed_steps"] == 2
    assert first_day["failed_steps"] == 238
    assert first_day["failure_step"] == 2
    assert result.summary["metric_denominators"]["vrate"] == 7200
    assert result.summary["metric_denominators"]["cr"] == 7200
    assert result.summary["metric_denominators"]["destroy"] == 7200


def test_each_day_has_unique_three_minute_timestamps_from_0600_to_1757(
    complete_policy_result,
):
    _, result = complete_policy_result

    for day_position in range(30):
        start = day_position * 240
        timestamps = result.timestamps[start : start + 240]
        assert len(np.unique(timestamps)) == 240
        assert str(timestamps[0]).endswith("06:00:00.000000000")
        assert str(timestamps[-1]).endswith("17:57:00.000000000")
        assert np.all(np.diff(timestamps) == np.timedelta64(3, "m"))


def test_controller_and_policy_share_the_same_accounting(complete_policy_result):
    _, policy_result = complete_policy_result
    controller_env = _FakeFixedDayEnv()

    controller_result = evaluate_controller_fixed_days(
        env=controller_env,
        controller=lambda env: True,
        days=TEST_DAYS,
        max_steps=240,
        start_hour=6,
    )

    for key in (
        "planned_steps",
        "executed_steps",
        "failed_steps",
        "percentage_of_v_out_of_control",
        "totally_controllable_ratio",
        "q_loss",
        "total_line_loss",
    ):
        assert controller_result.summary[key] == policy_result.summary[key]
    assert (
        controller_result.summary["metric_denominators"]
        == policy_result.summary["metric_denominators"]
    )
    assert set(controller_env.noise_values) == {False}


def test_controller_cannot_bypass_protocol_with_transition_tuple():
    env = _FakeFixedDayEnv()

    with pytest.raises(TypeError, match="boolean convergence"):
        evaluate_controller_fixed_days(
            env=env,
            controller=lambda _env: (0.0, False, _env._info()),
            days=TEST_DAYS,
            max_steps=240,
            start_hour=6,
        )


def test_known_powerflow_nonconvergence_uses_conservative_failure_fill():
    class LoadflowNotConverged(Exception):
        pass

    LoadflowNotConverged.__module__ = "pandapower.auxiliary"
    env = _FakeFixedDayEnv()

    def controller(controlled_env):
        if controlled_env.current_day == 1030 and controlled_env.steps == 2:
            raise LoadflowNotConverged("did not converge")
        return True

    result = evaluate_controller_fixed_days(
        env=env,
        controller=controller,
        days=TEST_DAYS,
        max_steps=240,
        start_hour=6,
    )

    assert result.daily_rows[0]["executed_steps"] == 2
    assert result.daily_rows[0]["failed_steps"] == 238
    assert result.daily_rows[0]["failure_step"] == 2


def test_controller_programming_error_is_not_reclassified_as_nonconvergence():
    with pytest.raises(RuntimeError, match="controller bug"):
        evaluate_controller_fixed_days(
            env=_FakeFixedDayEnv(),
            controller=lambda _env: (_ for _ in ()).throw(
                RuntimeError("controller bug")
            ),
            days=TEST_DAYS,
            max_steps=240,
            start_hour=6,
        )


def test_artifact_writer_emits_atomic_schema_and_refuses_overwrite(
    tmp_path, complete_policy_result
):
    _, result = complete_policy_result
    output_dir = tmp_path / "exp" / "q1_submission_v2_strict" / "test_best"

    paths = write_fixed_test_artifacts(result, output_dir, "fixed_test")

    assert set(paths) == {
        "step_metrics",
        "daily_metrics",
        "node_voltages",
        "trajectory_schema",
        "summary",
    }
    assert all(Path(path).is_file() for path in paths.values())
    assert not list(output_dir.glob("*.tmp"))
    assert not list(output_dir.parent.glob(".test_best.tmp-*"))
    schema = json.loads(Path(paths["trajectory_schema"]).read_text("utf-8"))
    assert schema["points_per_day"] == 240
    assert schema["date_window"] == {
        "start_day": 1030,
        "end_day": 1059,
        "day_count": 30,
    }
    assert schema["failure_fill"]["vrate"] == 1.0
    assert schema["planned_count"] == 7200
    assert schema["executed_count"] == 7200
    assert schema["bus_order"] == [0, 1]
    assert schema["schema_version"] == 2
    vectors = schema["vector_fields"]
    assert vectors["fields"] == [
        "inverter_p_mw",
        "inverter_q_mvar",
        "inverter_s_max_mva",
    ]
    assert vectors["units"] == {
        "inverter_p_mw": "MW",
        "inverter_q_mvar": "MVAr",
        "inverter_s_max_mva": "MVA",
    }
    assert vectors["encoding"] == "json-array-string"
    assert vectors["capability_constraint"]["tolerance"] == 1e-10
    assert "|p_i| <=" in vectors["capability_constraint"]["equation"]
    assert "sqrt(max(" in vectors["capability_constraint"]["equation"]
    with Path(paths["step_metrics"]).open(encoding="utf-8", newline="") as handle:
        first_step = next(csv.DictReader(handle))
    assert json.loads(first_step["inverter_p_mw"]) == [0.6]
    assert json.loads(first_step["inverter_q_mvar"]) == [0.2]
    assert json.loads(first_step["inverter_s_max_mva"]) == [1.0]
    arrays = np.load(paths["node_voltages"])
    assert arrays["node_voltages"].shape == (7200, 2)
    assert arrays["timestamps"].shape == (7200,)
    assert arrays["executed_mask"].all()

    with pytest.raises(FileExistsError, match="overwrite"):
        write_fixed_test_artifacts(result, output_dir, "fixed_test")


def test_artifact_set_failure_leaves_no_partial_or_empty_formal_directory(
    tmp_path, complete_policy_result, monkeypatch
):
    _, result = complete_policy_result
    output_dir = tmp_path / "exp" / "q1_submission_v2_strict" / "test_best"

    def fail_promotion(source, destination):
        raise OSError("forced directory promotion failure")

    monkeypatch.setattr(fixed_protocol.os, "replace", fail_promotion)

    with pytest.raises(OSError, match="forced directory promotion failure"):
        write_fixed_test_artifacts(result, output_dir, "fixed_test")

    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".test_best.tmp-*"))


def test_artifact_writer_rejects_legacy_or_non_submission_root(
    tmp_path, complete_policy_result
):
    _, result = complete_policy_result
    legacy = tmp_path / "exp" / "q1_risk_aware" / "test_best"

    with pytest.raises(ValueError, match="q1_submission_v2_strict"):
        write_fixed_test_artifacts(result, legacy, "fixed_test")
    assert not legacy.exists()


def test_runner_uses_summary_without_validation_trajectories_by_default(
    tmp_path, complete_policy_result
):
    _, result = complete_policy_result
    validation_dir = (
        tmp_path / "exp" / "q1_submission_v2_strict" / "validation" / "episode_0050"
    )

    summary = consume_fixed_test_result(
        result,
        output_dir=validation_dir,
        stem="fixed_test",
        retain_artifacts=False,
    )

    assert (
        summary["percentage_of_v_out_of_control"]
        == result.summary["percentage_of_v_out_of_control"]
    )
    assert summary["planned_steps"] == 7200.0
    assert "metric_denominators" not in summary
    assert "bus_order" not in summary
    assert not validation_dir.exists()


def test_runner_retains_full_final_test_artifacts(tmp_path, complete_policy_result):
    _, result = complete_policy_result
    test_dir = tmp_path / "exp" / "q1_submission_v2_strict" / "test_best"

    summary = consume_fixed_test_result(
        result,
        output_dir=test_dir,
        stem="fixed_test",
        retain_artifacts=True,
    )

    assert (
        summary["percentage_of_v_out_of_control"]
        == result.summary["percentage_of_v_out_of_control"]
    )
    assert "metric_denominators" not in summary
    assert (test_dir / "fixed_test_step_metrics.csv").is_file()
    assert (test_dir / "fixed_test_daily_metrics.csv").is_file()
    assert (test_dir / "fixed_test_node_voltages.npz").is_file()
    assert (test_dir / "fixed_test_trajectory_schema.json").is_file()
    assert (test_dir / "fixed_test_summary.json").is_file()


def _small_fixed_result(vrate):
    summary = {
        "planned_steps": 1,
        "executed_steps": 1,
        "failed_steps": 0,
        "vrate": vrate,
        "cr": 1.0,
        "destroy": 0.0,
        "percentage_of_v_out_of_control": vrate,
        "totally_controllable_ratio": 1.0,
        "average_voltage_deviation": 0.01,
        "q_loss": 0.2,
        "total_line_loss": 0.03,
        "mean_reward": -0.1,
        "metric_denominators": {
            "vrate": 1,
            "cr": 1,
            "destroy": 1,
            "average_voltage_deviation": 1,
            "q_loss": 1,
        },
        "planned_days": 1,
        "completed_days": 1,
        "failed_days": 0,
        "start_day": 1030,
        "end_day": 1030,
        "points_per_day": 1,
        "start_hour": 6,
        "cadence_minutes": 3,
        "bus_order": [0],
    }
    step = {
        "day_index": 1030,
        "day_position": 0,
        "step_index": 0,
        "timestamp": "2022-10-28T06:00:00",
        "executed": True,
        "vrate": vrate,
        "cr": 1.0,
        "destroy": 0.0,
        "percentage_of_v_out_of_control": vrate,
        "totally_controllable_ratio": 1.0,
        "average_voltage_deviation": 0.01,
        "q_loss": 0.2,
        "total_line_loss": 0.03,
        "reward": -0.1,
        "inverter_p_mw": np.asarray([0.6]),
        "inverter_q_mvar": np.asarray([0.2]),
        "inverter_s_max_mva": np.asarray([1.0]),
    }
    daily = {
        **summary,
        "day_index": 1030,
        "day_position": 0,
        "date": "2022-10-28",
        "failure_step": None,
    }
    return fixed_protocol.FixedTestResult(
        summary=summary,
        step_rows=[step],
        daily_rows=[daily],
        node_voltages=np.asarray([[1.0]]),
        timestamps=np.asarray(["2022-10-28T06:00:00"], dtype="datetime64[ns]"),
        executed_mask=np.asarray([True]),
    )


def _two_step_result_with_failure():
    result = _small_fixed_result(0.0)
    failed = fixed_protocol._failure_row(
        day=1030,
        day_position=0,
        step=1,
        timestamp=np.datetime64("2022-10-28T06:03:00", "ns"),
    )
    summary = dict(result.summary)
    summary.update(
        {
            "planned_steps": 2,
            "executed_steps": 1,
            "failed_steps": 1,
            "points_per_day": 2,
        }
    )
    daily = dict(result.daily_rows[0])
    daily.update(summary)
    daily["failure_step"] = 1
    return fixed_protocol.FixedTestResult(
        summary=summary,
        step_rows=[result.step_rows[0], failed],
        daily_rows=[daily],
        node_voltages=np.asarray([[1.0], [np.nan]]),
        timestamps=np.asarray(
            ["2022-10-28T06:00:00", "2022-10-28T06:03:00"],
            dtype="datetime64[ns]",
        ),
        executed_mask=np.asarray([True, False]),
    )


def _tamper_staged_step_csv(monkeypatch, transform):
    original_csv_text = fixed_protocol._csv_text

    def tampered_csv_text(rows):
        text = original_csv_text(rows)
        if not rows or "executed" not in rows[0]:
            return text
        reader = csv.DictReader(io.StringIO(text, newline=""))
        persisted = list(reader)
        fieldnames = list(reader.fieldnames or [])
        transform(fieldnames, persisted)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(persisted)
        return buffer.getvalue()

    monkeypatch.setattr(fixed_protocol, "_csv_text", tampered_csv_text)


def _blank_inverter_evidence(row):
    row["inverter_p_mw"] = ""
    row["inverter_q_mvar"] = ""
    row["inverter_s_max_mva"] = ""


def test_artifact_writer_rejects_active_power_above_nameplate(tmp_path):
    result = _small_fixed_result(0.0)
    result.step_rows[0]["inverter_p_mw"] = np.asarray([1.1])
    result.step_rows[0]["inverter_q_mvar"] = np.asarray([0.0])
    output = tmp_path / "writer-p-limit" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(ValueError, match="active power.*nameplate"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_staged_csv_rejects_active_power_above_nameplate(tmp_path, monkeypatch):
    result = _small_fixed_result(0.0)

    def inject_active_power_violation(_fieldnames, rows):
        rows[0]["inverter_p_mw"] = "[1.1]"
        rows[0]["inverter_q_mvar"] = "[0.0]"

    _tamper_staged_step_csv(monkeypatch, inject_active_power_violation)
    output = tmp_path / "staged-p-limit" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(RuntimeError, match="active power.*nameplate"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_staged_csv_requires_executed_column(tmp_path, monkeypatch):
    result = _small_fixed_result(0.0)

    def remove_executed(fieldnames, rows):
        fieldnames.remove("executed")
        rows[0].pop("executed")
        _blank_inverter_evidence(rows[0])

    _tamper_staged_step_csv(monkeypatch, remove_executed)
    output = tmp_path / "missing-executed" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(RuntimeError, match="executed column"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


@pytest.mark.parametrize(
    "invalid_value", ["", "executed-ish"], ids=["empty", "garbage"]
)
def test_staged_csv_rejects_noncanonical_executed_values(
    tmp_path, monkeypatch, invalid_value
):
    result = _small_fixed_result(0.0)

    def corrupt_executed(_fieldnames, rows):
        rows[0]["executed"] = invalid_value
        _blank_inverter_evidence(rows[0])

    _tamper_staged_step_csv(monkeypatch, corrupt_executed)
    output = (
        tmp_path
        / f"invalid-executed-{invalid_value or 'empty'}"
        / "q1_submission_v2_strict"
        / "test_best"
    )

    with pytest.raises(RuntimeError, match="canonical executed"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_staged_csv_rejects_true_rewritten_as_false(tmp_path, monkeypatch):
    result = _small_fixed_result(0.0)

    def hide_executed_evidence(_fieldnames, rows):
        rows[0]["executed"] = "false"
        _blank_inverter_evidence(rows[0])

    _tamper_staged_step_csv(monkeypatch, hide_executed_evidence)
    output = tmp_path / "true-to-false" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(RuntimeError, match="execution state mismatch"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_staged_csv_rejects_false_rewritten_as_true(tmp_path, monkeypatch):
    result = _two_step_result_with_failure()

    def fabricate_executed_evidence(_fieldnames, rows):
        rows[1]["executed"] = "true"
        rows[1]["inverter_p_mw"] = "[0.6]"
        rows[1]["inverter_q_mvar"] = "[0.2]"
        rows[1]["inverter_s_max_mva"] = "[1.0]"

    _tamper_staged_step_csv(monkeypatch, fabricate_executed_evidence)
    output = tmp_path / "false-to-true" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(RuntimeError, match="execution state mismatch"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_staged_csv_rejects_execution_mask_mismatch(tmp_path):
    result = _small_fixed_result(0.0)
    result.executed_mask[0] = False
    output = tmp_path / "mask-mismatch" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(RuntimeError, match="execution state mismatch"):
        write_fixed_test_artifacts(result, output, "fixed_test")
    assert not output.exists()


def test_artifact_writer_rejects_missing_or_over_capacity_inverter_evidence(
    tmp_path,
):
    missing = _small_fixed_result(0.0)
    missing.step_rows[0].pop("inverter_p_mw")
    missing_output = tmp_path / "missing" / "q1_submission_v2_strict" / "test_best"

    with pytest.raises(ValueError, match="inverter_p_mw"):
        write_fixed_test_artifacts(missing, missing_output, "fixed_test")
    assert not missing_output.exists()

    over_capacity = _small_fixed_result(0.0)
    over_capacity.step_rows[0]["inverter_q_mvar"] = np.asarray([0.81])
    over_capacity_output = (
        tmp_path / "over-capacity" / "q1_submission_v2_strict" / "test_best"
    )

    with pytest.raises(ValueError, match="capability"):
        write_fixed_test_artifacts(over_capacity, over_capacity_output, "fixed_test")
    assert not over_capacity_output.exists()


def test_run_single_ranks_validation_without_retaining_its_trajectories(
    tmp_path, monkeypatch
):
    import experiments.run_risk_aware_multiseed as runner

    class FakeWriter:
        def __init__(self, _path):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeEnvironment:
        def close(self):
            pass

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
    registry_module.Model = {"matd3": object}
    trainer_module = types.ModuleType("utilities.trainer")
    trainer_module.PGTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "tensorboardX", tensorboard_module)
    monkeypatch.setitem(sys.modules, "experiments.run_q1_ml_compare", compare_module)
    monkeypatch.setitem(sys.modules, "models.model_registry", registry_module)
    monkeypatch.setitem(sys.modules, "utilities.trainer", trainer_module)

    evaluations = iter(
        [_small_fixed_result(0.2), _small_fixed_result(0.1), _small_fixed_result(0.05)]
    )
    calls = []

    def fake_evaluate(env, net, args, days, max_steps, start_hour):
        calls.append((list(days), max_steps, start_hour))
        return next(evaluations)

    monkeypatch.setattr(runner, "_evaluate_fixed_result", fake_evaluate)
    output_dir = (
        tmp_path
        / "exp"
        / "q1_submission_v2_strict"
        / "case33_3min_final"
        / "matd3"
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
        "test_days": TEST_DAYS,
    }

    result = _run_single(
        {
            "scenario": "case33_3min_final",
            "algorithm": "matd3",
            "seed": 0,
            "output_dir": str(output_dir),
        },
        split,
        args_cli,
    )

    history = json.loads(
        (output_dir / "validation_history.json").read_text(encoding="utf-8")
    )
    assert [
        row["validation_metrics"]["percentage_of_v_out_of_control"] for row in history
    ] == [
        0.2,
        0.1,
    ]
    assert result["selected_episode"] == 2
    assert result["validation_metrics"]["percentage_of_v_out_of_control"] == 0.1
    assert result["test_metrics"]["percentage_of_v_out_of_control"] == 0.05
    assert calls == [
        (list(range(1000, 1030)), 240, 6),
        (list(range(1000, 1030)), 240, 6),
        (TEST_DAYS, 240, 6),
    ]
    assert not (output_dir / "validation_trajectories").exists()
    assert (output_dir / "validation_history.json").is_file()
    assert (output_dir / "test_best" / "fixed_test_step_metrics.csv").is_file()
    assert (output_dir / "test_best" / "fixed_test_daily_metrics.csv").is_file()
    assert (output_dir / "test_best" / "fixed_test_node_voltages.npz").is_file()
    assert (output_dir / "test_best" / "fixed_test_trajectory_schema.json").is_file()
    assert (output_dir / "test_best" / "fixed_test_summary.json").is_file()
