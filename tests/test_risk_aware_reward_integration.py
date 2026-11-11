from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# The reward unit does not need pandapower's heavy optional plotting stack.
fake_pp = types.ModuleType("pandapower")
fake_pp.ppException = RuntimeError
sys.modules.setdefault("pandapower", fake_pp)
fake_plot = types.ModuleType("environments.var_voltage_control.pf_res_plot")
fake_plot.pf_res_plotly = lambda *args, **kwargs: None
sys.modules.setdefault("environments.var_voltage_control.pf_res_plot", fake_plot)

from environments.var_voltage_control.voltage_control_env import VoltageControl


def make_reward_only_env(voltage, reactive_power, violation_weight=100.0):
    env = VoltageControl.__new__(VoltageControl)
    env.v_lower = 0.95
    env.v_upper = 1.05
    env.voltage_weight = 1.0
    env.violation_weight = violation_weight
    env.q_weight = 0.1
    env.line_weight = None
    env.powergrid = SimpleNamespace(
        res_bus=pd.DataFrame({"vm_pu": voltage}),
        res_sgen=pd.DataFrame({"q_mvar": reactive_power}),
        res_line=pd.DataFrame({"pl_mw": [0.01, 0.02]}),
    )
    return env


def test_calc_reward_adds_explicit_voltage_violation_term():
    env = make_reward_only_env([1.00, 1.07], [0.2, -0.4])

    reward, info = env._calc_reward({})

    assert info["voltage_deviation_loss"] == pytest.approx(0.035)
    assert info["voltage_violation_loss"] == pytest.approx(0.02)
    assert info["q_loss_weighted"] == pytest.approx(0.03)
    assert reward == pytest.approx(-0.085)


def test_zero_violation_weight_reproduces_legacy_l1_q_reward():
    env = make_reward_only_env([1.00, 1.07], [0.2, -0.4], violation_weight=0.0)

    reward, info = env._calc_reward({})

    assert info["voltage_violation_loss"] == 0.0
    assert reward == pytest.approx(-(0.035 + 0.03))


def test_violation_loss_is_reported_separately_from_violation_rate():
    env = make_reward_only_env([0.94, 1.00, 1.06], [0.0])

    _, info = env._calc_reward({})

    assert info["percentage_of_v_out_of_control"] == pytest.approx(2 / 3)
    assert info["voltage_violation_loss"] == pytest.approx(100 * (0.0001 + 0 + 0.0001) / 3)
    assert np.isfinite(info["voltage_violation_loss"])


def test_training_day_selection_respects_configured_candidates():
    env = VoltageControl.__new__(VoltageControl)
    env.start_day_candidates = np.array([10, 20, 30])
    env.pv_data = pd.DataFrame(
        index=pd.date_range("2012-01-01", periods=40 * 480, freq="3min")
    )
    env.episode_limit = 10

    selected = {env._select_start_day() for _ in range(30)}

    assert selected <= {10, 20, 30}
    assert selected
