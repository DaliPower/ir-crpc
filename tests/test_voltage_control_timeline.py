from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Rendering is not part of the timeline contract; avoid importing pandapower's
# optional plotting stack in the lightweight state-machine tests.
fake_pp = types.ModuleType("pandapower")
fake_pp.ppException = RuntimeError
fake_pp.runpp = lambda powergrid: None
sys.modules["pandapower"] = fake_pp
fake_plot = types.ModuleType("environments.var_voltage_control.pf_res_plot")
fake_plot.pf_res_plotly = lambda *args, **kwargs: None
sys.modules["environments.var_voltage_control.pf_res_plot"] = fake_plot

lightweight_env_module = importlib.import_module(
    "environments.var_voltage_control.voltage_control_env"
)
VoltageControl = lightweight_env_module.VoltageControl
sys.modules.pop("environments.var_voltage_control.voltage_control_env", None)
sys.modules.pop("environments.var_voltage_control.pf_res_plot", None)
sys.modules.pop("pandapower", None)


def make_minimal_voltage_control(episode_limit):
    env = VoltageControl.__new__(VoltageControl)
    env.episode_limit = episode_limit
    env.steps = 0
    env.sum_rewards = 0.0
    env.powergrid = SimpleNamespace()
    env._take_action = lambda actions: True
    return env


@pytest.fixture(scope="module")
def real_case33_index():
    csv_path = (
        ROOT
        / "environments"
        / "var_voltage_control"
        / "data"
        / "case33_3min_final"
        / "pv_active.csv"
    )
    timestamps = pd.read_csv(csv_path, usecols=["DateTime"], parse_dates=["DateTime"])
    return pd.DatetimeIndex(timestamps["DateTime"])


def make_index_only_env(index, day=0, hour=0, interval=0, episode_limit=240):
    env = VoltageControl.__new__(VoltageControl)
    timeline = pd.DataFrame({"sample": np.arange(len(index))}, index=index)
    env.pv_data = timeline
    env.active_demand_data = timeline
    env.reactive_demand_data = timeline
    env._episode_start_day = day
    env._episode_start_hour = hour
    env._episode_start_interval = interval
    env.episode_limit = episode_limit
    env.history = 1
    env.time_delta = 3
    env.steps = 0
    env._loaded_step = 0
    return env


def test_step_records_240_current_indices_and_only_advances_while_continuing():
    env = make_minimal_voltage_control(episode_limit=240)
    env.pv_data = pd.DataFrame(index=pd.date_range("2020-01-01 06:00", periods=240, freq="3min"))
    env._episode_start_index = 0
    env._episode_start_day = 0
    env._episode_start_hour = 6
    env._episode_start_interval = 0
    env.time_delta = 3
    env._loaded_step = 0
    observed = []
    env._calc_reward = lambda: (0.0, {"time_index": env.steps})
    env._set_demand_and_pv = lambda add_noise=True: observed.append(env.steps)

    with contextlib.redirect_stdout(io.StringIO()):
        stamps = []
        transitions = []
        for _ in range(240):
            stamps.append(env.current_timestamp())
            transitions.append(env.step(np.zeros(1), add_noise=False))

    infos = [transition[2] for transition in transitions]
    assert [row["time_index"] for row in infos] == list(range(240))
    assert [transition[1] for transition in transitions] == [False] * 239 + [True]
    assert observed == list(range(1, 240))
    assert stamps[0].strftime("%H:%M") == "06:00"
    assert stamps[-1].strftime("%H:%M") == "17:57"
    assert env.current_timestamp() == stamps[-1]


def test_failed_step_keeps_timestamp_of_last_loaded_state():
    env = make_minimal_voltage_control(episode_limit=240)
    env.pv_data = pd.DataFrame(index=pd.date_range("2020-01-01 06:00", periods=2, freq="3min"))
    env._episode_start_index = 0
    env._episode_start_day = 0
    env._episode_start_hour = 6
    env._episode_start_interval = 0
    env.time_delta = 3
    env._loaded_step = 0
    env.powergrid = SimpleNamespace(
        sgen=pd.DataFrame({"q_mvar": [0.0]})
    )
    env._take_action = lambda actions: False
    env._calc_reward = lambda: (0.0, {})

    before = env.current_timestamp()
    with contextlib.redirect_stdout(io.StringIO()):
        _, done, _ = env.step(np.zeros(1), add_noise=False)

    assert done is True
    assert env.steps == 1
    assert env.current_timestamp() == before


@pytest.mark.parametrize("reset_method", ["reset", "manual_reset"])
def test_reset_action_false_zeros_reactive_power_before_power_flow(
    monkeypatch, reset_method
):
    env = VoltageControl.__new__(VoltageControl)
    env.args = SimpleNamespace(reset_action=False)
    env.history = 1
    env.n_agents = 1
    env.base_powergrid = SimpleNamespace(
        sgen=pd.DataFrame({"p_mw": [1.0], "q_mvar": [9.0]})
    )
    env._episode_start_day = 0
    env._episode_start_hour = 0
    env._episode_start_interval = 0
    env.time_delta = 3
    timeline = pd.DataFrame(
        index=pd.date_range("2012-01-01 00:00", periods=2, freq="3min")
    )
    env.pv_data = timeline
    env.active_demand_data = timeline
    env.reactive_demand_data = timeline
    env._get_episode_pv_history = lambda: np.zeros((2, 1))
    env._get_episode_active_demand_history = lambda: np.zeros((2, 1))
    env._get_episode_reactive_demand_history = lambda: np.zeros((2, 1))
    env._set_demand_and_pv = lambda add_noise=True: None
    env.get_obs = lambda: [np.zeros(1)]
    env.get_state = lambda: np.zeros(1)
    q_at_power_flow = []

    def record_power_flow(powergrid):
        q_at_power_flow.append(powergrid.sgen["q_mvar"].to_list())

    monkeypatch.setattr(lightweight_env_module.pp, "runpp", record_power_flow)

    if reset_method == "reset":
        env.reset(reset_time=False)
    else:
        env.manual_reset(0, 0, 0)

    assert q_at_power_flow == [[0.0]]
    assert env.powergrid.sgen["q_mvar"].to_list() == [0.0]
    assert env.steps == 0


def test_real_case33_index_loads_240_states_from_0600_through_1757(real_case33_index):
    env = make_index_only_env(real_case33_index, day=1030, hour=6, interval=0)

    env._episode_start_index = env._resolve_episode_start_index()
    pv = env._get_episode_pv_history()
    active = env._get_episode_active_demand_history()
    reactive = env._get_episode_reactive_demand_history()
    stamps = []
    for loaded_step in range(240):
        env._loaded_step = loaded_step
        stamps.append(env.current_timestamp())

    assert pv[0, 0] == active[0, 0] == reactive[0, 0] == env._episode_start_index
    assert stamps[0] == pd.Timestamp("2014-10-27 06:00:00")
    assert stamps[-1] == pd.Timestamp("2014-10-27 17:57:00")
    assert len(set(stamps)) == 240


def test_day_zero_midnight_bfills_only_to_first_available_sample():
    index = pd.date_range("2012-01-01 00:03", periods=4, freq="3min")
    env = make_index_only_env(index)

    assert env._resolve_episode_start_index() == 0


def test_missing_non_boundary_timestamp_fails_fast():
    index = pd.date_range("2012-01-01 00:03", periods=4, freq="6min")
    env = make_index_only_env(index, interval=2)

    with pytest.raises(ValueError, match="episode start timestamp"):
        env._resolve_episode_start_index()


def test_misaligned_source_indexes_fail_fast():
    index = pd.date_range("2012-01-01 00:00", periods=4, freq="3min")
    env = make_index_only_env(index)
    env.active_demand_data = pd.DataFrame(index=index + pd.Timedelta(minutes=3))

    with pytest.raises(ValueError, match="indexes must match"):
        env._resolve_episode_start_index()
