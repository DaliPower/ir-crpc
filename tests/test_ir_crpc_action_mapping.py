from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

# Action mapping does not need a real power-flow backend or pandapower's
# optional plotting stack.  Import a private lightweight environment module,
# then restore the global import cache so later tests can choose their own
# backend in the same Python process.
fake_pp = types.ModuleType("pandapower")
fake_pp.ppException = RuntimeError
fake_pp.runpp = lambda powergrid: None
fake_plot = types.ModuleType("environments.var_voltage_control.pf_res_plot")
fake_plot.pf_res_plotly = lambda *args, **kwargs: None
env_module_name = "environments.var_voltage_control.voltage_control_env"
plot_module_name = "environments.var_voltage_control.pf_res_plot"
module_names = ("pandapower", plot_module_name, env_module_name)
missing = object()
saved_modules = {name: sys.modules.get(name, missing) for name in module_names}
sys.modules["pandapower"] = fake_pp
sys.modules[plot_module_name] = fake_plot
sys.modules.pop(env_module_name, None)
try:
    env_module = importlib.import_module(env_module_name)
finally:
    for name, saved in saved_modules.items():
        if saved is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

VoltageControl = env_module.VoltageControl
ROOT = Path(__file__).resolve().parents[1]


def make_mapping_env(*, p_mw=(0.6,), s_max=(1.0,), mapping="capacity_adaptive"):
    env = VoltageControl.__new__(VoltageControl)
    env.args = SimpleNamespace(action_mapping=mapping)
    env.s_max = np.asarray(s_max, dtype=float)
    env.powergrid = SimpleNamespace(
        sgen=pd.DataFrame(
            {
                "p_mw": np.asarray(p_mw, dtype=float),
                "q_mvar": np.zeros(len(p_mw), dtype=float),
            }
        )
    )
    return env


def make_step_env(*, mapping="static_projected"):
    env = make_mapping_env(p_mw=(0.8,), s_max=(1.0,), mapping=mapping)
    env.powergrid.res_bus = pd.DataFrame({"vm_pu": [1.0]})
    env.powergrid.res_sgen = pd.DataFrame({"q_mvar": [0.0]})
    env.powergrid.res_line = pd.DataFrame({"pl_mw": [0.0]})
    env.powergrid.load = pd.DataFrame({"p_mw": [0.0], "q_mvar": [0.0]})
    env.v_lower = 0.95
    env.v_upper = 1.05
    env.voltage_weight = 1.0
    env.violation_weight = 0.0
    env.q_weight = 0.1
    env.line_weight = None
    env.episode_limit = 1
    env.steps = 0
    env.sum_rewards = 0.0
    env._last_action_diagnostics = None
    return env


def test_default_environment_config_selects_capacity_adaptive_mapping():
    with (ROOT / "args" / "env_args" / "var_voltage_control.yaml").open(
        encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    assert config["env_args"]["action_mapping"] == "capacity_adaptive"


def test_capacity_adaptive_mapping_scales_by_instantaneous_capacity():
    env = make_mapping_env(p_mw=(0.6,), s_max=(1.0,))

    executed, stats = env._map_reactive_action(np.array([0.8]))

    np.testing.assert_allclose(executed, [0.64])
    assert stats["action_projection_ratio"] == 0.0
    assert stats["reactive_capacity_utilization"] == pytest.approx(0.8)
    assert stats["reactive_saturation_ratio"] == 0.0


def test_static_mapping_projects_static_candidate_to_instantaneous_capacity():
    env = make_mapping_env(
        p_mw=(0.8,), s_max=(1.0,), mapping="static_projected"
    )

    executed, stats = env._map_reactive_action(np.array([0.8]))

    np.testing.assert_allclose(executed, [0.6])
    assert stats["action_projection_ratio"] > 0.0
    assert stats["reactive_capacity_utilization"] == pytest.approx(1.0)
    assert stats["reactive_saturation_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("mapping", ["capacity_adaptive", "static_projected"])
def test_both_mappings_execute_inside_instantaneous_capacity_circle(mapping):
    env = make_mapping_env(
        p_mw=(0.6, 0.8), s_max=(1.0, 1.0), mapping=mapping
    )

    executed, _ = env._map_reactive_action(np.array([0.8, -0.8]))

    capacity = np.array([0.8, 0.6])
    assert np.all(np.abs(executed) <= capacity + 1e-12)


@pytest.mark.parametrize("mapping", ["capacity_adaptive", "static_projected"])
def test_zero_capacity_diagnostics_are_finite_and_not_falsely_saturated(mapping):
    env = make_mapping_env(
        p_mw=(1.0, 2.0), s_max=(1.0, 2.0), mapping=mapping
    )

    executed, stats = env._map_reactive_action(np.array([0.0, 0.5]))

    np.testing.assert_allclose(executed, [0.0, 0.0])
    assert all(np.isfinite(value) for value in stats.values())
    assert stats["reactive_capacity_utilization"] == 0.0
    assert stats["reactive_saturation_ratio"] == 0.0


@pytest.mark.parametrize(
    "actions",
    [
        np.array([0.5]),
        np.array([[0.5, -0.5]]),
        np.array([0.5, np.nan]),
        np.array([0.5, np.inf]),
    ],
    ids=["single-element-broadcast", "two-dimensional", "nan", "inf"],
)
def test_multi_inverter_invalid_action_shapes_or_values_fail_fast(actions):
    env = make_mapping_env(p_mw=(0.2, 0.3), s_max=(1.0, 1.0))

    with pytest.raises(ValueError):
        env._map_reactive_action(actions)


def test_single_inverter_scalar_action_is_coerced_to_length_one():
    env = make_mapping_env(p_mw=(0.6,), s_max=(1.0,))

    executed, stats = env._map_reactive_action(0.5)

    assert executed.shape == (1,)
    np.testing.assert_allclose(executed, [0.4])
    assert all(np.isfinite(value) for value in stats.values())


@pytest.mark.parametrize(
    ("p_mw", "s_max"),
    [
        ((0.2, 0.3), (1.0,)),
        ((0.2,), (-1.0,)),
        ((0.2,), (np.nan,)),
        ((np.inf,), (1.0,)),
    ],
    ids=["length-mismatch", "negative-s-max", "nan-s-max", "nonfinite-p"],
)
def test_invalid_inverter_limits_or_active_power_fail_fast(p_mw, s_max):
    env = make_mapping_env(p_mw=p_mw, s_max=s_max)

    with pytest.raises(ValueError):
        env._map_reactive_action(np.zeros(len(p_mw)))


def test_unknown_mapping_fails_before_network_loading(monkeypatch):
    def fail_if_network_load_starts(self):
        pytest.fail("network loading started before action_mapping validation")

    monkeypatch.setattr(VoltageControl, "_load_network", fail_if_network_load_starts)

    with pytest.raises(ValueError, match="unsupported action_mapping: unknown"):
        VoltageControl(
            {
                "action_mapping": "unknown",
                "data_path": "unused",
                "seed": 0,
            }
        )


def test_successful_step_exposes_current_controlled_action_diagnostics(monkeypatch):
    env = make_step_env()

    def successful_power_flow(powergrid):
        powergrid.res_sgen["q_mvar"] = powergrid.sgen["q_mvar"].to_numpy()

    monkeypatch.setattr(env_module.pp, "runpp", successful_power_flow)

    reward, terminated, info = env.step(np.array([0.8]), add_noise=False)

    assert terminated is True
    assert reward == pytest.approx(-0.06)
    assert info["action_projection_ratio"] == pytest.approx(1.0)
    assert info["reactive_capacity_utilization"] == pytest.approx(1.0)
    assert info["reactive_saturation_ratio"] == pytest.approx(1.0)


def test_failed_step_still_exposes_attempted_controlled_action_diagnostics(
    monkeypatch,
):
    env = make_step_env()
    from pandapower import LoadflowNotConverged
    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)

    def failed_power_flow(powergrid):
        raise LoadflowNotConverged("forced power-flow failure")

    monkeypatch.setattr(env_module.pp, "runpp", failed_power_flow)

    reward, terminated, info = env.step(np.array([0.8]), add_noise=False)

    assert terminated is True
    assert reward == pytest.approx(-200.0)
    assert info["action_projection_ratio"] == pytest.approx(1.0)
    assert info["reactive_capacity_utilization"] == pytest.approx(1.0)
    assert info["reactive_saturation_ratio"] == pytest.approx(1.0)
    assert env.last_powerflow_failure['exception']['type'] == 'LoadflowNotConverged'


def test_unknown_runtime_error_from_powerflow_is_not_training_termination(monkeypatch):
    env = make_step_env()
    error = RuntimeError('unrelated implementation error')
    monkeypatch.setattr(env_module, 'ppException', RuntimeError)
    monkeypatch.setattr(env_module.pp, 'runpp', lambda grid: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError) as caught:
        env.step(np.array([.8]), add_noise=False)
    assert caught.value is error


@pytest.mark.parametrize("reset_method", ["reset", "manual_reset"])
def test_reset_clipping_clears_and_does_not_fabricate_controlled_diagnostics(
    monkeypatch, reset_method
):
    env = VoltageControl.__new__(VoltageControl)
    env.args = SimpleNamespace(reset_action=True)
    env.history = 1
    env.n_agents = 1
    env.base_powergrid = SimpleNamespace(
        sgen=pd.DataFrame({"p_mw": [0.6], "q_mvar": [9.0]})
    )
    env._episode_start_day = 0
    env._episode_start_hour = 0
    env._episode_start_interval = 0
    env._resolve_episode_start_index = lambda: 0
    env._get_episode_pv_history = lambda: np.zeros((2, 1))
    env._get_episode_active_demand_history = lambda: np.zeros((2, 1))
    env._get_episode_reactive_demand_history = lambda: np.zeros((2, 1))
    env._set_demand_and_pv = lambda add_noise=True: None
    env.get_action = lambda: np.array([0.5])
    env.get_obs = lambda: [np.zeros(1)]
    env.get_state = lambda: np.zeros(1)
    env.s_max = np.array([1.0])
    env._last_action_diagnostics = {
        "action_projection_ratio": 1.0,
        "reactive_capacity_utilization": 1.0,
        "reactive_saturation_ratio": 1.0,
    }
    monkeypatch.setattr(env_module.pp, "runpp", lambda powergrid: None)

    if reset_method == "reset":
        env.reset(reset_time=False)
    else:
        env.manual_reset(day=0, hour=0, interval=0)

    np.testing.assert_allclose(env.powergrid.sgen["q_mvar"], [0.4])
    assert env._last_action_diagnostics is None
