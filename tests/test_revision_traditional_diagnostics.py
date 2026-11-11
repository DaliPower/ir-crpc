"""Regression tests for physical fairness and retained failure evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_revision_environment_protocol import construct, observation_env, env_module, LoadflowNotConverged
from test_submission_v2_traditional import _DroopEnv
from experiments import run_q1_fixed_test_traditional_baselines as traditional


def test_traditional_saturation_separates_physical_and_control_limits():
    env = _DroopEnv(.6)
    qmax = .6 * np.array([1., .6])
    traditional._set_action_diagnostics(env, qmax, qmax, -qmax, qmax)
    assert env._last_action_diagnostics['reactive_saturation_ratio'] == 0.
    assert env._last_action_diagnostics['control_limit_saturation_ratio'] == 1.
    assert env._last_action_diagnostics['control_limit_utilization'] == 1.


def test_revision_traditional_config_matches_physical_and_time_protocol():
    cfg = traditional.load_env_config('case141_3min_final', 1.5, strict_revision=True,
                                      max_steps=480, start_hour=0, days=[1000])
    assert cfg['statistics_end_day_exclusive'] == 999
    assert cfg['capacity_reference_pv_scale'] == 1.5
    assert cfg['episode_limit'] == 480 and cfg['start_hour'] == 0
    assert cfg['start_day_candidates'] == [1000]
    assert cfg['nighttime_reactive_power'] == 'enabled'
    assert cfg['observation_timing'] == 'current_pre_control'
    assert cfg['strict_episode_timestamps'] and not cfg['reset_action']


def test_revision_traditional_plan_preserves_requested_day_window(tmp_path):
    plan = traditional.build_revision_traditional_plan(
        repo_root=tmp_path, scenario='case141_3min_final', methods=['droop'],
        days=[1000], points=480, output=tmp_path / 'exp/revision_access_41572/traditional/check')
    assert plan['test_days'] == [1000] and plan['start_hour'] == 0
    assert plan['revision_protocol'] and not plan['strict_shape']
    with pytest.raises(ValueError, match='validation or test'):
        traditional.build_revision_traditional_plan(
            repo_root=tmp_path, scenario='case141_3min_final', methods=['droop'],
            days=[998], points=240, output=tmp_path / 'exp/revision_access_41572/traditional/bad')


def test_control_failure_preserves_attempted_q_before_step_rolls_back(monkeypatch):
    env, _ = observation_env(monkeypatch)
    original_q = env.powergrid.sgen.q_mvar.to_numpy().copy()
    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)
    error = LoadflowNotConverged('numerical failure detail')
    monkeypatch.setattr(env_module.pp, 'runpp', lambda net: (_ for _ in ()).throw(error))
    _, done, info = env.step([.8], add_noise=False)
    assert done and info['destroy'] == 1.
    evidence = env.last_powerflow_failure
    assert evidence['phase'] == 'controlled_action'
    assert evidence['exception']['message'] == 'numerical failure detail'
    assert evidence['snapshot']['inverter_q_mvar'] != original_q.tolist()
    np.testing.assert_array_equal(env.powergrid.sgen.q_mvar, original_q)
    json.dumps(evidence, allow_nan=False)
    assert len(env.powerflow_failures) == 1


def test_precontrol_failure_is_recorded_then_original_exception_propagates(monkeypatch):
    env, _ = observation_env(monkeypatch)
    env.step([.8], add_noise=False)
    error = LoadflowNotConverged('held q fails')
    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)
    monkeypatch.setattr(env_module.pp, 'runpp', lambda net: (_ for _ in ()).throw(error))
    with pytest.raises(LoadflowNotConverged) as caught:
        env.get_obs()
    assert caught.value is error
    assert env.last_powerflow_failure['phase'] == 'pre_control_measurement'
    assert env.last_powerflow_failure['loaded_step'] == 1


def test_unexpected_pandapower_error_is_never_converted_to_termination(monkeypatch):
    env = construct(monkeypatch)
    class ProgrammingError(Exception):
        pass
    monkeypatch.setattr(env_module, 'ppException', ProgrammingError)
    monkeypatch.setattr(env_module.pp, 'runpp', lambda net: (_ for _ in ()).throw(ProgrammingError('bug')))
    with pytest.raises(ProgrammingError):
        env._take_action([0.])


def test_traditional_limits_honor_nighttime_disabled(monkeypatch):
    env = construct(monkeypatch, nighttime_reactive_power='disabled', inverter_s_max_mva=[1.])
    env._last_nominal_pv_mw = np.zeros(1)
    env.powergrid.sgen.p_mw = .001  # Positive source noise must not enable night Q.
    low, high = traditional._env_reactive_limits(env)
    np.testing.assert_array_equal(low, [0.])
    np.testing.assert_array_equal(high, [0.])


def test_small_real_ac_witness_keeps_inputs_and_checks_balance():
    import pandapower as pp
    from types import SimpleNamespace
    from experiments.revision_diagnostics import diagnose_snapshot, physical_inputs
    net = pp.create_empty_network()
    b0, b1 = pp.create_bus(net, vn_kv=11), pp.create_bus(net, vn_kv=11)
    pp.create_ext_grid(net, b0, vm_pu=1.)
    pp.create_line_from_parameters(net, b0, b1, length_km=1., r_ohm_per_km=.2,
                                   x_ohm_per_km=.4, c_nf_per_km=0., max_i_ka=1.)
    pp.create_load(net, b1, p_mw=.2, q_mvar=.1)
    pp.create_sgen(net, b1, p_mw=.1, q_mvar=0.)
    pp.runpp(net, numba=False)
    from environments.var_voltage_control.voltage_control_env import VoltageControl
    env = VoltageControl.__new__(VoltageControl)
    env.powergrid, env.args = net, SimpleNamespace(action_scale=.6, action_bias=0.)
    env.s_max, env.v_lower, env.v_upper = np.array([.3]), .95, 1.05
    before = physical_inputs(env)
    report = diagnose_snapshot(env)
    for key in ('policy_snapshot', 'droop', 'ac_opf_witness'):
        outcome = report['outcomes'][key]
        assert outcome['status'] == 'solved'
        assert outcome['feasible_voltage_reactive_witness']
        assert outcome['ac_balance_max_mva'] < 1e-6
        assert outcome['inputs_unchanged']
        assert not outcome['infeasibility_proven']
    assert physical_inputs(env) == before
    np.testing.assert_array_equal(net.sgen.q_mvar, [0.])


def test_opf_nonconvergence_cannot_be_used_as_infeasibility_certificate(monkeypatch):
    from experiments import revision_diagnostics as diagnostics
    from types import SimpleNamespace
    import pandapower as pp
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=11)
    pp.create_ext_grid(net, b0)
    pp.create_load(net, b0, p_mw=.1)
    pp.create_sgen(net, b0, p_mw=.1)
    from environments.var_voltage_control.voltage_control_env import VoltageControl
    env = VoltageControl.__new__(VoltageControl)
    env.powergrid, env.args = net, SimpleNamespace(action_scale=.6, action_bias=0.)
    env.s_max, env.v_lower, env.v_upper = np.array([1.]), .95, 1.05
    def failure(env):
        raise pp.OPFNotConverged('local solver stopped')
    monkeypatch.setattr(diagnostics, '_run_witness_opf', failure)
    report = diagnostics.diagnose_snapshot(env)
    opf = report['outcomes']['ac_opf_witness']
    assert opf['status'] == 'solver_not_converged'
    assert not opf['feasible_voltage_reactive_witness']
    assert not opf['infeasibility_proven']
