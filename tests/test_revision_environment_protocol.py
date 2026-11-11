from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import pytest

from test_ir_crpc_action_mapping import VoltageControl, env_module

ORIGINAL_GET_OBS = VoltageControl.get_obs
ORIGINAL_GET_STATE = VoltageControl.get_state


class LoadflowNotConverged(Exception):
    """Power-flow exception double; assert the exact object survives propagation."""



def construct(monkeypatch, *, future_multiplier=1.0, **overrides):
    index = pd.date_range('2020-01-01', periods=4 * 480, freq='3min')
    raw = pd.DataFrame({'pv0': 0.2 + np.arange(len(index)) % 40 / 100}, index=index)
    raw.iloc[960:] *= future_multiplier
    network = SimpleNamespace(
        sgen=pd.DataFrame({'p_mw': [0.2], 'q_mvar': [0.0], 'bus': [0], 'name': ['zone1']}),
        load=pd.DataFrame({'p_mw': [0.1], 'q_mvar': [0.02]}),
        res_bus=pd.DataFrame({'vm_pu': [1.0]}),
    )
    monkeypatch.setattr(VoltageControl, '_load_network', lambda self: copy.deepcopy(network))
    monkeypatch.setattr(VoltageControl, '_load_pv_data', lambda self: raw * self.args.pv_scale)
    monkeypatch.setattr(VoltageControl, '_load_active_demand_data', lambda self: raw * 0.3)
    monkeypatch.setattr(VoltageControl, '_load_reactive_demand_data', lambda self: raw * 0.1)
    monkeypatch.setattr(VoltageControl, 'get_obs', lambda self: [np.array([1.0])])
    monkeypatch.setattr(VoltageControl, 'get_state', lambda self: np.array([1.0]))
    monkeypatch.setattr(env_module.pp, 'runpp', lambda net: None)
    args = dict(data_path='unused', seed=12, pv_scale=1., demand_scale=1., episode_limit=240,
                action_scale=.8, action_bias=0., mode='distributed', reset_action=True,
                start_day_candidates=[0, 1], start_hour=6, start_minute=0,
                statistics_end_day_exclusive=2, capacity_reference_pv_scale=1.5,
                strict_episode_timestamps=True)
    args.update(overrides)
    return VoltageControl(args)


def test_constructor_reset_noise_initial_q_do_not_touch_global_rng(monkeypatch):
    np.random.seed(71)
    before = copy.deepcopy(np.random.get_state())
    env = construct(monkeypatch)
    env.reset()
    env.get_action()
    env.manual_reset(1, 6, 0)
    env._set_demand_and_pv(add_noise=True)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_inserting_validation_environment_does_not_change_training_stream(monkeypatch):
    training = construct(monkeypatch)
    control = construct(monkeypatch)
    validation = construct(monkeypatch, seed=999)
    validation.reset()
    for _ in range(3):
        training.reset()
        control.reset()
        assert training.current_timestamp() == control.current_timestamp()
        np.testing.assert_array_equal(training.powergrid.sgen.values, control.powergrid.sgen.values)
        np.testing.assert_array_equal(training.powergrid.load.values, control.powergrid.load.values)


def test_future_mutation_does_not_change_fitted_statistics_or_ratings(monkeypatch):
    first = construct(monkeypatch)
    mutated = construct(monkeypatch, future_multiplier=1000.)
    for name in ('pv_std', 'active_demand_std', 'reactive_demand_std', 's_max'):
        np.testing.assert_array_equal(getattr(first, name), getattr(mutated, name))
    metadata = first.environment_metadata()
    assert metadata['statistics']['end_day_exclusive'] == 2
    assert metadata['capacity']['source'] == 'synthetic_training_peak'


def test_fixed_rating_is_independent_of_operating_kappa(monkeypatch):
    first = construct(monkeypatch, pv_scale=.5)
    second = construct(monkeypatch, pv_scale=1.5)
    np.testing.assert_allclose(first.s_max, second.s_max)
    np.testing.assert_allclose(first.pv_data * 3, second.pv_data)


def test_explicit_ratings_have_declared_source(monkeypatch):
    env = construct(monkeypatch, inverter_s_max_mva=[2.5], capacity_source='benchmark_table')
    np.testing.assert_array_equal(env.s_max, [2.5])
    assert env.environment_metadata()['capacity']['source'] == 'benchmark_table'


def test_reject_active_power_above_rating_including_direct_action_path(monkeypatch):
    env = construct(monkeypatch)
    env.powergrid.sgen['p_mw'] = env.s_max * 1.01
    with pytest.raises(ValueError, match='exceeds.*rating'):
        env._map_reactive_action([0.0])
    with pytest.raises(ValueError, match='exceeds.*rating'):
        construct(monkeypatch, inverter_s_max_mva=[.01])


def test_curtailment_is_explicit_and_recorded(monkeypatch):
    env = construct(monkeypatch, inverter_s_max_mva=[.01], active_power_overflow='curtail')
    np.testing.assert_allclose(env.powergrid.sgen.p_mw, [.01])
    assert np.sum(env._last_active_power_curtailment_mw) > 0
    q, _ = env._map_reactive_action([.8])
    np.testing.assert_array_equal(q, [0.])


@pytest.mark.parametrize('night_mode, expected', [('enabled', .8), ('disabled', 0.)])
def test_nighttime_capacity_assumption(monkeypatch, night_mode, expected):
    env = construct(monkeypatch, inverter_s_max_mva=[1.], nighttime_reactive_power=night_mode)
    env.powergrid.sgen['p_mw'] = 0.
    q, _ = env._map_reactive_action([.8])
    np.testing.assert_allclose(q, [expected])


def test_static_projection_obeys_same_alpha_boundary(monkeypatch):
    env = construct(monkeypatch, inverter_s_max_mva=[1.], action_mapping='static_projected')
    env.powergrid.sgen['p_mw'] = .8
    q, _ = env._map_reactive_action([.8])
    np.testing.assert_allclose(q, [.8 * .6])


def test_configured_start_time_is_really_used_and_reported(monkeypatch):
    env = construct(monkeypatch, start_hour=7, start_minute=12)
    for _ in range(5):
        env.reset()
        assert env.current_timestamp().strftime('%H:%M') == '07:12'
        assert env.environment_metadata()['episode']['start_timestamp'] == env.current_timestamp().isoformat()


def test_full_day_has_480_unique_snapshots_without_next_day(monkeypatch):
    env = construct(monkeypatch, episode_limit=480, start_day_candidates=[1], start_hour=0, start_minute=0)
    env._take_action = lambda action: True
    env._calc_reward = lambda: (0., {})
    stamps = []
    for _ in range(480):
        stamps.append(env.current_timestamp())
        _, done, _ = env.step([0.], add_noise=False)
    assert done
    assert len(set(stamps)) == 480
    assert stamps[0] == pd.Timestamp('2020-01-02 00:00')
    assert stamps[-1] == pd.Timestamp('2020-01-02 23:57')


def test_strict_midnight_rejects_missing_first_sample(monkeypatch):
    env = construct(monkeypatch, episode_limit=480)
    for name in ('pv_data', 'active_demand_data', 'reactive_demand_data'):
        setattr(env, name, getattr(env, name).iloc[1:])
    with pytest.raises(ValueError, match='missing'):
        env.manual_reset(0, 0, 0)


def test_strict_episode_rejects_internal_gap_and_short_tail(monkeypatch):
    env = construct(monkeypatch, episode_limit=480)
    for name in ('pv_data', 'active_demand_data', 'reactive_demand_data'):
        setattr(env, name, getattr(env, name).drop(pd.Timestamp('2020-01-02 12:00')))
    with pytest.raises(ValueError, match='contiguous|complete'):
        env.manual_reset(1, 0, 0)
    with pytest.raises(ValueError, match='contiguous|complete'):
        env.manual_reset(3, 12, 0)


def test_nighttime_disabled_stays_off_under_positive_source_noise(monkeypatch):
    env = construct(monkeypatch, nighttime_reactive_power='disabled')
    env.pv_histories[0, :] = 0.
    env._set_demand_and_pv(add_noise=True)
    assert np.all(env.powergrid.sgen.p_mw > 0.)
    q, _ = env._map_reactive_action([.8])
    np.testing.assert_array_equal(q, [0.])


def test_deterministic_manual_reset_does_not_retry_unsolvable_identical_state(monkeypatch):

    env = construct(monkeypatch, reset_action=False)
    original_error = LoadflowNotConverged('deterministic initial snapshot fails')
    calls = []

    class InfiniteRetryGuard(BaseException):
        pass

    def failed_power_flow(net):
        calls.append(1)
        if len(calls) > 1:
            raise InfiniteRetryGuard('same deterministic snapshot was retried')
        raise original_error

    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)
    monkeypatch.setattr(env_module.pp, 'runpp', failed_power_flow)
    with pytest.raises(LoadflowNotConverged) as caught:
        env.manual_reset(1, 6, 0)
    assert caught.value is original_error
    assert len(calls) == 1


def test_random_reset_retries_are_bounded_and_preserve_final_exception(monkeypatch):

    env = construct(monkeypatch, max_reset_attempts=3)
    original_error = LoadflowNotConverged('all sampled initial snapshots fail')
    calls = []

    class InfiniteRetryGuard(BaseException):
        pass

    def failed_power_flow(net):
        calls.append(1)
        if len(calls) > 3:
            raise InfiniteRetryGuard('random reset exceeded retry budget')
        raise original_error

    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)
    monkeypatch.setattr(env_module.pp, 'runpp', failed_power_flow)
    with pytest.raises(LoadflowNotConverged) as caught:
        env.reset()
    assert caught.value is original_error
    assert len(calls) == 3


def observation_env(monkeypatch, timing='current_pre_control'):
    env = construct(monkeypatch, episode_limit=2, inverter_s_max_mva=[1.],
                    reset_action=False, observation_timing=timing)
    monkeypatch.setattr(VoltageControl, 'get_obs', ORIGINAL_GET_OBS)
    monkeypatch.setattr(VoltageControl, 'get_state', ORIGINAL_GET_STATE)
    env.state_space = ['demand', 'pv', 'reactive', 'vm_pu']
    env.powergrid.bus = pd.DataFrame({'zone': ['zone0', 'zone1']})
    env.powergrid.sgen['bus'] = 1
    env.powergrid.load = pd.DataFrame({'bus': [1, 1], 'p_mw': [.3, .1], 'q_mvar': [.06, .04]})
    env.pv_histories = np.array([[.2], [.8]])
    env.active_demand_histories = np.array([[.3, .1], [.5, .4]])
    env.reactive_demand_histories = np.array([[.06, .04], [.15, .05]])
    env.pv_std = np.zeros(1)
    env.active_demand_std = env.reactive_demand_std = np.zeros(2)
    calls = []

    def power_flow(net):
        p, q = net.sgen.p_mw.sum(), net.sgen.q_mvar.sum()
        load_p, load_q = net.load.p_mw.sum(), net.load.q_mvar.sum()
        calls.append((float(p), float(q), float(load_p), float(load_q)))
        net.res_bus = pd.DataFrame({'p_mw': [0., load_p - p], 'q_mvar': [0., load_q - q],
                                   'vm_pu': [1., 1. + p * .01 - load_p * .02 + q * .03]})
        net.res_sgen = net.sgen[['p_mw', 'q_mvar']].copy()
        net.res_line = pd.DataFrame({'pl_mw': [.001]})

    monkeypatch.setattr(env_module.pp, 'runpp', power_flow)
    env._set_demand_and_pv(add_noise=False)
    power_flow(env.powergrid)
    env._pending_measurements = False
    return env, calls


def test_current_observation_is_lazy_uses_true_load_and_projects_held_q(monkeypatch):
    env, calls = observation_env(monkeypatch)
    _, done, _ = env.step([.8], add_noise=False)
    assert not done
    controlled_voltage = env.powergrid.res_bus.vm_pu.to_numpy().copy()
    controlled_calls = len(calls)
    # Evaluator reads the just-controlled voltage immediately after step.
    np.testing.assert_array_equal(env._get_res_bus_v(), controlled_voltage)
    assert len(calls) == controlled_calls
    obs = env.get_obs()[0]
    assert len(calls) == controlled_calls + 1
    # Current snapshot: aggregate loads, P, clipped physical held Q, current V.
    np.testing.assert_allclose(obs, [.9, .2, .8, .48, 1. + .008 - .018 + .0144])
    assert env._last_hold_q_projection_ratio == 1.
    state = env.get_state()
    np.testing.assert_allclose(state[:4], [0., .9, 0., .2])
    assert len(calls) == controlled_calls + 1
    _, done, info = env.step([0.], add_noise=False)
    assert done
    assert info['hold_q_projection_ratio'] == 1.
    assert not env._pending_measurements


def test_current_precontrol_failure_propagates_and_stays_pending(monkeypatch):
    env, _ = observation_env(monkeypatch)
    env.step([.8], add_noise=False)
    original_error = LoadflowNotConverged('pre-control held-Q power flow failed')
    monkeypatch.setattr(env_module, 'ppException', LoadflowNotConverged)
    def fail(net):
        raise original_error
    monkeypatch.setattr(env_module.pp, 'runpp', fail)
    with pytest.raises(LoadflowNotConverged) as caught:
        env.get_obs()
    assert caught.value is original_error
    assert env._pending_measurements


def test_legacy_observation_timing_remains_explicit_and_does_not_add_power_flow(monkeypatch):
    env, calls = observation_env(monkeypatch, timing='legacy')
    env.step([.8], add_noise=False)
    count = len(calls)
    env.get_obs()
    assert len(calls) == count
    assert env.environment_metadata()['observation']['timing'] == 'legacy'


def test_control_limit_saturation_is_separate_from_physical_saturation(monkeypatch):
    env = construct(monkeypatch, inverter_s_max_mva=[1.])
    env.powergrid.sgen['p_mw'] = .6
    q, stats = env._map_reactive_action([.8])
    np.testing.assert_allclose(q, [.64])
    assert stats['reactive_saturation_ratio'] == 0.
    assert stats['reactive_capacity_utilization'] == pytest.approx(.8)
    assert stats['control_limit_saturation_ratio'] == 1.
    assert stats['control_limit_utilization'] == pytest.approx(1.)
    env.powergrid.sgen['p_mw'] = 1.
    _, stats = env._map_reactive_action([.8])
    assert stats['control_limit_saturation_ratio'] == 0.
    assert stats['control_limit_utilization'] == 0.
