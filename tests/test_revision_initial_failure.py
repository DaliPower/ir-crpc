from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pandapower.auxiliary import LoadflowNotConverged

from experiments.fixed_test_protocol import _evaluate_fixed_days


class FailedInitialEnvironment:
    def __init__(self, error):
        self.args = SimpleNamespace(reset_action=False)
        self.error = error
        self.powergrid = SimpleNamespace(
            bus=pd.DataFrame(index=[0, 1]),
            res_bus=pd.DataFrame({'vm_pu': [np.nan, np.nan]}, index=[0, 1]))

    def manual_reset(self, day, hour, interval):
        self.day, self.hour = day, hour
        raise self.error

    def current_timestamp(self):
        return np.datetime64('2012-01-01') + np.timedelta64(self.day, 'D') + np.timedelta64(self.hour, 'h')


def test_initial_powerflow_failure_retains_full_planned_denominator():
    env = FailedInitialEnvironment(LoadflowNotConverged('initial voltage failure'))
    def no_controller(_state):
        raise AssertionError('must not build policy for an unsolved initial state')
    result = _evaluate_fixed_days(env=env, days=[1000, 1001], max_steps=4,
                                 start_hour=6, transition_factory=no_controller, strict_shape=False)
    assert result.summary['planned_steps'] == 8
    assert result.summary['failed_steps'] == 8
    assert result.summary['vrate'] == 1
    assert result.summary['cr'] == 0
    assert result.node_voltages.shape == (8, 2)
    assert np.isnan(result.node_voltages).all()
    assert not result.executed_mask.any()


def test_initial_programming_error_is_not_silently_scored_as_grid_failure():
    env = FailedInitialEnvironment(ValueError('invalid schema'))
    with pytest.raises(ValueError, match='invalid schema'):
        _evaluate_fixed_days(env=env, days=[1000], max_steps=4, start_hour=6,
                             transition_factory=lambda _: None, strict_shape=False)


def test_next_precontrol_failure_does_not_erase_successful_current_snapshot():
    class Env(FailedInitialEnvironment):
        def manual_reset(self, day, hour, interval):
            self.day, self.hour = day, hour
            self.powergrid.res_bus['vm_pu'] = [1.0, 1.01]
            return [], []

        def get_obs(self):
            raise LoadflowNotConverged('next pre-control state failed')

    info = {'percentage_of_v_out_of_control': 0.0,
            'totally_controllable_ratio': 1.0, 'destroy': 0.0,
            'inverter_p_mw': [0.1], 'inverter_q_mvar': [0.01],
            'inverter_s_max_mva': [0.2]}
    result = _evaluate_fixed_days(
        env=Env(None), days=[1000], max_steps=4, start_hour=6,
        transition_factory=lambda _: lambda __: (-0.1, False, info), strict_shape=False)
    assert result.executed_mask.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(result.node_voltages[0], [1.0, 1.01])
    assert np.isnan(result.node_voltages[1:]).all()
    assert result.daily_rows[0]['failure_step'] == 1
    assert result.summary['vrate'] == 0.75
