import contextlib
import io
import numpy as np
import pytest

from experiments.revision_scenario_pool import ScenarioSampler
from experiments.revision_training_environment import ScenarioTrainingEnv
from experiments.run_revision_pilot import environment_config


@pytest.mark.parametrize('scenario', ['case33_3min_final','case141_3min_final'])
def test_sampled_scale_changes_pv_but_never_rating_or_date_partition(scenario):
    rows = [{'day':d,'pv_scale':k,'stratum':'low'} for d in (1,2) for k in (.5,1.,1.5)]
    sampler = ScenarioSampler(rows,'B1',123)
    config = environment_config(scenario,100,phase='training',points=480,start_hour=0)
    with contextlib.redirect_stdout(io.StringIO()):
        env = ScenarioTrainingEnv(config,sampler)
    try:
        assert sampler.exposures.sum() == 0
        ratings = env.s_max.copy()
        seen = set()
        for _ in range(12):
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset()
            selected = env.current_scenario
            seen.add(selected['pv_scale'])
            np.testing.assert_array_equal(env.s_max,ratings)
            start = env._episode_start_index
            expected = env.pv_data.iloc[start:start+len(env.pv_histories)].to_numpy()
            np.testing.assert_allclose(env.pv_histories,expected*selected['pv_scale']/1.5)
            assert 1 <= env._episode_start_day < 999
            assert env.current_timestamp().hour == 0
            np.testing.assert_allclose(env.pv_std,env.reference_pv_std*selected['pv_scale']/1.5)
        assert seen == {.5,1.,1.5}
        assert sampler.exposures.sum() == 12
        state = env.checkpoint_state_dict()
        next_expected = sampler.sample()
        env.load_checkpoint_state_dict(state)
        assert sampler.sample() == next_expected
    finally:
        env.close()
