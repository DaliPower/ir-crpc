import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.revision_evaluation import (ObservationStress, PolicyObservationEnv,
    observation_layout, validate_evaluation_request)


def observation_fixture():
    return SimpleNamespace(args=SimpleNamespace(mode='distributed', reset_action=False),
        history=1, n_agents=2, obs_size=10, s_max=np.array([1.,2.]),
        state_space=['pv','demand','reactive','vm_pu','va_degree'],
        powergrid=SimpleNamespace(
            bus=pd.DataFrame({'zone':['zone0','zone1','zone2','zone2']},index=[0,1,2,3]),
            sgen=pd.DataFrame({'name':['zone1','zone2'],'bus':[1,2], 'p_mw':[.1,.2],'q_mvar':[0.,0.]})))


def observations(value=0):
    obs = np.arange(20,dtype=float).reshape(2,10)+value
    obs[0,6:] = 0
    return obs


def test_masks_follow_actual_layout_and_exclude_padding():
    layout = observation_layout(observation_fixture())
    np.testing.assert_array_equal(np.flatnonzero(layout['valid'][0]),range(6))
    np.testing.assert_array_equal(np.flatnonzero(layout['remote'][1]),[1,3,7,9])
    assert not layout['remote'][0].any()
    np.testing.assert_array_equal(np.flatnonzero(layout['voltage'][1]),[6,7])
    np.testing.assert_array_equal(np.flatnonzero(layout['power_scale'][1]),range(6))


def test_sensor_noise_does_not_mutate_inputs_padding_or_angle():
    layout = observation_layout(observation_fixture())
    stress = ObservationStress(layout, {'voltage_std':.005,'power_rating_fraction':.05},seed=9)
    obs = observations()
    before = obs.copy()
    received = np.asarray(stress.apply(obs,0))
    np.testing.assert_array_equal(obs,before)
    np.testing.assert_array_equal(received[~layout['valid']],obs[~layout['valid']])
    unchanged = ~(layout['voltage'] | (layout['power_scale'] > 0))
    np.testing.assert_array_equal(received[unchanged],obs[unchanged])
    assert np.any(received != obs)
    np.testing.assert_array_equal(stress.apply(obs,0),received)


@pytest.mark.parametrize('delay',[0,1,2])
def test_only_remote_channels_are_delayed(delay):
    layout = observation_layout(observation_fixture())
    stress = ObservationStress(layout,{'remote_delay_steps':delay})
    history = [observations(10*step) for step in range(4)]
    for step,obs in enumerate(history):
        received = np.asarray(stress.apply(obs,step))
        np.testing.assert_array_equal(received[layout['remote']],history[max(0,step-delay)][layout['remote']])
        np.testing.assert_array_equal(received[~layout['remote']],obs[~layout['remote']])


def test_missing_data_holds_last_received_and_preserves_local_channels():
    layout = observation_layout(observation_fixture())
    stress = ObservationStress(layout,{'remote_packet_loss':1.})
    first = np.asarray(stress.apply(observations(),0))
    received = np.asarray(stress.apply(observations(100),1))
    np.testing.assert_array_equal(received[layout['remote']],first[layout['remote']])
    np.testing.assert_array_equal(received[~layout['remote']],observations(100)[~layout['remote']])
    assert stress.counts['missing_agent_packets'] == 1


def test_five_step_burst_holds_then_recovers_without_future_information():
    layout = observation_layout(observation_fixture())
    stress = ObservationStress(layout,{'burst_start_step':2,'burst_length':5})
    for step in range(8):
        received = np.asarray(stress.apply(observations(step),step))
        source = 1 if 2 <= step < 7 else step
        np.testing.assert_array_equal(received[layout['remote']],observations(source)[layout['remote']])
    assert stress.counts['missing_agent_packets'] == 5


def test_observation_wrapper_preserves_physics_and_exit_affects_selected_action_only():
    base = observation_fixture()
    base.steps = 0
    base.manual_reset = lambda *args: (observations().tolist(),np.ones(5))
    base.get_obs = lambda: observations(base.steps).tolist()
    calls = []
    base.step = lambda actions,add_noise: (calls.append(np.asarray(actions).copy()) or (1.,False,{'physical_voltage':1.}))
    before = base.powergrid.sgen.copy(deep=True)
    stress = ObservationStress(observation_layout(base),{'voltage_std':.005,'disabled_controller':1})
    wrapper = PolicyObservationEnv(base,stress)
    corrupted,state = wrapper.manual_reset(1000,0,0)
    np.testing.assert_array_equal(state,np.ones(5))
    assert not np.array_equal(corrupted,observations())
    pd.testing.assert_frame_equal(base.powergrid.sgen,before)
    reward,done,info = wrapper.step([.3,.4])
    np.testing.assert_array_equal(calls[0],[.3,0.])
    assert reward == 1. and info['physical_voltage'] == 1.


def test_reset_repeats_seeded_corruption_for_same_day():
    stress = ObservationStress(observation_layout(observation_fixture()), {'voltage_std':.001},seed=3)
    stress.reset(1000)
    first = stress.apply(observations(),0)
    stress.reset(1001)
    assert not np.array_equal(first,stress.apply(observations(),0))
    stress.reset(1000)
    np.testing.assert_array_equal(first,stress.apply(observations(),0))


@pytest.mark.parametrize('phase,days,points',[('formal',[1000],480),('formal',list(range(1030,1059)),480),
    ('development',[1030],480),('development',[1000],240),('development',[1000,1000],480)])
def test_evaluation_rejects_date_budget_or_split_changes(phase,days,points):
    with pytest.raises(ValueError):
        validate_evaluation_request(phase,days,points)


def test_valid_formal_and_development_shapes():
    validate_evaluation_request('formal',list(range(1030,1060)))
    validate_evaluation_request('development',[1000,1029])


def test_unsupported_history_cannot_silently_corrupt_mask():
    env = observation_fixture()
    env.history = 2
    with pytest.raises(ValueError,match='history=1'):
        observation_layout(env)


def test_physical_outage_removes_p_profile_and_available_s_not_controller_identity(monkeypatch):
    from experiments.revision_evaluation import RevisionEvaluationEnv, VoltageControl
    env = RevisionEvaluationEnv.__new__(RevisionEvaluationEnv)
    env.physical_exit_controller = 1
    env.steps,env.history = 0,1
    env.pv_histories = np.array([[.4,.6],[.3,.5]])
    np.testing.assert_array_equal(env._get_pv_history(),[[.4,0.]])
    np.testing.assert_array_equal(env.pv_histories,[[.4,.6],[.3,.5]])
    monkeypatch.setattr(VoltageControl,'_set_reactive_power_boundary',lambda self:setattr(self,'s_max',np.array([1.,2.])))
    env._set_reactive_power_boundary()
    np.testing.assert_array_equal(env.s_max,[1.,0.])
    np.testing.assert_array_equal(env.installed_ratings_before_outage,[1.,2.])


def test_formal_policy_rejects_pilot_artifacts_before_loading_weight(tmp_path):
    import json
    from experiments.revision_evaluation import load_policy_artifacts
    for name,value in [('manifest.json',{'scenario':'case141_3min_final','solver':'matd3','seed':100}),
                       ('effective_configuration.json',{}),('pilot_result.json',{'status':'completed'})]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='completed formal training'):
        load_policy_artifacts(tmp_path,phase='formal')


def test_development_policy_rejects_changed_checkpoint(tmp_path):
    import json
    from experiments.revision_evaluation import load_policy_artifacts
    for name,value in [('manifest.json',{'scenario':'case141_3min_final','solver':'matd3','seed':100}),
                       ('effective_configuration.json',{}),('pilot_result.json',{'status':'completed','weight_sha256':{'best':'wrong'}})]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    (tmp_path/'best_model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint hash'):
        load_policy_artifacts(tmp_path,phase='development')


def test_evaluation_provenance_locks_executed_dependencies_not_standalone_reporting():
    from experiments.revision_evaluation import evaluation_source_manifest
    sources=evaluation_source_manifest()['source_sha256']
    assert 'experiments/revision_evaluation.py' in sources
    assert 'experiments/revision_pv_layouts.py' in sources
    assert 'experiments/run_q1_fixed_test_traditional_baselines.py' in sources
    assert 'experiments/revision_benchmark.py' not in sources
    assert 'experiments/revision_diagnostics.py' not in sources
    assert any(k.startswith('critics/') for k in sources)


@pytest.mark.parametrize('solver',['matd3_reference','matd3','maddpg','facmaddpg','mappo'])
def test_cuda_loader_moves_all_solver_parameters_and_can_execute_policy(solver,tmp_path):
    import json
    import torch
    from experiments.run_q1_ml_compare import load_alg_config
    from experiments.revision_evaluation import load_policy_artifacts
    from experiments.run_revision_pilot import file_hash
    from models.model_registry import Model
    from utilities.util import convert
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    configuration={'action_scale':.8,'action_bias':0.}
    values=load_alg_config(solver,configuration)
    values.update(cuda=False,agent_num=2,obs_size=3,action_dim=1,hid_size=8,
                  batch_size=2,layernorm=False,seed=100)
    args=convert(values)
    target=Model[solver](args) if args.target else None
    original=Model[solver](args,target) if args.target else Model[solver](args)
    checkpoint=tmp_path/'best_model.pt'
    torch.save({'model_state_dict':original.state_dict(),'episode':1},checkpoint)
    data=tmp_path/'data'
    data.mkdir()
    (data/'test.csv').write_text('fixed input',encoding='utf-8')
    manifest={'scenario':'case33_3min_final','solver':solver,'seed':100,
              'validation_environment':{'data_path':str(data)},
              'data_sha256':{'test.csv':file_hash(data/'test.csv')}}
    report={'status':'completed','weight_sha256':{'best':file_hash(checkpoint)}}
    for name,value in [('manifest.json',manifest),('effective_configuration.json',{'train_args':values}),
                       ('pilot_result.json',report)]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    loaded,loaded_args,_,_=load_policy_artifacts(tmp_path,phase='development',cpu=False)
    assert loaded_args.cuda
    assert not loaded.training
    assert {p.device.type for p in loaded.parameters()} == {'cuda'}
    observation=torch.zeros((1,2,3),device='cuda')
    with torch.no_grad():
        action,*_=loaded.get_actions(observation,status='test',exploration=False,
            actions_avail=torch.ones((1,2,1),device='cuda'),target=False,
            last_hid=torch.zeros((1,2,8),device='cuda'))
    assert action.device.type=='cuda' and torch.isfinite(action).all()
