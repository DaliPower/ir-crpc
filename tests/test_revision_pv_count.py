import json
from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp
import pytest

from experiments.revision_pv_count_export import export_dataset,validate_dataset_manifest,derive_equivalent_pool
from experiments.revision_pv_count_environment import ParentPVNoiseStream
from experiments.revision_provenance import data_hashes


def tiny_source(tmp_path):
    source=tmp_path/'original'
    source.mkdir()
    net=pp.create_empty_network()
    b0=pp.create_bus(net,vn_kv=11,zone='zone0')
    b1=pp.create_bus(net,vn_kv=11,zone='zone1')
    pp.create_ext_grid(net,b0)
    pp.create_line_from_parameters(net,b0,b1,1.,.1,.2,0.,1.)
    pp.create_load(net,b1,p_mw=.3,q_mvar=.04)
    pp.create_sgen(net,b1,p_mw=.1,q_mvar=.02,name='zone1')
    pp.create_sgen(net,b1,p_mw=.2,q_mvar=-.01,name='zone1')
    pp.to_pickle(net,str(source/'model.p'))
    times=pd.date_range('2020-01-01 00:03',periods=1440,freq='3min',name='time')
    pv=pd.DataFrame({'pv0':np.linspace(.01,.1,len(times)),'pv1':np.linspace(.02,.2,len(times))},index=times)
    pv.to_csv(source/'pv_active.csv')
    pd.DataFrame({'load0':.3},index=times).to_csv(source/'load_active.csv')
    pd.DataFrame({'load0':.04},index=times).to_csv(source/'load_reactive.csv')
    return source


def test_serialized_count_dataset_preserves_load_bytes_bus_pq_s_and_timeline(tmp_path):
    source=tiny_source(tmp_path)
    before=data_hashes(source)
    exported=export_dataset(source,tmp_path/'count3',3,scenario='tiny')
    verified=validate_dataset_manifest(tmp_path/'count3/dataset_manifest.json')
    assert verified['agent_count']==3
    assert verified['source_timeline']==verified['derived_timeline']
    assert verified['source_timeline']['day_zero_complete'] is False
    assert verified['equivalence']['maximum_residual']<1e-10
    assert data_hashes(source)==before
    for name in ['load_active.csv','load_reactive.csv']:
        assert (source/name).read_bytes()==(Path(exported['data_directory'])/name).read_bytes()
    with pytest.raises(FileExistsError):
        export_dataset(source,tmp_path/'count3',3)


def test_modified_derived_input_or_share_mapping_is_rejected(tmp_path):
    source=tiny_source(tmp_path)
    export_dataset(source,tmp_path/'count3',3)
    path=tmp_path/'count3/dataset_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['mapping'][0]['share']=1.
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='shares'):
        validate_dataset_manifest(path)


def test_derived_pool_inherits_only_matched_original_screening(tmp_path):
    source=tiny_source(tmp_path)
    manifest=export_dataset(source,tmp_path/'count3',3)
    original_pool=tmp_path/'original_pool.json'
    rows=[{'day':1,'pv_scale':k,'stratum':'low'} for k in [.5,1.,1.5]]
    original_pool.write_text(json.dumps({'data_directory':str(source),'data_sha256':data_hashes(source),
        's_max_mva':manifest['original_s_max_mva'],'rows':rows}))
    derived=derive_equivalent_pool(original_pool,tmp_path/'count3/dataset_manifest.json',tmp_path/'derived_pool.json')
    assert derived['rows']==rows
    assert derived['screening_reused_not_rerun']
    assert len(derived['s_max_mva'])==3


def test_parent_noise_keeps_per_bus_perturbation_and_subsequent_load_draws_identical():
    original=np.random.default_rng(19)
    child=np.random.default_rng(19)
    shared=ParentPVNoiseStream(child,[0,0,1],2)
    parent_draw=original.standard_normal(2)
    child_draw=shared.standard_normal((3,))
    np.testing.assert_array_equal(child_draw,parent_draw[[0,0,1]])
    # Share-scaled std makes the summed half-normal PV perturbation identical.
    perturb_original=np.abs(parent_draw)*np.array([.02,.05])
    perturb_children=np.abs(child_draw)*np.array([.01,.01,.05])
    np.testing.assert_allclose([sum(perturb_children[:2]),perturb_children[2]],perturb_original,rtol=0,atol=1e-16)
    np.testing.assert_array_equal(shared.standard_normal(4),original.standard_normal(4))
    np.testing.assert_array_equal(shared.standard_normal(4),original.standard_normal(4))


def test_parent_noise_rejects_changed_noise_call_layout():
    stream=ParentPVNoiseStream(np.random.default_rng(1),[0,0,1],2)
    with pytest.raises(ValueError,match='PV layout'):
        stream.standard_normal((4,))


def test_count_reward_preserves_total_q_penalty_with_new_device_mean_exposed():
    from experiments.revision_pv_count_environment import CountNoiseMixin
    from experiments.risk_aware_protocol import reward_components
    class Base:
        def _calc_reward(self,info=None):
            components=reward_components([.97,1.01],[.1,.1,-.3],v_lower=.95,v_upper=1.05,
                voltage_weight=1.,violation_weight=0.,q_weight=self.q_weight)
            return components['reward'],components
    class Count(CountNoiseMixin,Base):
        pass
    env=object.__new__(Count)
    env.n_agents=3
    env.pv_count_original_agent_count=2
    env.q_weight=.3
    reward,info=env._calc_reward()
    expected=reward_components([.97,1.01],[.2,-.3],v_lower=.95,v_upper=1.05,
        voltage_weight=1.,violation_weight=0.,q_weight=.3)
    assert reward==pytest.approx(expected['reward'])
    assert info['q_loss_weighted']==pytest.approx(expected['q_loss_weighted'])
    assert info['q_loss']==pytest.approx(.5/3)
    assert info['total_abs_q_mvar']==pytest.approx(.5)
    assert info['q_loss_original_count_normalized']==pytest.approx(.25)
