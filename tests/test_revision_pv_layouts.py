from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from experiments.revision_pv_layouts import split_inverters,relocate_within_zones


def network():
    return SimpleNamespace(
        sgen=pd.DataFrame({'bus':[1,2], 'name':['zone1','zone1'],
                           'p_mw':[.2,.3],'q_mvar':[0.,0.],'sn_mva':[.4,.6]}),
        bus=pd.DataFrame({'zone':['zone0','zone1','zone1','zone1']},index=[0,1,2,3]),
        ext_grid=pd.DataFrame({'bus':[0]}))


def test_split_preserves_every_snapshot_total_and_bus_injections():
    net=network()
    pv=pd.DataFrame([[.1,.2],[.2,.3]])
    new,profile,ratings,mapping=split_inverters(net,pv,np.array([.4,.6]),3)
    assert len(new.sgen)==3 and len(net.sgen)==2
    np.testing.assert_allclose(profile.sum(axis=1),pv.sum(axis=1))
    np.testing.assert_allclose(profile.iloc[:,0]+profile.iloc[:,1],pv.iloc[:,0])
    assert sum(ratings)==pytest.approx(1)
    assert mapping[0]['original_agent_id']==mapping[1]['original_agent_id']==0
    assert new.sgen.bus.tolist()==[1,1,2]


def test_relocation_preserves_zone_and_equipment_identity():
    net=network()
    new,mapping=relocate_within_zones(net,1)
    assert net.sgen.bus.tolist()==[1,2]
    assert new.sgen.bus.tolist()==[2,3]
    assert new.sgen.name.tolist()==net.sgen.name.tolist()
    np.testing.assert_array_equal(new.sgen.p_mw,net.sgen.p_mw)


def test_split_does_not_claim_arbitrary_new_device_counts():
    with pytest.raises(ValueError):
        split_inverters(network(),pd.DataFrame([[1,2]]),[1,2],5)
