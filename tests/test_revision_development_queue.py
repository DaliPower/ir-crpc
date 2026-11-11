import json
from pathlib import Path
import pytest
from experiments import revision_development_queue as queue


def test_original_grid_stays_150_without_changing_seeds_or_plans():
    registry=queue.read(queue.ROOT/'exp/revision_access_41572/protocol/dev_fair_q_01/fair_q_registry.json')
    rows=queue.jobs_for(registry)
    assert len(rows)==150
    assert sum(r['group']=='d01' for r in rows)==30
    assert {r['run']['seed'] for r in rows}=={100,101,102}
    assert {r['group'] for r in rows}=={'d01',*[p+q for p in ('tune_q','fair_q') for q in ('0','0.03','0.1','0.3')]}


@pytest.fixture
def fake_queue(tmp_path,monkeypatch):
    output=tmp_path/'result';registry_path=tmp_path/'registry.json';registry_path.write_text('{}')
    registry={'results_binding':{'results_root':str(output.resolve()),'source_sha256':{'source.py':'hash'}}}
    monkeypatch.setattr(queue,'load_registry',lambda p:registry)
    monkeypatch.setattr(queue,'training_source_manifest',lambda:{'source_sha256':{'source.py':'hash'}})
    monkeypatch.setattr(queue,'jobs_for',lambda r:[{'plan':'plan.json','group':'d01','run':{'run_id':str(i)}} for i in range(3)])
    calls=[]
    def worker(plan,run,folder,resume,cuda):
        calls.append((run['run_id'],resume,cuda));return {'run_id':run['run_id'],'status':'completed'}
    monkeypatch.setattr(queue,'worker',worker)
    return registry_path,output,registry,calls


def test_failure_stops_new_launches_and_preserves_failure_record(fake_queue,monkeypatch):
    registry,output,_,calls=fake_queue
    def fail(plan,run,folder,resume,cuda):
        calls.append(run['run_id']);raise RuntimeError('implementation failure, not a score')
    monkeypatch.setattr(queue,'worker',fail)
    assert queue.dispatch(registry,output,workers=1)==1
    assert calls==['0']
    result=json.loads(next((output/'queue/invocations').glob('*/result.json')).read_text())
    assert result['status']=='stopped_for_diagnosis' and result['failed']==1
    assert result['records'][0]['message']=='implementation failure, not a score'


def test_resume_is_same_root_identity_and_explicit(fake_queue):
    registry,output,_,calls=fake_queue
    assert queue.dispatch(registry,output,workers=1)==0
    assert all(not c[1] and not c[2] for c in calls)
    with pytest.raises(FileExistsError): queue.dispatch(registry,output,workers=1)
    calls.clear()
    assert queue.dispatch(registry,output,workers=1,resume=True)==0
    assert all(c[1] and not c[2] for c in calls)
    with pytest.raises(ValueError,match='identity changed'):
        queue.dispatch(registry,output,workers=2,resume=True)


def test_wrong_root_or_source_never_launches(fake_queue,monkeypatch):
    registry,output,_,calls=fake_queue
    with pytest.raises(ValueError,match='root'):
        queue.dispatch(registry,output.parent/'old')
    monkeypatch.setattr(queue,'training_source_manifest',lambda:{'source_sha256':{'source.py':'changed'}})
    with pytest.raises(ValueError,match='source'):
        queue.dispatch(registry,output)
    assert not calls and not output.exists()


def test_source_change_between_runs_stops_before_next_seed(fake_queue,monkeypatch):
    registry,output,_,calls=fake_queue
    def mutate(plan,run,folder,resume,cuda):
        calls.append(run['run_id'])
        monkeypatch.setattr(queue,'training_source_manifest',lambda:{'source_sha256':{'source.py':'changed'}})
        return {'run_id':run['run_id'],'status':'completed'}
    monkeypatch.setattr(queue,'worker',mutate)
    with pytest.raises(ValueError,match='changed during execution'):
        queue.dispatch(registry,output,workers=1)
    assert calls==['0']
    assert not list((output/'queue/invocations').glob('*/result.json'))
