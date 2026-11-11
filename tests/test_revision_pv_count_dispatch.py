import json
from types import SimpleNamespace
import threading
from pathlib import Path
import pytest

from experiments import revision_pv_count_pipeline as pipeline
from experiments import revision_pv_count_evaluation as evaluation


def setup_queue(tmp_path,monkeypatch,*,available=64.):
    import psutil
    plan={'protocol':{'phase':'formal'},'runs':[{'run_id':f'r{i:02d}'} for i in range(40)],
        'plan_sha256':'plan','source':{'source_sha256':{'training':'fixed'}},
        'evaluation_source_sha256':{'evaluation':'fixed'},'evaluation_registry_sha256':'registry'}
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    root=tmp_path/'training'
    monkeypatch.setattr(pipeline,'validate_count_plan',lambda plan:None)
    monkeypatch.setattr(pipeline,'validate_registry',lambda plan:None)
    monkeypatch.setattr(pipeline,'checked_queue_path',lambda path:Path(path).resolve())
    monkeypatch.setattr(evaluation,'evaluation_source_manifest',lambda:{'source_sha256':{'evaluation':'fixed'}})
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=available*1024**3))
    verified=[]
    monkeypatch.setattr(pipeline,'verify_training_artifacts',lambda directory,run,plan:verified.append(run['run_id']))
    return path,root,plan,verified


def test_count_parallel_dispatch_is_bounded_and_keeps_all_forty_records(tmp_path,monkeypatch):
    path,root,_,_=setup_queue(tmp_path,monkeypatch)
    barrier=threading.Barrier(2)
    calls=[]
    lock=threading.Lock()
    def work(run,*args,**kwargs):
        with lock:
            calls.append(run['run_id'])
        barrier.wait(timeout=3)
        return {'run_id':run['run_id'],'status':'completed','exit_code':0}
    monkeypatch.setattr(pipeline,'_train_one',work)
    result=pipeline.train(path,root,max_runs=2,workers=2)
    assert set(calls)=={'r00','r01'} and len(calls)==2
    assert result['launched_this_invocation']==2
    assert len(result['records'])==40
    assert sum(r['status']=='planned_not_dispatched' for r in result['records'])==38
    assert not result['all_training_complete']


def test_count_failure_stops_new_launches_without_replacing_or_deleting_seed(tmp_path,monkeypatch):
    path,root,_,_=setup_queue(tmp_path,monkeypatch)
    calls=[]
    def work(run,*args,**kwargs):
        calls.append(run['run_id'])
        return {'run_id':run['run_id'],'status':'unresolved_training_exception','exit_code':1}
    monkeypatch.setattr(pipeline,'_train_one',work)
    result=pipeline.train(path,root,max_runs=40,workers=1)
    assert calls==['r00']
    assert result['records'][0]['status']=='unresolved_training_exception'
    assert len(result['records'])==40 and result['status']=='unresolved'
    assert not result['all_training_complete']


def test_count_completed_verified_and_partial_seed_resumed_not_skipped(tmp_path,monkeypatch):
    path,root,_,verified=setup_queue(tmp_path,monkeypatch)
    (root/'r00').mkdir(parents=True)
    (root/'r00/training_result.json').write_text('{}')
    (root/'r01').mkdir()
    with pytest.raises(FileExistsError,match='explicit resume'):
        pipeline.train(path,root,max_runs=1,workers=2)
    calls=[]
    def work(run,*args,**kwargs):
        calls.append((run['run_id'],kwargs['resume']))
        return {'run_id':run['run_id'],'status':'completed','exit_code':0}
    monkeypatch.setattr(pipeline,'_train_one',work)
    result=pipeline.train(path,root,max_runs=1,workers=2,resume=True)
    assert calls==[('r01',True)]
    assert verified.count('r00')==2
    assert result['records'][0]['status']=='verified_completed'
    assert result['records'][2]['status']=='planned_not_dispatched'


def test_count_memory_guard_reserves_pending_process_capacity(tmp_path,monkeypatch):
    path,root,_,_=setup_queue(tmp_path,monkeypatch,available=17.)
    monkeypatch.setattr(pipeline,'_train_one',lambda *a,**kw:pytest.fail('must not launch below reservation'))
    result=pipeline.train(path,root,max_runs=40,workers=4,min_free_gib=10,memory_reserve_gib=8)
    assert result['launched_this_invocation']==0
    assert result['records'][0]['status']=='waiting_for_memory'
    assert len(result['records'])==40


@pytest.mark.parametrize('workers',[0,5,True])
def test_count_parallel_limit_rejects_invalid_workers(tmp_path,monkeypatch,workers):
    path,root,_,_=setup_queue(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='resource limit'):
        pipeline.train(path,root,workers=workers)


def test_count_duplicate_run_ids_and_reviewed_failure_records_are_rejected(tmp_path,monkeypatch):
    path,root,plan,_=setup_queue(tmp_path,monkeypatch)
    plan['runs'][-1]=plan['runs'][0]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='unique'):
        pipeline.train(path,root)
    plan['runs'][-1]={'run_id':'r39'}
    path.write_text(json.dumps(plan))
    failure=tmp_path/'failure.json'
    failure.write_text(json.dumps({'run_directory':str(root/'r00')}))
    monkeypatch.setattr(pipeline,'validate_failure_record',lambda *a:{'run_id':'r00'})
    with pytest.raises(ValueError,match='duplicate'):
        pipeline.train(path,root,failure_records=[failure,failure])


def test_count_training_cli_returns_nonzero_for_unresolved_child(monkeypatch):
    monkeypatch.setattr(pipeline.sys,'argv',['count','train','--plan','fake.json','--training-root','fake'])
    monkeypatch.setattr(pipeline,'train',lambda *a,**kw:{'status':'unresolved',
        'launched_this_invocation':1,'all_training_complete':False})
    assert pipeline.main()==1
