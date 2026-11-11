"""Observer-only regression: preserve failures while recognizing a precise live resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import revision_execution_status as status
from experiments.revision_live_file_reader import open_shared_read


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    protocol=tmp_path/'protocol'; root=tmp_path/'results'
    run={'run_id':'fac_B2__case33__seed100','solver':'facmaddpg','scenario':'case33','seed':100}
    plan={'protocol':{'training_steps':4800},'runs':[run]}
    plan_path=protocol/'development_plans_01/all_solvers.json'
    write_json(plan_path,plan)
    for folder in ('development_plans_01','dev_fair_q_01'):
        for q in ('0','0.03','0.1','0.3'):
            write_json(protocol/folder/f'q{q}.json',{'runs':[]})
    source={'fixture.py':'abc'}
    archive=protocol/'source.json'; write_json(archive,{'source':{'source_sha256':source}})
    directory=root/run['run_id']
    write_json(directory/'manifest.json',{'run':run,'plan_sha256':status.canonical_hash(plan),
                                         'source_sha256':source})
    (directory/'training_history.jsonl').write_text(json.dumps({'attempt':1,'episode':1,
        'environment_steps':1,'episode_steps':1,'reason':'controlled_failure'})+'\n',encoding='utf-8')
    # It must never be opened by the observer, whether active or stopped.
    (directory/'progress.json').write_bytes(b'UNREADABLE AS JSON')
    monkeypatch.setattr(status,'training_source_manifest',lambda:{'source_sha256':source})
    monkeypatch.setattr(status,'active_training_processes',lambda:[])
    original_text,original_bytes=status.read_shared_text,status.read_shared_bytes
    def permitted(path):
        assert Path(path).name not in ('progress.json','progress.json.tmp','training_state.pt',
                                       'queue_status.json','queue_results.json')
    def checked_text(path):
        permitted(path); return original_text(path)
    def checked_bytes(path):
        permitted(path); return original_bytes(path)
    monkeypatch.setattr(status,'read_shared_text',checked_text)
    monkeypatch.setattr(status,'read_shared_bytes',checked_bytes)
    return SimpleNamespace(protocol=protocol,root=root,run=run,plan=plan,plan_path=plan_path,
                           directory=directory,source=source,archive=archive)


def inspect(f):
    return status.execution_status(f.protocol,f.root,f.archive)['runs'][0]


def failure(f):
    (f.directory/'failure_history.jsonl').write_text(json.dumps({'error_type':'PermissionError',
        'message':'WinError 5 progress.json.tmp -> progress.json','wall_seconds':1})+'\n',encoding='utf-8')


def live(f,**changes):
    return dict({'pid':123,'create_time':10.,'run_id':f.run['run_id'],'plan':f.plan_path.resolve(),
                 'output':f.directory.resolve(),'resume':True},**changes)


def test_inactive_old_failure_stays_visible(fixture):
    failure(fixture)
    row=inspect(fixture)
    assert row['status']=='exception_requires_diagnosis_or_resume'
    assert row['prior_exception_count']==1
    assert row['last_error']['error_type']=='PermissionError'
    assert row['actual_steps']==1


def test_verified_live_resume_preserves_error_and_reverts_to_failure_after_exit(fixture,monkeypatch):
    failure(fixture)
    before=(fixture.directory/'failure_history.jsonl').read_bytes()
    monkeypatch.setattr(status,'active_training_processes',lambda:[live(fixture)])
    row=inspect(fixture)
    assert row['status']=='in_progress' and row['prior_exception_count']==1
    assert row['active_training_processes'][0]['pid']==123
    monkeypatch.setattr(status,'active_training_processes',lambda:[])
    assert inspect(fixture)['status']=='exception_requires_diagnosis_or_resume'
    assert (fixture.directory/'failure_history.jsonl').read_bytes()==before


@pytest.mark.parametrize('field,value',[('run_id','other'),('plan',Path('wrong_plan')),
    ('output',Path('wrong_output')),('resume',False)])
def test_unrelated_process_cannot_hide_failure(fixture,monkeypatch,field,value):
    failure(fixture)
    monkeypatch.setattr(status,'active_training_processes',lambda:[live(fixture,**{field:value})])
    assert inspect(fixture)['status']=='exception_requires_diagnosis_or_resume'


def test_changed_runtime_source_cannot_make_old_failure_live(fixture,monkeypatch):
    failure(fixture)
    monkeypatch.setattr(status,'active_training_processes',lambda:[live(fixture)])
    monkeypatch.setattr(status,'training_source_manifest',lambda:{'source_sha256':{'fixture.py':'different'}})
    assert inspect(fixture)['status']=='exception_requires_diagnosis_or_resume'


def test_wrong_manifest_identity_is_invalid_even_with_matching_pid(fixture,monkeypatch):
    failure(fixture)
    path=fixture.directory/'manifest.json'; value=json.loads(path.read_text()); value['run']['seed']=101
    write_json(path,value)
    monkeypatch.setattr(status,'active_training_processes',lambda:[live(fixture)])
    assert inspect(fixture)['status']=='invalid_source_or_plan'


def test_completion_is_allowed_with_retained_failure_history_without_reading_large_logs(fixture,monkeypatch):
    failure(fixture)
    write_json(fixture.directory/'training_result.json',{'status':'completed','environment_steps':4800,
        'formal_test_evaluated':False,'wall_seconds':10.})
    def reject_history(_):
        raise AssertionError('completed runs must not reread training history')
    monkeypatch.setattr(status,'history_progress',reject_history)
    row=inspect(fixture)
    assert row['status']=='completed' and row['prior_exception_count']==1


def test_append_history_honors_resume_boundary_and_discloses_torn_tail(fixture):
    path=fixture.directory/'training_history.jsonl'
    rows=[{'attempt':1,'episode':n,'environment_steps':n,'episode_steps':1} for n in (1,2,3)]
    path.write_bytes(('\n'.join(json.dumps(r) for r in rows)+'\n').encode()+b'{"bad":"\xe4\xb8')
    (fixture.directory/'resume_history.jsonl').write_text(json.dumps(
        {'attempt':2,'episode':1,'steps':1})+'\n',encoding='utf-8')
    row=inspect(fixture)
    assert row['actual_steps']==1 and row['episodes']==1
    assert row['history_fragments']==[4] and row['uncommitted_or_superseded_records']==2
    with path.open('ab') as h:
        h.write(b'\n'+json.dumps({'attempt':2,'episode':2,'environment_steps':2,'episode_steps':1}).encode()+b'\n')
    row=inspect(fixture)
    assert row['actual_steps']==2 and row['history_fragments']==[4]


def test_process_snapshot_requires_exact_worktree_script_and_arguments(monkeypatch,tmp_path):
    import psutil
    script=status.ROOT/'experiments/run_revision_training.py'
    plan=tmp_path/'plan.json'; output=tmp_path/'result'
    base=[sys.executable,'-u',str(script),'--plan',str(plan),'--run-id','wanted','--output',str(output),'--resume']
    commands=[base,[sys.executable,'-c',' '.join(base)],
              [sys.executable,'-u',str(tmp_path/'experiments/run_revision_training.py')]+base[3:],
              base+['--run-id','duplicate']]
    processes=[SimpleNamespace(info={'pid':n,'cmdline':cmd,'cwd':str(status.ROOT),
        'status':psutil.STATUS_RUNNING,'create_time':10.}) for n,cmd in enumerate(commands,1)]
    monkeypatch.setattr(psutil,'process_iter',lambda attrs:processes)
    rows=status.active_training_processes()
    assert len(rows)==1 and rows[0]['pid']==1
    assert rows[0]['plan']==plan.resolve() and rows[0]['output']==output.resolve()


def test_real_process_repeated_progress_replacements_survive_active_history_observer(fixture):
    """200 real replace calls without retries while status repeatedly reads logs."""
    script='''
import json, os, sys, time
from pathlib import Path
p=Path(sys.argv[1])
for n in range(2,202):
    with (p/'training_history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'attempt':1,'episode':n,'environment_steps':n,'episode_steps':1})+'\\n')
        f.flush(); os.fsync(f.fileno())
    t=p/'progress.json.tmp'
    t.write_text(json.dumps({'steps':n}),encoding='utf-8')
    os.replace(t,p/'progress.json')
    time.sleep(.002)
print('all_200_replacements_succeeded')
'''
    child=subprocess.Popen([sys.executable,'-u','-c',script,str(fixture.directory)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    reads=0
    try:
        # A permanently held log reader demonstrates that observer sharing is
        # compatible with append; no replace target is opened at any point.
        with open_shared_read(fixture.directory/'training_history.jsonl'):
            while child.poll() is None:
                row=inspect(fixture)
                assert 1<=row['actual_steps']<=201
                reads+=1
        stdout,stderr=child.communicate(timeout=10)
        assert child.returncode==0,stderr
        assert stdout.strip()=='all_200_replacements_succeeded' and reads>1
        assert inspect(fixture)['actual_steps']==201
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=10)
