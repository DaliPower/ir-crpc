"""Resumable subprocess queue. Never truncates budgets, changes seeds or retries failures silently."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.round2_ablation import read,validate_plan,verify_reuse,file_hash,RUN_ROOT
from experiments.run_revision_pilot import json_write


def schedule_runs(plan,runs):
    scenarios={'case33_3min_final':0,'case141_3min_final':1}
    return sorted(runs,key=lambda r:(r['run_id'] in plan.get('reuse_runs',{}),
        r['seed'],r['round2_variant'],scenarios[r['scenario']]))


def read_control(path):
    control=read(path)
    if type(control.get('workers')) is not int or not 1<=control['workers']<=8:
        raise ValueError('control workers must be an integer in 1..8')
    if type(control.get('pause_dispatch')) is not bool:
        raise ValueError('pause_dispatch must be Boolean')
    return dict(workers=control['workers'],pause_dispatch=control['pause_dispatch'])


def available_gib():
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong)
            for name in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('cannot read physical memory')
    return status.available/2**30


def verify_training(directory,plan,run):
    report=read(directory/'training_result.json'); manifest=read(directory/'manifest.json')
    if report['status']!='completed' or report['environment_steps']!=plan['protocol']['training_steps']:
        raise ValueError('incomplete training cannot be reused')
    if manifest['plan_sha256']!=plan['plan_sha256'] or manifest['run']!=run:
        raise ValueError('training belongs to another plan/run')
    for name,digest in report['weight_sha256'].items():
        if file_hash(directory/f'{name}_model.pt')!=digest: raise ValueError('checkpoint mismatch')


def execute(plan_path,plan,run,root,retry_failed,cuda):
    directory=root/'tr'/run['run_id']; evaluation=root/'ev'/run['run_id']; logs=root/'logs'/run['run_id']
    logs.mkdir(parents=True,exist_ok=True)
    attempts=list(logs.glob('attempt*.json'))
    if attempts and not retry_failed:
        last=read(sorted(attempts)[-1])
        if last.get('status')=='failed': return dict(run_id=run['run_id'],status='failed_requires_explicit_retry')
    number=len(attempts)+1; receipt=logs/f'attempt{number:04d}.json'
    status=dict(run_id=run['run_id'],plan_sha256=plan['plan_sha256'],status='running',pid=os.getpid(),started_epoch=time.time())
    environment=dict(os.environ)
    environment.update(plan.get('execution',{}).get('environment',{}))
    status['worker_environment']={key:environment.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    json_write(receipt,status)
    worker=ROOT/'experiments/round2_ablation.py'
    try:
        reused=run['run_id'] in plan.get('reuse_runs',{})
        if reused:directory=verify_reuse(plan,run,plan['reuse_runs'][run['run_id']])
        if not reused and not (directory/'training_result.json').exists():
            command=[sys.executable,str(worker),'train','--plan',str(plan_path),'--run-id',run['run_id'],'--output',str(directory)]
            if directory.exists():
                if not (directory/'training_state.pt').exists(): raise ValueError('existing partial run has no restorable checkpoint; inspect before retry')
                command.append('--resume')
            if cuda:command.append('--cuda')
            with (logs/f'train{number:04d}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise RuntimeError(f'training exit code {result.returncode}')
        if not reused:verify_training(directory,plan,run)
        if not (evaluation/'evaluation.json').exists():
            if evaluation.exists():
                # Retain failed/partial evaluation evidence in place; next invocation
                # must resolve it explicitly, never replace it with an easier case.
                raise ValueError('partial evaluation output exists; inspect before retry')
            command=[sys.executable,str(worker),'evaluate','--plan',str(plan_path),'--run-id',run['run_id'],'--run-dir',str(directory),'--output',str(evaluation)]
            if cuda: command.append('--cuda')
            with (logs/f'evaluate{number:04d}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise RuntimeError(f'evaluation exit code {result.returncode}')
        e=read(evaluation/'evaluation.json')
        if e['plan_sha256']!=plan['plan_sha256'] or e['run']!=run:raise ValueError('evaluation identity differs')
        for key,path in e['artifacts'].items():
            if file_hash(path)!=e['artifact_sha256'][key]:raise ValueError('evaluation artifact hash differs')
        status.update(status='completed',reused_original_A0=reused)
    except Exception as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    status['finished_epoch']=time.time();json_write(receipt,status)
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--run-id',action='append');p.add_argument('--retry-failed',action='store_true');p.add_argument('--cuda',action='store_true')
    a=p.parse_args(argv);plan=read(a.plan);validate_plan(plan)
    if not 1<=a.workers<=8:raise ValueError('queue allows 1..8 workers; measure before increasing')
    if a.cuda and plan['protocol']['phase']=='formal':raise ValueError('formal plan is CPU-only')
    root=a.output.resolve()
    if RUN_ROOT.resolve() not in root.parents:raise ValueError('queue output must be below round-two root')
    root.mkdir(parents=True,exist_ok=True)
    registered=root/'registered_plan.json'
    if registered.exists() and read(registered)!=plan:raise ValueError('cannot replace registered plan')
    if not registered.exists():json_write(registered,plan)
    runs=plan['runs']
    if a.run_id:
        wanted=set(a.run_id);runs=[r for r in runs if r['run_id'] in wanted]
        if {r['run_id'] for r in runs}!=wanted:raise ValueError('unknown run requested')
    runs=schedule_runs(plan,runs)
    lock=root/'queue.lock'
    # Windows byte-range lock is released by the OS after interruption; file
    # existence alone is never treated as an active lock.
    handle=lock.open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
    import msvcrt
    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    try:
        control_path=root/'queue_control.json';control=dict(workers=a.workers,pause_dispatch=False)
        json_write(control_path,control)
        with ThreadPoolExecutor(max_workers=8) as executor:
            pending=list(runs);active={};results=[]
            while pending or active:
                try:control=read_control(control_path)
                except (ValueError,KeyError,OSError) as exc:
                    print(json.dumps(dict(control_error=str(exc),retaining=control)),flush=True)
                free=available_gib()
                while pending and not control['pause_dispatch'] and len(active)<control['workers'] and free>=6:
                    run=pending.pop(0)
                    future=executor.submit(execute,a.plan.resolve(),plan,run,root,a.retry_failed,a.cuda)
                    active[future]=run['run_id']
                    # Reserve a conservative 3 GiB for processes just launched,
                    # whose physical allocations may not yet appear in OS usage.
                    free-=3
                if active:
                    done,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                    for future in done:
                        result=future.result();results.append(result);del active[future]
                        print(json.dumps(result),flush=True)
                elif pending:time.sleep(15)
                json_write(root/'queue_summary.json',dict(plan_sha256=plan['plan_sha256'],requested=len(runs),
                    returned=results,running=list(active.values()),pending=len(pending),control=control,
                    available_physical_gib=available_gib(),updated_epoch=time.time()))
    finally:
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()

if __name__=='__main__':main()
