"""One finite local workflow: finish development, freeze, train, evaluate, report.

This does not submit a manuscript or mark reviewer responses complete. Any
missing/failed development arm, failed verification, or source change stops the
dependent stages. Long computations run as ordinary local child processes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
import importlib.metadata
import math
import hashlib
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
BASE=ROOT/'exp/revision_access_41572'
AUXILIARY_FILES=(
    'experiments/revision_continue_pipeline.py','experiments/revision_execution_status.py',
    'experiments/revision_live_file_reader.py',
    'experiments/revision_logging_migration.py','experiments/revision_archive.py','experiments/run_revision_resume_integration.py',
    'experiments/dispatch_revision_training.py','experiments/revision_development_queue.py','experiments/revision_failure_accounting.py',
    'experiments/revision_evaluation.py','experiments/revision_evaluation_queue.py',
    'experiments/revision_stress_statistics.py','experiments/revision_statistics.py',
    'experiments/revision_benchmark.py','experiments/revision_traditional_benchmark.py',
    'experiments/revision_pv_layouts.py','experiments/run_q1_fixed_test_traditional_baselines.py',
    'experiments/traditional_action_limits.py','manuscript_revision/revision_fair_q_selection.py',
    'manuscript_revision/revision_results_report.py',
    'experiments/revision_pv_count_export.py','experiments/revision_pv_count_environment.py',
    'experiments/revision_pv_count_plan.py','experiments/run_revision_pv_count_training.py',
    'experiments/revision_pv_count_check.py','experiments/revision_pv_count_checkpoint_check.py',
    'experiments/revision_pv_count_evaluation.py','experiments/revision_pv_count_queue.py',
    'experiments/revision_pv_count_pipeline.py','experiments/revision_pv_count_reporting.py')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    from experiments.revision_logging_migration import _publish_json
    _publish_json(path,value)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def publish_or_verify(path,value):
    if Path(path).exists():
        if read(path)!=value: raise ValueError('existing immutable evidence differs: '+str(path))
    else: write_new(path,value)


class IncompleteStage(RuntimeError):
    """No completed artifact exists; preserve this attempt and use a fresh one."""


def need_file(path):
    path=Path(path)
    if not path.is_file(): raise IncompleteStage('missing completion artifact: '+str(path))
    return path


def snapshot(paths):
    files={}
    for item in paths:
        path=Path(item)
        selected=path.rglob('*') if path.is_dir() else [path]
        for file in selected:
            if file.is_file(): files[str(file.resolve())]=digest(file)
    return dict(sorted(files.items()))


def child_still_running(attempt):
    import psutil
    for path in Path(attempt).glob('*.process.json'):
        record=read(path)
        try:
            process=psutil.Process(record['pid'])
            if abs(process.create_time()-record['create_time'])<.01 and process.is_running():
                return True
        except psutil.NoSuchProcess: pass
        except psutil.AccessDenied:
            raise RuntimeError('cannot establish whether an old stage child is still active')
    # Cover a crash between Popen and publishing the PID/create-time receipt.
    normalize=lambda command:[str(part).replace('\\','/').lower() for part in command]
    commands=[normalize(read(path)['command']) for path in Path(attempt).glob('*.command.json')]
    if commands:
        for process in psutil.process_iter(['pid','cmdline']):
            if process.pid==os.getpid(): continue
            try:
                if normalize(process.info['cmdline'] or []) in commands: return True
            except psutil.NoSuchProcess: pass
            except psutil.AccessDenied:
                raise RuntimeError('cannot exclude a surviving stage child')
    return False


class StageLedger:
    """Immutable receipts, exact-input resumption, and no choosing among successes."""
    def __init__(self,directory,identity):
        self.directory=Path(directory);self.identity=identity

    def run(self,name,inputs,action,validate,*,retry=True):
        stage=self.directory/'stages'/name
        request={'schema':1,'workflow_identity':self.identity,'stage':name,'inputs':inputs}
        attempts=sorted(stage.glob('a[0-9]*'))
        successes=[p for p in attempts if (p/'completed.json').exists()]
        if len(successes)>1: raise ValueError('multiple completed stage attempts: '+name)
        for attempt in attempts:
            if not (attempt/'request.json').exists():
                if list(attempt.iterdir()): raise ValueError('orphan stage data without immutable request')
                continue
            if read(attempt/'request.json')!=request:
                raise ValueError('stage inputs/source changed: '+name)
        if successes:
            attempt=successes[0];receipt=read(attempt/'completed.json')
            if receipt.get('request_sha256')!=digest(attempt/'request.json'):
                raise ValueError('stage request receipt mismatch: '+name)
            if any(not Path(path).is_file() or digest(path)!=sha for path,sha in receipt['files'].items()):
                raise ValueError('completed stage artifact hash changed: '+name)
            value,files=validate(attempt)
            if value!=receipt['value'] or snapshot(files)!=receipt['validated_files']:
                raise ValueError('completed stage semantics changed: '+name)
            return value
        # Interrupted stages may already have complete output. Recover it rather
        # than repeating model selection/freeze or collecting a better timing.
        for attempt in attempts:
            if not (attempt/'request.json').exists(): continue
            if child_still_running(attempt): raise RuntimeError('old stage process remains active: '+name)
            if (attempt/'failed.json').exists(): continue
            exits=list(attempt.glob('*.exit.json'))
            if any(read(path)['exit_code']!=0 for path in exits): continue
            try:
                value,files=validate(attempt)
            except IncompleteStage:
                continue
            return self._complete(attempt,value,files)
        if attempts and not retry:
            raise RuntimeError('unresolved stage requires explicit retry: '+name)
        number=max([int(p.name[1:]) for p in attempts],default=0)+1
        attempt=stage/f'a{number:04d}'
        attempt.mkdir(parents=True,exist_ok=False)
        write_new(attempt/'request.json',request)
        try:
            action(attempt)
            value,files=validate(attempt)
            return self._complete(attempt,value,files)
        except Exception as exc:
            write_new(attempt/'failed.json',{'error_type':type(exc).__name__,'message':str(exc),
                                            'traceback':traceback.format_exc()})
            raise

    def _complete(self,attempt,value,files):
        validated=snapshot(files)
        own=[p for p in attempt.rglob('*') if p.is_file() and p.name!='completed.json' and not p.name.endswith('.tmp')]
        write_new(attempt/'completed.json',{'schema':1,'request_sha256':digest(attempt/'request.json'),
            'value':value,'validated_files':validated,'files':{**snapshot(own),**validated}})
        return value


def update_state(directory,stage,**fields):
    value={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields}
    temporary=directory/'state.tmp'
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(directory/'state.json')
    with (directory/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def run_command(command,log_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    write_new(Path(log_path).with_suffix('.command.json'),{'command':command})
    with Path(log_path).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        import psutil
        try: created=psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess: created=0.
        write_new(Path(log_path).with_suffix('.process.json'),{'pid':process.pid,'create_time':created,'command':command})
        code=process.wait()
    write_new(Path(log_path).with_suffix('.exit.json'),{'exit_code':code})
    if code:
        raise RuntimeError(f'command failed with exit {code}; retained log {log_path}')


def archive_auxiliary(directory,expected):
    from experiments.run_revision_pilot import file_hash
    target=directory/'formal_evaluation_reporting_source.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,digest in expected.items():
            if file_hash(ROOT/name)!=digest: raise ValueError('workflow dependency changed: '+name)
            archive.writestr(name,(ROOT/name).read_bytes())
        archive.writestr('auxiliary_manifest.json',json.dumps(expected,indent=2))
    write_new(target.with_suffix('.json'),{'archive':str(target),'sha256':file_hash(target),
                                          'source_sha256':expected})


def run_isolated_benchmarks(plan,training_root,directory,ledger,devices):
    records=[]
    variants={'reference_B2','derived_B2','maddpg_B2','mappo_B2','facmaddpg_B2'}
    jobs=[]
    for run in plan['runs']:
        if run['variant'] not in variants or run['seed']!=0: continue
        for device in devices:
            name=run['run_id']+'_'+device
            command=[sys.executable,'experiments/revision_benchmark.py','--run-dir',str(training_root/run['run_id']),'--isolated']
            if device=='cuda': command.append('--cuda')
            jobs.append((name,command,run,device))
    for scenario in ('case33_3min_final','case141_3min_final'):
        for method in ('droop','opf'):
            name=scenario+'_'+method
            jobs.append((name,[sys.executable,'experiments/revision_traditional_benchmark.py',
                '--scenario',scenario,'--method',method,'--isolated'],None,None))
    for index,(name,command,run,device) in enumerate(jobs):
        def validate(attempt):
            output=attempt/'result';report=read(need_file(output/'benchmark.json'))
            need_file(output/('raw_timings.npz' if run else 'raw_calls.json'))
            script=ROOT/('experiments/revision_benchmark.py' if run else 'experiments/revision_traditional_benchmark.py')
            if (report.get('source',{}).get('source_sha256')!=plan['evaluation_source_sha256'] or
                report.get('benchmark_source_sha256')!=digest(script) or report.get('publication_timing_candidate') is not True):
                raise ValueError('benchmark source/isolation provenance differs')
            for key in ('load_context_before','load_context_after'):
                if report.get(key,{}).get('isolated_from_other_worktree_experiment_workers') is not True:
                    raise ValueError('benchmark was not isolated')
            if run:
                provenance=report['provenance']
                if (provenance.get('run')!=run or provenance.get('plan_sha256')!=plan['plan_sha256'] or
                    provenance.get('checkpoint_sha256')!=digest(training_root/run['run_id']/'best_model.pt') or
                    report.get('device')!=device or not finite_number(report.get('actual_input_count')) or
                    report['actual_input_count']<1000):
                    raise ValueError('benchmark model/device/input evidence differs')
                import numpy as np
                try:
                    with np.load(output/'raw_timings.npz',allow_pickle=False) as raw:
                        for key in ('full_policy_s','agent_zero_forward_s','preparation_s','simulation_s'):
                            values=raw[key]
                            if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
                                raise ValueError('invalid raw benchmark timings')
                except (zipfile.BadZipFile,EOFError) as exc:
                    raise IncompleteStage('interrupted benchmark raw archive') from exc
            else:
                raw=read(output/'raw_calls.json')
                if report.get('measured_calls')!=1000 or len(raw)<1000:
                    raise ValueError('incomplete conventional benchmark calls')
            return {'name':name,'status':'completed','output':str(output.resolve())},[output]
        try:
            value=ledger.run(f'bench{index:02d}',{'plan_sha256':plan['plan_sha256'],'name':name,'command':command},
                lambda a:run_command([*command,'--output',str(a/'result')],a/'command.log'),validate)
            records.append(value)
        except RuntimeError as exc:
            # Timing gaps do not invalidate completed learning/evaluation. Keep
            # the gap explicit and continue collecting scientific evidence.
            records.append({'name':name,'status':'unresolved','error':str(exc)})
    return records


def verify_junit(path,minimum=300):
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
    counts={name:sum(int(s.get(name,0)) for s in suites) for name in ('tests','failures','errors','skipped')}
    if counts['tests']<minimum or counts['failures'] or counts['errors'] or counts['skipped']:
        raise ValueError('verification suite incomplete or failing: '+str(counts))
    return counts


def active_development_consumers():
    import psutil
    root=str(ROOT).replace('\\','/').lower()
    active=[]
    for process in psutil.process_iter(['pid','cmdline','cwd']):
        try:
            command=' '.join(process.info['cmdline'] or []).replace('\\','/').lower()
            cwd=(process.info['cwd'] or '').replace('\\','/').lower()
            if any(name in command for name in ('dispatch_revision_training.py','run_revision_training.py','revision_development_queue.py')) and (
                    root in command or cwd==root): active.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess): continue
    return active


def verify_diagnosis(path):
    report=read(path)
    if report.get('formal_evidence') is not False or len(report.get('snapshots',[]))!=8:
        raise ValueError('validation same-state diagnosis is incomplete')
    for snapshot in report['snapshots']:
        for method in ('droop','ac_opf_witness'):
            witness=snapshot['outcomes'][method]
            residuals=[witness[key] for key in ('voltage_violation_max_pu','control_bound_violation_max_mvar',
                                               'inverter_circle_violation_max_mva2','ac_balance_max_mva')]
            if (witness['status']!='solved' or witness['inputs_unchanged'] is not True or
                any(not finite_number(value) or not 0<=value<=1e-6 for value in residuals)):
                raise ValueError('traditional-control feasibility witness failed its recorded tolerances')


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def make_gate(selection_path,verification,development_source,migration_path,directory,verification_source=None,
              development_root=None,source_revision=False):
    from experiments.run_revision_pilot import file_hash
    from experiments.revision_run_plan import canonical_hash
    workflow_plan=read(BASE/'protocol/development_plans_01/all_solvers.json')
    import torch
    import numpy as np
    expected_runtime={'device':'cpu','gpu':None,'python':sys.version,'torch':str(torch.__version__),
                      'numpy':np.__version__,'pandapower':importlib.metadata.version('pandapower')}
    development_root=Path(development_root or BASE/'development')
    for manifest_path in development_root.rglob('manifest.json'):
        manifest=read(manifest_path)
        if manifest.get('protocol',{}).get('phase')=='development' and manifest.get('runtime')!=expected_runtime:
            raise ValueError('development/formal runtime drift: '+str(manifest_path))
    reports=[]
    for run in workflow_plan['runs']:
        path=development_root/'d01'/run['run_id']/'training_result.json'
        result=read(path);manifest=read(path.parent/'manifest.json')
        if (result.get('run')!=run or result.get('status')!='completed' or
                result.get('environment_steps')!=4800 or result.get('phase')!='development' or
                result.get('formal_test_evaluated') is not False or not finite_number(result.get('wall_seconds')) or
                result['wall_seconds']<=0 or
                manifest.get('source_sha256')!=development_source or
                manifest.get('plan_sha256')!=canonical_hash(workflow_plan)):
            raise ValueError('workflow development evidence invalid: '+run['run_id'])
        for tag,digest in result['weight_sha256'].items():
            if file_hash(path.parent/(tag+'_model.pt'))!=digest:
                raise ValueError('development checkpoint changed')
        reports.append(path)
    selected=read(selection_path)
    if (selected.get('status')!='development_common_q_selected' or selected.get('formal_test_used') is not False
            or selected['audit']['valid_result_count']!=120 or selected['audit']['errors'] or selected['audit']['missing_run_ids']):
        raise ValueError('equal-budget common reward selection is incomplete')
    resume=(Path(read(migration_path)['real_resume_evidence']) if source_revision else
            BASE/'diagnostics/resume_integration_01/resume_integration_result.json')
    value=read(resume)
    required_equal={'model','optimizers','replay','global_rng','solver_rng','environment','artifacts','trainer'}
    if (value.get('status')!='passed' or value.get('bitwise_parameter_equal') is not True or
            not isinstance(value.get('starting_steps'),int) or value['starting_steps']<0 or
            value.get('ending_steps')!=value['starting_steps']+960 or
            (source_revision and (value.get('source_sha256')!=development_source or value.get('selection_history_entries',0)<1)) or
            not required_equal<=set(value.get('fields_exactly_equal',[]))):
        raise ValueError('real-process resume evidence is incomplete')
    diagnosis=BASE/'diagnostics/foundation_02_same_state_02/diagnostics.json'
    verify_diagnosis(diagnosis)
    failure_diagnosis=BASE/'diagnostics/actual_training_failure_diagnostics_01/diagnostics.json'
    if len(read(failure_diagnosis).get('failures',[]))!=3:
        raise ValueError('training failure diagnosis is incomplete')
    verification_counts=verify_junit(verification)
    runtime=directory/'runtime_budget_evidence.json'
    total_wall=sum(read(p)['wall_seconds'] for p in reports)
    publish_or_verify(runtime,{'source':'all 30 fixed-budget workflow development reports',
        'completed_runs':30,'training_steps_per_run':4800,'total_process_wall_seconds':total_wall,
        'formal_core_runs':160,'formal_steps_per_run':240000,
        'formal_total_training_steps':38400000,
        'formal_evaluation_jobs':3132,'planned_snapshots_per_evaluation':14400,
        'runtime_scope':'measured short development end-to-end, including IO/validation/concurrent load',
        'exact_completion_time_not_predicted':True})
    mapping={
        'implementation_tests_passed':[verification,migration_path,verification_source or directory/'verification_source.json'],
        'all_five_solvers_exercised':reports,
        'resume_verified':[resume,verification],
        'traditional_protocol_aligned':[diagnosis,verification],
        'validation_diagnosis_completed':[diagnosis,failure_diagnosis],
        'configuration_selected_on_validation':[selection_path],
        'runtime_budget_measured':[runtime,*reports],
    }
    files=sorted({Path(p).resolve() for paths in mapping.values() for p in paths})
    return {**{name:True for name in mapping},
        'evidence_files':[{'path':str(p),'sha256':file_hash(p)} for p in files],
        'assertion_evidence':{name:[str(Path(p).resolve()) for p in paths] for name,paths in mapping.items()},
        'development_runs':[{'path':str(p.resolve()),'sha256':file_hash(p)} for p in reports],
        'implementation_test_counts':verification_counts,
        'development_to_formal_change':('none; all 150 development runs rerun under corrected validation ranking and logging source'
            if source_revision else 'logging display only; transition rewards/updates/RNG unchanged, verified across five solvers'),
        'selected_alpha_interpretation':'retain legacy engineering allowances 0.8/0.6, not claimed optimal',
        'risk_mixture_interpretation':'predeclared 0.5, not claimed optimal'}



def test_files():
    paths=sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'tests').glob('test_revision*.py'))
    paths += ['tests/test_submission_v2_split.py','tests/test_voltage_control_timeline.py',
        'tests/test_ir_crpc_action_mapping.py','tests/test_fixed_test_protocol.py',
        'tests/test_risk_aware_runner.py','tests/test_submission_provenance.py',
        'tests/test_submission_v2_traditional.py']
    paths += sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'manuscript_revision').glob('test_*.py'))
    return sorted(set(paths))


def source_snapshot(tests):
    from experiments.revision_provenance import training_source_manifest
    from experiments.revision_evaluation import evaluation_source_manifest
    return {'training':training_source_manifest()['source_sha256'],
            'evaluation':evaluation_source_manifest()['source_sha256'],
            'tests':{name:digest(ROOT/name) for name in tests}}


def development_status_arguments(manifest):
    if not manifest: return {}
    return {'results_root':manifest['development_results_root'],
            'source_archive':manifest['development_source_manifest']}


def verify_source_revision(path,expected):
    """Bind a complete fresh development restart to its audited source change."""
    from experiments.revision_provenance import training_source_manifest
    value=read(path)
    if (value.get('status')!='verified_full_development_restart' or
            value.get('training_source_after')!=expected or
            value.get('old_results_used_for_selection') is not False or
            value.get('same_declared_plans_seeds_and_budgets') is not True or
            value.get('planned_development_runs')!=150):
        raise ValueError('source revision must declare the complete consistent development restart')
    before=value['training_source_before']
    actual_changes=sorted(name for name in set(before)|set(expected) if before.get(name)!=expected.get(name))
    if value.get('changed_training_files')!=actual_changes or not actual_changes:
        raise ValueError('source revision change inventory differs')
    for entry in value['evidence_files']:
        if digest(entry['path'])!=entry['sha256']:
            raise ValueError('source revision evidence changed')
    evidence_paths={str(Path(e['path']).resolve()) for e in value['evidence_files']}
    if not {str(Path(value[k]).resolve()) for k in ('verification_xml','real_resume_evidence')}<=evidence_paths:
        raise ValueError('source revision tests and real resume must be hash-bound evidence')
    verify_junit(value['verification_xml'])
    if training_source_manifest()['source_sha256']!=expected:
        raise ValueError('training source changed after complete development restart')
    return value


def open_workflow(directory,*,resume=False,training_workers=None,evaluation_workers=None,
                  development_root=None,development_archive=None,fair_q_registry=None,source_revision=None,
                  count_evidence=None):
    """Recover only the same directory, workflow source and development archive."""
    import torch
    from experiments.revision_logging_migration import inspect_migration
    directory=Path(directory).resolve()
    if BASE.resolve() not in directory.parents:
        raise ValueError('pipeline directory must stay under revision evidence')
    if directory.exists() and not resume:
        raise FileExistsError('existing workflow requires --resume')
    if resume and not directory.exists():
        raise FileNotFoundError('resume requires the original workflow directory')
    saved_path=directory/'workflow_manifest.json'
    prior=read(saved_path) if saved_path.exists() else {}
    def chosen(value,key,default):
        return Path(value or prior.get(key) or default).resolve()
    archive_path=chosen(development_archive,'development_source_manifest',BASE/'protocol/source_snapshots/development_01.json')
    development_root=chosen(development_root,'development_results_root',BASE/'development')
    fair_q_registry=chosen(fair_q_registry,'fair_q_registry',BASE/'protocol/dev_fair_q_01/fair_q_registry.json')
    source_revision=Path(source_revision or prior['source_revision']).resolve() if source_revision or prior.get('source_revision') else None
    count_evidence=Path(count_evidence or prior['count_evidence']).resolve() if count_evidence or prior.get('count_evidence') else None
    archive=read(archive_path)
    if digest(archive['archive'])!=archive['sha256']:
        raise ValueError('development source archive changed')
    # Verify archived content, not merely a checksum copied from another file.
    with zipfile.ZipFile(archive['archive']) as bundle:
        for name,sha in archive['source']['source_sha256'].items():
            if hashlib.sha256(bundle.read(name)).hexdigest()!=sha:
                raise ValueError('development source archive content mismatch')
    manifest_path=directory/'workflow_manifest.json'
    saved=read(manifest_path) if manifest_path.exists() else None
    if resume and saved is None:
        raise ValueError('orphan output has no workflow identity; cannot infer ownership')
    tw=training_workers if training_workers is not None else (saved['training_workers'] if saved else 8)
    ew=evaluation_workers if evaluation_workers is not None else (saved['evaluation_workers'] if saved else 8)
    if any(isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=12 for n in (tw,ew)):
        raise ValueError('worker limit is integer 1..12')
    identity={'schema':2,'workflow_directory':str(directory),'workflow_sha256':digest(__file__),
        'auxiliary_source_sha256':{name:digest(ROOT/name) for name in AUXILIARY_FILES},
        'development_source_manifest':str(archive_path.resolve()),
        'development_source_manifest_sha256':digest(archive_path),
        'development_source_archive':archive['archive'],'development_source_archive_sha256':archive['sha256'],
        'development_source_sha256':archive['source']['source_sha256'],
        'development_results_root':str(development_root),'fair_q_registry':str(fair_q_registry),
        'fair_q_registry_sha256':digest(fair_q_registry),
        'source_revision':str(source_revision) if source_revision else None,
        'source_revision_sha256':digest(source_revision) if source_revision else None,
        'count_evidence':str(count_evidence) if count_evidence else None,
        'count_evidence_sha256':digest(count_evidence) if count_evidence else None,
        'training_workers':tw,'evaluation_workers':ew,
        'benchmark_devices':saved['benchmark_devices'] if saved else (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']),
        'scope':'finite local computation through numerical drafts; no manuscript submission'}
    if saved is not None and saved!=identity:
        raise ValueError('workflow/auxiliary source, archive, directory or worker identity changed')
    if source_revision:
        verify_source_revision(source_revision,identity['development_source_sha256'])
        from manuscript_revision.revision_fair_q_selection import load_registry,validate_results_binding
        binding=validate_results_binding(load_registry(fair_q_registry),development_root)
        if (Path(binding['source_archive']).resolve()!=archive_path or
                binding['source_sha256']!=identity['development_source_sha256']):
            raise ValueError('source-bound development registry belongs to another archive')
        status={'state':'not_started'}
    else:
        status=inspect_migration(expected_source_sha256=identity['development_source_sha256'],
                                 evidence_path=directory/'logging_migration.json')
    if saved is None and status['state']!='not_started':
        raise ValueError('resume the original workflow that owns the migration')
    if count_evidence: checked_count_evidence(identity)
    directory.mkdir(parents=True,exist_ok=resume)
    if saved is None: write_new(manifest_path,identity)
    return identity


def development_complete(manifest=None):
    from experiments.revision_execution_status import execution_status
    state=execution_status(**development_status_arguments(manifest))
    if state['counts']!={'completed':150}:
        raise IncompleteStage('150 development runs are not all completed: '+str(state['counts']))
    paths=[]
    for row in state['runs']:
        location=Path(row['directory'])
        paths.extend([location/'manifest.json',location/'training_result.json'])
    return {'completed_runs':150,'evidence_sha256':snapshot(paths)},paths


def wait_development(directory,manifest):
    from experiments.revision_execution_status import execution_status
    from experiments.revision_logging_migration import inspect_migration
    previous=None;inactive=0
    while True:
        if manifest.get('source_revision'):
            verify_source_revision(manifest['source_revision'],manifest['development_source_sha256'])
            status={'state':'not_started'}
        else:
            status=inspect_migration(expected_source_sha256=manifest['development_source_sha256'],
                                     evidence_path=directory/'logging_migration.json')
        if any(digest(ROOT/name)!=sha for name,sha in manifest['auxiliary_source_sha256'].items()):
            raise ValueError('workflow/evaluation/reporting source changed while waiting')
        state=execution_status(**development_status_arguments(manifest))
        if set(state['counts'])-{'completed','in_progress','not_started'}:
            raise ValueError('development requires diagnosis: '+str(state['counts']))
        if state['counts']!=previous:
            update_state(directory,'waiting_development',counts=state['counts'])
            previous=state['counts']
        if state['counts']=={'completed':150}: return
        if status['state']!='not_started':
            raise ValueError('migration exists but development evidence became incomplete')
        inactive=0 if active_development_consumers() else inactive+1
        if inactive>=3:
            raise RuntimeError('development incomplete without a live worktree training/dispatcher process')
        time.sleep(30)


def validate_selection(attempt,registry_path,results_root=None):
    from manuscript_revision.revision_fair_q_selection import load_registry,collect,summarize_and_select
    folder=attempt/'result'
    path=need_file(folder/'development_common_q_selection.json')
    need_file(folder/'available_validated_seed_results.json')
    need_file(folder/'raw_seed_results.csv')
    rows,missing,errors,_=collect(load_registry(registry_path),Path(results_root or BASE/'development'))
    if missing or errors or len(rows)!=120:
        raise ValueError('selection input evidence is no longer complete/valid')
    actual=read(path)
    if {k:v for k,v in actual.items() if k!='audit'}!=summarize_and_select(rows):
        raise ValueError('recorded common-q selection differs from the complete declared grid')
    audit=actual['audit']
    if (audit['registry_sha256']!=digest(registry_path) or audit['valid_result_count']!=120 or
        audit['missing_run_ids'] or audit['errors'] or audit['formal_test_used'] is not False):
        raise ValueError('selection audit differs from registered validation-only scope')
    return {'path':str(path.resolve()),'sha256':digest(path)},[folder]


def verify_source_archive(path,expected,manifest_name):
    need_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None: raise ValueError('archive CRC failed')
            for name,sha in expected.items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=sha:
                    raise ValueError('source archive content differs: '+name)
            manifest=json.loads(archive.read(manifest_name))
            declared=manifest['source_sha256'] if manifest_name=='training_source_manifest.json' else manifest
            if declared!=expected: raise ValueError('source archive manifest content differs')
    except (zipfile.BadZipFile,KeyError) as exc:
        raise IncompleteStage('interrupted source archive') from exc


def queue_command(kind,plan_path,output,workers,*,training_root=None,retry_failed=False,failure_records=()):
    """Queue reuse is explicit; never request a retry on a brand-new queue."""
    output=Path(output)
    if kind=='training':
        command=[sys.executable,'-u','experiments/dispatch_revision_training.py','--plan',str(plan_path),
                 '--output',str(output),'--workers',str(workers)]
    elif kind=='evaluation':
        command=[sys.executable,'-u','experiments/revision_evaluation_queue.py','dispatch',
            '--plan',str(plan_path),'--training-root',str(training_root),'--output',str(output),'--workers',str(workers)]
        for path in failure_records: command.extend(['--failure-record',str(path)])
    else: raise ValueError('unknown queue kind')
    if output.exists():
        command.append('--resume')
        if kind=='evaluation' and retry_failed: command.append('--retry-failed')
    return command


def training_complete(plan,training_root,failure_records):
    from experiments.revision_evaluation_queue import preflight_training
    try: evidence=preflight_training(plan,training_root,failure_records)
    except FileNotFoundError as exc: raise IncompleteStage('training is not fully accounted for') from exc
    return {'output':str(training_root),'evidence':evidence},[]


def evaluation_complete(plan,evaluation_root):
    from experiments.revision_evaluation_queue import collect_verified_results
    if not (evaluation_root/'queue_identity.json').exists():
        raise IncompleteStage('evaluation queue has not started')
    try: rows=collect_verified_results(plan,evaluation_root)
    except FileNotFoundError as exc: raise IncompleteStage('evaluation output is incomplete') from exc
    except ValueError as exc:
        if 'must complete exactly once' in str(exc): raise IncompleteStage(str(exc)) from exc
        raise
    compact=[{'job_id':r['job']['job_id'],'status':r['status'],
              'evaluation_sha256':r.get('evaluation_sha256'),
              'receipt_sha256':r.get('receipt_sha256'),'failure_record_sha256':r.get('failure_record_sha256')}
             for r in rows]
    return {'output':str(evaluation_root),'job_count':len(rows),'evidence_sha256':canonical(compact)},[]


def checked_count_evidence(manifest):
    path=manifest['count_evidence']
    if digest(path)!=manifest['count_evidence_sha256']:
        raise ValueError('count development evidence index changed')
    evidence=read(path)
    required={'export_index','physical_check','checkpoint_check','development_plan'}
    if set(evidence.get('files',{}))!=required:
        raise ValueError('count evidence index is incomplete')
    for entry in evidence['files'].values():
        if digest(entry['path'])!=entry['sha256']:
            raise ValueError('count development input changed')
    return evidence


def prepare_count_workflow(core_plan_path,manifest,ledger):
    from experiments.revision_pv_count_pipeline import prepare,freeze
    from experiments.revision_pv_count_plan import validate_count_plan
    from experiments.revision_pv_count_queue import validate_registry
    evidence=checked_count_evidence(manifest)
    paths={k:v['path'] for k,v in evidence['files'].items()}
    def action(a):
        candidate=a/'candidate.json'
        prepare(core_plan_path,paths['export_index'],candidate)
        freeze(candidate,paths['physical_check'],evidence['development_root'],
               paths['checkpoint_check'],a/'formal_plan.json')
    def validate(a):
        path=need_file(a/'formal_plan.json');plan=read(path)
        validate_count_plan(plan);validate_registry(plan)
        if Path(plan['base_protocol']['path']).resolve()!=Path(core_plan_path).resolve():
            raise ValueError('count experiment refers to another core selection')
        return {'path':str(path.resolve()),'plan_sha256':plan['plan_sha256']},[a/'candidate.json',path]
    return ledger.run('count_freeze',{'core_plan_sha256':read(core_plan_path)['plan_sha256'],
        'development_evidence_sha256':manifest['count_evidence_sha256']},action,validate)


def run_count_workflow(frozen,manifest,ledger,args):
    from experiments.revision_pv_count_queue import preflight_training,collect_verified_results
    plan_path=Path(frozen['path']);plan=read(plan_path)
    base=BASE/'pv_count'/Path(args.output).name
    train_root=base/'training';queue_root=base/'evaluation'
    failures=[Path(p).resolve() for p in args.count_failure_record]
    def train_validate(a):
        try: value=preflight_training(plan,train_root,failures)
        except FileNotFoundError as exc: raise IncompleteStage('count training has missing runs') from exc
        return {'root':str(train_root.resolve()),'evidence':value},[]
    def train_action(a):
        try:
            train_validate(a);return
        except IncompleteStage: pass
        command=[sys.executable,'-u','experiments/revision_pv_count_pipeline.py','train',
            '--plan',str(plan_path),'--training-root',str(train_root),'--max-runs','40',
            '--workers',str(min(4,manifest['training_workers'])),'--min-free-gib','10']
        if train_root.exists(): command.append('--resume')
        for path in failures: command.extend(['--failure-record',str(path)])
        run_command(command,a/'command.log')
    training=ledger.run('count_training',{'plan_sha256':plan['plan_sha256']},train_action,train_validate)
    def eval_validate(a):
        if not (queue_root/'queue_identity.json').exists(): raise IncompleteStage('count queue not started')
        try: rows=collect_verified_results(plan,queue_root)
        except FileNotFoundError as exc: raise IncompleteStage('count evaluations missing') from exc
        except ValueError as exc:
            if 'exactly one completion' in str(exc) or 'must complete exactly once' in str(exc):
                raise IncompleteStage(str(exc)) from exc
            raise
        compact=[{'job_id':r['job']['job_id'],'status':r['status'],
            'evaluation_sha256':r.get('evaluation_sha256'),
            'receipt_sha256':r.get('receipt_sha256'),'failure_record_sha256':r.get('failure_record_sha256')} for r in rows]
        return {'root':str(queue_root.resolve()),'jobs':len(rows),'evidence_sha256':canonical(compact)},[]
    def eval_action(a):
        command=[sys.executable,'-u','experiments/revision_pv_count_queue.py','dispatch',
            '--plan',str(plan_path),'--training-root',str(train_root),'--output',str(queue_root),
            '--workers',str(min(4,manifest['evaluation_workers']))]
        if queue_root.exists():
            command.append('--resume')
            if args.retry_failed_evaluations: command.append('--retry-failed')
        for path in failures: command.extend(['--failure-record',str(path)])
        run_command(command,a/'command.log')
    evaluation=ledger.run('count_evaluation',{'plan_sha256':plan['plan_sha256'],'training':canonical(training)},
                          eval_action,eval_validate)
    def report_validate(a):
        path=need_file(a/'count_summary.json');value=read(path)
        if (value.get('plan_sha256')!=plan['plan_sha256'] or
                value.get('evaluation_registry_sha256')!=plan['evaluation_registry_sha256'] or
                value.get('planned_evaluation_jobs')!=86 or len(value.get('groups',[]))!=8 or
                any(row.get('n_planned_training_seeds')!=10 for row in value['groups']) or
                len(value.get('traditional',[]))!=6):
            raise ValueError('count report is not the complete registered independent-seed family')
        return {'path':str(path.resolve()),'sha256':digest(path)},[path]
    return ledger.run('count_report',{'plan_sha256':plan['plan_sha256'],'evaluation':evaluation},
        lambda a:run_command([sys.executable,'experiments/revision_pv_count_reporting.py','--plan',str(plan_path),
            '--queue',str(queue_root),'--output',str(a/'count_summary.json')],a/'command.log'),report_validate)


def run_workflow(args,manifest):
    from experiments.revision_evaluation_queue import register_evaluation_jobs
    from experiments.revision_logging_migration import apply_after_development
    from experiments.revision_run_plan import build_plan,freeze_plan,validate_frozen_plan,validate_development_evidence
    from experiments.revision_archive import archive_training_source
    from manuscript_revision.revision_fair_q_selection import select
    directory=Path(args.output).resolve()
    ledger=StageLedger(directory,canonical(manifest))
    expected=manifest['development_source_sha256']
    development_root=Path(manifest['development_results_root'])
    ledger.run('development',{'archive':manifest['development_source_archive_sha256']},
               lambda a:wait_development(directory,manifest),lambda a:development_complete(manifest))
    registry_path=Path(manifest['fair_q_registry'])
    def choose(a):
        if select(registry_path,development_root,a/'result'):
            raise RuntimeError('common-q selection refused; retained gap report')
    selection=ledger.run('selection',{'registry_sha256':digest(registry_path)},choose,
                         lambda a:validate_selection(a,registry_path,development_root))
    migration_path=Path(manifest['source_revision']) if manifest.get('source_revision') else directory/'logging_migration.json'
    # Its own intent transaction handles crashes on either side of source replace.
    if manifest.get('source_revision'):
        verify_source_revision(migration_path,expected)
    else:
        apply_after_development(expected_source_sha256=expected,evidence_path=migration_path)
    migration=read(migration_path)
    tests=test_files()
    tested_source=source_snapshot(tests)
    def test_action(a):
        write_new(a/'verification_source.json',tested_source)
        temp=ROOT.parents[1]/'.tmp'/('rf'+uuid.uuid4().hex[:8])
        run_command([sys.executable,'-m','pytest',*tests,'-q','--basetemp='+str(temp),
            '--junitxml='+str(a/'verification.xml')],a/'command.log')
    def test_validate(a):
        source=read(need_file(a/'verification_source.json'))
        if source!=tested_source or source_snapshot(tests)!=source:
            raise ValueError('tested training/evaluation/test source changed')
        path=need_file(a/'verification.xml')
        try: counts=verify_junit(path)
        except ET.ParseError as exc: raise IncompleteStage('interrupted pytest XML') from exc
        return {'xml':str(path.resolve()),'source':str((a/'verification_source.json').resolve()),'counts':counts},[path,a/'verification_source.json']
    verification=ledger.run('verification',tested_source,test_action,test_validate)
    def gate_action(a):
        value=make_gate(Path(selection['path']),Path(verification['xml']),expected,migration_path,a,
                        verification_source=Path(verification['source']),development_root=development_root,
                        source_revision=bool(manifest.get('source_revision')))
        write_new(a/'development_gate.json',value)
    def gate_validate(a):
        path=need_file(a/'development_gate.json');gate=read(path)
        validate_development_evidence(gate)
        if (gate['assertion_evidence']['configuration_selected_on_validation']!=[selection['path']] or
            str(Path(verification['xml']).resolve()) not in gate['assertion_evidence']['implementation_tests_passed'] or
            str(migration_path.resolve()) not in gate['assertion_evidence']['implementation_tests_passed']):
            raise ValueError('gate belongs to different selection/tests/migration')
        return {'path':str(path.resolve()),'sha256':digest(path)},[path,a/'runtime_budget_evidence.json']
    gate_info=ledger.run('gate',{'selection':selection,'verification':verification,
                                'migration_sha256':digest(migration_path)},gate_action,gate_validate)
    gate=read(gate_info['path'])
    candidate=build_plan(BASE/'protocol/training_pool_bound_01')
    selected=read(selection['path'])['selected_common_q_by_network']
    candidate['protocol']['reward_by_scenario']={scenario:{'voltage_weight':1.,'violation_weight':0.,
        'q_weight':row['q_weight']} for scenario,row in selected.items()}
    candidate=register_evaluation_jobs(candidate)
    plan_path=directory/'formal_plan.json'
    def freeze_action(a):
        # Atomic final path can exist after a crash just before its stage receipt.
        if not plan_path.exists(): write_new(plan_path,freeze_plan(candidate,gate))
    def freeze_validate(a):
        formal=read(need_file(plan_path));validate_frozen_plan(formal)
        if formal['development_gate']!=gate or any(formal.get(key)!=value for key,value in candidate.items() if key!='status'):
            raise ValueError('frozen plan differs from selected candidate/development gate')
        if formal['source']['source_sha256']!=tested_source['training']:
            raise ValueError('frozen source differs from verified source')
        return {'path':str(plan_path),'plan_sha256':formal['plan_sha256']},[plan_path]
    frozen=ledger.run('freeze',{'candidate_sha256':canonical(candidate),'gate':gate_info},freeze_action,freeze_validate)
    formal=read(frozen['path'])
    count_frozen=prepare_count_workflow(plan_path,manifest,ledger) if manifest.get('count_evidence') else None
    def archive_action(a):
        archive_training_source(a/'training.zip',formal['source']['source_sha256'])
        archive_auxiliary(a,manifest['auxiliary_source_sha256'])
    def archive_validate(a):
        train=a/'training.zip';aux=a/'formal_evaluation_reporting_source.zip'
        verify_source_archive(train,formal['source']['source_sha256'],'training_source_manifest.json')
        verify_source_archive(aux,manifest['auxiliary_source_sha256'],'auxiliary_manifest.json')
        # Sidecars may be missing after a crash. Their content is fully reconstructable
        # from verified bytes, so publish only the missing records.
        publish_or_verify(a/'archive_index.json',{'training':{'path':str(train),'sha256':digest(train)},
            'auxiliary':{'path':str(aux),'sha256':digest(aux)}})
        return {'training':str(train),'auxiliary':str(aux)},[train,aux,a/'archive_index.json']
    ledger.run('archives',{'plan_sha256':formal['plan_sha256'],'auxiliary':manifest['auxiliary_source_sha256']},
               archive_action,archive_validate)
    update_state(directory,'formal_frozen',**frozen)
    if args.stop_after_freeze: return 0
    training_root=BASE/'formal'/directory.name
    failure_records=[Path(p).resolve() for p in args.failure_record]
    def train_action(a):
        try:
            training_complete(formal,training_root,failure_records)
            return
        except IncompleteStage:
            if failure_records:
                raise RuntimeError('reviewed failures plus other unfinished training require a targeted recovery; do not restart adjudicated runs')
        command=queue_command('training',plan_path,training_root,manifest['training_workers'])
        if active_development_consumers():
            raise RuntimeError('training/dispatcher process still active; refuse duplicate dispatch')
        run_command(command,a/'command.log')
    training=ledger.run('training',{'plan_sha256':formal['plan_sha256']},train_action,
                       lambda a:training_complete(formal,training_root,failure_records))
    benchmark_records=run_isolated_benchmarks(formal,training_root,directory,ledger,manifest['benchmark_devices'])
    evaluation_root=BASE/'evaluation'/directory.name
    def evaluate_action(a):
        command=queue_command('evaluation',plan_path,evaluation_root,manifest['evaluation_workers'],
            training_root=training_root,retry_failed=args.retry_failed_evaluations,failure_records=failure_records)
        run_command(command,a/'command.log')
    evaluation=ledger.run('evaluation',{'plan_sha256':formal['plan_sha256'],'training':canonical(training)},
        evaluate_action,lambda a:evaluation_complete(formal,evaluation_root))
    def report_validate(a):
        output=a/'result';report=read(need_file(output/'report_manifest.json'))
        if (report.get('status')!='formal_draft_generated' or report.get('plan_sha256')!=formal['plan_sha256'] or
            report.get('reporter_sha256')!=digest(ROOT/'manuscript_revision/revision_results_report.py') or
            report.get('holm_family_size')!=8 or report.get('independent_training_seed_unit') is not True):
            raise ValueError('primary report provenance changed')
        if any(digest(path)!=sha for path,sha in report['evaluation_files'].items()):
            raise ValueError('report source evaluations changed')
        return {'output':str(output.resolve()),'manifest_sha256':digest(output/'report_manifest.json')},[output]
    reports=ledger.run('report',{'plan_sha256':formal['plan_sha256'],'evaluation':evaluation},
        lambda a:run_command([sys.executable,'manuscript_revision/revision_results_report.py','--plan',str(plan_path),
            '--evaluations-root',str(evaluation_root),'--output',str(a/'result')],a/'command.log'),report_validate)
    def stress_validate(a):
        path=need_file(a/'stress_summary.json');value=read(path)
        if (value.get('plan_sha256')!=formal['plan_sha256'] or value.get('evaluation_registry_sha256')!=formal['evaluation_registry_sha256'] or
            len(value.get('groups',[]))!=68 or any(g.get('n_training_seeds')!=10 for g in value['groups'])):
            raise ValueError('stress summary differs from registered seed-level analysis')
        return {'path':str(path.resolve()),'sha256':digest(path)},[path]
    stress=ledger.run('stress',{'plan_sha256':formal['plan_sha256'],'evaluation':evaluation},
        lambda a:run_command([sys.executable,'experiments/revision_stress_statistics.py','--plan',str(plan_path),
            '--queue',str(evaluation_root),'--output',str(a/'stress_summary.json')],a/'command.log'),stress_validate)
    # Each invocation gets its own status; it never overwrites an earlier gap report.
    count_report=run_count_workflow(count_frozen,manifest,ledger,args) if count_frozen else None
    completion={'report_directory':reports['output'],'stress_summary':stress['path'],'count_report':count_report,
        'benchmark_records':benchmark_records,'unresolved_benchmarks':sum(r['status']!='completed' for r in benchmark_records),
        'final_submission_ready':False,'remaining':([] if count_report else ['PV-count appendix training/evaluation'])+
            (['resolve benchmark gaps'] if any(r['status']!='completed' for r in benchmark_records) else [])+
            ['scientific interpretation and manuscript/response integration','final release and page-line cross-check']}
    write_new(directory/'invocations'/(uuid.uuid4().hex+'.completed.json'),completion)
    update_state(directory,'core_numerical_draft_ready',**completion)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-workers',type=int)
    parser.add_argument('--evaluation-workers',type=int)
    parser.add_argument('--stop-after-freeze',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-failed-evaluations',action='store_true')
    parser.add_argument('--failure-record',type=Path,action='append',default=[])
    parser.add_argument('--development-root',type=Path)
    parser.add_argument('--development-archive',type=Path)
    parser.add_argument('--fair-q-registry',type=Path)
    parser.add_argument('--source-revision',type=Path)
    parser.add_argument('--count-evidence',type=Path)
    parser.add_argument('--count-failure-record',type=Path,action='append',default=[])
    args=parser.parse_args(argv)
    if args.retry_failed_evaluations and not args.resume:
        raise ValueError('explicit evaluation retries require --resume')
    from experiments.revision_evaluation_queue import exclusive_queue_lock
    directory=args.output.resolve()
    # Directory/identity checks precede the lock; only fresh creation or explicit
    # resume is allowed. The OS lock releases on process death.
    manifest=open_workflow(directory,resume=args.resume,training_workers=args.training_workers,
                           evaluation_workers=args.evaluation_workers,development_root=args.development_root,
                           development_archive=args.development_archive,fair_q_registry=args.fair_q_registry,
                           source_revision=args.source_revision,count_evidence=args.count_evidence)
    with exclusive_queue_lock(directory):
        write_new(directory/'invocations'/(uuid.uuid4().hex+'.request.json'),{
            'pid':os.getpid(),'resume':args.resume,'retry_failed_evaluations':args.retry_failed_evaluations,
            'failure_records':[str(p.resolve()) for p in args.failure_record]})
        try:
            return run_workflow(args,manifest)
        except Exception as exc:
            update_state(directory,'stopped_requires_diagnosis',error_type=type(exc).__name__,
                         message=str(exc),traceback=traceback.format_exc(),final_submission_ready=False)
            raise


if __name__=='__main__': raise SystemExit(main())
