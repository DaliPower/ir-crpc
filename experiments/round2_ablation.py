"""Isolated, frozen six-mechanism study; old training code is not edited.

Subcommands: prepare, freeze, audit-a0, train, evaluate. One run per process.
The injection context only routes the old trainer to an explicitly declared
round-two model/validator/source boundary and is restored on exit. No old gates
are silently bypassed: round-two formal budgets, source, data, pools and tests
are checked by validate_plan before any training or evaluation.
"""
from __future__ import annotations
import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.run_revision_pilot import file_hash as _file_hash,json_write
from experiments.revision_provenance import training_source_manifest,derive_training_streams,data_hashes,validate_pool_binding
from experiments.revision_run_plan import canonical_hash

BASE_PLAN=ROOT/'exp/revision_access_41572/workflows/continuation_v02_io_recovery/formal_plan.json'
BASE_RUNS=ROOT/'exp/revision_access_41572/formal/continuation_v02_io_recovery'
EVIDENCE=ROOT/'manuscript_revision/round2_20261026/analysis/matd3_ablation'
RUN_ROOT=ROOT/'exp/revision_access_41572/r2a'
SCENARIOS=('case33_3min_final','case141_3min_final')
ROUND2_VARIANTS=tuple(f'A{i}' for i in range(6))


def file_hash(path): return _file_hash(Path(path))
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def plan_hash(plan): return canonical_hash({k:v for k,v in plan.items() if k!='plan_sha256'})


def source_manifest():
    result=training_source_manifest()
    for path in sorted((ROOT/'experiments').glob('round2_ablation*.py')):
        result['source_sha256'][path.relative_to(ROOT).as_posix()]=file_hash(path)
    # Formal policy reconstruction is used by the evaluation command.
    for relative in ('experiments/revision_evaluation.py','experiments/revision_pv_layouts.py'):
        result['source_sha256'][relative]=file_hash(ROOT/relative)
    result['scope']='round-two model, runner and complete inherited training/evaluation dependency boundary'
    return result


def seed_streams(variant,scenario,seed):
    # Existing helper hashes the full variant name. Round-two names create an
    # independent domain; no claim of paired trials from equal seed labels.
    return derive_training_streams(variant,scenario,seed)


def build_plan(phase='formal',smoke_steps=120):
    if phase not in ('formal','development'): raise ValueError('unsupported phase')
    original=read(BASE_PLAN)
    p=deepcopy(original['protocol'])
    for key in tuple(p):
        if key.startswith('evaluation_'): del p[key]
    p.update(phase=phase,evaluation_pv_scale=1.5,
        training_steps=240000 if phase=='formal' else int(smoke_steps),
        validation_days=list(range(1000,1030)) if phase=='formal' else [1000],
        validation_every_steps=48000 if phase=='formal' else int(smoke_steps),
        multiple_testing='Holm across ten network-specific primary tests; training seed is the unit',
        planned_comparisons_per_network=[['A1','A0'],['A2','A0'],['A3-A2','A1-A0'],['A4','A0'],['A5','A0']],
        inference_design='independent_training_repeats; equal seed labels do not imply paired observations',
        historical_test_exposure_disclosed=True)
    if p['training_steps']<60: raise ValueError('smoke budget must exercise at least one 60-step learning trigger')
    seeds=range(10) if phase=='formal' else [100]
    runs=[dict(run_id=f'{v}_{s}_s{seed}',variant=f'round2_52275_{v}',round2_variant=v,
        solver='matd3_reference',sampling='B2',mapping='capacity_adaptive',scenario=s,seed=seed)
        for v in ROUND2_VARIANTS for s in SCENARIOS for seed in seeds]
    return dict(schema=1,study='Access-2026-52275-controlled-MATD3-ablation',status='candidate',
        base_plan=dict(path=str(BASE_PLAN),sha256=file_hash(BASE_PLAN)),protocol=p,
        pools=deepcopy(original['pools']),runs=runs,
        reuse_policy='No A0 reuse until per-run source/data/configuration/checkpoint audit is approved; default all 120 retrain',
        primary_design='A0..A3 delay x TD actor-source factorial; A4 historical noise; A5 conditional shared critic',
        performance_claims_allowed=False)


def check_runtime(plan,cpu):
    """Runtime scheduling is declared separately from algorithm and old provenance."""
    if plan['protocol']['phase']!='formal':return
    if not cpu:raise ValueError('formal ablation is CPU-only to match original A0 device')
    for key,value in plan['execution']['environment'].items():
        if os.environ.get(key)!=value:raise ValueError('frozen worker runtime requires '+key+'='+value)


def validate_plan(plan,verify_source=True):
    if plan.get('status')!='frozen': raise ValueError('candidate plan cannot execute')
    if plan.get('plan_sha256')!=plan_hash(plan): raise ValueError('frozen plan hash changed')
    p=plan['protocol']; phase=p['phase']
    expected=build_plan(phase,p['training_steps'])
    if plan['runs']!=expected['runs']: raise ValueError('run matrix differs from fixed six-configuration design')
    if p!=expected['protocol']: raise ValueError('protocol budget/dates/physical settings differ')
    if plan['pools']!=expected['pools']: raise ValueError('pool bindings changed')
    if phase=='formal' and (p['training_steps']!=240000 or len(plan['runs'])!=120):
        raise ValueError('formal matrix/budget must be 120 complete 240000-transition runs')
    if not plan.get('source'): raise ValueError('source freeze missing')
    if phase=='formal':
        gate=plan.get('implementation_gate',{})
        if not gate.get('passed') or file_hash(gate['path'])!=gate.get('sha256'):
            raise ValueError('formal implementation gate missing or changed')
        execution=plan.get('execution',{})
        if execution.get('device')!='cpu' or execution.get('environment')!={'OPENBLAS_NUM_THREADS':'1'}:
            raise ValueError('formal CPU/OpenBLAS runtime declaration missing')
        for key in ('runtime_probe','equivalence_audit'):
            evidence=execution[key]
            if file_hash(evidence['path'])!=evidence['sha256']:raise ValueError('runtime evidence changed')
    if verify_source:
        if plan['source']['source_sha256']!=source_manifest()['source_sha256']:
            raise ValueError('source changed after freeze')
        for pool in plan['pools'].values():
            if file_hash(pool['path'])!=pool['sha256']: raise ValueError('pool checksum differs')
            validate_pool_binding(pool['path'])
    for run_id,record in plan.get('reuse_runs',{}).items():
        matches=[r for r in plan['runs'] if r['run_id']==run_id and r['round2_variant']=='A0']
        if phase!='formal' or len(matches)!=1:raise ValueError('only formal A0 may reuse original evidence')
        if verify_source: verify_reuse(plan,matches[0],record)


def freeze(plan,test_report=None,reuse_a0_audit=None):
    result=deepcopy(plan)
    if plan.get('status')!='candidate': raise ValueError('only a candidate may be frozen')
    result['source']=source_manifest()
    if test_report is not None:
        tree=ET.parse(test_report)
        suites=list(tree.getroot().iter('testsuite'))
        tests=sum(int(s.get('tests','0')) for s in suites)
        failed=sum(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped'))
        if tests<17 or failed: raise ValueError('implementation report must contain >=17 passing, unskipped tests')
        result['implementation_gate']=dict(path=str(Path(test_report).resolve()),sha256=file_hash(test_report),tests=tests,passed=True)
    if reuse_a0_audit:
        audit=read(reuse_a0_audit)
        if audit['artifact_eligible_count']!=20:raise ValueError('A0 reuse requires all 20 unchanged original repeats')
        result['reuse_runs']={}
        for entry in audit['runs']:
            run_id=f"A0_{entry['scenario']}_s{entry['seed']}"
            directory=Path(entry['path'])
            result['reuse_runs'][run_id]=dict(path=str(directory),files={name:file_hash(directory/name)
                for name in ('manifest.json','effective_configuration.json','training_result.json','best_model.pt','final_model.pt')})
        result['reuse_policy']='20 complete original independent A0 repeats retained, 100 new runs; original RNG domains and timestamps retained'
    if result['protocol']['phase']=='formal':
        runtime_path=EVIDENCE/'runtime_single.json';equivalence_path=EVIDENCE/'blas_equivalence.json'
        probe=read(runtime_path);comparison=read(equivalence_path)
        if probe['torch_intraop']!=1 or any(p['threads']!=1 for p in probe['blas']):
            raise ValueError('single-thread runtime probe did not pass')
        if {r['scenario'] for r in comparison['results']}!=set(SCENARIOS) or any(
            r['points']!=480 or not r['voltage_violation_equal'] or not r['all_recorded_step_fields_equal']
            or not r['voltage_bitwise_equal'] for r in comparison['results']):
            raise ValueError('two-network full-day runtime comparison did not pass')
        result['execution']=dict(device='cpu',environment={'OPENBLAS_NUM_THREADS':'1'},
            torch_intraop=1,torch_interop=probe['torch_interop'],
            runtime_probe=dict(path=str(runtime_path),sha256=file_hash(runtime_path)),
            equivalence_audit=dict(path=str(equivalence_path),sha256=file_hash(equivalence_path)),
            original_training_threads='Not recorded in original manifests; do not infer retrospective bitwise training equivalence',
            current_runtime_change='Explicit OpenBLAS thread limit for new training and all fresh evaluation; no algorithm change')
    result['status']='frozen'; result['plan_sha256']=plan_hash(result)
    validate_plan(result)
    return result


def verify_reuse(plan,run,record):
    directory=Path(record['path'])
    for name,digest in record['files'].items():
        if file_hash(directory/name)!=digest:raise ValueError('reused artifact changed')
    manifest=read(directory/'manifest.json'); report=read(directory/'training_result.json')
    original=read(BASE_PLAN)
    if run['round2_variant']!='A0' or manifest['run']['variant']!='reference_B2':raise ValueError('reuse must be original reference B2')
    if manifest['run']['scenario']!=run['scenario'] or manifest['run']['seed']!=run['seed']:raise ValueError('reuse identity differs')
    if manifest['protocol']!=original['protocol'] or manifest['plan_sha256']!=original['plan_sha256']:raise ValueError('original plan binding differs')
    if report['status']!='completed' or report['environment_steps']!=240000 or report['phase']!='formal':raise ValueError('reuse needs complete formal training')
    if manifest['rng_domain']!=derive_training_streams('reference_B2',run['scenario'],run['seed']):raise ValueError('original RNG domains differ')
    current=source_manifest()['source_sha256']
    if any(current.get(path)!=digest for path,digest in manifest['source_sha256'].items()):raise ValueError('original training dependency changed')
    if data_hashes(manifest['train_environment']['data_path'])!=manifest['data_sha256']:raise ValueError('original data changed')
    for key in ('training_steps','episode_points','validation_days','validation_every_steps','formal_test_days',
                'risk_mixture','reward_by_scenario','action_scales','nighttime_reactive_power','evaluation_pv_scale'):
        if plan['protocol'][key]!=original['protocol'][key]:raise ValueError('reused A0 common protocol differs: '+key)
    return directory


@contextlib.contextmanager
def runtime(variant,reused=False):
    from models.matd3_round2_ablation import MATD3Round2Ablation,VARIANTS
    from models.model_registry import Model
    from experiments import run_q1_ml_compare as common,run_revision_training as train,revision_evaluation as evaluation
    original_loader=common.load_alg_config
    def loader(alg,config):
        if alg!='matd3_reference': raise ValueError('round two accepts reference solver only')
        values=original_loader(alg,config)
        values.update(round2_variant=variant,policy_delay=VARIANTS[variant]['policy_delay'])
        return values
    targets=[(common,'load_alg_config',loader),(train,'validate_frozen_plan',validate_plan),
        (train,'training_source_manifest',source_manifest),(evaluation,'validate_frozen_plan',validate_plan)]
    previous=[(module,key,getattr(module,key)) for module,key,_ in targets]
    old_class=Model['matd3_reference']
    try:
        Model['matd3_reference']=old_class if reused else MATD3Round2Ablation
        for module,key,value in targets: setattr(module,key,value)
        yield
    finally:
        Model['matd3_reference']=old_class
        for module,key,value in previous: setattr(module,key,value)


def run_training(plan,run_id,output,resume=False,cpu=True,stop_after_steps=None):
    validate_plan(plan)
    check_runtime(plan,cpu)
    run=next(r for r in plan['runs'] if r['run_id']==run_id)
    if run_id in plan.get('reuse_runs',{}):raise ValueError('registered original A0 cannot be retrained under the same plan identity')
    from experiments.run_revision_training import run_training as inherited
    with runtime(run['round2_variant']):
        return inherited(plan,run,Path(output),resume=resume,cpu=cpu,stop_after_steps=stop_after_steps)


def evaluate(plan,run_id,run_dir,output,cpu=True):
    validate_plan(plan)
    check_runtime(plan,cpu)
    import torch
    from experiments.revision_evaluation import load_policy_artifacts,RevisionEvaluationEnv
    from experiments.fixed_test_protocol import evaluate_policy_fixed_days,write_fixed_test_artifacts
    run=next(r for r in plan['runs'] if r['run_id']==run_id)
    phase=plan['protocol']['phase']
    days=plan['protocol']['formal_test_days'] if phase=='formal' else plan['protocol']['validation_days']
    output=Path(output).resolve()
    if RUN_ROOT.resolve() not in output.parents: raise ValueError('evaluation must remain in round-two evidence root')
    if output.exists(): raise FileExistsError('evaluation outputs cannot be overwritten')
    source=source_manifest()
    reused=run_id in plan.get('reuse_runs',{})
    if reused:
        expected=verify_reuse(plan,run,plan['reuse_runs'][run_id])
        if Path(run_dir).resolve()!=expected.resolve():raise ValueError('wrong reused directory')
    with runtime(run['round2_variant'],reused=reused):
        # Reuse has its own strict original-source/plan/budget checks above.
        # The inherited reconstruction helper's development mode avoids falsely
        # rebinding original training to a round-two training-plan hash.
        model,args,config,provenance=load_policy_artifacts(run_dir,phase='development' if reused else phase,plan=plan,checkpoint='best',cpu=cpu)
        provenance['reused_original_formal_training']=reused
        if not reused and getattr(args,'round2_variant',None)!=run['round2_variant']: raise ValueError('checkpoint variant differs')
        config.update(episode_limit=480,start_hour=0,start_minute=0,start_day_candidates=days,
            reset_action=False,add_noise=False)
        with contextlib.redirect_stdout(io.StringIO()): env=RevisionEvaluationEnv(config)
        try:
            started=time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                result=evaluate_policy_fixed_days(env=env,behaviour_net=model,args=args,days=days,max_steps=480,start_hour=0,_strict_shape=False)
            if source_manifest()['source_sha256']!=source['source_sha256']: raise RuntimeError('evaluation source changed')
            if data_hashes(config['data_path'])!=provenance['data_sha256']: raise RuntimeError('data changed')
            result.summary.update(phase=phase,round2_variant=run['round2_variant'],performance_claims_allowed=phase=='formal')
            artifacts=write_fixed_test_artifacts(result,output/'trajectory','fixed_test')
            report=dict(status='completed',run=run,plan_sha256=plan['plan_sha256'],days=days,
                source=source,provenance=provenance,summary=result.summary,wall_seconds=time.perf_counter()-started,
                artifact_sha256={k:file_hash(v) for k,v in artifacts.items()},artifacts=artifacts)
            json_write(output/'evaluation.json',report)
            return report
        finally: env.close()


def audit_a0():
    """Conservative reuse decision: changed inherited sources require review, not waiver."""
    current=source_manifest()['source_sha256']; rows=[]
    for scenario in SCENARIOS:
        for seed in range(10):
            directory=BASE_RUNS/f'reference_B2__{scenario}__seed{seed}'
            m=read(directory/'manifest.json'); r=read(directory/'training_result.json')
            changed=[p for p,h in m['source_sha256'].items() if current.get(p)!=h]
            missing=[]
            for name,h in r.get('weight_sha256',{}).items():
                path=directory/f'{name}_model.pt'
                if not path.exists() or file_hash(path)!=h: missing.append(name)
            data_ok=data_hashes(m['train_environment']['data_path'])==m['data_sha256']
            rows.append(dict(scenario=scenario,seed=seed,path=str(directory),
                original_source_matches=not changed,changed_sources=changed,
                data_matches=data_ok,missing_or_changed_checkpoints=missing,
                environment_steps=r['environment_steps'],update_counts=r['update_counts'],
                runtime=m.get('runtime'),original_rng_domain=m['rng_domain'],
                eligible_by_artifact_checks=not changed and not missing and data_ok and r['environment_steps']==240000))
    return dict(status='audit_only_no_automatic_reuse',runs=rows,
        artifact_eligible_count=sum(r['eligible_by_artifact_checks'] for r in rows),
        remaining_checks='A0 runtime/torch configuration equivalence and complete synthetic/real replay gate must be reviewed before formal reuse selection',
        default_formal_action='retrain all 120 until reuse is separately frozen')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--phase',choices=['formal','development'],default='formal');p.add_argument('--smoke-steps',type=int,default=120);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('freeze');p.add_argument('--plan',type=Path,required=True);p.add_argument('--test-report',type=Path);p.add_argument('--reuse-a0-audit',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('audit-a0');p.add_argument('--output',type=Path,required=True)
    for command in ('train','evaluate'):
        p=sub.add_parser(command);p.add_argument('--plan',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda',action='store_true')
        if command=='train':p.add_argument('--resume',action='store_true');p.add_argument('--development-stop-after-steps',type=int)
        else:p.add_argument('--run-dir',type=Path,required=True)
    a=parser.parse_args(argv)
    if a.command in ('prepare','freeze','audit-a0'):
        if a.output.exists(): raise FileExistsError('refusing to overwrite plan/audit')
        result=build_plan(a.phase,a.smoke_steps) if a.command=='prepare' else freeze(read(a.plan),a.test_report,a.reuse_a0_audit) if a.command=='freeze' else audit_a0()
        json_write(a.output,result); print(json.dumps({'output':str(a.output),'status':result.get('status'),'runs':len(result.get('runs',[]))}))
    elif a.command=='train':
        run_training(read(a.plan),a.run_id,a.output,resume=a.resume,cpu=not a.cuda,stop_after_steps=a.development_stop_after_steps)
    else:
        r=evaluate(read(a.plan),a.run_id,a.run_dir,a.output,cpu=not a.cuda)
        print(json.dumps({'status':r['status'],'summary':r['summary']}))

if __name__=='__main__': main()
