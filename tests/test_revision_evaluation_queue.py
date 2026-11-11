import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import revision_evaluation_queue as queue
from experiments import revision_evaluation as evaluation
from experiments import revision_stress_statistics as stress_stats
from experiments.revision_run_plan import build_plan, canonical_hash, SCENARIOS


@pytest.fixture
def plan(tmp_path, monkeypatch):
    # The tests isolate queue contracts; no fake gate is passed to real evaluation.
    pools = tmp_path/'pools'
    pools.mkdir()
    for scenario in SCENARIOS:
        directory = tmp_path/scenario
        directory.mkdir()
        (directory/'data.csv').write_text('synthetic test fixture', encoding='utf-8')
        queue.write_new_json(pools/(scenario+'_pool.json'),
            {'data_directory': str(directory), 'data_sha256': queue.data_hashes(directory)})
    candidate = build_plan(pools)
    monkeypatch.setattr(evaluation, 'evaluation_source_manifest', lambda: {'source_sha256': {'test.py': 'synthetic'}})
    registered = queue.register_evaluation_jobs(candidate)
    registered.update(status='frozen', source={'source_sha256': {'train.py': 'synthetic'}})
    registered['plan_sha256'] = canonical_hash(registered)
    def check_plan(value):
        assert value['status'] == 'frozen'
        assert value['plan_sha256'] == canonical_hash({k:v for k,v in value.items() if k != 'plan_sha256'})
    monkeypatch.setattr(queue, 'validate_frozen_plan', check_plan)
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    return registered


def _rehash(plan):
    plan['plan_sha256'] = canonical_hash({k:v for k,v in plan.items() if k != 'plan_sha256'})


def _metadata(job, plan, *, vrate=.2, cr=.8, failed_steps=0):
    run = next((r for r in plan['runs'] if r['run_id'] == job['run_id']), None)
    data_hashes=queue.read_json(plan['pools'][job['scenario']]['path'])['data_sha256']
    summary = {'planned_days':30, 'planned_steps':14400, 'executed_steps':14400-failed_steps,
        'failed_steps':failed_steps, 'vrate':vrate, 'cr':cr,
        'metric_denominators':{'vrate':14400, 'cr':14400}}
    for metric in stress_stats.PHYSICAL_METRICS:
        summary[metric] = None if failed_steps == 14400 else .1
        summary['metric_denominators'][metric] = 14400-failed_steps
    return {'evaluation_job':job, 'evaluation_registry_sha256':plan['evaluation_registry_sha256'],
        'phase':'formal', 'days':queue.DAYS, 'points_per_day':480,
        'source':{'source_sha256':plan['evaluation_source_sha256']},
        'summary':summary, 'environment_configuration':{'nighttime_reactive_power':job['nighttime_reactive_power']},
        'provenance': {'plan_sha256':plan['plan_sha256'], 'run':run, 'checkpoint':job['checkpoint'],
                       'checkpoint_sha256':'best', 'training_manifest_sha256':'manifest', 'data_sha256':data_hashes},
        'data_sha256':data_hashes,
        'plan_sha256':plan['plan_sha256'], 'method':job['method'], 'scenario':job['scenario'],
        'stress_case':job['stress_case'], 'stress_seed':job['stress_seed'],
        'evaluation_night_mode':job['nighttime_reactive_power'], 'relocation_offset':job['relocation_offset'],
        'physical_exit_controller':job['physical_exit_controller']}


def _write_training(run_dir, run, plan):
    run_dir.mkdir(parents=True)
    data = queue.read_json(plan['pools'][run['scenario']]['path'])
    manifest = {'run':run, 'protocol':plan['protocol'], 'plan_sha256':plan['plan_sha256'],
        'source_sha256':plan['source']['source_sha256'], 'data_sha256':data['data_sha256'],
        'validation_environment':{'data_path':data['data_directory']}}
    queue.write_new_json(run_dir/'manifest.json', manifest)
    queue.write_new_json(run_dir/'effective_configuration.json', {'args':{}})
    for checkpoint in ('best','final'):
        (run_dir/(checkpoint+'_model.pt')).write_bytes(checkpoint.encode())
    queue.write_new_json(run_dir/'training_result.json', {'run':run, 'status':'completed', 'phase':'formal',
        'environment_steps':240000, 'weight_sha256':{c:queue.file_hash(run_dir/(c+'_model.pt')) for c in ('best','final')}})


def test_registry_exact_core_count_and_one_factor_design(plan):
    jobs = list(queue.validate_registry(plan).values())
    assert len(jobs) == 3132
    assert Counter(j['role'] for j in jobs) == {'primary_best_clean':160,'final_clean':160,'stress':2800,'traditional_clean':12}
    for job in jobs:
        assert job['days'] == list(range(1030,1060)) and job['points_per_day'] == 480
        if job['role'] == 'stress':
            assert job['variant'] in ('reference_B2','maddpg_B2')
            changes = sum([job['stress_case'] != 'clean', job['nighttime_reactive_power'] != 'enabled',
                           job['relocation_offset'] != 0, job['physical_exit_controller'] is not None])
            assert changes == 1
    conventional = [j for j in jobs if j['kind']=='traditional']
    assert Counter(j['method'] for j in conventional) == {'zero':4,'droop':4,'opf':4}
    assert all(j['checkpoint'] is None and j['seed'] is None for j in conventional)
    assert 'plan_sha256' not in str(plan['evaluation_registry'])


def test_external_rng_common_conditions_independent_of_training_seeds(plan):
    rows = [j for j in plan['evaluation_registry']['jobs'] if j['stress_case']=='voltage_noise_0.005']
    for scenario in SCENARIOS:
        values = {j['realization']:j['stress_seed'] for j in rows if j['scenario']==scenario}
        assert len(set(values.values())) == 10
        assert all(j['stress_seed']==values[j['realization']] for j in rows if j['scenario']==scenario)
    assert all(j['realizations']==1 and j['realization']==0 for j in plan['evaluation_registry']['jobs']
               if j['stress_case'] in queue.DETERMINISTIC_CASES)


def test_unregistered_mixed_stress_or_seed_is_rejected(plan):
    job = next(j for j in plan['evaluation_registry']['jobs'] if j['stress_case']=='voltage_noise_0.001')
    request = {key:job[key] for key in queue.REQUEST_KEYS}
    assert queue.require_registered_request(plan, job['job_id'], **request) == job
    for change in ({'relocation_offset':1}, {'stress_seed':job['stress_seed']+1}, {'checkpoint':'final'},
                   {'nighttime_reactive_power':'disabled'}, {'days':[1030]}):
        with pytest.raises(ValueError, match='invocation'):
            queue.require_registered_request(plan, job['job_id'], **dict(request, **change))
    with pytest.raises(ValueError, match='job_id'):
        queue.require_registered_request(plan, None, **request)


def test_rehashed_short_registry_is_rejected(plan):
    plan['evaluation_registry']['jobs'].pop()
    plan['evaluation_registry_sha256'] = canonical_hash(plan['evaluation_registry'])
    _rehash(plan)
    with pytest.raises(ValueError, match='complete preregistered'):
        queue.validate_registry(plan)


def test_external_registry_file_is_immutable(plan,tmp_path):
    path=tmp_path/'registry.json'
    queue.write_new_json(path,plan['evaluation_registry'])
    plan['evaluation_registry_file']={'path':str(path),'sha256':queue.file_hash(path)}
    _rehash(plan)
    queue.validate_registry(plan)
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='registry file'):
        queue.validate_registry(plan)


def test_registration_cannot_rewrite_frozen_plan(plan):
    with pytest.raises(ValueError, match='before freezing'):
        queue.register_evaluation_jobs(plan)


def test_formal_evaluator_requires_registered_job_before_model_loading(plan,tmp_path,monkeypatch):
    run=plan['runs'][0]
    directory=tmp_path/'training'
    directory.mkdir()
    queue.write_new_json(directory/'manifest.json',{'run':run})
    monkeypatch.setattr(evaluation,'load_policy_artifacts',lambda *a,**k: pytest.fail('model loaded before registry check'))
    with pytest.raises(ValueError,match='job_id'):
        evaluation.evaluate_run(run_dir=directory,output=tmp_path/'unused',phase='formal',plan=plan)
    job=next(j for j in plan['evaluation_registry']['jobs'] if j['kind']=='traditional')
    monkeypatch.setattr(evaluation,'validate_frozen_plan',lambda p:None)
    with pytest.raises(ValueError,match='invocation'):
        evaluation.evaluate_traditional(scenario=job['scenario'],output=tmp_path/'unused',days=queue.DAYS,
            phase='formal',plan=plan,method=job['method'],job_id=job['job_id'],nighttime_reactive_power='disabled')


@pytest.mark.parametrize('corruption',['source','budget','weights','data'])
def test_training_preflight_rejects_mismatched_evidence(plan,tmp_path,corruption):
    run=plan['runs'][0]
    directory=tmp_path/'training'
    _write_training(directory,run,plan)
    queue.verify_training_artifacts(directory,run,plan)
    if corruption=='weights':
        (directory/'best_model.pt').write_bytes(b'changed')
    elif corruption=='data':
        data=queue.read_json(plan['pools'][run['scenario']]['path'])
        (Path(data['data_directory'])/'data.csv').write_text('changed')
    else:
        path=directory/('manifest.json' if corruption=='source' else 'training_result.json')
        value=queue.read_json(path)
        value['source_sha256' if corruption=='source' else 'environment_steps'] = {} if corruption=='source' else 239999
        path.write_text(__import__('json').dumps(value),encoding='utf-8')
    with pytest.raises(ValueError):
        queue.verify_training_artifacts(directory,run,plan)


def test_missing_training_blocks_entire_preflight(plan,tmp_path):
    with pytest.raises(FileNotFoundError):
        queue.preflight_training(plan,tmp_path/'missing')


def _attempt_fixture(plan,tmp_path):
    job=plan['evaluation_registry']['jobs'][0]
    attempt=tmp_path/'exp/revision_access_41572/q/jobs'/job['job_id']/'attempt0001'
    inputs={'best_model.pt':'best','manifest.json':'manifest'}
    queue.write_new_json(attempt/'result/evaluation.json',_metadata(job,plan))
    queue.write_new_json(attempt/'completed.json',{'status':'completed','job':job,'plan_sha256':plan['plan_sha256'],
        'evaluation_registry_sha256':plan['evaluation_registry_sha256'],
        'evaluation_source_sha256':plan['evaluation_source_sha256'],'training_input_sha256':inputs,
        'artifacts':queue.result_inventory(attempt/'result')})
    return job,attempt,inputs


def test_completed_receipt_rechecks_artifacts_inputs_and_denominators(plan,tmp_path):
    job,attempt,inputs=_attempt_fixture(plan,tmp_path)
    assert queue.verify_completed_attempt(attempt,job,plan,inputs)['status']=='completed'
    with pytest.raises(ValueError,match='hash mismatch'):
        queue.verify_completed_attempt(attempt,job,plan,{'best_model.pt':'changed'})
    (attempt/'result/extra.txt').write_text('unregistered mutation')
    with pytest.raises(ValueError,match='hash mismatch'):
        queue.verify_completed_attempt(attempt,job,plan,inputs)
    meta=_metadata(job,plan)
    meta['summary']['metric_denominators']['vrate']=14399
    with pytest.raises(ValueError,match='denominators'):
        queue.validate_result_metadata(meta,job,plan)


def test_retry_is_fresh_and_failed_attempt_is_not_overwritten(plan,tmp_path,monkeypatch):
    job=plan['evaluation_registry']['jobs'][0]
    output=tmp_path/'exp/revision_access_41572/retries'
    prior=output/'jobs'/job['job_id']/'attempt0001'
    queue.write_new_json(prior/'failed.json',{'immutable':'old failure'})
    before=(prior/'failed.json').read_bytes()
    calls=[]
    monkeypatch.setattr(queue.subprocess,'run',lambda command,**kw:(calls.append(command) or SimpleNamespace(returncode=7)))
    record=queue.dispatch_one(tmp_path/'plan.json',tmp_path/'train',output,job,plan,{},resume=True,retry_failed=False,cpu=True)
    assert record['status']=='unresolved_attempt' and calls==[]
    record=queue.dispatch_one(tmp_path/'plan.json',tmp_path/'train',output,job,plan,{},resume=True,retry_failed=True,cpu=True)
    assert record['status']=='failed' and len(calls)==1
    assert Path(record['attempt']).name=='attempt0002'
    assert (prior/'failed.json').read_bytes()==before
    assert (Path(record['attempt'])/'exit.json').exists()


def test_duplicate_completion_never_selects_favourable_attempt(plan,tmp_path):
    job,attempt,inputs=_attempt_fixture(plan,tmp_path)
    second=attempt.parent/'attempt0002'
    queue.write_new_json(second/'completed.json',{})
    with pytest.raises(ValueError,match='multiple completed'):
        queue.dispatch_one(tmp_path/'plan.json',tmp_path/'train',attempt.parents[2],job,plan,inputs,
                           resume=True,retry_failed=False,cpu=True)


def test_collector_rejects_changed_aggregation_source_before_reading_results(plan,tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'evaluation_source_manifest',lambda:{'source_sha256':{'changed.py':'new'}})
    with pytest.raises(ValueError,match='source changed'):
        queue.collect_verified_results(plan,tmp_path/'absent')


def test_collector_requires_every_registered_job_not_just_successful_glob(plan,tmp_path,monkeypatch):
    job,attempt,_=_attempt_fixture(plan,tmp_path)
    output=attempt.parents[2]
    inputs={'completed':{},'reviewed_failures':{}}
    queue.write_new_json(output/'queue_identity.json',{'plan_sha256':plan['plan_sha256'],
        'evaluation_registry_sha256':plan['evaluation_registry_sha256'],
        'evaluation_source_sha256':plan['evaluation_source_sha256'],'training_root':str(tmp_path/'t'),
        'training_inputs':inputs})
    monkeypatch.setattr(queue,'preflight_training',lambda *args:inputs)
    with pytest.raises(ValueError,match='must complete exactly once'):
        queue.collect_verified_results(plan,output)


def test_new_json_never_overwrites_or_leaves_partial_publication(tmp_path):
    path=tmp_path/'record.json'
    queue.write_new_json(path,{'old':1})
    with pytest.raises(FileExistsError):
        queue.write_new_json(path,{'new':2})
    assert queue.read_json(path)=={'old':1}
    assert not list(tmp_path.glob('*.tmp'))


def test_worker_publishes_verified_receipt_and_refuses_reexecution(plan,tmp_path,monkeypatch):
    run=plan['runs'][0]
    job=plan['evaluation_registry']['jobs'][0]
    train_root=tmp_path/'t'
    _write_training(train_root/run['run_id'],run,plan)
    inputs=queue.verify_training_artifacts(train_root/run['run_id'],run,plan)
    plan_path=tmp_path/'plan.json'
    queue.write_new_json(plan_path,plan)
    attempt=tmp_path/'exp/revision_access_41572/q/attempt0001'
    calls=[]
    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        meta=_metadata(job,plan)
        meta['provenance'].update(checkpoint_sha256=inputs['best_model.pt'],
                                  training_manifest_sha256=inputs['manifest.json'])
        queue.write_new_json(kwargs['output']/'evaluation.json',meta)
    monkeypatch.setattr(evaluation,'evaluate_run',fake_evaluate)
    record=queue.execute_job(plan_path,train_root,job['job_id'],attempt)
    assert record['status']=='completed' and len(calls)==1
    assert calls[0]['days']==list(range(1030,1060)) and calls[0]['job_id']==job['job_id']
    with pytest.raises(FileExistsError):
        queue.execute_job(plan_path,train_root,job['job_id'],attempt)
    assert len(calls)==1


def test_worker_crash_retains_partial_evidence_without_completion(plan,tmp_path,monkeypatch):
    run=plan['runs'][0]
    job=plan['evaluation_registry']['jobs'][0]
    train_root=tmp_path/'t'
    _write_training(train_root/run['run_id'],run,plan)
    plan_path=tmp_path/'plan.json'
    queue.write_new_json(plan_path,plan)
    attempt=tmp_path/'exp/revision_access_41572/q/attempt0001'
    def fail(**kwargs):
        queue.write_new_json(kwargs['output']/'partial.json',{'stopped':True})
        raise RuntimeError('synthetic process failure')
    monkeypatch.setattr(evaluation,'evaluate_run',fail)
    with pytest.raises(RuntimeError,match='synthetic'):
        queue.execute_job(plan_path,train_root,job['job_id'],attempt)
    assert (attempt/'failed.json').exists() and not (attempt/'completed.json').exists()
    assert queue.read_json(attempt/'failed.json')['partial_artifacts']==queue.result_inventory(attempt/'result')


def test_reviewed_failure_preflight_requires_explicit_validated_evidence(plan,tmp_path,monkeypatch):
    from experiments import revision_failure_accounting as failure_accounting
    training_root=tmp_path/'t'
    run=plan['runs'][0]
    record_path=tmp_path/'review.json'
    queue.write_new_json(record_path,{'run_directory':str(training_root/run['run_id'])})
    hashes=queue.read_json(plan['pools'][run['scenario']]['path'])['data_sha256']
    calls=[]
    def reviewed(path,passed_plan):
        calls.append(path)
        assert passed_plan is plan
        return {'run_id':run['run_id'],'data_sha256':hashes,'training_failure':True}
    monkeypatch.setattr(failure_accounting,'validate_failure_record',reviewed)
    verified=[]
    monkeypatch.setattr(queue,'verify_training_artifacts',lambda path,row,p,**kw:(verified.append(row['run_id']) or {'synthetic':'fixture'}))
    result=queue.preflight_training(plan,training_root,[record_path])
    assert calls==[record_path] and len(verified)==159 and run['run_id'] not in verified
    assert len(result['reviewed_failures'])==1
    monkeypatch.setattr(failure_accounting,'validate_failure_record',lambda *args:(_ for _ in ()).throw(ValueError('unreviewed')))
    with pytest.raises(ValueError,match='unreviewed'):
        queue.preflight_training(plan,training_root,[record_path])


@pytest.mark.parametrize('workers',[0,13,1.5,True])
def test_worker_bounds_before_any_dispatch(workers,tmp_path):
    with pytest.raises(ValueError,match='workers'):
        queue.dispatch(tmp_path/'absent',tmp_path,tmp_path/'out',workers=workers)


def _stress_rows(plan):
    return [{'job':job,'status':'completed','metadata':_metadata(job,plan,
        vrate=(job['seed']+job['realization']/10)/10,cr=1-(job['seed']+job['realization']/10)/10)}
        for job in plan['evaluation_registry']['jobs'] if job['role'] in ('primary_best_clean','stress')
        and job['variant'] in queue.STRESS_VARIANTS]


def test_stress_averages_realizations_before_n10_summary(plan):
    result=stress_stats.aggregate_stress_records(plan,_stress_rows(plan),bootstrap_resamples=100)
    group=next(g for g in result['groups'] if g['condition']=='voltage_noise_0.001')
    assert group['n_training_seeds']==10
    assert group['n_executed_realizations']==100
    assert group['vrate_fraction']['n_training_seeds']==10
    assert group['vrate_fraction']['raw_seed_values']==pytest.approx([s/10+.045 for s in range(10)])
    assert group['vrate_fraction']['mean']==pytest.approx(.495)
    deterministic=next(g for g in result['groups'] if g['condition']=='zone_offset_1')
    assert deterministic['n_executed_realizations']==10


@pytest.mark.parametrize('corruption',['missing','duplicate','unresolved'])
def test_stress_never_drops_missing_or_failed_external_realizations(plan,corruption):
    rows=_stress_rows(plan)
    if corruption=='missing': rows.pop()
    elif corruption=='duplicate': rows.append(rows[0])
    else: rows[0]['status']='failed'
    with pytest.raises(ValueError):
        stress_stats.aggregate_stress_records(plan,rows,bootstrap_resamples=10)


def test_reviewed_training_failure_counts_seed_once_without_fake_trajectories(plan):
    rows=_stress_rows(plan)
    for row in rows:
        job=row['job']
        if job['variant']=='reference_B2' and job['scenario']==SCENARIOS[0] and job['seed']==3:
            row.update(status=queue.FAILED_TRAINING,failure_record='review.json',failure_record_sha256='reviewed',
                failure={'seed':3,'variant':job['variant'],'scenario':job['scenario'],'training_failure':True,
                         'planned_steps':14400,'vrate':1.,'cr':0.})
            del row['metadata']
    result=stress_stats.aggregate_stress_records(plan,rows,bootstrap_resamples=100)
    group=next(g for g in result['groups'] if g['variant']=='reference_B2' and g['scenario']==SCENARIOS[0]
               and g['condition']=='remote_missing_0.05')
    assert group['n_training_seeds']==10 and group['n_reviewed_training_failures']==1
    assert group['n_executed_realizations']==90
    failed=group['seeds'][3]
    assert failed['vrate']==1 and failed['executed_realizations']==0
    assert 'realization_vrate' not in failed and 'job_ids' not in failed
    assert group['physical']['q_loss']['summary'] is None
    assert group['physical']['q_loss']['seed_values'][3] is None


def test_powerflow_failure_keeps_safety_denominator_and_physical_null(plan):
    rows=_stress_rows(plan)
    for row in rows:
        if row['job']['condition']=='power_noise_0.05' and row['job']['seed']==0 and row['job']['realization']==0:
            row['metadata']=_metadata(row['job'],plan,vrate=1.,cr=0.,failed_steps=14400)
    result=stress_stats.aggregate_stress_records(plan,rows,bootstrap_resamples=10)
    group=next(g for g in result['groups'] if g['condition']=='power_noise_0.05')
    seed=group['seeds'][0]
    assert seed['planned_condition_steps']==144000 and seed['powerflow_failed_steps']==14400
    assert seed['vrate']==pytest.approx(.145)
    assert seed['physical']['q_loss']['mean_of_realization_means'] is None
    assert seed['physical']['q_loss']['finite_realization_means']==9
    assert group['physical']['q_loss']['summary'] is None
