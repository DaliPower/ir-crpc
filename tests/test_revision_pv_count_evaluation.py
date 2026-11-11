import copy
import json
from pathlib import Path
import pytest

from experiments.revision_run_plan import build_plan,SCENARIOS,canonical_hash
from experiments.revision_pv_count_plan import build_count_plan
from experiments.revision_pv_count_queue import register_evaluation_jobs,validate_registry,FAILED_TRAINING
from experiments.revision_pv_count_evaluation import validate_dates,evaluation_source_manifest
from experiments.revision_pv_count_reporting import aggregate_count_records,physical_values


def registered(tmp_path):
    for s in SCENARIOS:
        (tmp_path/(s+'_pool.json')).write_text(json.dumps({'data_sha256':{}}))
    base=tmp_path/'base.json'
    base.write_text(json.dumps(build_plan(tmp_path)))
    index=tmp_path/'index.json'
    index.write_text(json.dumps({s:{'dataset_manifest':str(tmp_path/(s+'_pool.json')),
        'pool':str(tmp_path/(s+'_pool.json'))} for s in SCENARIOS}))
    plan=register_evaluation_jobs(build_count_plan(base,index))
    plan['plan_sha256']='synthetic-unit-fixture'
    return plan


def synthetic_rows(plan):
    rows=[]
    for job in plan['evaluation_registry']['jobs']:
        summary={'planned_days':30,'planned_steps':14400,'executed_steps':14400,'failed_steps':0,
            'vrate':.1,'cr':.9,'average_voltage_deviation':.01,'q_loss':.2,'total_line_loss':.3,
            'metric_denominators':{name:14400 for name in ('vrate','cr','average_voltage_deviation','q_loss','total_line_loss')}}
        meta={'evaluation_job':job,'evaluation_registry_sha256':plan['evaluation_registry_sha256'],
            'phase':'formal','days':list(range(1030,1060)),'points_per_day':480,
            'source':{'source_sha256':plan['evaluation_source_sha256']},'summary':summary,
            'environment_configuration':{'nighttime_reactive_power':job['nighttime_reactive_power']},
            'plan_sha256':plan['plan_sha256'],'method':job['method'],'scenario':job['scenario'],
            'data_sha256':{},'stress_case':'clean','stress_seed':41572,
            'evaluation_night_mode':job['nighttime_reactive_power'],'relocation_offset':0,'physical_exit_controller':None,
            'provenance':{'plan_sha256':plan['plan_sha256'],'run':{'run_id':job['run_id']},
                'checkpoint':job['checkpoint'],'data_sha256':{}}}
        rows.append({'job':job,'status':'completed','metadata':meta})
    return rows


def test_count_registry_preregisters_every_seed_checkpoint_and_traditional_once(tmp_path):
    plan=registered(tmp_path)
    jobs=validate_registry(plan,verify_plan=False)
    assert len(jobs)==86
    assert sum(j['kind']=='policy' for j in jobs.values())==80
    assert sum(j['kind']=='traditional' for j in jobs.values())==6
    assert {j['method'] for j in jobs.values() if j['kind']=='traditional'}=={'zero','droop','opf'}
    for run in plan['runs']:
        assert {j['checkpoint'] for j in jobs.values() if j['run_id']==run['run_id']}=={'best','final'}
    plan['evaluation_registry']['jobs'].pop()
    plan['evaluation_registry_sha256']=canonical_hash(plan['evaluation_registry'])
    with pytest.raises(ValueError,match='registry'):
        validate_registry(plan,verify_plan=False)


@pytest.mark.parametrize('phase,days',[('formal',[1030]),('development',[1030]),('formal',list(range(1000,1030))),('development',[1000,1000])])
def test_count_dates_cannot_use_heldout_days_in_development_or_shorten_formal(phase,days):
    with pytest.raises(ValueError):
        validate_dates(phase,days)


def test_count_evaluation_sources_include_own_scope_and_reused_receipt_helpers():
    sources=evaluation_source_manifest()['source_sha256']
    assert 'experiments/revision_pv_count_queue.py' in sources
    assert 'experiments/revision_evaluation_queue.py' in sources
    assert 'experiments/revision_statistics.py' in sources
    assert 'experiments/revision_benchmark.py' not in sources


def test_count_reporting_preserves_ten_seeds_and_failure_nulls_best_final_separate(tmp_path):
    plan=registered(tmp_path)
    rows=synthetic_rows(plan)
    failed_run=plan['runs'][0]
    for row in rows:
        job=row['job']
        if job['run_id']==failed_run['run_id']:
            row.clear()
            row.update(job=job,status=FAILED_TRAINING,failure_record='synthetic-reviewed-failure',
                failure={'run_id':job['run_id'],'training_failure':True,'vrate':1.,'cr':0.,'planned_steps':14400})
    report=aggregate_count_records(plan,rows,bootstrap_resamples=20)
    assert len(report['groups'])==8
    assert len(report['traditional'])==6
    assert len(report['predeclared_best_comparisons'])==2
    affected=[g for g in report['groups'] if g['scenario']==failed_run['scenario'] and g['variant']==failed_run['variant']]
    assert {g['checkpoint'] for g in affected}=={'best','final'}
    for group in affected:
        assert group['n_planned_training_seeds']==10 and group['n_training_failures']==1
        assert group['vrate_fraction']['n_training_seeds']==10
        assert group['physical']['q_loss']['summary'] is None
        assert group['seeds'][0]['physical']['total_abs_q_mvar']['mean_executed_only'] is None


def test_missing_count_seed_or_traditional_job_blocks_report(tmp_path):
    plan=registered(tmp_path)
    rows=synthetic_rows(plan)
    with pytest.raises(ValueError,match='86'):
        aggregate_count_records(plan,rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):
        aggregate_count_records(plan,rows+[rows[0]])


def test_physical_report_does_not_invent_full_energy_for_partial_coverage():
    value=physical_values({'average_voltage_deviation':None,'q_loss':.2,'total_line_loss':.3,
        'metric_denominators':{'average_voltage_deviation':0,'q_loss':100,'total_line_loss':100}},9)
    assert value['total_abs_q_mvar']['mean_executed_only']==pytest.approx(1.8)
    assert value['loss_energy_mwh']['full_30day_energy'] is None
    assert value['loss_energy_mwh']['observed_partial_energy']==pytest.approx(1.5)
    with pytest.raises(ValueError):
        physical_values({'average_voltage_deviation':0.,'q_loss':None,'total_line_loss':None,'metric_denominators':{}},9)
