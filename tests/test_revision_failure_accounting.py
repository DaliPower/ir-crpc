"""Synthetic failure records exercise safeguards; they are not experiment data."""
import copy
import json
from pathlib import Path
import pytest
from experiments.revision_failure_accounting import (
    FAILURE_POLICY, canonical, digest, evidence, validate_failure_record)


def write(path, value):
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def record(tmp_path):
    run={'run_id':'reference_B2__case33_3min_final__seed0','variant':'reference_B2',
         'solver':'matd3_reference','scenario':'case33_3min_final','seed':0,
         'sampling':'B2','mapping':'capacity_adaptive'}
    plan={'status':'frozen','runs':[run],
          'source':{'source_sha256':{'synthetic_fixture.py':'test_only'}},
          'protocol':{'phase':'formal','training_steps':240000,
                      'formal_test_days':list(range(1030,1060)),
                      'training_failure_scoring':copy.deepcopy(FAILURE_POLICY)}}
    plan['plan_sha256']=canonical(plan)
    identity={'domain':'revision-access-41572-independent-training-v1',
              'variant':run['variant'],'scenario':run['scenario'],'replicate_seed':0}
    manifest={'run':run,'plan_sha256':plan['plan_sha256'],'protocol':plan['protocol'],
              'source_sha256':plan['source']['source_sha256'],
              'data_sha256':{'test_only.csv':'synthetic_fixture'},
              'independent_training_rng_domain':{'identity':identity,'domain_sha256':canonical(identity)}}
    write(tmp_path/'manifest.json',manifest)
    failure=tmp_path/'failure_history.jsonl'
    write(failure,{'error_type':'FloatingPointError','message':'synthetic nonfinite gradient',
                  'traceback':'SYNTHETIC UNIT-TEST TRACEBACK; NOT REAL EVIDENCE'})
    audit=tmp_path/'audit.json'
    write(audit,{'run_id':run['run_id'],'failure_history_sha256':digest(failure),
                 'nonfinite_learning_state_confirmed':True,
                 'finite_input_and_valid_configuration_confirmed':True,
                 'fixture_only':True})
    meta={'phase':'formal','status':'reviewed_terminal_algorithm_failure',
          'plan_sha256':plan['plan_sha256'],'run':run,'run_directory':str(tmp_path),
          'training_manifest':evidence(tmp_path/'manifest.json'),'failure_history':evidence(failure),
          'diagnosis':{'classification':'algorithm_numerical_failure',
             'system_interruption_excluded':True,'implementation_error_excluded':True,
             'rationale':'Synthetic unit-test fixture: numerical learner divergence on validated finite inputs.',
             'failure_record_index':0,'evidence':[evidence(audit)],'numerical_state_audit':evidence(audit)}}
    path=tmp_path/'training_failure.json';write(path,meta)
    return path,plan,meta


def test_failed_seed_retains_denominator_but_no_physical_measurements(record):
    path,plan,_=record
    row=validate_failure_record(path,plan)
    assert row['planned_steps']==14400 and row['planned_days']==30
    assert row['vrate']==1 and row['cr']==0 and row['executed_steps']==0
    assert row['training_failure'] and row['safety_values_are_conservative_penalties']
    for key in ('trajectory_directory','full_30day_energy_mwh','observed_energy_mwh',
                'mean_voltage_deviation_pu','mean_q_mvar_executed_only','mean_loss_mw_executed_only'):
        assert row[key] is None


@pytest.mark.parametrize('status',['running','system_interruption','failed','missing'])
def test_incomplete_is_never_automatically_scored(record,status):
    path,plan,meta=record;meta['status']=status;write(path,meta)
    with pytest.raises(ValueError,match='not formal algorithm failures'):
        validate_failure_record(path,plan)


def test_undeclared_penalty_is_rejected(record):
    path,plan,_=record
    del plan['protocol']['training_failure_scoring']
    plan['plan_sha256']=canonical({k:v for k,v in plan.items() if k!='plan_sha256'})
    with pytest.raises(ValueError,match='predeclared'):
        validate_failure_record(path,plan)


def test_completed_and_failed_same_seed_cannot_both_count(record):
    path,plan,_=record;write(path.parent/'training_result.json',{'status':'completed'})
    with pytest.raises(ValueError,match='simultaneously'):
        validate_failure_record(path,plan)


def test_evidence_cannot_be_changed_after_diagnosis(record):
    path,plan,_=record;write(path.parent/'audit.json',{'modified':True})
    with pytest.raises(ValueError,match='hash mismatch'):
        validate_failure_record(path,plan)


def test_unreviewed_numerical_failure_does_not_count(record):
    path,plan,meta=record;meta['diagnosis']['implementation_error_excluded']=False;write(path,meta)
    with pytest.raises(ValueError,match='explicit diagnosis'):
        validate_failure_record(path,plan)


def test_runtime_resource_failure_not_algorithm_failure(record):
    path,plan,meta=record
    failure=path.parent/'failure_history.jsonl'
    write(failure,{'error_type':'MemoryError','message':'out of RAM','traceback':'TEST ONLY'})
    meta['failure_history']=evidence(failure);write(path,meta)
    with pytest.raises(ValueError,match='infrastructure'):
        validate_failure_record(path,plan)


def test_final_checkpoint_clean_evaluation_is_excluded_from_primary(tmp_path):
    from manuscript_revision.revision_results_report import collect_formal
    write(tmp_path/'evaluation.json',{'phase':'formal','stress_case':'clean',
                                    'provenance':{'checkpoint':'final'}})
    _,gap=collect_formal({'runs':[],'protocol':{}},tmp_path)
    assert len(gap['excluded_evaluations'])==1
    assert not any('evaluation.json:' in item for item in gap['invalid_or_unfrozen_reasons'])


def test_queue_failure_external_record_is_not_lost_by_filesystem_glob(tmp_path,monkeypatch):
    from manuscript_revision.revision_results_report import collect_formal
    import experiments.revision_evaluation_queue as queue
    write(tmp_path/'queue_identity.json',{})
    row={'run_id':'test0','scenario':'case33','variant':'ref','seed':0,
         'training_failure':True,'rng_domain_sha256':'domain','data_sha256':{'test':'hash'}}
    monkeypatch.setattr(queue,'collect_verified_results',lambda plan,root:[
        {'job':{'role':'primary_best_clean'},'status':queue.FAILED_TRAINING,'failure':row},
        {'job':{'role':'final_clean'},'status':queue.FAILED_TRAINING,'failure':row}])
    rows,gap=collect_formal({'runs':[{'run_id':'test0','scenario':'case33','variant':'ref','seed':0}],
                           'protocol':{}},tmp_path)
    assert rows==[row] and gap['reviewed_training_failures']==1
    assert gap['verified_formal_runs']==0 and gap['missing_run_ids']==[]


def test_partial_queue_never_globs_a_failed_attempt_as_a_result(tmp_path,monkeypatch):
    from manuscript_revision.revision_results_report import collect_formal
    import experiments.revision_evaluation_queue as queue
    write(tmp_path/'queue_identity.json',{})
    write(tmp_path/'evaluation.json',{'phase':'formal','stress_case':'clean',
                                    'provenance':{'checkpoint':'best'}})
    def incomplete(*args): raise ValueError('registered job incomplete')
    monkeypatch.setattr(queue,'collect_verified_results',incomplete)
    rows,gap=collect_formal({'runs':[],'protocol':{}},tmp_path)
    assert not rows and any('queue is incomplete' in x for x in gap['invalid_or_unfrozen_reasons'])
    assert not any('evaluation.json:' in x for x in gap['invalid_or_unfrozen_reasons'])


def test_resigned_shortened_formal_plan_is_not_accepted_by_standalone_report(tmp_path):
    from manuscript_revision.revision_results_report import collect_formal,canonical
    plan={'status':'frozen','protocol':{'phase':'formal','training_steps':1},'runs':[]}
    plan['plan_sha256']=canonical(plan)
    _,gap=collect_formal(plan,tmp_path)
    assert gap['status']=='incomplete'
    assert any('structure/evidence' in item for item in gap['invalid_or_unfrozen_reasons'])
