from pathlib import Path
import json
import pytest
from experiments.revision_continue_pipeline import verify_junit,verify_diagnosis
from experiments import revision_continue_pipeline as pipeline


@pytest.mark.parametrize('attribute',['failures','errors','skipped'])
def test_pipeline_cannot_freeze_with_failed_or_unexecuted_checks(tmp_path,attribute):
    file=tmp_path/'suite.xml'
    file.write_text(f'<testsuites><testsuite tests="320" {attribute}="1"/></testsuites>')
    with pytest.raises(ValueError,match='incomplete or failing'): verify_junit(file)


def test_pipeline_cannot_accept_smoke_suite_as_full_gate(tmp_path):
    file=tmp_path/'suite.xml';file.write_text('<testsuite tests="5"/>')
    with pytest.raises(ValueError,match='incomplete'): verify_junit(file)


def test_pipeline_verifies_witness_numeric_residuals_not_success_label(tmp_path):
    witness={'status':'solved','inputs_unchanged':True,'voltage_violation_max_pu':0.,
        'control_bound_violation_max_mvar':0.,'inverter_circle_violation_max_mva2':0.,'ac_balance_max_mva':.01}
    data={'formal_evidence':False,'snapshots':[{'outcomes':{'droop':witness,'ac_opf_witness':witness}}]*8}
    file=tmp_path/'diagnosis.json';file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='tolerances'): verify_diagnosis(file)


def test_pipeline_real_diagnosis_artifact_still_passes_recorded_contract():
    root=Path(__file__).resolve().parents[1]
    path=root/'exp/revision_access_41572/diagnostics/foundation_02_same_state_02/diagnostics.json'
    if not path.exists(): pytest.skip('local development evidence is not installed')
    verify_diagnosis(path)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),-1.,True,'0'])
def test_pipeline_rejects_nonfinite_or_invalid_residuals(tmp_path,bad):
    witness={'status':'solved','inputs_unchanged':True,'voltage_violation_max_pu':0.,
        'control_bound_violation_max_mvar':0.,'inverter_circle_violation_max_mva2':0.,'ac_balance_max_mva':bad}
    file=tmp_path/'diagnosis.json'
    file.write_text(json.dumps({'formal_evidence':False,'snapshots':[{'outcomes':{'droop':witness,'ac_opf_witness':witness}}]*8}))
    with pytest.raises(ValueError,match='tolerances'): verify_diagnosis(file)


@pytest.mark.parametrize('flag',[1,'true',[],None])
def test_pipeline_requires_boolean_unchanged_inputs(tmp_path,flag):
    witness={'status':'solved','inputs_unchanged':flag,'voltage_violation_max_pu':0.,
        'control_bound_violation_max_mvar':0.,'inverter_circle_violation_max_mva2':0.,'ac_balance_max_mva':0.}
    file=tmp_path/'diagnosis.json'
    file.write_text(json.dumps({'formal_evidence':False,'snapshots':[{'outcomes':{'droop':witness,'ac_opf_witness':witness}}]*8}))
    with pytest.raises(ValueError,match='tolerances'): verify_diagnosis(file)


def _stage_validation(attempt):
    path=pipeline.need_file(attempt/'result.json')
    result=pipeline.read(path)
    if result.get('verified') is not True: raise ValueError('invalid result')
    return result,[path]


def test_stage_completed_skip_requires_unchanged_input_and_artifact(tmp_path):
    ledger=pipeline.StageLedger(tmp_path,'identity');calls=[]
    def action(a):
        calls.append(a);pipeline.write_new(a/'result.json',{'verified':True,'result':3})
    first=ledger.run('selection',{'source':'fixed'},action,_stage_validation)
    assert ledger.run('selection',{'source':'fixed'},action,_stage_validation)==first and len(calls)==1
    with pytest.raises(ValueError,match='inputs/source changed'):
        ledger.run('selection',{'source':'other'},action,_stage_validation)
    (calls[0]/'result.json').write_text('{"verified":true,"result":4}',encoding='utf-8')
    with pytest.raises(ValueError,match='artifact hash changed'):
        ledger.run('selection',{'source':'fixed'},action,_stage_validation)


def test_crash_after_complete_output_is_adopted_without_second_selection(tmp_path,monkeypatch):
    ledger=pipeline.StageLedger(tmp_path,'identity');calls=[]
    class Crash(BaseException): pass
    original=ledger._complete
    monkeypatch.setattr(ledger,'_complete',lambda *args:(_ for _ in ()).throw(Crash()))
    def action(a): calls.append(a);pipeline.write_new(a/'result.json',{'verified':True})
    with pytest.raises(Crash): ledger.run('selection',{},action,_stage_validation)
    monkeypatch.setattr(ledger,'_complete',original)
    assert ledger.run('selection',{},action,_stage_validation)=={'verified':True}
    assert len(calls)==1 and (calls[0]/'completed.json').exists()


def test_failed_report_gets_fresh_attempt_and_keeps_old_failure(tmp_path):
    ledger=pipeline.StageLedger(tmp_path,'identity');calls=[]
    def action(a):
        calls.append(a)
        if len(calls)==1:
            pipeline.write_new(a/'partial.json',{'partial':True});raise RuntimeError('stopped')
        pipeline.write_new(a/'result.json',{'verified':True})
    with pytest.raises(RuntimeError): ledger.run('report',{},action,_stage_validation)
    prior=(calls[0]/'failed.json').read_bytes()
    ledger.run('report',{},action,_stage_validation)
    assert calls[0]!=calls[1] and (calls[0]/'failed.json').read_bytes()==prior
    ledger.run('report',{},lambda a:pytest.fail('completed report must not be regenerated'),_stage_validation)


def test_existing_failure_can_require_explicit_retry(tmp_path):
    ledger=pipeline.StageLedger(tmp_path,'identity')
    with pytest.raises(RuntimeError):
        ledger.run('job',{},lambda a:(_ for _ in ()).throw(RuntimeError('first')),_stage_validation)
    with pytest.raises(RuntimeError,match='explicit retry'):
        ledger.run('job',{},lambda a:pytest.fail('must not retry'),_stage_validation,retry=False)


def test_stage_does_not_run_a_second_child_after_parent_crash(tmp_path,monkeypatch):
    ledger=pipeline.StageLedger(tmp_path,'identity')
    class Crash(BaseException): pass
    with pytest.raises(Crash):
        ledger.run('job',{},lambda a:(_ for _ in ()).throw(Crash()),_stage_validation)
    monkeypatch.setattr(pipeline,'child_still_running',lambda a:True)
    with pytest.raises(RuntimeError,match='remains active'):
        ledger.run('job',{},lambda a:pytest.fail('duplicate child'),_stage_validation)


def test_multiple_successes_do_not_allow_favourable_selection(tmp_path):
    import shutil
    ledger=pipeline.StageLedger(tmp_path,'identity')
    ledger.run('bench',{},lambda a:pipeline.write_new(a/'result.json',{'verified':True}),_stage_validation)
    folder=tmp_path/'stages/bench'
    shutil.copytree(folder/'a0001',folder/'a0002')
    with pytest.raises(ValueError,match='multiple completed'):
        ledger.run('bench',{},lambda a:pytest.fail('must stop'),_stage_validation)


def test_fresh_run_command_logs_can_resume_as_new_attempt(tmp_path):
    import sys
    pipeline.run_command([sys.executable,'-c','print("real short subprocess")'],tmp_path/'command.log')
    assert pipeline.read(tmp_path/'command.exit.json')['exit_code']==0
    assert 'real short subprocess' in (tmp_path/'command.log').read_text()
    assert not pipeline.child_still_running(tmp_path)
    with pytest.raises(FileExistsError):
        pipeline.run_command([sys.executable,'-c','print("must not overwrite")'],tmp_path/'command.log')


@pytest.fixture
def workflow_workspace(tmp_path,monkeypatch):
    import zipfile
    from experiments import revision_logging_migration as migration
    root=tmp_path/'repository';(root/'models').mkdir(parents=True)
    model=root/'models/model.py';model.write_text('# synthetic fixture\n'+migration.OLD_BLOCK+'\n',encoding='utf-8')
    (root/'aux.py').write_text('# immutable auxiliary')
    base=root/'exp/revision_access_41572'
    pipeline.write_new(base/'protocol/dev_fair_q_01/fair_q_registry.json',{'fixture':'registry'})
    archive_dir=base/'protocol/source_snapshots';archive_dir.mkdir(parents=True)
    source={'models/model.py':pipeline.digest(model)}
    archive=archive_dir/'development.zip'
    with zipfile.ZipFile(archive,'w') as handle:
        handle.writestr('models/model.py',model.read_bytes())
    pipeline.write_new(archive_dir/'development_01.json',{'source':{'source_sha256':source},
        'archive':str(archive),'sha256':pipeline.digest(archive)})
    monkeypatch.setattr(pipeline,'ROOT',root);monkeypatch.setattr(pipeline,'BASE',base)
    monkeypatch.setattr(pipeline,'AUXILIARY_FILES',('aux.py',))
    monkeypatch.setattr(migration,'ROOT',root)
    monkeypatch.setattr(migration,'_current_source',lambda:{'models/model.py':pipeline.digest(model)})
    monkeypatch.setattr(migration,'_assert_quiescent_development',lambda:None)
    return migration,root,base,source


def test_original_workflow_resumes_after_migration_but_new_workflow_cannot_claim_it(workflow_workspace):
    migration,root,base,source=workflow_workspace
    folder=base/'workflow'
    identity=pipeline.open_workflow(folder,training_workers=2,evaluation_workers=3)
    migration.apply_after_development(expected_source_sha256=source,evidence_path=folder/'logging_migration.json')
    assert pipeline.open_workflow(folder,resume=True)==identity
    assert pipeline.read(folder/'workflow_manifest.json')==identity
    with pytest.raises(FileExistsError,match='--resume'): pipeline.open_workflow(folder)
    with pytest.raises(ValueError,match='identity changed'):
        pipeline.open_workflow(folder,resume=True,training_workers=4)
    with pytest.raises(ValueError,match='without the original migration intent'):
        pipeline.open_workflow(base/'other_workflow')


def test_workflow_resume_refuses_auxiliary_or_archive_mutation(workflow_workspace):
    migration,root,base,source=workflow_workspace
    folder=base/'workflow';pipeline.open_workflow(folder)
    (root/'aux.py').write_text('# changed source')
    with pytest.raises(ValueError,match='identity changed'): pipeline.open_workflow(folder,resume=True)


@pytest.mark.parametrize('kind',['training','evaluation'])
def test_existing_child_queue_gets_resume_and_only_explicit_evaluation_retry(tmp_path,kind):
    output=tmp_path/'queue'
    first=pipeline.queue_command(kind,tmp_path/'plan.json',output,2,training_root=tmp_path/'train',retry_failed=True)
    assert '--resume' not in first and '--retry-failed' not in first
    output.mkdir()
    ordinary=pipeline.queue_command(kind,tmp_path/'plan.json',output,2,training_root=tmp_path/'train')
    assert '--resume' in ordinary and '--retry-failed' not in ordinary
    retry=pipeline.queue_command(kind,tmp_path/'plan.json',output,2,training_root=tmp_path/'train',retry_failed=True)
    assert ('--retry-failed' in retry)==(kind=='evaluation')


def test_archive_interruption_uses_fresh_attempt_and_preserves_partial_archive(tmp_path):
    import zipfile
    root=tmp_path/'source.py';root.write_text('source')
    expected={'source.py':pipeline.digest(root)}
    ledger=pipeline.StageLedger(tmp_path/'workflow','identity');calls=[]
    class Crash(BaseException): pass
    def action(a):
        calls.append(a)
        with zipfile.ZipFile(a/'source.zip','x') as archive:
            archive.writestr('source.py',root.read_bytes())
            if len(calls)==1: raise Crash()
            archive.writestr('auxiliary_manifest.json',json.dumps(expected))
    def validate(a):
        path=a/'source.zip';pipeline.verify_source_archive(path,expected,'auxiliary_manifest.json')
        return {'sha256':pipeline.digest(path)},[path]
    with pytest.raises(Crash): ledger.run('archive',{},action,validate)
    first=(calls[0]/'source.zip').read_bytes()
    ledger.run('archive',{},action,validate)
    assert len(calls)==2 and (calls[0]/'source.zip').read_bytes()==first
    ledger.run('archive',{},lambda a:pytest.fail('archive must be reused'),validate)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),True,'1',None])
def test_gate_numeric_validator_rejects_nonfinite_walltimes(value):
    assert not pipeline.finite_number(value)
