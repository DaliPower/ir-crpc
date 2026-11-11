"""Both logging versions execute the same learning/RNG/replay updates."""
from copy import deepcopy
import inspect
import random
import textwrap
from types import MethodType
import numpy as np
import pytest
import torch

import models.model as base
from experiments.revision_checkpointing import save_training_checkpoint
from experiments.revision_logging_migration import OLD_BLOCK,NEW_BLOCK,patched_text
from tests.test_revision_solver_resume import trainer_for,TinyEnvironment,assert_nested_equal


class RewardInfoEnvironment(TinyEnvironment):
    def __init__(self,fail):
        super().__init__();self.fail=fail;self.actual_rewards=[]
    def step(self,action):
        value,done,info=super().step(action)
        failure=float(self.fail and self.count==3)
        reward=value-200*failure
        self.actual_rewards.append(reward)
        return reward,done,dict(info,reward=value,total_loss=-value,destroy=failure)


def method_versions():
    source=inspect.getsource(base.Model.train_process)
    if NEW_BLOCK in source:
        old=source.replace(NEW_BLOCK,OLD_BLOCK)
    else:
        old=source
    new=patched_text(old)
    methods=[]
    for version in (old,new):
        namespace={}
        exec(compile(textwrap.dedent(version),'<logging-equivalence-only>','exec'),base.__dict__,namespace)
        methods.append(namespace['train_process'])
    return methods


@pytest.mark.parametrize('solver',['matd3_reference','matd3','maddpg','mappo','facmaddpg'])
@pytest.mark.parametrize('fail',[False,True])
def test_logging_patch_preserves_learning_including_failure_penalty(tmp_path,solver,fail):
    versions=method_versions();states=[];stats=[]
    for index,method in enumerate(versions):
        random.seed(103);np.random.seed(107);torch.manual_seed(101)
        trainer=trainer_for(solver)
        trainer.env=RewardInfoEnvironment(fail)
        trainer.behaviour_net.train_process=MethodType(method,trainer.behaviour_net)
        stat={}
        for _ in range(2): trainer.behaviour_net.train_process(stat,trainer)
        path=save_training_checkpoint(tmp_path/f'v{index}.pt',trainer,progress={},metadata={})
        states.append(torch.load(path,map_location='cpu',weights_only=False));stats.append(stat)
        if index==1:
            assert stat['mean_train_reward']==pytest.approx(np.mean(trainer.env.actual_rewards[-3:]))
    assert_nested_equal(states[0],states[1])
    old,new=deepcopy(stats[0]),deepcopy(stats[1])
    old.pop('mean_train_reward');new.pop('mean_train_reward')
    assert_nested_equal(old,new)


def test_migration_refuses_unrecognized_or_reapplied_source():
    with pytest.raises(ValueError): patched_text('unknown source')
    with pytest.raises(ValueError): patched_text(NEW_BLOCK)


@pytest.fixture
def migration_workspace(tmp_path,monkeypatch):
    from experiments import revision_logging_migration as migration
    root=tmp_path/'repository';(root/'models').mkdir(parents=True)
    path=root/'models/model.py'
    path.write_text('# audited synthetic source fixture\n'+OLD_BLOCK+'\n',encoding='utf-8')
    monkeypatch.setattr(migration,'ROOT',root)
    monkeypatch.setattr(migration,'_current_source',lambda:{'models/model.py':migration._digest(path.read_bytes()),'unchanged.py':'fixed'})
    monkeypatch.setattr(migration,'_assert_quiescent_development',lambda:None)
    return migration,path,migration._current_source(),tmp_path/'workflow/migration.json'


@pytest.mark.parametrize('moment',['intent','backup','source','completion'])
def test_migration_recovers_each_atomic_boundary_without_double_patch(migration_workspace,monkeypatch,moment):
    migration,path,before,evidence=migration_workspace
    original=path.read_bytes()
    publish=migration._publish
    class Crash(BaseException): pass
    crashed=[]
    def interrupted(target,payload,**kwargs):
        publish(target,payload,**kwargs)
        name=str(target)
        hit=(moment=='intent' and name.endswith('.intent.json') or
             moment=='backup' and name.endswith('.model_before.py') or
             moment=='source' and kwargs.get('replace') or
             moment=='completion' and target==evidence)
        if hit and not crashed:
            crashed.append(True);raise Crash()
    monkeypatch.setattr(migration,'_publish',interrupted)
    with pytest.raises(Crash):
        migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)
    monkeypatch.setattr(migration,'_publish',publish)
    record=migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)
    assert path.read_bytes()==migration.patched_bytes(original)
    assert evidence.with_suffix('.model_before.py').read_bytes()==original
    evidence_bytes=evidence.read_bytes()
    assert migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)==record
    assert evidence.read_bytes()==evidence_bytes
    assert migration.inspect_migration(expected_source_sha256=before,evidence_path=evidence)['state']=='completed'


def test_migration_refuses_live_training_before_writing_intent(migration_workspace,monkeypatch):
    migration,path,before,evidence=migration_workspace
    def live(): raise ValueError('training consumer active')
    monkeypatch.setattr(migration,'_assert_quiescent_development',live)
    with pytest.raises(ValueError,match='active'):
        migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)
    assert not evidence.with_suffix('.intent.json').exists()
    assert migration._current_source()==before


@pytest.mark.parametrize('kind',['foreign_source','backup','record'])
def test_migration_rejects_changed_evidence_and_unaudited_source(migration_workspace,kind):
    migration,path,before,evidence=migration_workspace
    migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)
    if kind=='foreign_source': path.write_bytes(path.read_bytes()+b'# unauthorized change')
    elif kind=='backup':
        backup=evidence.with_suffix('.model_before.py');backup.write_bytes(backup.read_bytes()+b'# changed')
    else: evidence.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):
        migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)


def test_new_output_cannot_claim_already_patched_source_without_intent(migration_workspace):
    migration,path,before,evidence=migration_workspace
    path.write_bytes(migration.patched_bytes(path.read_bytes()))
    with pytest.raises(ValueError,match='without the original migration intent'):
        migration.apply_after_development(expected_source_sha256=before,evidence_path=evidence)


def test_patch_preserves_crlf_and_all_unchanged_bytes():
    from experiments.revision_logging_migration import patched_bytes
    original=('# header\n'+OLD_BLOCK+'\n# footer\n').replace('\n','\r\n').encode()
    expected=('# header\n'+NEW_BLOCK+'\n# footer\n').replace('\n','\r\n').encode()
    assert patched_bytes(original)==expected
