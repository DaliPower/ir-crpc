import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.revision_validation_selection import (METRICS, ensure_finite_learning_state,
    normalize_validation_metrics, validation_metrics_from_summary, validation_rank)


def all_failed():
    return dict(zip(METRICS,(1.,None,None)), planned_steps=1440,executed_steps=0,failed_steps=1440,
        planned_days=3,points_per_day=480,cr=0.,destroy=1.,
        metric_denominators={METRICS[0]:1440,METRICS[1]:0,METRICS[2]:0,'cr':1440,'destroy':1440})


def test_all_failed_is_explicit_worst_and_keeps_physical_nulls():
    summary=all_failed();before=copy.deepcopy(summary)
    missing=validation_metrics_from_summary(summary,learner_finite=True)
    assert summary==before
    assert missing[METRICS[1]] is None and missing[METRICS[2]] is None
    assert json.loads(json.dumps(missing,allow_nan=False))==missing
    for values in ((0.,0.,0.),(.999,1e9,1e9),(1.,1e300,1e300)):
        assert validation_rank(dict(zip(METRICS,values)),9999)<validation_rank(missing,0)
    assert validation_rank(missing,1)<validation_rank(missing,2)


def test_every_pair_of_finite_candidates_keeps_legacy_lexicographic_order():
    candidates=[(dict(zip(METRICS,values)),episode)
        for values in itertools.product((0.,.2,1.),(0.,.3),(0.,2.)) for episode in (1,2)]
    old=lambda item:(*(item[0][key] for key in METRICS),float(item[1]))
    assert sorted(candidates,key=old)==sorted(candidates,key=lambda item:validation_rank(*item))
    for metrics,_ in candidates: assert normalize_validation_metrics(metrics)==metrics


@pytest.mark.parametrize('change',[{'executed_steps':1,'failed_steps':1439}, {'failed_steps':1439},
    {'planned_days':2}, {'cr':.1}, {'destroy':0.}, {'average_voltage_deviation':float('inf')},
    {'percentage_of_v_out_of_control':float('nan')}, {'q_loss':.2}, {'planned_steps':1440.0}])
def test_unexpected_nonfinite_or_invalid_failure_support_still_stops(change):
    summary=all_failed();summary.update(change)
    with pytest.raises(ValueError): validation_metrics_from_summary(summary,learner_finite=True)


def test_failure_rank_requires_finite_learner_and_zero_physical_denominators():
    with pytest.raises(ValueError): validation_metrics_from_summary(all_failed())
    summary=all_failed();summary['metric_denominators'][METRICS[2]]=1
    with pytest.raises(ValueError): validation_metrics_from_summary(summary,learner_finite=True)
    summary=all_failed();summary[METRICS[1]]=float('nan');summary[METRICS[2]]=np.float64('nan')
    result=validation_metrics_from_summary(summary,learner_finite=True)
    assert result[METRICS[1]] is None and result[METRICS[2]] is None


@pytest.mark.parametrize('place',['weights','gradient','optimizer'])
def test_nonfinite_learning_state_is_not_relabelled_as_powerflow_failure(place):
    model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    trainer=SimpleNamespace(behaviour_net=model,optimizer=optimizer)
    assert ensure_finite_learning_state(trainer) is True
    if place=='weights':
        with torch.no_grad(): model.weight[0,0]=float('nan')
    elif place=='gradient': model.weight.grad[0,0]=float('inf')
    else: optimizer.state[model.weight]['exp_avg'][0,0]=float('nan')
    with pytest.raises(FloatingPointError): ensure_finite_learning_state(trainer)


def test_finite_guard_does_not_change_rng_parameters_or_optimizer():
    from tests.test_revision_solver_resume import assert_nested_equal
    model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    before=copy.deepcopy((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
    ensure_finite_learning_state(SimpleNamespace(behaviour_net=model,optimizer=optimizer))
    assert_nested_equal(before,(model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))


def test_manager_can_save_allfailed_checkpoint_and_restore_its_selection_order(tmp_path):
    from experiments.risk_aware_checkpointing import CheckpointManager
    failed=validation_metrics_from_summary(all_failed(),learner_finite=True)
    manager=CheckpointManager(tmp_path)
    assert manager.consider({'weight':torch.tensor([1.])},failed,1)
    assert not manager.consider({'weight':torch.tensor([2.])},failed,2)
    history=json.loads(manager.history_path.read_text())
    assert history[0]['validation_metrics'][METRICS[1]] is None
    assert 'NaN' not in manager.history_path.read_text()
    payload=torch.load(manager.finalize(),map_location='cpu',weights_only=False)
    assert payload['validation_metrics'][METRICS[2]] is None and payload['episode']==1
    resumed=CheckpointManager(tmp_path)
    resumed._history=history
    resumed._best_rank=min(validation_rank(row['validation_metrics'],row['episode']) for row in history)
    finite=dict(zip(METRICS,(1.,100.,100.)))
    assert resumed.consider({'weight':torch.tensor([3.])},finite,3)
    assert torch.load(resumed.finalize(),map_location='cpu',weights_only=False)['episode']==3


def test_complete_training_checkpoint_roundtrip_retains_unobserved_best_payload(tmp_path):
    from experiments.revision_checkpointing import save_training_checkpoint,load_training_checkpoint
    from tests.test_revision_solver_resume import trainer_for,assert_nested_equal
    from experiments.risk_aware_checkpointing import CheckpointManager
    trainer=trainer_for()
    trainer.behaviour_net.train_process({},trainer)
    manager=CheckpointManager(tmp_path/'selection')
    metrics=validation_metrics_from_summary(all_failed(),learner_finite=ensure_finite_learning_state(trainer))
    manager.consider(trainer.behaviour_net.state_dict(),metrics,1)
    trainer.checkpoint_artifacts={'best_model':torch.load(manager.best_path,map_location='cpu',weights_only=False)}
    checkpoint=save_training_checkpoint(tmp_path/'state.pt',trainer,progress={'validation_history':manager._history},metadata={})
    restored=trainer_for()
    progress=load_training_checkpoint(checkpoint,restored,expected_metadata={})
    assert_nested_equal(trainer.checkpoint_artifacts,restored.checkpoint_artifacts)
    assert progress['validation_history'][0]['validation_metrics'][METRICS[1]] is None
    assert validation_rank(progress['validation_history'][0]['validation_metrics'],1)==validation_rank(metrics,1)
