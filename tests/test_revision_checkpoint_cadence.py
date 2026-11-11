"""Checkpoint I/O scheduling preserves learning and identifies replayed log tails."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from experiments.revision_checkpointing import (CHECKPOINT_POLICY, filter_committed_history,
    load_training_checkpoint, record_history_boundary, save_training_checkpoint_if_due)
from test_revision_solver_resume import assert_nested_equal, trainer_for


def trainer():
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    value = trainer_for()
    return value


def one_step_trainer(solver):
    value = trainer_for(solver)
    step = value.env.step
    def failed_step(action):
        reward, _, info = step(action)
        return reward - 200, True, {**info, 'destroy': 1.}
    value.env.step = failed_step
    return value


def episode(value, progress, attempt, rows):
    start = value.steps
    stat = {}
    value.behaviour_net.train_process(stat, value)
    progress['episode'] += 1
    record_history_boundary(progress, attempt=attempt, episode=progress['episode'], steps=value.steps)
    rows.append({'attempt': attempt, 'episode': progress['episode'], 'environment_steps': value.steps,
                 'episode_steps': value.steps-start, **stat})


@pytest.mark.parametrize('solver', ['maddpg', 'facmaddpg', 'mappo', 'matd3', 'matd3_reference'])
def test_periodic_save_and_crash_replay_preserve_complete_learning_state(tmp_path, solver):
    torch.set_num_threads(1)
    def run(name, interval, crash=False):
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        value = one_step_trainer(solver)
        path = tmp_path / (name+'.pt')
        progress = {'episode': 0}
        record_history_boundary(progress, attempt=1, episode=0, steps=0)
        assert save_training_checkpoint_if_due(path, value, progress=progress)
        rows = []
        attempt, saves = 1, 1
        for _ in range(7 if crash else 9):
            episode(value, progress, attempt, rows)
            saves += save_training_checkpoint_if_due(path, value, progress=progress,
                interval_steps=interval, force=value.steps == 9)
        if crash:
            assert value.steps == 7 and progress['checkpoint_steps'] == 4
            value = one_step_trainer(solver)
            progress = load_training_checkpoint(path, value)
            assert value.steps == progress['checkpoint_steps'] == 4
            attempt = 2
            record_history_boundary(progress, attempt=attempt, episode=progress['episode'], steps=value.steps)
            while value.steps < 9:
                episode(value, progress, attempt, rows)
                saves += save_training_checkpoint_if_due(path, value, progress=progress,
                    interval_steps=interval, force=value.steps == 9, attempt=attempt)
        selected = filter_committed_history(rows, progress['history_segments'])
        assert len(rows) == (12 if crash else 9)
        assert len(selected) == 9
        assert [r['attempt'] for r in selected] == ([1]*4+[2]*5 if crash else [1]*9)
        payload = torch.load(path, weights_only=False)
        assert payload['trainer']['steps'] == progress['checkpoint_steps'] == 9
        del payload['progress']  # wall/provenance lineage differs, learner state must not.
        return payload, selected, saves
    each, each_rows, each_saves = run('each_episode', 1)
    periodic, periodic_rows, periodic_saves = run('periodic', 4)
    resumed, resumed_rows, _ = run('resumed', 4, crash=True)
    assert each_saves == 10 and periodic_saves == 4
    assert_nested_equal(each, periodic)
    assert_nested_equal(each, resumed)
    for expected, observed in zip(each_rows, resumed_rows):
        assert_nested_equal({k:v for k,v in expected.items() if k != 'attempt'},
                            {k:v for k,v in observed.items() if k != 'attempt'})


def test_initial_forced_gates_and_episode_boundary_overshoot(tmp_path):
    value, path = trainer(), tmp_path/'state.pt'
    progress = {'episode': 0}
    assert save_training_checkpoint_if_due(path, value, progress=progress)
    assert progress['checkpoint_steps'] == 0
    value.steps, progress['episode'] = 479, 479
    assert not save_training_checkpoint_if_due(path, value, progress=progress)
    value.steps, progress['episode'] = 959, 480  # next episode consumes all 480 steps
    assert save_training_checkpoint_if_due(path, value, progress=progress)
    assert CHECKPOINT_POLICY['maximum_crash_replay_steps'] == 959
    value.steps, progress['episode'] = 960, 481
    assert not save_training_checkpoint_if_due(path, value, progress=progress)
    assert save_training_checkpoint_if_due(path, value, progress=progress, force=True)
    assert progress['checkpoint_steps'] == 960
    value.at_episode_boundary = False
    before = deepcopy(progress)
    with pytest.raises(ValueError, match='episode boundary'):
        save_training_checkpoint_if_due(path, value, progress=progress, force=True)
    assert progress == before


def test_history_repeated_crash_and_initial_checkpoint_exclude_superseded_tails():
    progress = {}
    record_history_boundary(progress, attempt=1, episode=0, steps=0)
    initial = deepcopy(progress)
    record_history_boundary(progress, attempt=1, episode=1, steps=1)
    committed = deepcopy(progress)
    rows = [{'attempt':1,'episode':n,'environment_steps':n,'episode_steps':1} for n in (1,2,3)]
    record_history_boundary(committed, attempt=2, episode=1, steps=1)
    record_history_boundary(committed, attempt=2, episode=2, steps=2)
    rows += [{'attempt':2,'episode':2,'environment_steps':2,'episode_steps':1}]
    # Attempt 2 never committed: reload the attempt 1 checkpoint and use attempt 3.
    progress['history_segments'][-1].update(end_episode=1, end_steps=1)
    record_history_boundary(progress, attempt=3, episode=1, steps=1)
    record_history_boundary(progress, attempt=3, episode=2, steps=2)
    rows += [{'attempt':3,'episode':2,'environment_steps':2,'episode_steps':1}]
    assert [r['attempt'] for r in filter_committed_history(rows, progress['history_segments'])] == [1,3]
    assert filter_committed_history(rows, initial['history_segments']) == []
    with pytest.raises(ValueError, match='missing or duplicate'):
        filter_committed_history(rows+[rows[-1]], progress['history_segments'])
    with pytest.raises(ValueError, match='missing or duplicate'):
        filter_committed_history(rows[1:], progress['history_segments'])
