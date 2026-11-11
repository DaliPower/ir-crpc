"""Resume all six mechanisms, checking next episode, replay, optimizers and RNG."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import random
import runpy
import numpy as np
import pytest
import torch
from models.matd3_round2_ablation import MATD3Round2Ablation,VARIANTS
from utilities.trainer import PGTrainer
from experiments.revision_checkpointing import save_training_checkpoint,load_training_checkpoint

H=runpy.run_path(str(Path(__file__).with_name('test_revision_solver_resume.py')))


@pytest.mark.parametrize('variant',list(VARIANTS))
def test_resume_matches_next_episode_and_all_random_streams(tmp_path,variant):
    def trainer():
        args=H['arguments']('matd3_reference',round2_variant=variant,policy_delay=VARIANTS[variant]['policy_delay'])
        return PGTrainer(args,MATD3Round2Ablation,H['TinyEnvironment'](),SimpleNamespace(add_scalar=lambda *args:None))
    torch.manual_seed(101);random.seed(103);np.random.seed(107)
    original=trainer();original.behaviour_net.train_process({},original)
    path=save_training_checkpoint(tmp_path/'state.pt',original,progress={'episode':1},metadata={'variant':variant})
    original.behaviour_net.train_process({},original)
    expected=deepcopy(original.behaviour_net.state_dict())
    draws=(random.random(),np.random.random(),torch.rand(4))
    restored=trainer();load_training_checkpoint(path,restored,expected_metadata={'variant':variant})
    restored.behaviour_net.train_process({},restored)
    eq=H['assert_nested_equal']
    eq(expected,restored.behaviour_net.state_dict())
    eq(original.policy_optimizer.state_dict(),restored.policy_optimizer.state_dict())
    eq(original.value_optimizer.state_dict(),restored.value_optimizer.state_dict())
    eq(original.replay_buffer.buffer,restored.replay_buffer.buffer)
    eq(draws,(random.random(),np.random.random(),torch.rand(4)))
    for name in ('exploration_generator','target_noise_generator'):
        eq(getattr(original.behaviour_net,name).get_state(),getattr(restored.behaviour_net,name).get_state())
    assert restored.steps==original.steps==6
    assert restored.update_counts==original.update_counts
    assert restored.env.exposure==original.env.exposure
