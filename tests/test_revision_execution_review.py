from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.revision_provenance import (derive_training_streams, training_source_manifest,
                                             validate_pool_binding)
from experiments.run_revision_training import append_json
from experiments.revision_run_plan import build_plan, freeze_plan, validate_frozen_plan, canonical_hash
from experiments.revision_checkpointing import save_training_checkpoint, load_training_checkpoint
from test_revision_run_plan import pool_dir, gate
from test_revision_solver_resume import trainer_for, assert_nested_equal


def test_all_training_streams_have_independent_variant_and_network_domains():
    variants = ('reference_B0', 'reference_B1', 'reference_B2', 'reference_B2_static',
                'derived_B2', 'maddpg_B2', 'mappo_B2', 'facmaddpg_B2')
    domains = [derive_training_streams(variant, scenario, 0) for variant in variants
               for scenario in ('case33_3min_final', 'case141_3min_final')]
    flat = [value for domain in domains for value in domain['streams'].values()]
    assert len(set(flat)) == len(flat)
    assert domains[0] == derive_training_streams('reference_B0', 'case33_3min_final', 0)
    assert domains[0] != derive_training_streams('reference_B0', 'case33_3min_final', 1)


def test_training_source_manifest_includes_real_learning_dependencies_and_excludes_reporting():
    files = training_source_manifest()['source_sha256']
    for name in ('learning_algorithms/ppo.py', 'critics/qmix.py',
                 'experiments/revision_provenance.py', 'experiments/revision_checkpointing.py',
                 'experiments/fixed_test_protocol.py'):
        assert name in files
    assert 'experiments/revision_statistics.py' not in files
    assert 'experiments/dispatch_revision_training.py' not in files


def test_torn_jsonl_record_does_not_destroy_next_valid_record(tmp_path):
    path = tmp_path / 'trace.jsonl'
    path.write_text('{"attempt": 1, "part', encoding='utf-8')
    append_json(path, {'attempt': 2, 'status': 'resumed'})
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == '{"attempt": 1, "part'
    assert json.loads(lines[1]) == {'attempt': 2, 'status': 'resumed'}


def test_best_selection_is_restored_from_atomic_training_state(tmp_path):
    trainer = trainer_for()
    trainer.checkpoint_artifacts = {'best_model': {'episode': 4,
        'model_state_dict': deepcopy(trainer.behaviour_net.state_dict()),
        'validation_metrics': {'vrate': 0.1}}}
    expected = deepcopy(trainer.checkpoint_artifacts)
    path = save_training_checkpoint(tmp_path / 'training_state.pt', trainer,
                                    progress={'episode': 5, 'validation_history': [4]})
    # Simulate an uncommitted newer validation result overwriting the external
    # best file before a crash. The authoritative training checkpoint stays old.
    trainer.checkpoint_artifacts['best_model']['episode'] = 6
    torch.save(trainer.checkpoint_artifacts['best_model'], tmp_path / 'best_model.pt')
    restored = trainer_for()
    progress = load_training_checkpoint(path, restored)
    assert progress['episode'] == 5
    assert_nested_equal(expected, restored.checkpoint_artifacts)


@pytest.mark.parametrize('kwargs', [dict(steps=1), dict(validation_days=[1000]),
    dict(seeds=list(range(1, 11))), dict(seeds=[-1]+list(range(9)))])
def test_formal_protocol_cannot_silently_reduce_predeclared_budget(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_plan(pool_dir(tmp_path), **kwargs)


def test_resigned_plan_still_requires_full_formal_matrix(tmp_path):
    plan = freeze_plan(build_plan(pool_dir(tmp_path)), gate(tmp_path), source={'source_sha256': {}})
    plan['runs'].pop()
    plan['plan_sha256'] = canonical_hash({key: value for key, value in plan.items() if key != 'plan_sha256'})
    with pytest.raises(ValueError, match='160'):
        validate_frozen_plan(plan, verify_source=False)


def test_gate_cannot_pass_using_only_a_generic_file_and_true_flags(tmp_path):
    evidence = gate(tmp_path)
    evidence['development_runs'] = []
    with pytest.raises(ValueError, match='all five'):
        freeze_plan(build_plan(pool_dir(tmp_path)), evidence, source={'source_sha256': {}})


def test_pool_binding_rejects_source_data_mutation(tmp_path):
    pools = pool_dir(tmp_path)
    pool = pools / 'case33_3min_final_pool.json'
    validate_pool_binding(pool)
    (pools / 'case33_3min_final' / 'model.p').write_text('changed')
    with pytest.raises(ValueError, match='source data changed'):
        validate_pool_binding(pool)
