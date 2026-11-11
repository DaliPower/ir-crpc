"""Auditable revision training with exact interaction budgets and resume.

Development runs never sample or evaluate final-test dates. Formal training requires a frozen
plan and checks its source, pool and gate hashes before starting or resuming.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_revision_pilot import (environment_config,
    finite_json, json_write, file_hash)
from experiments.revision_provenance import (training_source_manifest, derive_training_streams,
    validate_pool_binding, data_hashes)
from experiments.revision_run_plan import canonical_hash, validate_frozen_plan
from experiments.revision_scenario_pool import ScenarioSampler


def append_json(path, value):
    path = Path(path)
    # Preserve a torn last record as its own line instead of concatenating the
    # next valid JSON object onto it. Readers can explicitly report that fragment.
    needs_newline = path.exists() and path.stat().st_size > 0
    if needs_newline:
        with path.open('rb') as previous:
            previous.seek(-1, os.SEEK_END)
            needs_newline = previous.read(1) != b'\n'
    with path.open('a', encoding='utf-8') as handle:
        if needs_newline:
            handle.write('\n')
        handle.write(json.dumps(finite_json(value), ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def checked_output(path):
    path = Path(path).resolve()
    boundary = (ROOT / 'exp/revision_access_41572').resolve()
    if boundary not in path.parents:
        raise ValueError('output must stay in the revision evidence tree')
    return path


def run_training(plan, run, output, *, resume=False, cpu=True, stop_after_steps=None):
    import torch
    from tensorboardX import SummaryWriter
    from experiments.run_q1_ml_compare import load_alg_config, make_args, make_env
    from experiments.revision_training_environment import ScenarioTrainingEnv
    from experiments.revision_checkpointing import (save_training_checkpoint_if_due, load_training_checkpoint,
        preserve_training_random_state, CHECKPOINT_POLICY, record_history_boundary)
    from experiments.fixed_test_protocol import evaluate_policy_fixed_days, write_fixed_test_artifacts
    from experiments.risk_aware_checkpointing import CheckpointManager
    from experiments.risk_aware_protocol import validation_rank
    from experiments.revision_validation_selection import (validation_metrics_from_summary,
                                                           ensure_finite_learning_state, VALIDATION_SELECTION_POLICY)
    from models.model_registry import Model
    from utilities.trainer import PGTrainer

    protocol = plan['protocol']
    formal = protocol['phase'] == 'formal'
    if formal:
        validate_frozen_plan(plan)
    elif protocol['phase'] != 'development':
        raise ValueError('invalid phase')
    if run not in plan['runs']:
        raise ValueError('run is not registered in plan')
    if formal and stop_after_steps is not None:
        raise ValueError('formal budgets cannot be reduced using development stop flags')
    output = checked_output(output)
    if output.exists() and not resume:
        raise FileExistsError('existing run requires explicit resume; results are never overwritten')
    if resume and not (output / 'training_state.pt').exists():
        raise FileNotFoundError('resume checkpoint is missing')
    if (output / 'training_result.json').exists():
        raise FileExistsError('completed training cannot be resumed or overwritten')
    output.mkdir(parents=True, exist_ok=True)
    source = training_source_manifest()
    rng_domain = derive_training_streams(run['variant'], run['scenario'], run['seed'])
    streams = rng_domain['streams']
    pool_path = Path(plan['pools'][run['scenario']]['path'])
    if file_hash(pool_path) != plan['pools'][run['scenario']]['sha256']:
        raise ValueError('scenario pool checksum mismatch')
    pool = validate_pool_binding(pool_path)
    sampler_seed = streams['scenario_sampler']
    sampler = ScenarioSampler(pool['rows'], run['sampling'], sampler_seed,
                              mixture=protocol['risk_mixture'])
    train_cfg = environment_config(run['scenario'], run['seed'], phase='training',
                                   points=480, start_hour=0)
    val_cfg = environment_config(run['scenario'], run['seed'], phase='validation',
                                 points=480, start_hour=0)
    train_cfg['seed'] = streams['training_environment']
    val_cfg['seed'] = streams['validation_environment']
    for config in (train_cfg, val_cfg):
        config.update(protocol.get('reward_by_scenario', {}).get(run['scenario'], protocol['reward']))
        config.update(action_scale=protocol['action_scales'][run['scenario']],
                      nighttime_reactive_power=protocol['nighttime_reactive_power'],
                      action_mapping=run['mapping'])
    # A failed chosen scenario is never silently replaced with an easier one.
    train_cfg['max_reset_attempts'] = 1
    metadata = {'plan_sha256': plan.get('plan_sha256', canonical_hash(plan)),
                'run': run, 'source_sha256': source['source_sha256'],
                'pool_sha256': file_hash(pool_path),
                'data_sha256': data_hashes(train_cfg['data_path']),
                'validation_selection_policy': VALIDATION_SELECTION_POLICY,
                'checkpoint_policy': CHECKPOINT_POLICY,
                'rng_domain': rng_domain,
                'runtime': {'device': 'cpu' if cpu or not torch.cuda.is_available() else 'cuda',
                            'gpu': None if cpu or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
                            'python': sys.version, 'torch': str(torch.__version__),
                            'numpy': np.__version__,
                            'pandapower': importlib.metadata.version('pandapower')}}
    if metadata['data_sha256'] != pool['data_sha256']:
        raise ValueError('training data differs from bound screening source')
    started = time.perf_counter()
    resources = []
    attempt = 1
    if (output / 'attempt_history.jsonl').exists():
        attempt += sum(1 for line in (output / 'attempt_history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
    append_json(output / 'attempt_history.jsonl', {'attempt': attempt, 'resume': resume})
    try:
        torch.set_num_threads(1)
        random.seed(streams['initialization'])
        np.random.seed(streams['initialization'])
        torch.manual_seed(streams['initialization'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(streams['initialization'])
        with contextlib.redirect_stdout(io.StringIO()):
            env = ScenarioTrainingEnv(train_cfg, sampler)
            val_env = make_env(val_cfg)
        resources += [env, val_env]
        if ('s_max_mva' not in pool or
                not np.allclose(env.s_max, pool['s_max_mva'], rtol=0, atol=1e-12)):
            raise ValueError('training inverter ratings differ from screened fixed ratings')
        algorithm = load_alg_config(run['solver'], train_cfg)
        # Validation is controlled here; old runner frequency is disabled.
        args = make_args(train_cfg, algorithm, env, {
            'max_steps': 480, 'train_episodes_num': protocol['training_steps'],
            'eval_freq': protocol['training_steps'] + 1, 'seed': run['seed'],
            'cuda': not cpu and torch.cuda.is_available(),
            'replay_seed': streams['replay'], 'exploration_seed': streams['exploration'],
            'target_noise_seed': streams['target_noise']})
        logger = SummaryWriter(str(output / 'tensorboard'))
        resources.append(logger)
        trainer = PGTrainer(args, Model[run['solver']], env, logger)
        torch.manual_seed(streams['learning'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(streams['learning'])
        manager = CheckpointManager(output)
        progress = {'episode': 0, 'next_validation_step': protocol['validation_every_steps'],
                    'validation_history': [], 'wall_seconds': 0., 'failed_episodes': 0}
        if resume:
            progress = load_training_checkpoint(output / 'training_state.pt', trainer,
                                                 expected_metadata=metadata)
            if (progress['checkpoint_steps'] != trainer.steps or
                    progress['checkpoint_episode'] != progress['episode']):
                raise ValueError('checkpoint progress boundary is inconsistent')
            manager._history = progress['validation_history']
            ranks = [validation_rank(r['validation_metrics'], r['episode'])
                     for r in manager._history]
            manager._best_rank = min(ranks) if ranks else None
            if manager._best_rank is not None:
                selected = trainer.checkpoint_artifacts.get('best_model')
                if selected is None:
                    raise ValueError('resume checkpoint lacks authoritative selected model')
                manager._write_checkpoint(manager.best_path, selected)
            manager._write_history()
            append_json(output / 'resume_history.jsonl', {
                'attempt': attempt, 'episode': progress['episode'], 'steps': trainer.steps,
                'checkpoint_attempt': progress['checkpoint_attempt'],
                'checkpoint_sha256': file_hash(output / 'training_state.pt')})
        else:
            json_write(output / 'manifest.json', {
                **metadata, 'source': source, 'seed_streams': streams,
                'outer_replicate_seed': run['seed'],
                'independent_training_rng_domain': rng_domain,
                'scenario_sampler_seed': sampler_seed, 'protocol': protocol,
                'train_environment': train_cfg, 'validation_environment': val_cfg,
                'historical_test_exposure_disclosed': True,
                'formal_test_dates_not_sampled_or_evaluated_during_training': True})
            json_write(output / 'effective_configuration.json', {
                'args': args._asdict(), 'environment': env.environment_metadata(),
                'torch_version': torch.__version__, 'device': str(trainer.device)})
            with (output / 'inverter_parameters.csv').open('w', encoding='utf-8-sig', newline='') as h:
                writer = csv.DictWriter(h, fieldnames=('agent_id','bus_id','zone','s_max_mva',
                    'synthetic_reference_peak_mw','action_scale','nighttime_reactive_power'))
                writer.writeheader()
                for i, (_, row) in enumerate(env.base_powergrid.sgen.iterrows()):
                    writer.writerow({'agent_id':i,'bus_id':row['bus'],'zone':row.get('name',''),
                        's_max_mva':float(env.s_max[i]),'synthetic_reference_peak_mw':float(env.p_max[i]),
                        'action_scale':train_cfg['action_scale'],
                        'nighttime_reactive_power':train_cfg['nighttime_reactive_power']})
        record_history_boundary(progress, attempt=attempt, episode=progress['episode'], steps=trainer.steps)
        if not resume:
            save_training_checkpoint_if_due(output / 'training_state.pt', trainer,
                progress=progress, metadata=metadata, force=True, attempt=attempt)
        base_wall = progress['wall_seconds']
        while trainer.steps < protocol['training_steps']:
            before_steps = trainer.steps
            before_failures = len(getattr(env, 'powerflow_failures', []))
            stat, tick = {}, time.perf_counter()
            trainer.behaviour_net.train()
            remaining = protocol['training_steps'] - trainer.steps
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.behaviour_net.train_process(stat, trainer, step_limit=min(480, remaining))
            if trainer.steps <= before_steps:
                raise RuntimeError('training episode made no progress')
            trainer.logging(stat)
            progress['episode'] += 1
            record_history_boundary(progress, attempt=attempt, episode=progress['episode'], steps=trainer.steps)
            reason = trainer.last_episode_reason
            if 'failure' in reason or float(stat.get('mean_train_destroy', 0)) > 0:
                progress['failed_episodes'] += 1
            record = {'attempt': attempt, 'episode': progress['episode'], 'environment_steps': trainer.steps,
                      'episode_steps': trainer.steps - before_steps, 'reason': reason,
                      'scenario': env.current_scenario, 'wall_seconds': time.perf_counter()-tick,
                      'update_counts': dict(trainer.update_counts), **stat}
            if len(getattr(env, 'powerflow_failures', [])) > before_failures:
                record['powerflow_failures'] = env.powerflow_failures[before_failures:]
            append_json(output / 'training_history.jsonl', record)
            needs_eval = (trainer.steps >= progress['next_validation_step'] or
                          trainer.steps == protocol['training_steps'])
            if needs_eval:
                learner_finite = ensure_finite_learning_state(trainer)
                validation_failure_start = len(getattr(val_env, 'powerflow_failures', []))
                with preserve_training_random_state(trainer), contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate_policy_fixed_days(env=val_env,
                        behaviour_net=trainer.behaviour_net, args=args,
                        days=protocol['validation_days'], max_steps=480,
                        start_hour=0, _strict_shape=False)
                metrics = validation_metrics_from_summary(result.summary, learner_finite=learner_finite)
                if 'validation_support' in metrics:
                    json_write(output / f'validation_unobserved_{trainer.steps}_attempt{attempt}.json', finite_json({
                        'summary': result.summary, 'selection_metrics': metrics,
                        'physical_values_are_unobserved': True,
                        'powerflow_failures': getattr(val_env, 'powerflow_failures', [])[validation_failure_start:]}))
                selected = manager.consider(trainer.behaviour_net.state_dict(), metrics, progress['episode'])
                if selected:
                    trainer.checkpoint_artifacts = {'best_model': torch.load(
                        manager.best_path, map_location='cpu', weights_only=False)}
                progress['validation_history'] = manager._history
                append_json(output / 'validation_summaries.jsonl',
                            {'attempt':attempt, 'steps':trainer.steps, 'episode':progress['episode'], **result.summary})
                while progress['next_validation_step'] <= trainer.steps:
                    progress['next_validation_step'] += protocol['validation_every_steps']
            progress['wall_seconds'] = base_wall + time.perf_counter() - started
            pause_requested = stop_after_steps is not None and trainer.steps >= stop_after_steps
            save_training_checkpoint_if_due(output / 'training_state.pt', trainer,
                progress=progress, metadata=metadata, force=needs_eval or pause_requested,
                interval_steps=CHECKPOINT_POLICY['completed_step_interval'], attempt=attempt)
            json_write(output / 'progress.json', {
                **progress, 'status':'training', 'steps':trainer.steps,
                'unsaved_completed_steps':trainer.steps - progress['checkpoint_steps'],
                'step_budget':protocol['training_steps'], 'run_id':run['run_id']})
            print(json.dumps({'run_id':run['run_id'], 'episode':progress['episode'],
                              'steps':trainer.steps,'budget':protocol['training_steps'],
                              'failed_episodes':progress['failed_episodes']}), flush=True)
            if pause_requested and trainer.steps < protocol['training_steps']:
                return {'status':'development_paused', 'steps':trainer.steps, 'output':str(output)}
        manager.save_final(trainer.behaviour_net.state_dict(), progress['episode'])
        manager.finalize()
        payload = torch.load(output/'best_model.pt', map_location=trainer.device, weights_only=False)
        trainer.behaviour_net.load_state_dict(payload['model_state_dict'])
        with preserve_training_random_state(trainer), contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_policy_fixed_days(env=val_env, behaviour_net=trainer.behaviour_net,
                args=args, days=protocol['validation_days'], max_steps=480,start_hour=0,_strict_shape=False)
        validation_output = output/f'validation_best_attempt_{attempt}'
        write_fixed_test_artifacts(result, validation_output, 'fixed_test')
        if training_source_manifest()['source_sha256'] != source['source_sha256']:
            raise RuntimeError('source changed during run; evidence is not a frozen experiment')
        if data_hashes(train_cfg['data_path']) != metadata['data_sha256']:
            raise RuntimeError('source data changed during training')
        json_write(output/'scenario_exposure.json', {
            'pool_sha256':sampler.pool_sha256, 'probabilities':sampler.probability.tolist(),
            'exposure_counts':sampler.exposures.tolist(), 'rows':sampler.rows})
        report = {'status':'completed', 'phase':protocol['phase'], 'run':run,
            'environment_steps':trainer.steps, 'episodes':progress['episode'],
            'failed_episodes':progress['failed_episodes'], 'best_episode':payload['episode'],
            'update_counts':trainer.update_counts, 'validation':result.summary,
            'wall_seconds':base_wall+time.perf_counter()-started,
            'validation_artifacts':str(validation_output), 'completion_attempt':attempt,
            'checkpoint_steps':progress['checkpoint_steps'], 'checkpoint_episode':progress['checkpoint_episode'],
            'checkpoint_attempt':progress['checkpoint_attempt'], 'unsaved_completed_steps':0,
            'history_segments':progress['history_segments'],
            'history_recovery_rule':CHECKPOINT_POLICY['history_authority'],
            'weight_sha256':{name:file_hash(output/(name+'_model.pt')) for name in ('best','final')},
            'formal_test_evaluated':False, 'performance_claims_allowed':False}
        json_write(output/'training_result.json', finite_json(report))
        json_write(output/'progress.json', finite_json(report))
        return report
    except Exception as exc:
        append_json(output/'failure_history.jsonl', {'error_type':type(exc).__name__,
                    'message':str(exc),'traceback':traceback.format_exc(),
                    'last_powerflow_failure':getattr(locals().get('env'), 'last_powerflow_failure', None),
                    'wall_seconds':time.perf_counter()-started})
        raise
    finally:
        for resource in reversed(resources):
            if hasattr(resource, 'close'):
                resource.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--development-stop-after-steps', type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    matches = [r for r in plan['runs'] if r['run_id'] == args.run_id]
    if len(matches) != 1:
        raise ValueError('run ID must uniquely identify a planned run')
    result = run_training(plan,matches[0],args.output,resume=args.resume,cpu=not args.cuda,
                          stop_after_steps=args.development_stop_after_steps)
    print(json.dumps({'status':result['status'],'output':str(args.output)}),flush=True)


if __name__ == '__main__':
    main()
