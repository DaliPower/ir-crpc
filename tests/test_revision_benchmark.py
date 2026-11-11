"""Timing contracts with synthetic inputs; no power flow or training is run.

The real benchmark loops run against one-element policies and a fake clock.
These fixtures test scheduling/provenance, not numerical performance.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import revision_benchmark as policy
from experiments import revision_traditional_benchmark as traditional


class SyntheticPowerflowFailure(Exception):
    pass


class Clock:
    def __init__(self):
        self.value = 0.

    def perf_counter(self):
        self.value += 1.
        return self.value

    def advance(self, seconds):
        self.value += seconds


def module(monkeypatch, name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, result)
    return result


def harness(monkeypatch, tmp_path, target, drift=None):
    """All writes stay in pytest's temporary tree; actual source stays read-only."""
    root = tmp_path / 'synthetic_root'
    monkeypatch.setattr(target, 'ROOT', root)
    clock = Clock()
    monkeypatch.setattr(target, 'time', clock)
    monkeypatch.setattr(target, 'measurement_context', lambda isolated: {
        'isolation_requested': isolated,
        'isolated_from_other_worktree_experiment_workers': True,
    })
    calls = {'source': 0, 'script': 0, 'helper': 0}

    def source_manifest():
        calls['source'] += 1
        sha = 'changed' if drift == 'source' and calls['source'] > 1 else 'frozen'
        return {'source_sha256': {'dependency.py': sha}}

    def file_hash(path):
        assert isinstance(path, Path), 'production file_hash accepts pathlib.Path, not str'
        key = ('script' if path.resolve() == Path(target.__file__).resolve()
               else 'helper' if path == root / 'experiments/revision_benchmark.py' else None)
        if key:
            calls[key] += 1
            return key + ('-changed' if drift == key and calls[key] > 1 else '-frozen')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(target, 'file_hash', file_hash)
    evaluator = module(monkeypatch, 'experiments.revision_evaluation',
                       evaluation_source_manifest=source_manifest)
    module(monkeypatch, 'pandapower.powerflow', LoadflowNotConverged=SyntheticPowerflowFailure)
    return SimpleNamespace(root=root, output=root / 'exp/revision_access_41572/timing',
                           clock=clock, evaluator=evaluator, calls=calls)


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def init_hidden(self):
        return torch.zeros(1, 1, 1)

    def forward(self, tensor, hidden):
        self.seen.append(int(tensor.item()))
        return tensor, None, hidden


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_dicts = torch.nn.ModuleList([TinyActor()])
        self.seen = []

    def get_actions(self, tensor, *, last_hid, **kwargs):
        self.seen.append(int(tensor.item()))
        return torch.zeros_like(tensor), None, None, None, last_hid + 1


class SyntheticPolicyEnvironment:
    def __init__(self, clock, failure=None):
        self.clock = clock
        self.failure = failure
        self.closed = False
        self.resets = []

    def state(self):
        return [[self.day * 1000 + self.step_index]]

    def manual_reset(self, day, hour, minute):
        assert (hour, minute) == (0, 0)
        self.day, self.step_index = day, 0
        self.resets.append(day)
        return self.state(), None

    def get_avail_actions(self):
        return [[1.]]

    def step(self, action, add_noise):
        assert add_noise is False
        failed = self.failure == 'controlled' and (self.day, self.step_index) == (1000, 100)
        if failed:
            self.clock.advance(7)
        return 0., failed or self.step_index == 479, {'destroy': int(failed)}

    def get_obs(self):
        if self.failure == 'precontrol' and (self.day, self.step_index) == (1000, 100):
            self.clock.advance(7)
            raise SyntheticPowerflowFailure('synthetic failed measurement')
        self.step_index += 1
        return self.state()

    def close(self):
        self.closed = True


def policy_fixture(monkeypatch, tmp_path, *, failure=None, drift=None):
    fixture = harness(monkeypatch, tmp_path, policy, drift)
    fixture.env = SyntheticPolicyEnvironment(fixture.clock, failure)
    fixture.model = TinyPolicy()
    args = SimpleNamespace(cuda=False, shared_params=True, agent_type='rnn',
                           agent_num=1, obs_size=1, agent_id=False)
    fixture.evaluator.load_policy_artifacts = lambda *a, **k: (
        fixture.model, args, {'mode': 'distributed'}, {'legacy_pilot': False})
    module(monkeypatch, 'experiments.run_q1_ml_compare', make_env=lambda config: fixture.env)
    module(monkeypatch, 'utilities.util', prep_obs=lambda state: torch.as_tensor(state, dtype=torch.float32),
           translate_action=lambda args, action, env: (None, action))
    return fixture


def traditional_fixture(monkeypatch, tmp_path, *, failure=False, drift=None, omit=None):
    fixture = harness(monkeypatch, tmp_path, traditional, drift)
    fixture.env = SimpleNamespace(_episode_start_day=0, _loaded_step=0, closed=False)
    fixture.env.current_timestamp = lambda: f'{fixture.env._episode_start_day}:{fixture.env._loaded_step}'
    fixture.env.close = lambda: setattr(fixture.env, 'closed', True)

    def solve_pf(*args, **kwargs):
        if failure and (fixture.env._episode_start_day, fixture.env._loaded_step) == (1000, 100):
            fixture.clock.advance(7)
            raise SyntheticPowerflowFailure('synthetic nonconvergence inside controller')

    pp = module(monkeypatch, 'pandapower', runpp=solve_pf, runopp=lambda *a, **k: None)

    def controller(env):
        try:
            pp.runpp(None)
        except SyntheticPowerflowFailure:
            pass  # A controller may catch a failed solve and choose its fallback.
        pp.runopp(None)
        return np.zeros(1)

    def load_config(scenario, scale, **kwargs):
        assert kwargs['days'] == [1000, 1014, 1029]
        assert kwargs['max_steps'] == 480 and kwargs['start_hour'] == 0
        return kwargs

    module(monkeypatch, 'experiments.run_q1_fixed_test_traditional_baselines',
           load_env_config=load_config, make_env=lambda config: fixture.env,
           CONTROLLERS={'droop': controller, 'opf': controller})

    def evaluate(*, env, controller, days, max_steps, start_hour, _strict_shape):
        assert days == [1000, 1014, 1029] and max_steps == 480 and start_hour == 0
        for day in days:
            for step in range(max_steps):
                if (day, step) == omit:
                    continue
                env._episode_start_day, env._loaded_step = day, step
                controller(env)
        return SimpleNamespace(summary={'synthetic_fixture_only': True})

    module(monkeypatch, 'experiments.fixed_test_protocol', evaluate_controller_fixed_days=evaluate)
    return fixture


def test_common_schedule_is_validation_only_and_excludes_fifty_warmup_inputs():
    schedule = policy.planned_timing_inputs()
    assert policy.BENCHMARK_DAYS == (1000, 1014, 1029)
    assert policy.WARMUP_INPUTS == 50
    assert len(schedule) == 1000 and len(set(schedule)) == 1000
    assert schedule[0] == (1000, 50) and schedule[-1] == (1029, 89)
    assert schedule[429:431] == [(1000, 479), (1014, 0)]
    assert schedule[909:911] == [(1014, 479), (1029, 0)]
    assert all(1000 <= day <= 1029 and 0 <= step < 480 for day, step in schedule)


def test_policy_and_traditional_measure_the_same_planned_day_step_schedule(monkeypatch, tmp_path):
    first = policy_fixture(monkeypatch, tmp_path / 'policy')
    report = policy.benchmark('synthetic-run', first.output, isolated=True)
    planned = policy.planned_timing_inputs()
    encoded = [day * 1000 + step for day, step in planned]
    assert report['warmup_inputs'] == 50 and report['actual_input_count'] == 1000
    assert report['matches_common_planned_day_step_schedule'] is True
    assert first.model.seen[1050:1100] == list(range(1000000, 1000050))
    assert first.model.seen[1100:] == encoded
    assert first.model.policy_dicts[0].seen == encoded
    assert first.env.closed
    with np.load(first.output / 'raw_timings.npz') as raw:
        np.testing.assert_array_equal(raw['input_day_step'], planned)
        for name in ('full_policy_s', 'agent_zero_forward_s', 'preparation_s', 'simulation_s', 'simulation_status'):
            assert len(raw[name]) == 1000
    second = traditional_fixture(monkeypatch, tmp_path / 'traditional')
    result = traditional.benchmark('case33_3min_final', 'droop', second.output, isolated=True)
    calls = json.loads((second.output / 'raw_calls.json').read_text(encoding='utf-8'))
    assert result['warmup_calls'] == 50 and result['measured_calls'] == 1000
    assert result['all_executed_controller_calls'] == 1440
    assert result['matches_common_planned_day_step_schedule'] is True
    assert [(r['day'], r['step']) for r in calls[50:1050]] == planned
    assert second.env.closed


@pytest.mark.parametrize('phase,status', [('precontrol', 'next_precontrol_failure'),
                                          ('controlled', 'controlled_failure')])
def test_policy_failed_powerflow_attempt_cost_is_retained_and_schedule_gap_disclosed(monkeypatch, tmp_path, phase, status):
    fixture = policy_fixture(monkeypatch, tmp_path, failure=phase)
    report = policy.benchmark('synthetic-run', fixture.output, isolated=True)
    assert report['matches_common_planned_day_step_schedule'] is False
    assert report['simulation_timing_retains_failed_attempts'] is True
    assert report['simulation_status_counts'][status] == 1
    with np.load(fixture.output / 'raw_timings.npz') as raw:
        index = np.flatnonzero(raw['simulation_status'] == status)
        assert len(index) == 1
        assert tuple(raw['input_day_step'][index[0]]) == (1000, 100)
        assert raw['simulation_s'][index[0]] == pytest.approx(8.)
        assert len(raw['simulation_s']) == len(raw['input_day_step']) == 1000
    assert fixture.env.closed


def test_traditional_failed_solver_call_cost_survives_controller_fallback(monkeypatch, tmp_path):
    fixture = traditional_fixture(monkeypatch, tmp_path, failure=True)
    report = traditional.benchmark('case33_3min_final', 'droop', fixture.output, isolated=True)
    raw = json.loads((fixture.output / 'raw_calls.json').read_text(encoding='utf-8'))
    row = next(r for r in raw if (r['day'], r['step']) == (1000, 100))
    assert row['powerflow_calls'] == 1 and row['opf_calls'] == 1
    assert row['powerflow_seconds'] == pytest.approx(8.)
    assert row['whole_controller_seconds'] >= row['powerflow_seconds'] + row['opf_seconds']
    assert report['powerflow_in_controller']['max_ms'] == pytest.approx(8000.)
    assert report['matches_common_planned_day_step_schedule'] is True


def test_traditional_omitted_input_is_not_claimed_to_match_common_schedule(monkeypatch, tmp_path):
    fixture = traditional_fixture(monkeypatch, tmp_path, omit=(1000, 100))
    report = traditional.benchmark('case33_3min_final', 'opf', fixture.output, isolated=True)
    assert report['all_executed_controller_calls'] == 1439
    assert report['matches_common_planned_day_step_schedule'] is False


@pytest.mark.parametrize('drift', ['source', 'script'])
def test_policy_source_drift_prevents_publication_and_closes_environment(monkeypatch, tmp_path, drift):
    fixture = policy_fixture(monkeypatch, tmp_path, drift=drift)
    with pytest.raises(RuntimeError, match='dependencies changed'):
        policy.benchmark('synthetic-run', fixture.output, isolated=True)
    assert fixture.env.closed and not fixture.output.exists()


@pytest.mark.parametrize('drift', ['source', 'script', 'helper'])
def test_traditional_source_or_timing_helper_drift_prevents_publication(monkeypatch, tmp_path, drift):
    fixture = traditional_fixture(monkeypatch, tmp_path, drift=drift)
    with pytest.raises(RuntimeError, match='source changed'):
        traditional.benchmark('case141_3min_final', 'opf', fixture.output, isolated=True)
    assert fixture.env.closed and not fixture.output.exists()


@pytest.mark.parametrize('samples', [999, 1391])
def test_policy_refuses_sample_count_outside_predeclared_schedule(monkeypatch, tmp_path, samples):
    fixture = policy_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='at least 1000|exceeds the common'):
        policy.benchmark('synthetic-run', fixture.output, samples=samples)
    assert fixture.env.resets == [] and not fixture.output.exists()


@pytest.mark.parametrize('script', [
    'run_revision_training.py', 'dispatch_revision_training.py',
    'run_revision_pv_count_training.py', 'revision_evaluation_queue.py',
    'revision_pv_count_queue.py', 'revision_pv_count_evaluation.py',
    'revision_pv_count_pipeline.py', 'revision_benchmark.py',
    'revision_traditional_benchmark.py',
])
def test_isolation_detects_actual_worker_names_and_ignores_launcher_ancestors(monkeypatch, tmp_path, script):
    root = tmp_path / 'Worktree'
    monkeypatch.setattr(policy, 'ROOT', root)
    monkeypatch.setattr(policy, 'os', SimpleNamespace(getpid=lambda: 991))

    def process(pid, command, cwd):
        return SimpleNamespace(pid=pid, info={'pid': pid, 'cmdline': command, 'cwd': str(cwd)})

    processes = [
        process(990, ['python', script], root),  # Launcher is an ignored ancestor.
        process(991, ['python', script], root),  # The benchmark itself is ignored.
        process(992, ['python', script], str(root).upper()),  # Windows case-insensitive path.
        process(993, ['python', script], tmp_path / 'different_worktree'),
        process(994, ['python', 'unrelated_script.py'], root),
    ]
    module(monkeypatch, 'psutil',
           Process=lambda: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=990)]),
           process_iter=lambda fields: processes,
           AccessDenied=PermissionError, NoSuchProcess=ProcessLookupError,
           cpu_percent=lambda interval: 12.5,
           virtual_memory=lambda: SimpleNamespace(available=1024 ** 3),
           cpu_count=lambda logical=True: 8 if logical else 4)
    context = policy.measurement_context(False)
    assert context['other_experiment_process_ids'] == [992]
    assert context['isolated_from_other_worktree_experiment_workers'] is False
    assert context['ignored_launcher_ancestor_ids'] == [990]
    with pytest.raises(ValueError, match='other worktree experiment processes'):
        policy.measurement_context(True)
