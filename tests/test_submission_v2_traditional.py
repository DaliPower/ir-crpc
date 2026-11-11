from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.fixed_test_protocol import evaluate_controller_fixed_days


SCENARIOS = {
    "case33_3min_final": 0.8,
    "case141_3min_final": 0.6,
}
ROOT = Path(__file__).resolve().parents[1]


def _args(scale: float, bias: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(action_scale=scale, action_bias=bias)


def test_importing_strict_runner_does_not_change_process_working_directory(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; before = os.getcwd(); "
                "import experiments.run_q1_fixed_test_traditional_baselines; "
                "assert os.getcwd() == before, (before, os.getcwd())"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("scenario,alpha", SCENARIOS.items())
def test_normalized_bounds_and_q_round_trip_match_each_system(scenario, alpha):
    from experiments.traditional_action_limits import (
        bounded_reactive_limits,
        normalized_action_bounds,
        q_to_bounded_actions,
        reactive_capability,
    )

    args = _args(alpha)
    active = np.array([0.0, 0.8])
    rating = np.array([1.0, 1.0])
    capability = reactive_capability(active, rating)
    requested_q = np.array([-0.25, 0.18])

    assert normalized_action_bounds(args) == pytest.approx((-alpha, alpha))
    actions = q_to_bounded_actions(requested_q, active, rating, args)
    assert capability * actions == pytest.approx(requested_q)
    q_min, q_max = bounded_reactive_limits(active, rating, args)
    assert q_min == pytest.approx(-alpha * capability)
    assert q_max == pytest.approx(alpha * capability)


@pytest.mark.parametrize("alpha", [0.8, 0.6])
def test_q_translation_clips_to_bound_and_zero_capacity_stays_finite(alpha):
    from experiments.traditional_action_limits import q_to_bounded_actions

    actions = q_to_bounded_actions(
        q_mvar=np.array([-10.0, 10.0, 1.0]),
        active_power=np.array([0.0, 0.0, 1.0]),
        s_max=np.array([1.0, 1.0, 1.0]),
        args=_args(alpha),
    )

    assert actions == pytest.approx([-alpha, alpha, 0.0])
    assert np.all(np.isfinite(actions))


@pytest.mark.parametrize(
    "call",
    [
        lambda module: module.reactive_capability([0.0, 0.1], [1.0]),
        lambda module: module.reactive_capability([np.nan], [1.0]),
        lambda module: module.reactive_capability([0.0], [-1.0]),
        lambda module: module.q_to_bounded_actions(
            [np.inf], [0.0], [1.0], _args(0.8)
        ),
        lambda module: module.q_to_bounded_actions(
            [0.0, 0.0], [0.0], [1.0], _args(0.8)
        ),
        lambda module: module.normalized_action_bounds(_args(-0.1)),
        lambda module: module.normalized_action_bounds(_args(np.nan)),
        lambda module: module.normalized_action_bounds(_args(0.8, np.inf)),
    ],
)
def test_action_limit_helpers_reject_malformed_or_nonfinite_inputs(call):
    import experiments.traditional_action_limits as limits

    with pytest.raises(ValueError):
        call(limits)


class _DroopEnv:
    def __init__(self, alpha: float):
        self.args = _args(alpha)
        self.s_max = np.array([1.0, 1.0])
        self.recorded_actions: list[np.ndarray] = []
        self.powergrid = SimpleNamespace(
            sgen=pd.DataFrame(
                {
                    "p_mw": [0.0, 0.8],
                    "q_mvar": [0.0, 0.0],
                    "bus": [1, 2],
                }
            ),
            res_bus=pd.DataFrame({"vm_pu": [1.0, 1.1, 0.9]}, index=[0, 1, 2]),
        )

    def _map_reactive_action(self, actions):
        from experiments.traditional_action_limits import reactive_capability

        action = np.asarray(actions, dtype=float)
        self.recorded_actions.append(action.copy())
        capability = reactive_capability(
            self.powergrid.sgen["p_mw"].to_numpy(), self.s_max
        )
        reactive = capability * action
        return reactive, {
            "action_projection_ratio": 0.0,
            "reactive_capacity_utilization": float(
                np.mean(
                    np.divide(
                        np.abs(reactive),
                        capability,
                        out=np.zeros_like(reactive),
                        where=capability > 0.0,
                    )
                )
            ),
            "reactive_saturation_ratio": 0.0,
        }

    def _take_action(self, actions):
        reactive, _diagnostics = self._map_reactive_action(actions)
        self.powergrid.sgen.loc[:, "q_mvar"] = reactive
        return True


@pytest.mark.parametrize("alpha", [0.8, 0.6])
def test_droop_controller_never_exceeds_alpha_times_instantaneous_capacity(
    alpha, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner
    from experiments.traditional_action_limits import reactive_capability

    env = _DroopEnv(alpha)
    monkeypatch.setattr(runner, "droop_q_single", lambda *args: 1e6)
    monkeypatch.setattr(runner.pp, "runpp", lambda _net, **_kwargs: None)

    assert runner.run_droop_step(env, max_ite=3, gain=1.0) is True
    recorded = np.vstack(env.recorded_actions)
    assert np.max(np.abs(recorded)) <= alpha
    capability = reactive_capability(
        env.powergrid.sgen["p_mw"].to_numpy(), env.s_max
    )
    assert np.all(
        np.abs(env.powergrid.sgen["q_mvar"].to_numpy())
        <= alpha * capability + 1e-12
    )
    assert env._last_action_diagnostics["reactive_saturation_ratio"] == 0.0
    assert env._last_action_diagnostics["control_limit_saturation_ratio"] == 1.0


def test_droop_reports_false_when_fixed_point_iteration_is_exhausted(monkeypatch):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    class NeverSettlingEnv(_DroopEnv):
        def _map_reactive_action(self, actions):
            outcome = super()._map_reactive_action(actions)
            iteration = len(self.recorded_actions)
            self.powergrid.res_bus.loc[:, "vm_pu"] = [
                1.0,
                1.0 + 0.01 * iteration,
                1.0 - 0.01 * iteration,
            ]
            return outcome

    monkeypatch.setattr(runner.pp, "runpp", lambda _net, **_kwargs: None)
    assert runner.run_droop_step(NeverSettlingEnv(0.8), max_ite=2) is False


def test_droop_only_converts_known_powerflow_nonconvergence_to_false(monkeypatch):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    known_error = _known_solver_error("LoadflowNotConverged")
    monkeypatch.setattr(
        runner.pp,
        "runpp",
        lambda _net, **_kwargs: (_ for _ in ()).throw(known_error()),
    )
    assert runner.run_droop_step(_DroopEnv(0.8), max_ite=2) is False

    unexpected_powerflow_error = type(
        "UnexpectedPandapowerError",
        (Exception,),
        {"__module__": "pandapower.fake_solver"},
    )
    monkeypatch.setattr(
        runner.pp,
        "runpp",
        lambda _net, **_kwargs: (_ for _ in ()).throw(
            unexpected_powerflow_error("unexpected")
        ),
    )
    with pytest.raises(unexpected_powerflow_error, match="unexpected"):
        runner.run_droop_step(_DroopEnv(0.8), max_ite=2)

    monkeypatch.setattr(
        runner.pp,
        "runpp",
        lambda _net, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad program")),
    )
    with pytest.raises(RuntimeError, match="bad program"):
        runner.run_droop_step(_DroopEnv(0.8), max_ite=2)


class _OpfEnv:
    def __init__(self, alpha: float):
        self.args = _args(alpha)
        self.s_max = np.array([1.0, 1.0])
        self.v_lower = 0.95
        self.v_upper = 1.05
        self.powergrid = SimpleNamespace(
            sgen=pd.DataFrame(
                {
                    "p_mw": [0.0, 0.8],
                    "q_mvar": [0.0, 0.0],
                    "bus": [1, 2],
                }
            ),
            res_sgen=pd.DataFrame({"q_mvar": [0.0, 0.0]}),
            bus=pd.DataFrame(index=[0, 1, 2]),
            line=pd.DataFrame(index=[0, 1]),
            poly_cost=pd.DataFrame(),
        )


@pytest.mark.parametrize("alpha", [0.8, 0.6])
def test_opf_controller_enforces_and_retains_bounded_reactive_limits(
    alpha, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner
    from experiments.traditional_action_limits import reactive_capability

    env = _OpfEnv(alpha)
    powerflow_q: list[np.ndarray] = []

    def fake_runpp(net, **_kwargs):
        powerflow_q.append(net.sgen["q_mvar"].to_numpy(dtype=float, copy=True))

    def fake_runopp(net, **_kwargs):
        capability = reactive_capability(net.sgen["p_mw"].to_numpy(), env.s_max)
        assert net.sgen["min_q_mvar"].to_numpy() == pytest.approx(
            -alpha * capability
        )
        assert net.sgen["max_q_mvar"].to_numpy() == pytest.approx(
            alpha * capability
        )
        net.res_sgen.loc[:, "q_mvar"] = [10.0, -10.0]

    monkeypatch.setattr(runner, "_ensure_opf_costs", lambda _net: None)
    monkeypatch.setattr(runner.pp, "runpp", fake_runpp)
    monkeypatch.setattr(runner.pp, "runopp", fake_runopp)

    assert runner.run_opf_step(env) is True
    capability = reactive_capability(
        env.powergrid.sgen["p_mw"].to_numpy(), env.s_max
    )
    final_q = env.powergrid.sgen["q_mvar"].to_numpy()
    assert np.abs(final_q) == pytest.approx(alpha * capability)
    assert powerflow_q[-1] == pytest.approx(final_q)
    assert env._last_action_diagnostics["reactive_saturation_ratio"] == 0.0
    assert env._last_action_diagnostics["control_limit_saturation_ratio"] == 1.0


def _known_solver_error(name: str):
    return type(name, (Exception,), {"__module__": "pandapower.fake_solver"})


def test_opf_known_nonconvergence_returns_false_but_programming_error_propagates(
    monkeypatch,
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    env = _OpfEnv(0.8)
    monkeypatch.setattr(runner, "_ensure_opf_costs", lambda _net: None)
    monkeypatch.setattr(runner.pp, "runpp", lambda _net, **_kwargs: None)
    known_error = _known_solver_error("OPFNotConverged")
    monkeypatch.setattr(
        runner.pp, "runopp", lambda _net, **_kwargs: (_ for _ in ()).throw(known_error())
    )
    assert runner.run_opf_step(env) is False

    monkeypatch.setattr(
        runner.pp,
        "runopp",
        lambda _net, **_kwargs: (_ for _ in ()).throw(ValueError("bad program")),
    )
    with pytest.raises(ValueError, match="bad program"):
        runner.run_opf_step(env)


class _ProtocolEnv:
    def __init__(self):
        self.args = SimpleNamespace(reset_action=True)
        self.episode_limit = 240
        self.steps = 0
        self._loaded_step = 0
        self.current_day = 0
        self.start_hour = 6
        self.reset_q: list[float] = []
        self.noise_flags: list[bool] = []
        self.powergrid = SimpleNamespace(
            sgen=pd.DataFrame({"q_mvar": [9.0]}),
            res_bus=pd.DataFrame({"vm_pu": [1.0, 1.01]}, index=[0, 1]),
        )

    def manual_reset(self, day, hour, interval):
        self.current_day = int(day)
        self.start_hour = int(hour)
        self.steps = 0
        self._loaded_step = 0
        if not self.args.reset_action:
            self.powergrid.sgen.loc[:, "q_mvar"] = 0.0
        self.reset_q.append(float(self.powergrid.sgen["q_mvar"].iloc[0]))
        return [[0.0]], None

    def current_timestamp(self):
        return (
            pd.Timestamp("2020-01-01")
            + pd.Timedelta(days=self.current_day)
            + pd.Timedelta(hours=self.start_hour)
            + pd.Timedelta(minutes=3 * self._loaded_step)
        )

    def _calc_reward(self):
        return 0.0, {
            "percentage_of_v_out_of_control": 0.0,
            "totally_controllable_ratio": 1.0,
            "destroy": 0.0,
            "average_voltage_deviation": 0.005,
            "q_loss": 0.0,
            "total_line_loss": 0.01,
            "inverter_p_mw": [0.2],
            "inverter_q_mvar": [0.1],
            "inverter_s_max_mva": [1.0],
        }

    def _set_demand_and_pv(self, *, add_noise):
        self.noise_flags.append(bool(add_noise))

    def get_obs(self):
        return [[float(self.steps)]]

    def close(self):
        return None


def _successful_fixed_result(env: _ProtocolEnv, *, days: list[int], strict: bool):
    return evaluate_controller_fixed_days(
        env=env,
        controller=lambda _env: True,
        days=days,
        max_steps=240,
        start_hour=6,
        _strict_shape=strict,
    )


def _prepare_provenance_inputs(repo_root: Path, scenario: str) -> None:
    config_path = repo_root / "args/env_args/var_voltage_control.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("env_args: {}\n", encoding="utf-8")
    data_dir = repo_root / "environments/var_voltage_control/data" / scenario
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.p", "pv_active.csv", "load_active.csv", "load_reactive.csv"):
        (data_dir / name).write_bytes(f"fixture:{name}\n".encode("utf-8"))
    for relative in (
        "experiments/run_q1_fixed_test_traditional_baselines.py",
        "experiments/traditional_action_limits.py",
        "experiments/fixed_test_protocol.py",
        "environments/var_voltage_control/voltage_control_env.py",
    ):
        source = repo_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture:{relative}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_controller_false_receives_conservative_failure_fill():
    env = _ProtocolEnv()
    result = evaluate_controller_fixed_days(
        env=env,
        controller=lambda _env: False,
        days=[1030],
        max_steps=3,
        start_hour=6,
        _strict_shape=False,
    )

    assert result.summary["planned_steps"] == 3
    assert result.summary["executed_steps"] == 0
    assert result.summary["vrate"] == 1.0
    assert result.summary["cr"] == 0.0
    assert all(row["executed"] is False for row in result.step_rows)


@pytest.mark.parametrize(
    ("scenario", "scale"),
    [
        ("case33_3min_final", 0.8),
        ("case141_3min_final", 0.6),
    ],
)
def test_strict_runner_writes_recomputable_30_by_240_artifacts(
    tmp_path, monkeypatch, scenario, scale
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    _prepare_provenance_inputs(tmp_path, scenario)
    monkeypatch.setattr(
        runner,
        "_git_snapshot",
        lambda _root: {"revision": "a" * 40, "dirty": False},
        raising=False,
    )
    strict_root = tmp_path / "exp" / "q1_submission_v2_strict"
    plan = runner.build_traditional_plan(
        repo_root=tmp_path,
        scenario=scenario,
        methods=["droop", "opf"],
        pv_scale=1.5,
        test_start_day=1030,
        test_days=30,
        max_steps=240,
        start_hour=6,
        smoke=False,
        save_path=strict_root,
    )
    environments: list[_ProtocolEnv] = []

    def env_factory(_config):
        env = _ProtocolEnv()
        environments.append(env)
        return env

    outputs = runner.execute_traditional_plan(
        plan,
        env_factory=env_factory,
        controllers={"droop": lambda _env: True, "opf": lambda _env: True},
    )

    assert len(outputs) == 2
    for output in outputs:
        directory = Path(output["output_dir"])
        assert {path.name for path in directory.iterdir()} == {
            "fixed_test_step_metrics.csv",
            "fixed_test_daily_metrics.csv",
            "fixed_test_node_voltages.npz",
            "fixed_test_trajectory_schema.json",
            "fixed_test_summary.json",
            "method_metadata.json",
        }
        with (directory / "fixed_test_step_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 7200
        timestamps = [row["timestamp"] for row in rows]
        assert len(set(timestamps)) == 7200
        assert timestamps[0].endswith("T06:00:00")
        assert timestamps[239].endswith("T17:57:00")
        summary = json.loads(
            (directory / "fixed_test_summary.json").read_text(encoding="utf-8")
        )
        assert summary["vrate"] == pytest.approx(
            np.mean([float(row["vrate"]) for row in rows])
        )
        metadata = json.loads(
            (directory / "method_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["action_scale"] == scale
        assert metadata["action_bias"] == 0.0
        assert metadata["normalized_action_bounds"] == [-scale, scale]
        assert metadata["zero_initial_q"] is True
        assert metadata["reset_action"] is False
        assert metadata["add_noise"] is False
        assert metadata["strict_shape"] is True
        assert metadata["reactive_saturation_definition"] == (
            "fraction at the nonzero physical instantaneous reactive capacity"
        )
        assert metadata["provenance"]["git"] == {
            "revision": "a" * 40,
            "dirty": False,
        }
        runtime = metadata["provenance"]["runtime"]
        assert all(runtime[name] for name in ("python", "numpy", "pandapower"))
        inputs = {
            item["role"]: item for item in metadata["provenance"]["inputs"]
        }
        assert set(inputs) == {
            "model",
            "pv_active",
            "load_active",
            "load_reactive",
            "environment_yaml",
            "runner_source",
            "action_limit_source",
            "fixed_test_protocol_source",
            "voltage_control_env_source",
        }
        expected_inputs = {
            "model": f"environments/var_voltage_control/data/{scenario}/model.p",
            "pv_active": f"environments/var_voltage_control/data/{scenario}/pv_active.csv",
            "load_active": f"environments/var_voltage_control/data/{scenario}/load_active.csv",
            "load_reactive": f"environments/var_voltage_control/data/{scenario}/load_reactive.csv",
            "environment_yaml": "args/env_args/var_voltage_control.yaml",
            "runner_source": "experiments/run_q1_fixed_test_traditional_baselines.py",
            "action_limit_source": "experiments/traditional_action_limits.py",
            "fixed_test_protocol_source": "experiments/fixed_test_protocol.py",
            "voltage_control_env_source": "environments/var_voltage_control/voltage_control_env.py",
        }
        assert {role: item["path"] for role, item in inputs.items()} == expected_inputs
        for item in inputs.values():
            path = Path(item["absolute_path"])
            assert path.resolve().is_relative_to(tmp_path.resolve())
            assert item["sha256"] == _sha256(path)
            assert item["size_bytes"] == path.stat().st_size
        assert metadata["fixed_test_protocol"] == {
            "date_window": {"start_day": 1030, "end_day": 1059, "day_count": 30},
            "points_per_day": 240,
            "window": {"start": "06:00", "end": "17:57", "cadence_minutes": 3},
            "trajectory_schema": {
                "path": "fixed_test_trajectory_schema.json",
                "schema_version": 2,
                "inverter_vector_fields": [
                    "inverter_p_mw",
                    "inverter_q_mvar",
                    "inverter_s_max_mva",
                ],
            },
            "instantaneous_capacity_circle": {
                "equation": "|q_i| <= sqrt(max(S_i^2 - p_i^2, 0)) + 1e-10",
                "tolerance": 1e-10,
            },
        }
        definitions = metadata["controller_definitions"]
        assert definitions["droop"] == {
            "gain": 0.1,
            "max_iterations": 100,
            "convergence_norm": "l2",
            "convergence_tolerance": 0.0001,
            "voltage_curve_pu": {
                "full_injection_at_or_below": 0.95,
                "zero_lower": 1.0,
                "zero_upper": 1.0,
                "full_absorption_at_or_above": 1.05,
            },
        }
        assert definitions["opf"]["objective"] == {
            "sgen_reactive_quadratic_coefficient": 1.0,
            "grid_active_linear_coefficient": 1.0,
            "fixed_exogenous_power": True,
        }
        assert definitions["opf"]["solver_calls"] == {
            "pre_power_flow": {"function": "runpp", "numba": False},
            "optimization": {
                "function": "runopp",
                "init": "results",
                "verbose": False,
                "numba": False,
            },
            "post_power_flow": {
                "function": "runpp",
                "init": "results",
                "numba": False,
            },
        }

    assert all(env.reset_q == [0.0] * 30 for env in environments)
    assert all(env.noise_flags and not any(env.noise_flags) for env in environments)


def test_input_manifest_rehashes_mutated_inputs_and_rejects_escape(tmp_path):
    """Catch stale or out-of-repository source records before publication."""
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    _prepare_provenance_inputs(tmp_path, "case33_3min_final")
    config = runner.load_env_config("case33_3min_final", 1.5, repo_root=tmp_path)
    before = {
        item["path"]: item["sha256"]
        for item in runner._input_manifest(config, tmp_path)
    }
    changed = tmp_path / "environments/var_voltage_control/data/case33_3min_final/pv_active.csv"
    changed.write_bytes(b"mutated fixture\n")
    after = {
        item["path"]: item["sha256"]
        for item in runner._input_manifest(config, tmp_path)
    }

    assert after["environments/var_voltage_control/data/case33_3min_final/pv_active.csv"] != before[
        "environments/var_voltage_control/data/case33_3min_final/pv_active.csv"
    ]
    outside = tmp_path.parent / "outside.csv"
    outside.write_bytes(b"outside\n")
    with pytest.raises(ValueError, match="outside repository"):
        runner._input_record("outside", outside, tmp_path)


def test_method_directory_is_not_published_when_metadata_write_fails(
    tmp_path, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    short_root = tmp_path.parent / "txn"
    _prepare_provenance_inputs(short_root, "case33_3min_final")
    monkeypatch.setattr(
        runner,
        "_git_snapshot",
        lambda _root: {"revision": "b" * 40, "dirty": True},
        raising=False,
    )
    smoke_parent = short_root / "exp" / "q1_submission_v2_traditional_smoke"
    plan = runner.build_traditional_plan(
        repo_root=short_root,
        scenario="case33_3min_final",
        methods=["droop"],
        pv_scale=1.5,
        test_start_day=1030,
        test_days=1,
        max_steps=240,
        start_hour=6,
        smoke=True,
        save_path=smoke_parent,
    )
    final_directory = Path(plan["runs"][0]["output_dir"])
    monkeypatch.setattr(
        runner,
        "_atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("metadata promotion failed")
        ),
    )

    with pytest.raises(RuntimeError, match="metadata promotion failed"):
        runner.execute_traditional_plan(
            plan,
            env_factory=lambda _config: _ProtocolEnv(),
            controllers={"droop": lambda _env: True},
        )

    assert not final_directory.exists()
    assert list(final_directory.parent.glob(".b-*")) == []


def _default_dual_method_smoke_plan(
    tmp_path: Path, monkeypatch, scenario: str = "case33_3min_final"
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    _prepare_provenance_inputs(tmp_path, scenario)
    monkeypatch.setattr(
        runner,
        "_git_snapshot",
        lambda _root: {"revision": "c" * 40, "dirty": False},
    )
    return runner.build_traditional_plan(
        repo_root=tmp_path,
        scenario=scenario,
        methods=["droop", "opf"],
        pv_scale=1.5,
        test_start_day=1030,
        test_days=1,
        max_steps=240,
        start_hour=6,
        smoke=True,
        save_path=tmp_path / "exp" / "q1_submission_v2_traditional_smoke",
    )


@pytest.mark.parametrize("failure", ["evaluation", "metadata", "bundle", "promotion"])
def test_default_dual_method_plan_failure_leaves_no_partial_formal_bundle(
    tmp_path, monkeypatch, failure
):
    """Catch a second-method failure that publishes the first method early."""
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    plan = _default_dual_method_smoke_plan(tmp_path, monkeypatch)
    output_root = Path(plan["output_root"])
    controllers = {"droop": lambda _env: True, "opf": lambda _env: True}
    if failure == "evaluation":
        controllers["opf"] = lambda _env: (_ for _ in ()).throw(
            RuntimeError("opf evaluation failed")
        )
    elif failure == "metadata":
        real_build = runner._build_method_metadata

        def fail_second_metadata(**kwargs):
            if kwargs["method"] == "opf":
                raise RuntimeError("opf metadata failed")
            return real_build(**kwargs)

        monkeypatch.setattr(runner, "_build_method_metadata", fail_second_metadata)
    elif failure == "bundle":
        real_validate = runner._validate_method_bundle
        validation_count = 0

        def fail_second_bundle(directory):
            nonlocal validation_count
            validation_count += 1
            if validation_count == 2:
                raise RuntimeError("opf bundle validation failed")
            return real_validate(directory)

        monkeypatch.setattr(runner, "_validate_method_bundle", fail_second_bundle)
    else:
        real_replace = runner.os.replace

        def fail_final_promotion(source, destination):
            if Path(destination).name in {
                "opf",
                "traditional",
                "case33_3min_final",
            }:
                raise RuntimeError("traditional promotion failed")
            return real_replace(source, destination)

        monkeypatch.setattr(runner.os, "replace", fail_final_promotion)

    with pytest.raises(RuntimeError, match=failure if failure != "promotion" else "promotion"):
        runner.execute_traditional_plan(
            plan,
            env_factory=lambda _config: _ProtocolEnv(),
            controllers=controllers,
        )

    assert not output_root.exists()
    assert list(output_root.parent.glob(".case33_3min_final-*")) == []


def test_default_dual_method_plan_promotes_both_exact_bundles_together(
    tmp_path, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    plan = _default_dual_method_smoke_plan(tmp_path, monkeypatch)
    output_root = Path(plan["output_root"])

    outputs = runner.execute_traditional_plan(
        plan,
        env_factory=lambda _config: _ProtocolEnv(),
        controllers={"droop": lambda _env: True, "opf": lambda _env: True},
    )

    assert {output["method"] for output in outputs} == {"droop", "opf"}
    for method in ("droop", "opf"):
        directory = output_root / "traditional/case33_3min_final" / method
        assert {path.name for path in directory.iterdir()} == {
            "fixed_test_step_metrics.csv",
            "fixed_test_daily_metrics.csv",
            "fixed_test_node_voltages.npz",
            "fixed_test_trajectory_schema.json",
            "fixed_test_summary.json",
            "method_metadata.json",
        }
    assert list(output_root.parent.glob(".case33_3min_final-*")) == []


def _exact_bundle_names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def test_scenarios_publish_independently_without_changing_prior_evidence(
    tmp_path, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    case33 = _default_dual_method_smoke_plan(
        tmp_path, monkeypatch, "case33_3min_final"
    )
    case141 = _default_dual_method_smoke_plan(
        tmp_path, monkeypatch, "case141_3min_final"
    )
    output_root = Path(case33["output_root"])
    controllers = {"droop": lambda _env: True, "opf": lambda _env: True}

    runner.execute_traditional_plan(
        case33, env_factory=lambda _config: _ProtocolEnv(), controllers=controllers
    )
    before = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in (output_root / "traditional/case33_3min_final").rglob("*")
        if path.is_file()
    }
    runner.execute_traditional_plan(
        case141, env_factory=lambda _config: _ProtocolEnv(), controllers=controllers
    )

    expected = {
        "fixed_test_step_metrics.csv",
        "fixed_test_daily_metrics.csv",
        "fixed_test_node_voltages.npz",
        "fixed_test_trajectory_schema.json",
        "fixed_test_summary.json",
        "method_metadata.json",
    }
    for scenario in ("case33_3min_final", "case141_3min_final"):
        for method in ("droop", "opf"):
            assert _exact_bundle_names(
                output_root / "traditional" / scenario / method
            ) == expected
    after = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in (output_root / "traditional/case33_3min_final").rglob("*")
        if path.is_file()
    }
    assert after == before


def test_second_scenario_failure_preserves_first_scenario_and_parent_sentinel(
    tmp_path, monkeypatch
):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    case33 = _default_dual_method_smoke_plan(
        tmp_path, monkeypatch, "case33_3min_final"
    )
    case141 = _default_dual_method_smoke_plan(
        tmp_path, monkeypatch, "case141_3min_final"
    )
    output_root = Path(case33["output_root"])
    traditional = output_root / "traditional"
    traditional.mkdir(parents=True)
    sentinel = traditional / "parent_sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    controllers = {"droop": lambda _env: True, "opf": lambda _env: True}
    runner.execute_traditional_plan(
        case33, env_factory=lambda _config: _ProtocolEnv(), controllers=controllers
    )
    before = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in (traditional / "case33_3min_final").rglob("*")
        if path.is_file()
    }

    with pytest.raises(RuntimeError, match="opf evaluation failed"):
        runner.execute_traditional_plan(
            case141,
            env_factory=lambda _config: _ProtocolEnv(),
            controllers={
                "droop": lambda _env: True,
                "opf": lambda _env: (_ for _ in ()).throw(
                    RuntimeError("opf evaluation failed")
                ),
            },
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    assert not (traditional / "case141_3min_final").exists()
    assert list(traditional.glob(".case141_3min_final-*")) == []
    after = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in (traditional / "case33_3min_final").rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"test_days": 29},
        {"max_steps": 239},
        {"start_hour": 7},
        {"test_start_day": 1031},
    ],
)
def test_formal_shape_or_window_is_rejected_before_output(tmp_path, overrides):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    kwargs = {
        "repo_root": tmp_path,
        "scenario": "case33_3min_final",
        "methods": ["droop"],
        "pv_scale": 1.5,
        "test_start_day": 1030,
        "test_days": 30,
        "max_steps": 240,
        "start_hour": 6,
        "smoke": False,
        "save_path": tmp_path / "exp" / "q1_submission_v2_strict",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        runner.build_traditional_plan(**kwargs)
    assert not (tmp_path / "exp").exists()


def test_smoke_output_is_explicitly_separated_from_formal_root(tmp_path):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    smoke_parent = tmp_path / "exp" / "q1_submission_v2_traditional_smoke"
    plan = runner.build_traditional_plan(
        repo_root=tmp_path,
        scenario="case33_3min_final",
        methods=["droop", "opf"],
        pv_scale=1.5,
        test_start_day=1030,
        test_days=1,
        max_steps=240,
        start_hour=6,
        smoke=True,
        save_path=smoke_parent,
    )

    assert Path(plan["output_root"]) == smoke_parent / "q1_submission_v2_strict"
    assert plan["strict_shape"] is False
    review_parent = (
        tmp_path / "exp" / "q1_submission_v2_traditional_review_smoke"
    )
    review_plan = runner.build_traditional_plan(
        repo_root=tmp_path,
        scenario="case33_3min_final",
        methods=["droop", "opf"],
        pv_scale=1.5,
        test_start_day=1030,
        test_days=1,
        max_steps=240,
        start_hour=6,
        smoke=True,
        save_path=review_parent,
    )
    assert Path(review_plan["output_root"]) == (
        review_parent / "q1_submission_v2_strict"
    )
    with pytest.raises(ValueError):
        runner.build_traditional_plan(
            repo_root=tmp_path,
            scenario="case33_3min_final",
            methods=["droop"],
            pv_scale=1.5,
            test_start_day=1030,
            test_days=2,
            max_steps=240,
            start_hour=6,
            smoke=True,
            save_path=smoke_parent,
        )


def test_dry_run_prints_json_without_writing(tmp_path, capsys):
    import experiments.run_q1_fixed_test_traditional_baselines as runner

    assert runner.main(["--dry-run"], repo_root=tmp_path) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["test_days"] == list(range(1030, 1060))
    assert printed["max_steps"] == 240
    assert not (tmp_path / "exp").exists()
