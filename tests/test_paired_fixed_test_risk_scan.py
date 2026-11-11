from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_paired_fixed_test_risk_scan import (
    DEFAULT_PV_SCALES,
    DEFAULT_SYSTEMS,
    build_scan_plan,
    evaluate_snapshot,
    run_environment_consistent_power_flow,
    summarize_step_rows,
)


def test_default_plan_is_six_paired_30_day_snapshot_runs(tmp_path):
    plan = build_scan_plan(
        systems=DEFAULT_SYSTEMS,
        pv_scales=DEFAULT_PV_SCALES,
        start_day=1030,
        day_count=30,
        start_hour=6,
        steps_per_day=240,
        output_root=tmp_path,
    )

    assert len(plan) == 6
    assert {run["days"] for run in plan} == {tuple(range(1030, 1060))}
    assert {run["first_sample_time"] for run in plan} == {"06:03:00"}
    assert {run["last_sample_time"] for run in plan} == {"18:00:00"}
    assert {run["steps_per_day"] for run in plan} == {240}
    assert {run["expected_step_count"] for run in plan} == {7200}
    assert all(
        run["control"] == {"reactive_action": "zero", "add_noise": False}
        for run in plan
    )


def test_plan_assigns_a_unique_directory_to_each_system_scale(tmp_path):
    plan = build_scan_plan(
        systems=DEFAULT_SYSTEMS,
        pv_scales=DEFAULT_PV_SCALES,
        start_day=1030,
        day_count=30,
        start_hour=6,
        steps_per_day=240,
        output_root=tmp_path,
    )

    directories = [run["output_dir"] for run in plan]
    assert len(directories) == len(set(directories)) == 6
    assert directories[0].endswith("case33_3min_final/pv_scale_0.5")
    assert directories[-1].endswith("case141_3min_final/pv_scale_1.5")


def test_summary_uses_deterministic_step_weighted_manuscript_metrics():
    rows = [
        {
            "percentage_of_v_out_of_control": 0.10,
            "average_voltage_deviation": 0.020,
            "total_line_loss": 0.30,
            "totally_controllable_ratio": 0.0,
            "q_loss": 0.0,
        },
        {
            "percentage_of_v_out_of_control": 0.30,
            "average_voltage_deviation": 0.040,
            "total_line_loss": 0.50,
            "totally_controllable_ratio": 1.0,
            "q_loss": 0.0,
        },
    ]

    summary = summarize_step_rows(rows)

    assert summary == {
        "voltage_violation_rate_pct": pytest.approx(20.0),
        "mean_voltage_deviation_pu": pytest.approx(0.03),
        "mean_total_active_line_loss_mw": pytest.approx(0.4),
        "violation_free_period_ratio_pct": pytest.approx(50.0),
        "mean_reactive_power_mvar": pytest.approx(0.0),
    }


def test_snapshot_failure_is_attempted_once_without_manual_reset_retry_loop():
    net = SimpleNamespace(
        sgen=pd.DataFrame({"p_mw": [1.0], "q_mvar": [9.0]}),
        load=pd.DataFrame({"p_mw": [1.0], "q_mvar": [0.2]}),
    )
    calls = []

    def fail_once(candidate):
        calls.append(candidate.sgen["q_mvar"].tolist())
        raise RuntimeError("unsolvable fixture")

    with pytest.raises(RuntimeError, match="single snapshot power flow failed"):
        evaluate_snapshot(
            net,
            pv_active=[2.0],
            load_active=[1.5],
            load_reactive=[0.3],
            v_lower=0.95,
            v_upper=1.05,
            run_power_flow=fail_once,
        )

    assert calls == [[0.0]]


def test_snapshot_power_flow_matches_environment_without_init_override(monkeypatch):
    import pandapower as pp

    net = object()
    calls = []

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(pp, "runpp", record_call)

    run_environment_consistent_power_flow(net)

    assert calls == [((net,), {})]


def test_dry_run_prints_protocol_and_does_not_create_output_root(tmp_path):
    output_root = tmp_path / "must_not_exist"
    command = [
        sys.executable,
        "experiments/run_paired_fixed_test_risk_scan.py",
        "--output-root",
        str(output_root),
        "--dry-run",
    ]

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
    )

    assert completed.returncode == 0, completed.stderr
    protocol = json.loads(completed.stdout)
    assert protocol["systems"] == DEFAULT_SYSTEMS
    assert protocol["pv_scales"] == DEFAULT_PV_SCALES
    assert protocol["days"] == list(range(1030, 1060))
    assert protocol["expected_step_count_per_scenario"] == 7200
    assert protocol["time_convention"]["first_sample_time"] == "06:03:00"
    assert protocol["time_convention"]["last_sample_time"] == "18:00:00"
    assert protocol["environment"]["power_flow_call"] == "pandapower.runpp(net)"
    assert protocol["environment"]["initialization"] == (
        "pandapower default; no init argument supplied"
    )
    assert not output_root.exists()


def test_dry_run_protocol_out_is_explicit_and_never_overwrites(tmp_path):
    output_root = tmp_path / "unused_output"
    protocol_out = tmp_path / "protocol.json"
    command = [
        sys.executable,
        "experiments/run_paired_fixed_test_risk_scan.py",
        "--output-root",
        str(output_root),
        "--protocol-out",
        str(protocol_out),
        "--dry-run",
    ]

    first = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
    )
    assert first.returncode == 0, first.stderr
    original = protocol_out.read_bytes()
    second = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
    )

    assert second.returncode != 0
    assert protocol_out.read_bytes() == original
    assert not output_root.exists()


def test_execution_refuses_existing_output_root_without_touching_sentinel(tmp_path):
    output_root = tmp_path / "existing"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    command = [
        sys.executable,
        "experiments/run_paired_fixed_test_risk_scan.py",
        "--systems",
        "case33_3min_final",
        "--pv-scales",
        "0.5",
        "--day-count",
        "1",
        "--steps-per-day",
        "1",
        "--output-root",
        str(output_root),
    ]

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False
    )

    assert completed.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert sorted(path.name for path in output_root.iterdir()) == ["sentinel.txt"]


def test_real_case33_one_day_has_240_distinct_three_minute_snapshots(tmp_path):
    output_root = tmp_path / "real_case33"
    command = [
        sys.executable,
        "experiments/run_paired_fixed_test_risk_scan.py",
        "--systems",
        "case33_3min_final",
        "--pv-scales",
        "0.5",
        "--start-day",
        "1030",
        "--day-count",
        "1",
        "--output-root",
        str(output_root),
    ]

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180, check=False
    )

    assert completed.returncode == 0, completed.stderr
    scenario = output_root / "case33_3min_final" / "pv_scale_0.5"
    with (scenario / "step_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = pd.to_datetime([row["timestamp"] for row in rows])
    assert len(rows) == timestamps.nunique() == 240
    assert timestamps[0].isoformat() == "2014-10-27T06:03:00"
    assert timestamps[-1].isoformat() == "2014-10-27T18:00:00"
    assert set(timestamps[1:] - timestamps[:-1]) == {pd.Timedelta(minutes=3)}
    assert all(float(row["max_abs_reactive_power_mvar"]) == 0.0 for row in rows)

    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    dataset = protocol["dataset_identity"]["case33_3min_final"]
    assert {item["role"] for item in dataset["files"]} == {
        "network",
        "pv_active",
        "load_active",
        "load_reactive",
    }
    assert all(len(item["sha256"]) == 64 for item in dataset["files"])
    assert protocol["environment"]["evaluation_semantics"] == "independent snapshot power flow"
    assert protocol["environment"]["v_lower_pu"] == 0.95
    assert protocol["environment"]["v_upper_pu"] == 1.05
    assert protocol["environment"]["seed"] == 0
    assert protocol["software"]["git_revision"]
    assert len(protocol["software"]["runner_sha256"]) == 64

    summary = json.loads((scenario / "summary.json").read_text(encoding="utf-8"))
    assert summary["day_count"] == 1
    assert summary["step_count"] == 240
    assert summary["first_timestamp"] == "2014-10-27T06:03:00"
    assert summary["last_timestamp"] == "2014-10-27T18:00:00"
