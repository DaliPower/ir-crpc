from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.analyze_budget_diagnostic as budget_analyzer
import experiments.run_risk_aware_multiseed as budget_runner
from experiments.analyze_budget_diagnostic import analyze_budget_diagnostic, write_diagnostic
from experiments.run_risk_aware_multiseed import (
    make_validation_record,
    write_validation_history_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUTPUT = "exp/q1_budget_diagnostic_2000ep_case141"
CANONICAL_ORIGINAL_HISTORY = (
    ROOT
    / "exp/q1_multiseed_1000ep_case141/case141_3min_final/mappo/seed_1/validation_history.json"
)
CANONICAL_DIAGNOSTIC_HISTORY = (
    ROOT
    / CANONICAL_OUTPUT
    / "case141_3min_final/mappo/seed_1/validation_history.json"
)
CANONICAL_DIAGNOSTIC_OUTPUT = ROOT / CANONICAL_OUTPUT / "diagnostic"


def _record(episode: int, violation: float, deviation: float = 0.01, q_loss: float = 0.2):
    return {
        "episode": episode,
        "validation_metrics": {
            "percentage_of_v_out_of_control": violation,
            "average_voltage_deviation": deviation,
            "q_loss": q_loss,
        },
    }


def _history(through_episode: int, *, violation: float = 0.05):
    return [_record(episode, violation) for episode in range(50, through_episode + 1, 50)]


def _run_validation_only(*extra_args: str):
    command = [
        sys.executable,
        "experiments/run_risk_aware_multiseed.py",
        "--scenario", "case141_3min_final",
        "--algorithms", "mappo",
        "--seeds", "1",
        "--train-episodes", "2000",
        "--eval-freq", "50",
        "--pv-scale", "1.5",
        "--max-steps", "240",
        "--eval-start-hour", "6",
        "--validation-start-day", "1000",
        "--validation-days", "30",
        "--test-start-day", "1030",
        "--test-days", "30",
        "--voltage-weight", "1",
        "--violation-weight", "0",
        "--q-weight", "0.1",
        "--save-path", CANONICAL_OUTPUT,
        "--validation-only",
        "--dry-run",
        *extra_args,
    ]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)


def test_validation_only_rejects_noncanonical_output_before_creating_it(tmp_path):
    output_root = tmp_path / "wrong_budget_root"
    command = [
        sys.executable,
        "experiments/run_risk_aware_multiseed.py",
        "--scenario", "case141_3min_final",
        "--algorithms", "mappo",
        "--seeds", "1",
        "--train-episodes", "2000",
        "--eval-freq", "50",
        "--violation-weight", "0",
        "--save-path", str(output_root),
        "--validation-only",
        "--dry-run",
    ]

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)

    assert completed.returncode != 0
    assert "canonical" in completed.stderr
    assert not output_root.exists()


def test_main_validates_canonical_configuration_before_reading_data_or_writing(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "must_not_be_created"
    args = argparse.Namespace(
        validation_only=True,
        aggregate_only=False,
        skip_existing=False,
        save_path=str(output_root),
        scenario="case33_3min_final",
        algorithms=["mappo"],
        seeds=[1],
        train_episodes=2000,
        eval_freq=50,
        pv_scale=1.5,
        max_steps=240,
        eval_start_hour=6,
        validation_start_day=1000,
        validation_days=30,
        test_start_day=1030,
        test_days=30,
        voltage_weight=1.0,
        violation_weight=0.0,
        q_weight=0.1,
    )
    monkeypatch.setattr(budget_runner, "parse_args", lambda: args)

    def fail_if_data_is_read(scenario):
        raise AssertionError("canonical validation ran after data access")

    monkeypatch.setattr(budget_runner, "infer_total_days", fail_if_data_is_read)

    with pytest.raises(ValueError, match="canonical"):
        budget_runner.main()

    assert not output_root.exists()


def test_budget_run_dry_run_writes_only_the_canonical_plan():
    completed = _run_validation_only()

    assert completed.returncode == 0, completed.stderr
    plan_path = ROOT / CANONICAL_OUTPUT / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["runs"] == [
        {
            "scenario": "case141_3min_final",
            "algorithm": "mappo",
            "seed": 1,
            "output_dir": (
                "exp/q1_budget_diagnostic_2000ep_case141/"
                "case141_3min_final/mappo/seed_1"
            ),
        }
    ]
    assert plan["training_protocol"]["validation_episodes"] == list(
        range(50, 2001, 50)
    )
    assert plan["training_protocol"]["initialization"] == (
        "seed initialization only; no checkpoint resume"
    )
    assert plan["training_protocol"]["prohibited_checkpoint_inputs"] == [
        "best_model.pt"
    ]
    assert "checkpoint_selection" not in plan["training_protocol"]
    assert "final_report" not in plan["training_protocol"]
    assert plan["split"]["validation_days"] == list(range(1000, 1030))
    assert plan["split"]["test_days"] == list(range(1030, 1060))
    assert plan["reward_weights"] == {
        "voltage_weight": 1.0,
        "violation_weight": 0.0,
        "q_weight": 0.1,
    }
    assert plan["training_protocol"]["max_steps"] == 240
    assert plan["training_protocol"]["eval_start_hour"] == 6


@pytest.mark.parametrize(
    ("override", "bad_value"),
    [
        ("--scenario", "case33_3min_final"),
        ("--algorithms", "matd3"),
        ("--seeds", "2"),
        ("--train-episodes", "1999"),
        ("--eval-freq", "100"),
        ("--pv-scale", "1.6"),
        ("--max-steps", "120"),
        ("--eval-start-hour", "7"),
        ("--validation-start-day", "999"),
        ("--validation-days", "29"),
        ("--test-start-day", "1029"),
        ("--test-days", "29"),
        ("--voltage-weight", "2"),
        ("--violation-weight", "1"),
        ("--q-weight", "0.2"),
    ],
)
def test_validation_only_rejects_noncanonical_experiment_configuration(
    override, bad_value
):
    completed = _run_validation_only(override, bad_value)

    assert completed.returncode != 0
    assert "canonical" in completed.stderr


@pytest.mark.parametrize("incompatible", ["--aggregate-only", "--skip-existing"])
def test_validation_only_rejects_incompatible_modes(incompatible):
    completed = _run_validation_only(incompatible)

    assert completed.returncode != 0
    assert "incompatible" in completed.stderr


def test_validation_history_record_has_no_checkpoint_selection_semantics():
    metrics = _record(50, 0.02)["validation_metrics"]

    record = make_validation_record(50, metrics)

    assert record == {"episode": 50, "validation_metrics": metrics}


def test_validation_history_atomic_failure_preserves_previous_file(tmp_path, monkeypatch):
    history_path = tmp_path / "validation_history.json"
    previous = [_record(50, 0.03)]
    history_path.write_text(json.dumps(previous), encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("experiments.run_risk_aware_multiseed.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_validation_history_atomic(history_path, [_record(50, 0.02)])

    assert json.loads(history_path.read_text(encoding="utf-8")) == previous


def test_budget_diagnostic_selects_lexicographic_pre_and_post_budget_best():
    original = _history(1000, violation=0.001)
    diagnostic = _history(2000, violation=0.05)
    diagnostic[-1] = _record(2000, 0.01)

    result = analyze_budget_diagnostic(original, diagnostic)

    assert result["pre_1000_best"]["episode"] == 50
    assert result["post_1000_best"]["episode"] == 2000
    assert result["outcome"] == "post_budget_improvement"
    assert result["original_comparison"]["differing_records"] == 20
    assert result["original_comparison"]["identical"] is False


def test_budget_diagnostic_reports_no_improvement_when_post_budget_rank_is_worse():
    original = _history(1000, violation=0.9)
    diagnostic = _history(2000, violation=0.04)
    diagnostic[0] = _record(50, 0.01)
    diagnostic[20] = _record(1050, 0.02)

    result = analyze_budget_diagnostic(original, diagnostic)

    assert result["pre_1000_best"]["episode"] == 50
    assert result["post_1000_best"]["episode"] == 1050
    assert result["outcome"] == "no_post_budget_improvement"


def test_original_comparison_ignores_legacy_checkpoint_selection_field():
    diagnostic = _history(2000)
    original = [
        {**record, "selected": index == 0}
        for index, record in enumerate(diagnostic[:20])
    ]

    result = analyze_budget_diagnostic(original, diagnostic)

    assert result["original_comparison"] == {
        "identical": True,
        "matching_records": 20,
        "differing_records": 0,
        "differing_episodes": [],
    }


def test_budget_diagnostic_requires_complete_50_episode_schedule():
    original = [_record(50, 0.02), _record(1000, 0.03)]
    diagnostic = [*original, _record(1050, 0.02)]

    with pytest.raises(ValueError, match="validation episodes"):
        analyze_budget_diagnostic(original, diagnostic)


@pytest.mark.parametrize(
    ("bad_field", "bad_path"),
    [
        ("original", ROOT / "exp/q1_multiseed_1000ep_case141"),
        ("diagnostic", ROOT / CANONICAL_OUTPUT),
        ("output", ROOT / "exp/q1_multiseed_1000ep_case141"),
        ("output", ROOT / "exp"),
    ],
)
def test_analyzer_rejects_noncanonical_paths_before_read_or_write(
    bad_field, bad_path, monkeypatch
):
    paths = {
        "original": CANONICAL_ORIGINAL_HISTORY,
        "diagnostic": CANONICAL_DIAGNOSTIC_HISTORY,
        "output": CANONICAL_DIAGNOSTIC_OUTPUT,
    }
    paths[bad_field] = bad_path

    def fail_if_read(path):
        raise AssertionError("path validation ran after input read")

    monkeypatch.setattr(budget_analyzer, "_read_history", fail_if_read)

    with pytest.raises(ValueError, match="canonical"):
        write_diagnostic(paths["original"], paths["diagnostic"], paths["output"])


def test_budget_diagnostic_writes_json_csv_and_plot_source_without_changing_inputs(tmp_path):
    original_path = tmp_path / "original.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    original = _history(1000)
    diagnostic = _history(2000)
    original_path.write_text(json.dumps(original), encoding="utf-8")
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

    output_path = tmp_path / "budget" / "diagnostic"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(budget_analyzer, "CANONICAL_ORIGINAL_HISTORY", original_path)
    monkeypatch.setattr(budget_analyzer, "CANONICAL_DIAGNOSTIC_HISTORY", diagnostic_path)
    monkeypatch.setattr(budget_analyzer, "CANONICAL_DIAGNOSTIC_OUTPUT", output_path)
    try:
        result = write_diagnostic(original_path, diagnostic_path, output_path)
    finally:
        monkeypatch.undo()

    assert result["outcome"] == "no_post_budget_improvement"
    assert (output_path / "budget_diagnostic.json").exists()
    assert (output_path / "budget_diagnostic.csv").exists()
    assert (output_path / "budget_diagnostic_plot_source.csv").read_text(
        encoding="utf-8"
    ).count("\n") == 41
    assert json.loads(original_path.read_text(encoding="utf-8")) == original
