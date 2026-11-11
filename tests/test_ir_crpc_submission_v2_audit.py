from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.audit_ir_crpc_submission_v2 import (
    _AuditStop,
    _contained,
    _validate_model_state,
    _validate_result_metric_mapping,
    audit_run_root,
    main,
)
from experiments.fixed_test_protocol import (
    FixedTestResult,
    PHYSICAL_METRICS,
    _summarize_rows,
    write_fixed_test_artifacts,
)
from experiments.ir_crpc_submission_v2 import build_submission_plan

CORE_METRICS = (
    "vrate",
    "cr",
    "destroy",
    "average_voltage_deviation",
    "q_loss",
    "total_line_loss",
    "action_projection_ratio",
    "reactive_capacity_utilization",
    "reactive_saturation_ratio",
)
BUS_COUNTS = {"case33_3min_final": 33, "case141_3min_final": 141}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".replace")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".replace")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    temporary = path.with_suffix(path.suffix + ".replace")
    temporary.write_text(value, encoding=encoding)
    os.replace(temporary, path)


def _preflight_entries() -> tuple[list[dict], list[dict], list[dict]]:
    config_paths = (
        "args/default.yaml",
        "args/env_args/var_voltage_control.yaml",
        "args/alg_args/matd3.yaml",
        "args/alg_args/maddpg.yaml",
        "args/alg_args/mappo.yaml",
        "args/alg_args/facmaddpg.yaml",
    )
    data_paths = tuple(
        f"environments/var_voltage_control/data/{scenario}/{name}"
        for scenario in BUS_COUNTS
        for name in ("model.p", "pv_active.csv", "load_active.csv", "load_reactive.csv")
    )
    runtime_paths = (
        "experiments/ir_crpc_submission_v2.py",
        "experiments/run_risk_aware_multiseed.py",
        "experiments/fixed_test_protocol.py",
        "experiments/risk_aware_checkpointing.py",
        "experiments/submission_provenance.py",
        "environments/var_voltage_control/voltage_control_env.py",
        "experiments/run_q1_fixed_test_traditional_baselines.py",
        "experiments/traditional_action_limits.py",
    )

    def entries(paths: tuple[str, ...]) -> list[dict]:
        return [
            {
                "path": path,
                "size": len(path),
                "sha256": hashlib.sha256(path.encode()).hexdigest(),
            }
            for path in paths
        ]

    return entries(config_paths), entries(data_paths), entries(runtime_paths)


def _trajectory(scenario: str, *, offset: float = 0.0) -> FixedTestResult:
    rows: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    timestamps: list[np.datetime64] = []
    for day_position, day in enumerate(range(1030, 1060)):
        date = np.datetime64("2020-01-01") + np.timedelta64(day_position, "D")
        day_rows: list[dict[str, object]] = []
        for step in range(240):
            timestamp = date + np.timedelta64(6, "h") + np.timedelta64(3 * step, "m")
            row = {
                "day_index": day,
                "day_position": day_position,
                "step_index": step,
                "timestamp": np.datetime_as_string(timestamp, unit="s"),
                "executed": True,
                "vrate": 0.01 + offset,
                "cr": 1.0,
                "destroy": 0.0,
                "percentage_of_v_out_of_control": 0.01 + offset,
                "totally_controllable_ratio": 1.0,
                "average_voltage_deviation": 0.02 + offset,
                "q_loss": 0.20 + offset,
                "total_line_loss": 0.30 + offset,
                "action_projection_ratio": 0.10,
                "reactive_capacity_utilization": 0.50,
                "reactive_saturation_ratio": 0.0,
                "inverter_p_mw": json.dumps([0.6, 0.6]),
                "inverter_q_mvar": json.dumps([0.3, -0.3]),
                "inverter_s_max_mva": json.dumps([1.0, 1.0]),
            }
            rows.append(row)
            day_rows.append(row)
            timestamps.append(timestamp)
        day_summary = _summarize_rows(day_rows)
        day_summary.update(
            {
                "day_index": day,
                "day_position": day_position,
                "date": str(date),
                "failure_step": None,
            }
        )
        daily.append(day_summary)
    summary = _summarize_rows(rows)
    summary.update(
        {
            "planned_days": 30,
            "completed_days": 30,
            "failed_days": 0,
            "start_day": 1030,
            "end_day": 1059,
            "points_per_day": 240,
            "start_hour": 6,
            "cadence_minutes": 3,
            "bus_order": list(range(BUS_COUNTS[scenario])),
            "physical_metrics_denominator": "executed finite transitions only",
            "safety_metrics_denominator": "all planned transitions",
        }
    )
    return FixedTestResult(
        summary=summary,
        step_rows=rows,
        daily_rows=daily,
        node_voltages=np.full((7200, BUS_COUNTS[scenario]), 1.0, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
        executed_mask=np.ones(7200, dtype=bool),
    )


def _provenance(
    *,
    plan_hash: str,
    scenario: str,
    solver: str,
    run_dir: Path,
    config_entries: list[dict],
    data_entries: list[dict],
) -> dict:
    config_names = {
        "args/default.yaml",
        "args/env_args/var_voltage_control.yaml",
        f"args/alg_args/{solver}.yaml",
    }
    data_prefix = f"environments/var_voltage_control/data/{scenario}/"

    def convert(item: dict) -> dict:
        return {
            "path": item["path"],
            "relative_to": "repo_root",
            "bytes": item["size"],
            "sha256": item["sha256"],
        }

    return {
        "schema_version": "1.0",
        "protocol_id": "ir-crpc-submission-v2-strict",
        "git": {"revision": "fixture-revision", "dirty": False},
        "run_plan_sha256": plan_hash,
        "checkpoints": {
            "best": {"episode": 950, "role": "validation-selected"},
            "final": {"episode": 1000, "role": "terminal-training-state"},
        },
        "files": {
            "config": [
                convert(item) for item in config_entries if item["path"] in config_names
            ],
            "data": [
                convert(item)
                for item in data_entries
                if item["path"].startswith(data_prefix)
            ],
            "weights": [
                {
                    "path": name,
                    "relative_to": "run_dir",
                    "bytes": (run_dir / name).stat().st_size,
                    "sha256": _sha(run_dir / name),
                    "role": role,
                    "episode": episode,
                }
                for name, role, episode in (
                    ("best_model.pt", "validation-selected", 950),
                    ("final_model.pt", "terminal-training-state", 1000),
                )
            ],
        },
    }


def _fixed_bundle(directory: Path, scenario: str, offset: float) -> dict[str, str]:
    return write_fixed_test_artifacts(
        _trajectory(scenario, offset=offset), directory, "fixed_test"
    )


def _persisted_test_metrics(directory: Path) -> dict[str, float]:
    summary = json.loads((directory / "fixed_test_summary.json").read_text("utf-8"))
    metrics = {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for key in PHYSICAL_METRICS:
        output = "mean_reward" if key == "reward" else key
        if output not in metrics and summary.get(output) is None:
            metrics[output] = float("nan")
    return metrics


def _build_complete_root(root: Path) -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)
    plan = build_submission_plan(root)
    root.mkdir()
    _write_json(root / "run_plan.json", plan)
    _write_text(root / "run_plan.sha256", plan["plan_sha256"] + "\n", encoding="ascii")
    config_entries, data_entries, runtime_entries = _preflight_entries()
    preflight_payload = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "git": {"revision": "fixture-revision", "dirty_paths": []},
        "configuration_files": config_entries,
        "dataset_files": data_entries,
        "runtime_files": runtime_entries,
    }
    preflight = {
        **preflight_payload,
        "preflight_sha256": _canonical_hash(preflight_payload),
    }
    _write_json(root / "execution_preflight.json", preflight)

    completed: list[dict] = []
    for run_index, run in enumerate(plan["runs"]):
        run_dir = root / run["output_dir"]
        run_dir.mkdir(parents=True)
        torch.save(
            {
                "model_state_dict": {"linear.weight": torch.tensor([1.0])},
                "episode": 950,
                "validation_metrics": {
                    "percentage_of_v_out_of_control": 0.01,
                    "average_voltage_deviation": 0.02,
                    "q_loss": 0.2,
                },
            },
            run_dir / "best_model.pt",
        )
        torch.save(
            {
                "model_state_dict": {"linear.weight": torch.tensor([2.0])},
                "episode": 1000,
            },
            run_dir / "final_model.pt",
        )
        history = [
            {
                "episode": episode,
                "validation_metrics": {
                    "percentage_of_v_out_of_control": (
                        0.01 if episode == 950 else 0.02
                    ),
                    "average_voltage_deviation": 0.02,
                    "q_loss": 0.2,
                },
                "selected": episode in {50, 950},
            }
            for episode in range(50, 1001, 50)
        ]
        _write_json(run_dir / "validation_history.json", history)
        _fixed_bundle(
            run_dir / "test_best",
            run["scenario"],
            offset=run_index * 1e-6,
        )
        if run["evaluate_terminal"]:
            _fixed_bundle(
                run_dir / "test_final",
                run["scenario"],
                offset=run_index * 1e-6 + 0.001,
            )
        best_test_metrics = _persisted_test_metrics(run_dir / "test_best")
        final_test_metrics = (
            _persisted_test_metrics(run_dir / "test_final")
            if run["evaluate_terminal"]
            else None
        )
        identity = {
            "run_id": run["run_id"],
            "variant_id": run["variant_id"],
            "scenario": run["scenario"],
            "algorithm": run["solver"],
            "seed": run["seed"],
        }
        _write_json(
            run_dir / "seed_result_best.json",
            {
                "status": "completed",
                **identity,
                "checkpoint": "best",
                "evaluated": True,
                "episode": 950,
                "validation_metrics": history[18]["validation_metrics"],
                "test_metrics": best_test_metrics,
            },
        )
        _write_json(
            run_dir / "seed_result_final.json",
            {
                "status": "completed",
                **identity,
                "checkpoint": "final",
                "evaluated": bool(run["evaluate_terminal"]),
                "episode": 1000,
                "validation_metrics": None,
                "test_metrics": final_test_metrics,
                **(
                    {}
                    if run["evaluate_terminal"]
                    else {
                        "evaluation_note": (
                            "terminal fixed testing was not registered for this run"
                        )
                    }
                ),
            },
        )
        provenance = _provenance(
            plan_hash=plan["plan_sha256"],
            scenario=run["scenario"],
            solver=run["solver"],
            run_dir=run_dir,
            config_entries=config_entries,
            data_entries=data_entries,
        )
        _write_json(run_dir / "provenance.json", provenance)
        artifacts = {
            path.relative_to(run_dir).as_posix(): _sha(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"completion_manifest.json", "seed_result.json"}
        }
        manifest = {
            "schema_version": 2,
            "status": "completed",
            "run_id": run["run_id"],
            "plan_sha256": plan["plan_sha256"],
            "artifacts": artifacts,
        }
        _write_json(run_dir / "completion_manifest.json", manifest)
        _write_json(
            run_dir / "seed_result.json",
            {
                "status": "completed",
                **identity,
                "selected_episode": 950,
                "terminal_episode": 1000,
                "evaluate_terminal": bool(run["evaluate_terminal"]),
                "validation_metrics": history[18]["validation_metrics"],
                "test_metrics": best_test_metrics,
                "reward_weights": {
                    "voltage_weight": 1.0,
                    "violation_weight": 0.0,
                    "q_weight": 0.1,
                },
                **(
                    {"terminal_test_metrics": final_test_metrics}
                    if run["evaluate_terminal"]
                    else {}
                ),
            },
        )
        completed.append(
            {
                "run_id": run["run_id"],
                "completion_manifest": f"{run['output_dir']}/completion_manifest.json",
                "completion_manifest_sha256": _sha(
                    run_dir / "completion_manifest.json"
                ),
            }
        )

    _write_json(
        root / "resume_metadata.json",
        {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "execution_state": "completed",
            "git_snapshot": {"revision": "fixture-revision", "dirty": False},
            "completed_runs": completed,
            "failed_runs": [],
            "pending_run_ids": [],
        },
    )

    preflight_map = {
        item["path"]: item for item in config_entries + data_entries + runtime_entries
    }
    for scenario_index, scenario in enumerate(BUS_COUNTS):
        for method_index, method in enumerate(("droop", "opf")):
            method_dir = root / "traditional" / scenario / method
            paths = _fixed_bundle(
                method_dir,
                scenario,
                offset=0.01 + scenario_index * 0.001 + method_index * 0.0001,
            )
            inputs = []
            for path in (
                *(
                    f"environments/var_voltage_control/data/{scenario}/{name}"
                    for name in (
                        "model.p",
                        "pv_active.csv",
                        "load_active.csv",
                        "load_reactive.csv",
                    )
                ),
                "args/env_args/var_voltage_control.yaml",
                "experiments/run_q1_fixed_test_traditional_baselines.py",
                "experiments/traditional_action_limits.py",
                "experiments/fixed_test_protocol.py",
                "environments/var_voltage_control/voltage_control_env.py",
            ):
                source = preflight_map.get(path)
                inputs.append(
                    {
                        "role": Path(path).stem,
                        "path": path,
                        "absolute_path": str(
                            Path(__file__).resolve().parents[1] / path
                        ),
                        "size_bytes": source["size"] if source else len(path),
                        "sha256": (
                            source["sha256"]
                            if source
                            else hashlib.sha256(path.encode()).hexdigest()
                        ),
                    }
                )
            _write_json(
                method_dir / "method_metadata.json",
                {
                    "schema_version": 2,
                    "method": method,
                    "selected_controller": method,
                    "scenario": scenario,
                    "pv_scale": 1.5,
                    "test_days": list(range(1030, 1060)),
                    "max_steps": 240,
                    "start_hour": 6,
                    "cadence_minutes": 3,
                    "reset_action": False,
                    "add_noise": False,
                    "action_mapping": "capacity_adaptive",
                    "action_scale": 0.8 if scenario.startswith("case33") else 0.6,
                    "action_bias": 0.0,
                    "normalized_action_bounds": (
                        [-0.8, 0.8] if scenario.startswith("case33") else [-0.6, 0.6]
                    ),
                    "zero_initial_q": True,
                    "control_protocol": {
                        "zero_initial_q": True,
                        "action_scale": (0.8 if scenario.startswith("case33") else 0.6),
                        "action_bias": 0.0,
                        "normalized_action_bounds": (
                            [-0.8, 0.8]
                            if scenario.startswith("case33")
                            else [-0.6, 0.6]
                        ),
                        "trajectory_schema": "fixed_test_trajectory_schema.json",
                        "step_metrics": "fixed_test_step_metrics.csv",
                    },
                    "strict_shape": True,
                    "smoke": False,
                    "summary": json.loads(
                        (method_dir / "fixed_test_summary.json").read_text("utf-8")
                    ),
                    "artifact_sha256": {
                        name: _sha(Path(path)) for name, path in paths.items()
                    },
                    "provenance": {
                        "git": {"revision": "fixture-revision", "dirty": False},
                        "runtime": {
                            "python": "fixture",
                            "numpy": "fixture",
                            "pandapower": "fixture",
                        },
                        "inputs": inputs,
                    },
                },
            )
    return root


@pytest.fixture(scope="module")
def complete_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_complete_root(
        tmp_path_factory.mktemp("audit-fixture") / "q1_submission_v2_strict"
    )


def _clone(complete_root: Path, tmp_path: Path) -> Path:
    target = tmp_path / "q1_submission_v2_strict"
    shutil.copytree(complete_root, target, copy_function=os.link)
    return target


def _first_run(root: Path) -> tuple[dict, Path]:
    plan = json.loads((root / "run_plan.json").read_text("utf-8"))
    run = plan["runs"][0]
    return run, root / run["output_dir"]


def _refresh_run_integrity(root: Path, run: dict, run_dir: Path) -> None:
    manifest_path = run_dir / "completion_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"] = {
        path.relative_to(run_dir).as_posix(): _sha(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name not in {"completion_manifest.json", "seed_result.json"}
    }
    _write_json(manifest_path, manifest)
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    for entry in metadata["completed_runs"]:
        if entry["run_id"] == run["run_id"]:
            entry["completion_manifest_sha256"] = _sha(manifest_path)
    _write_json(metadata_path, metadata)


def _traditional_dir(root: Path) -> Path:
    return root / "traditional/case33_3min_final/droop"


def _refresh_traditional_metadata(directory: Path) -> None:
    path = directory / "method_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    mapping = {
        "step_metrics": "fixed_test_step_metrics.csv",
        "daily_metrics": "fixed_test_daily_metrics.csv",
        "node_voltages": "fixed_test_node_voltages.npz",
        "trajectory_schema": "fixed_test_trajectory_schema.json",
        "summary": "fixed_test_summary.json",
    }
    metadata["artifact_sha256"] = {
        key: _sha(directory / name) for key, name in mapping.items()
    }
    metadata["summary"] = json.loads(
        (directory / "fixed_test_summary.json").read_text("utf-8")
    )
    _write_json(path, metadata)


def _replace_checkpoint_payload(
    root: Path, run: dict, run_dir: Path, name: str, payload: object
) -> None:
    temporary = run_dir / f"{name}.replace"
    torch.save(payload, temporary)
    os.replace(temporary, run_dir / name)
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text("utf-8"))
    for record in provenance["files"]["weights"]:
        if record["path"] == name:
            record["sha256"] = _sha(run_dir / name)
            record["bytes"] = (run_dir / name).stat().st_size
    _write_json(provenance_path, provenance)
    _refresh_run_integrity(root, run, run_dir)


def _assert_error(report, code: str) -> None:
    assert report.status == "FAIL"
    assert code in report.error_codes, report.errors


def test_complete_36_run_and_traditional_fixture_passes_and_recomputes_seed_sd(
    complete_root,
):
    report = audit_run_root(complete_root)
    assert report.status == "PASS", report.errors
    assert report.training_run_count == 36
    assert report.traditional_run_count == 4
    rows = [
        row
        for row in report.summary_long
        if row["system"] == "case33_3min_final"
        and row["variant"] == "full_matd3"
        and row["checkpoint"] == "best"
        and row["metric"] == "vrate"
    ]
    assert {row["statistic"] for row in rows} == {"seed", "mean", "sample_sd"}
    assert sum(row["statistic"] == "seed" for row in rows) == 3


def test_missing_seed_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    plan = json.loads((root / "run_plan.json").read_text("utf-8"))
    run = next(item for item in plan["runs"] if item["seed"] == 2)
    shutil.rmtree(root / run["output_dir"])
    _assert_error(audit_run_root(root), "MISSING_RUN")


def test_nineteen_validation_candidates_are_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    history_path = run_dir / "validation_history.json"
    history = json.loads(history_path.read_text("utf-8"))[:-1]
    _write_json(history_path, history)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "VALIDATION_COUNT")


@pytest.mark.parametrize("group", ["config", "data"])
def test_changed_config_or_data_hash_is_rejected(complete_root, tmp_path, group):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text("utf-8"))
    provenance["files"][group][0]["sha256"] = "f" * 64
    _write_json(provenance_path, provenance)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "PROVENANCE_HASH")


def test_duplicate_timestamp_is_rejected_even_with_refreshed_manifest(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rows[1]["timestamp"] = rows[0]["timestamp"]
    _write_csv(path, rows)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "TIMESTAMP_GRID")


def test_noncanonical_executed_value_is_rejected_even_if_truthy(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rows[0]["executed"] = "1"
    _write_csv(path, rows)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "ROW_SCHEMA")


def test_7199_rows_are_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))[:-1]
    _write_csv(path, rows)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "STEP_COUNT")


def test_unknown_variant_in_plan_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    plan_path = root / "run_plan.json"
    plan = json.loads(plan_path.read_text("utf-8"))
    plan["runs"][0]["variant_id"] = "invented_variant"
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = _canonical_hash(payload)
    _write_json(plan_path, plan)
    _write_text(root / "run_plan.sha256", plan["plan_sha256"] + "\n", encoding="ascii")
    _assert_error(audit_run_root(root), "PLAN_MATRIX")


def test_missing_retained_weight_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    _, run_dir = _first_run(root)
    (run_dir / "final_model.pt").unlink()
    _assert_error(audit_run_root(root), "MISSING_WEIGHT")


@pytest.mark.parametrize("mutation", ["deleted", "garbage", "all_nan"])
def test_nonfac_required_operating_metrics_are_strict(
    complete_root, tmp_path, mutation
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if mutation == "deleted":
        for row in rows:
            row.pop("average_voltage_deviation")
    elif mutation == "garbage":
        rows[0]["q_loss"] = "not-a-number"
    else:
        for row in rows:
            row["total_line_loss"] = "nan"
    _write_csv(path, rows)
    _refresh_run_integrity(root, run, run_dir)
    expected = "ROW_SCHEMA" if mutation == "deleted" else "OPERATING_DENOMINATOR"
    _assert_error(audit_run_root(root), expected)


def test_zero_operating_denominator_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    summary_path = run_dir / "test_best/fixed_test_summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["q_loss"] = None
    summary["metric_denominators"]["q_loss"] = 0
    _write_json(summary_path, summary)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "SUMMARY_MISMATCH")


def test_traditional_executed_operating_metric_must_be_finite(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    step_path = directory / "fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(step_path.open("r", encoding="utf-8", newline="")))
    rows[0]["average_voltage_deviation"] = "nan"
    _write_csv(step_path, rows)
    _refresh_traditional_metadata(directory)
    _assert_error(audit_run_root(root), "OPERATING_DENOMINATOR")


@pytest.mark.parametrize("state", ["running", "paused_cuda_oom", "aborted"], ids=str)
def test_noncompleted_execution_states_are_rejected(complete_root, tmp_path, state):
    root = _clone(complete_root, tmp_path)
    path = root / "resume_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["execution_state"] = state
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "EXECUTION_STATE")


def test_pending_run_and_git_snapshot_mismatch_are_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    path = root / "resume_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["pending_run_ids"] = [metadata["completed_runs"][0]["run_id"]]
    metadata["git_snapshot"]["revision"] = "other"
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "EXECUTION_STATE")


def test_lexicographic_validation_best_must_match_all_persisted_selection_evidence(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "validation_history.json"
    history = json.loads(path.read_text("utf-8"))
    history[17]["validation_metrics"]["percentage_of_v_out_of_control"] = 0.001
    _write_json(path, history)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "CHECKPOINT_RESULT")


def test_best_model_payload_episode_must_match_selection(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    payload = torch.load(run_dir / "best_model.pt", weights_only=True)
    payload["episode"] = 900
    temporary = run_dir / "best_model.pt.replace"
    torch.save(payload, temporary)
    os.replace(temporary, run_dir / "best_model.pt")
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text("utf-8"))
    provenance["files"]["weights"][0]["sha256"] = _sha(run_dir / "best_model.pt")
    _write_json(provenance_path, provenance)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "CHECKPOINT_SELECTION")


@pytest.mark.parametrize(
    "state",
    [
        pytest.param("missing", id="missing-state-dict"),
        pytest.param("nonmapping", id="nonmapping-state-dict"),
        pytest.param("empty", id="empty-state-dict"),
        pytest.param("metadata_only", id="metadata-only-state-dict"),
    ],
)
def test_checkpoint_requires_nonempty_parameter_state_mapping(
    complete_root, tmp_path, state
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    payload = torch.load(run_dir / "best_model.pt", weights_only=True)
    if state == "missing":
        payload.pop("model_state_dict")
    elif state == "nonmapping":
        payload["model_state_dict"] = "not-a-mapping"
    elif state == "empty":
        payload["model_state_dict"] = {}
    else:
        payload["model_state_dict"] = {"layer": {"label": "metadata"}}
    _replace_checkpoint_payload(root, run, run_dir, "best_model.pt", payload)
    _assert_error(audit_run_root(root), "CHECKPOINT_STATE")


@pytest.mark.parametrize(
    "state",
    [
        {"epoch": 950},
        {"metadata": np.asarray([950])},
        {"empty": torch.empty(0)},
        {"nested": [1, 2, 3]},
    ],
    ids=("numeric-shell", "array-shell", "empty-tensor", "numeric-list-shell"),
)
def test_model_state_rejects_nontensor_metadata_shells(state, tmp_path):
    with pytest.raises(_AuditStop) as captured:
        _validate_model_state(
            {"model_state_dict": state}, path=tmp_path / "checkpoint.pt"
        )
    assert captured.value.issue.code == "CHECKPOINT_STATE"


def test_model_state_accepts_nested_nonempty_tensor(tmp_path):
    _validate_model_state(
        {"model_state_dict": {"actor": {"weight": torch.ones(1)}}},
        path=tmp_path / "checkpoint.pt",
    )


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("seed_result_best.json", "missing_status"),
        ("seed_result_best.json", "swapped_run_id"),
        ("seed_result_best.json", "validation_metrics"),
        ("seed_result_best.json", "test_metrics"),
        ("seed_result_final.json", "test_metrics"),
        ("seed_result.json", "identity"),
        ("seed_result.json", "validation_metrics"),
        ("seed_result.json", "test_metrics"),
        ("seed_result.json", "terminal_test_metrics"),
        ("seed_result_best.json", "numeric_string"),
        ("seed_result_final.json", "boolean_metric"),
        ("seed_result.json", "corrupted_metric"),
        ("seed_result.json", "none_terminal_metric"),
    ],
)
def test_seed_result_evidence_must_match_identity_selection_and_raw_bundles(
    complete_root, tmp_path, filename, mutation
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / filename
    value = json.loads(path.read_text("utf-8"))
    if mutation == "missing_status":
        value.pop("status")
    elif mutation == "swapped_run_id":
        value["run_id"] = "full_matd3__case33_3min_final__seed_1"
    elif mutation == "validation_metrics":
        value["validation_metrics"]["q_loss"] += 1e-6
    elif mutation == "identity":
        value["seed"] = 1
    elif mutation == "terminal_test_metrics":
        value["terminal_test_metrics"]["vrate"] += 1e-6
    elif mutation == "numeric_string":
        value["test_metrics"]["planned_steps"] = "7200"
    elif mutation == "boolean_metric":
        value["test_metrics"]["vrate"] = True
    elif mutation == "corrupted_metric":
        value["test_metrics"]["vrate"] = "corrupted"
    elif mutation == "none_terminal_metric":
        value["terminal_test_metrics"]["q_loss"] = None
    else:
        value["test_metrics"]["vrate"] += 1e-6
    _write_json(path, value)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "CHECKPOINT_RESULT")


@pytest.mark.parametrize("invalid", ["nan", "corrupted", None, True, "7200"])
def test_nan_result_metric_accepts_only_actual_float_nan(invalid, tmp_path):
    with pytest.raises(_AuditStop) as captured:
        _validate_result_metric_mapping(
            {"q_loss": invalid},
            {"q_loss": float("nan")},
            label="fac.test_metrics",
            path=tmp_path / "seed_result.json",
        )
    assert captured.value.issue.code == "CHECKPOINT_RESULT"


def test_nan_result_metric_accepts_real_float_nan(tmp_path):
    _validate_result_metric_mapping(
        {"q_loss": float("nan")},
        {"q_loss": float("nan")},
        label="fac.test_metrics",
        path=tmp_path / "seed_result.json",
    )


def test_final_checkpoint_requires_real_model_state(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    payload = torch.load(run_dir / "final_model.pt", weights_only=True)
    payload["model_state_dict"] = {}
    _replace_checkpoint_payload(root, run, run_dir, "final_model.pt", payload)
    _assert_error(audit_run_root(root), "CHECKPOINT_STATE")


def test_final_model_payload_must_be_terminal_episode(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    payload = torch.load(run_dir / "final_model.pt", weights_only=True)
    payload["episode"] = 999
    temporary = run_dir / "final_model.pt.replace"
    torch.save(payload, temporary)
    os.replace(temporary, run_dir / "final_model.pt")
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text("utf-8"))
    provenance["files"]["weights"][1]["sha256"] = _sha(run_dir / "final_model.pt")
    _write_json(provenance_path, provenance)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "CHECKPOINT_SELECTION")


def test_provenance_best_episode_must_match_recomputed_selection(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "provenance.json"
    provenance = json.loads(path.read_text("utf-8"))
    provenance["checkpoints"]["best"]["episode"] = 900
    _write_json(path, provenance)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "PROVENANCE_HASH")


def test_extra_numeric_evidence_tree_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    (root / "full_matd3/case33_3min_final/seed_99").mkdir()
    _assert_error(audit_run_root(root), "EXTRA_EVIDENCE")


def test_extra_numeric_file_inside_registered_seed_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    _, run_dir = _first_run(root)
    (run_dir / "metrics_10000.csv").write_text("invented\n", encoding="utf-8")
    _assert_error(audit_run_root(root), "EXTRA_EVIDENCE")


def test_preflight_must_freeze_both_traditional_sources(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    path = root / "execution_preflight.json"
    preflight = json.loads(path.read_text("utf-8"))
    preflight["runtime_files"] = [
        item
        for item in preflight["runtime_files"]
        if item["path"] != "experiments/traditional_action_limits.py"
    ]
    payload = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    preflight["preflight_sha256"] = _canonical_hash(payload)
    _write_json(path, preflight)
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["preflight_sha256"] = preflight["preflight_sha256"]
    _write_json(metadata_path, metadata)
    _assert_error(audit_run_root(root), "PREFLIGHT_HASH")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_scale", 0.6),
        ("action_bias", 0.1),
        ("normalized_action_bounds", [-1.0, 1.0]),
        ("zero_initial_q", False),
    ],
)
def test_traditional_action_contract_is_exact(complete_root, tmp_path, field, value):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    path = directory / "method_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata[field] = value
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "TRADITIONAL_METADATA")


def test_traditional_provenance_absolute_path_escape_is_rejected(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    path = directory / "method_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["provenance"]["inputs"][0]["absolute_path"] = str(tmp_path / "outside.csv")
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "PATH_ESCAPE")


def test_traditional_metadata_must_record_both_runtime_sources(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    path = directory / "method_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["provenance"]["inputs"] = [
        item
        for item in metadata["provenance"]["inputs"]
        if item["path"] != "experiments/traditional_action_limits.py"
    ]
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "PROVENANCE_HASH")


def test_traditional_protocol_must_link_to_step_and_schema_evidence(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    path = directory / "method_metadata.json"
    metadata = json.loads(path.read_text("utf-8"))
    metadata["control_protocol"]["step_metrics"] = "elsewhere.csv"
    _write_json(path, metadata)
    _assert_error(audit_run_root(root), "TRADITIONAL_METADATA")


def test_traditional_symlink_or_junction_bundle_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    directory = _traditional_dir(root)
    external = tmp_path / "external-droop"
    shutil.copytree(directory, external)
    shutil.rmtree(directory)
    try:
        os.symlink(external, directory, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable on this Windows host: {exc}")
    _assert_error(audit_run_root(root), "PATH_ESCAPE")


def test_resolved_traditional_path_escape_is_rejected(monkeypatch, tmp_path):
    root = tmp_path / "formal"
    root.mkdir()
    requested = root / "traditional/case33_3min_final/droop"
    requested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = Path.resolve

    def fake_resolve(path: Path, *args, **kwargs):
        if path.absolute() == requested.absolute():
            return outside.absolute()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(_AuditStop) as captured:
        _contained(root, "traditional/case33_3min_final/droop")
    assert captured.value.issue.code == "PATH_ESCAPE"


def test_extra_traditional_method_directory_is_rejected(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    (root / "traditional/case33_3min_final/invented").mkdir()
    _assert_error(audit_run_root(root), "EXTRA_EVIDENCE")


@pytest.mark.parametrize(
    ("active", "reactive"),
    [
        ([0.6, 0.6], [0.9, -0.3]),
        ([1.01, 0.6], [0.0, -0.3]),
    ],
    ids=("q-over-capacity", "active-power-over-rating"),
)
def test_physical_inverter_capability_violation_is_rejected(
    complete_root, tmp_path, active, reactive
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    rows[0]["inverter_p_mw"] = json.dumps(active)
    rows[0]["inverter_q_mvar"] = json.dumps(reactive)
    _write_csv(path, rows)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "REACTIVE_CAPACITY")


def test_tampered_stored_summary_is_rejected_after_hashes_are_refreshed(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    run, run_dir = _first_run(root)
    path = run_dir / "test_best/fixed_test_summary.json"
    summary = json.loads(path.read_text("utf-8"))
    summary["vrate"] += 1e-6
    _write_json(path, summary)
    _refresh_run_integrity(root, run, run_dir)
    _assert_error(audit_run_root(root), "SUMMARY_MISMATCH")


def test_fac_failure_keeps_safety_but_disables_equal_time_ranking(
    complete_root, tmp_path
):
    root = _clone(complete_root, tmp_path)
    plan = json.loads((root / "run_plan.json").read_text("utf-8"))
    run = next(item for item in plan["runs"] if item["variant_id"] == "full_facmaddpg")
    run_dir = root / run["output_dir"]
    step_path = run_dir / "test_best/fixed_test_step_metrics.csv"
    rows = list(csv.DictReader(step_path.open("r", encoding="utf-8", newline="")))
    rows[239].update(
        {
            "executed": "false",
            "vrate": "1.0",
            "cr": "0.0",
            "destroy": "1.0",
            "percentage_of_v_out_of_control": "1.0",
            "totally_controllable_ratio": "0.0",
            "average_voltage_deviation": "nan",
            "q_loss": "nan",
            "total_line_loss": "nan",
            "action_projection_ratio": "nan",
            "reactive_capacity_utilization": "nan",
            "reactive_saturation_ratio": "nan",
            "inverter_p_mw": "",
            "inverter_q_mvar": "",
            "inverter_s_max_mva": "",
        }
    )
    for row in rows:
        row["q_loss"] = "nan"
    _write_csv(step_path, rows)
    daily_path = run_dir / "test_best/fixed_test_daily_metrics.csv"
    daily = list(csv.DictReader(daily_path.open("r", encoding="utf-8", newline="")))
    day_denominators = ast.literal_eval(daily[0]["metric_denominators"])
    original_day_vrate = float(daily[0]["vrate"])
    original_day_cr = float(daily[0]["cr"])
    original_day_destroy = float(daily[0]["destroy"])
    daily[0]["executed_steps"] = 239
    daily[0]["failed_steps"] = 1
    daily[0]["vrate"] = (original_day_vrate * 239 + 1.0) / 240
    daily[0]["percentage_of_v_out_of_control"] = daily[0]["vrate"]
    daily[0]["cr"] = original_day_cr * 239 / 240
    daily[0]["totally_controllable_ratio"] = daily[0]["cr"]
    daily[0]["destroy"] = (original_day_destroy * 239 + 1.0) / 240
    for key, denominator in tuple(day_denominators.items()):
        if (
            key
            not in {
                "vrate",
                "cr",
                "destroy",
                "percentage_of_v_out_of_control",
                "totally_controllable_ratio",
            }
            and denominator == 240
        ):
            day_denominators[key] = 239
    day_denominators["q_loss"] = 0
    daily[0]["q_loss"] = "nan"
    daily[0]["metric_denominators"] = json.dumps(day_denominators)
    daily[0]["failure_step"] = 239
    for later_day in daily[1:]:
        later_denominators = ast.literal_eval(later_day["metric_denominators"])
        later_denominators["q_loss"] = 0
        later_day["metric_denominators"] = json.dumps(later_denominators)
        later_day["q_loss"] = "nan"
    _write_csv(daily_path, daily)
    summary_path = run_dir / "test_best/fixed_test_summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    original_vrate = float(summary["vrate"])
    original_cr = float(summary["cr"])
    original_destroy = float(summary["destroy"])
    summary["executed_steps"] = 7199
    summary["failed_steps"] = 1
    summary["vrate"] = (original_vrate * 7199 + 1.0) / 7200
    summary["percentage_of_v_out_of_control"] = summary["vrate"]
    summary["cr"] = original_cr * 7199 / 7200
    summary["totally_controllable_ratio"] = summary["cr"]
    summary["destroy"] = (original_destroy * 7199 + 1.0) / 7200
    for key, denominator in tuple(summary["metric_denominators"].items()):
        if (
            key
            not in {
                "vrate",
                "cr",
                "destroy",
                "percentage_of_v_out_of_control",
                "totally_controllable_ratio",
            }
            and denominator == 7200
        ):
            summary["metric_denominators"][key] = 7199
    summary["q_loss"] = float("nan")
    summary["metric_denominators"]["q_loss"] = 0
    summary["completed_days"] = 29
    summary["failed_days"] = 1
    _write_json(summary_path, summary)
    schema_path = run_dir / "test_best/fixed_test_trajectory_schema.json"
    schema = json.loads(schema_path.read_text("utf-8"))
    schema["executed_count"] = 7199
    schema["failed_count"] = 1
    schema["metric_denominators"] = summary["metric_denominators"]
    _write_json(schema_path, schema)
    npz_path = run_dir / "test_best/fixed_test_node_voltages.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        payload = {key: arrays[key].copy() for key in arrays.files}
    payload["executed_mask"][239] = False
    payload["node_voltages"][239, :] = np.nan
    temporary = npz_path.with_suffix(".replace.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, npz_path)
    expected_test_metrics = _persisted_test_metrics(run_dir / "test_best")
    best_result_path = run_dir / "seed_result_best.json"
    best_result = json.loads(best_result_path.read_text("utf-8"))
    best_result["test_metrics"] = expected_test_metrics
    _write_json(best_result_path, best_result)
    marker_path = run_dir / "seed_result.json"
    marker = json.loads(marker_path.read_text("utf-8"))
    marker["test_metrics"] = expected_test_metrics
    _write_json(marker_path, marker)
    _refresh_run_integrity(root, run, run_dir)

    report = audit_run_root(root)
    assert report.status == "PASS", report.errors
    assert report.main_equal_time_rank_eligible is False
    assert report.fac_failed_steps
    fac_q = [
        row
        for row in report.summary_long
        if row["variant"] == "full_facmaddpg"
        and row["metric"] == "q_loss"
        and row["statistic"] == "seed"
        and row["seed"] == run["seed"]
        and row["system"] == run["scenario"]
    ]
    assert fac_q == []
    fac_voltage = [
        row
        for row in report.summary_long
        if row["variant"] == "full_facmaddpg"
        and row["metric"] == "average_voltage_deviation"
        and row["statistic"] == "seed"
        and row["seed"] == run["seed"]
        and row["system"] == run["scenario"]
    ]
    assert len(fac_voltage) == 1
    assert fac_voltage[0]["operating_denominator"] == 7199


def test_failed_cli_writes_reports_but_no_plot_ready_csv(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    _, run_dir = _first_run(root)
    (run_dir / "final_model.pt").unlink()
    assert main(["--root", str(root)]) == 1
    audit_dir = root / "audit"
    assert (audit_dir / "audit_report.json").is_file()
    assert (audit_dir / "audit_report.md").is_file()
    assert not (audit_dir / "summary_long.csv").exists()
    assert not (audit_dir / "daily_long.csv").exists()
    assert not (audit_dir / "time_of_day_long.csv").exists()


def test_passing_cli_transactionally_writes_five_audit_files(complete_root, tmp_path):
    root = _clone(complete_root, tmp_path)
    assert main(["--root", str(root)]) == 0
    assert {path.name for path in (root / "audit").iterdir()} == {
        "audit_report.json",
        "audit_report.md",
        "summary_long.csv",
        "daily_long.csv",
        "time_of_day_long.csv",
    }
