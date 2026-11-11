from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path

import pytest

from experiments.ir_crpc_submission_v2 import (
    FORMAL_OUTPUT_RELATIVE,
    build_submission_plan,
    build_submission_runs,
    main,
    submission_variants,
    validate_execution_root,
    validate_formal_output_root,
)


EXPECTED_VARIANTS = {
    "full_matd3": ("matd3", 1.5, 1.5, "capacity_adaptive", True),
    "nominal_train_matd3": (
        "matd3",
        1.0,
        1.5,
        "capacity_adaptive",
        False,
    ),
    "static_projected_matd3": (
        "matd3",
        1.5,
        1.5,
        "static_projected",
        False,
    ),
    "full_maddpg": ("maddpg", 1.5, 1.5, "capacity_adaptive", False),
    "full_mappo": ("mappo", 1.5, 1.5, "capacity_adaptive", False),
    "full_facmaddpg": (
        "facmaddpg",
        1.5,
        1.5,
        "capacity_adaptive",
        False,
    ),
}


EXPECTED_DATA_SHA256 = {
    "case33_3min_final/model.p": (
        39024,
        "b9452fa9896fc709258edd4cb7c431aec2ef348a090fd4184ada21dc821d3ef3",
    ),
    "case33_3min_final/pv_active.csv": (
        54296160,
        "a5d64f59e8e291f366768d601f16fe95662a5d25ccba86f21d31d04e940db9ef",
    ),
    "case33_3min_final/load_active.csv": (
        351655119,
        "35c7657e886e46d8f2685540db1ee0ae327ec9446e21270217bbde190fc2365f",
    ),
    "case33_3min_final/load_reactive.csv": (
        356496345,
        "0f296701a1f01be68cc63b19dc3787baa2883e101979dd5c5acc9e89b280cdc3",
    ),
    "case141_3min_final/model.p": (
        53924,
        "aa4b60b5535cac35c2938199267162e6d0bf3412c66a7c3e90a9cbfd63b8ed51",
    ),
    "case141_3min_final/pv_active.csv": (
        166714024,
        "730fa5f0819190603cf96eaf17257390b9cb42be67b30a2e38cdec01db7c3cc2",
    ),
    "case141_3min_final/load_active.csv": (
        890101896,
        "ada3263018344eb33aeec12f8f16ceab6dc9b7d348adc66fc74b58f78ae364d4",
    ),
    "case141_3min_final/load_reactive.csv": (
        899058260,
        "e55a62fa23f2dc51506e97b821bbc82e5c59bdfadc3d66d90f4304ba442b4fed",
    ),
}


def test_submission_data_are_materialized_inside_this_checkout():
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "environments" / "var_voltage_control" / "data"
    file_attributes = getattr(data_root.stat(), "st_file_attributes", 0)
    reparse_attribute = 0x400

    assert data_root.is_dir()
    assert file_attributes & reparse_attribute == 0
    for relative, (expected_size, expected_hash) in EXPECTED_DATA_SHA256.items():
        candidate = data_root / relative
        assert candidate.is_file(), relative
        assert candidate.resolve().is_relative_to(repo_root.resolve()), relative
        assert candidate.stat().st_size == expected_size, relative
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected_hash, relative


def test_submission_variants_are_exact_and_stably_ordered():
    variants = submission_variants()

    assert isinstance(variants, tuple)
    assert [variant.variant_id for variant in variants] == list(EXPECTED_VARIANTS)
    assert {
        variant.variant_id: (
            variant.solver,
            variant.train_pv_scale,
            variant.evaluation_pv_scale,
            variant.action_mapping,
            variant.evaluate_terminal,
        )
        for variant in variants
    } == EXPECTED_VARIANTS
    assert sum(variant.evaluate_terminal for variant in variants) == 1


def test_submission_matrix_contains_exactly_36_training_runs(tmp_path):
    runs = build_submission_runs(tmp_path)

    assert len(runs) == 36
    assert len({row["run_id"] for row in runs}) == 36
    assert len({row["output_dir"] for row in runs}) == 36
    assert {row["seed"] for row in runs} == {0, 1, 2}
    assert {row["scenario"] for row in runs} == {
        "case33_3min_final",
        "case141_3min_final",
    }
    assert Counter(row["variant_id"] for row in runs) == {
        "full_matd3": 6,
        "nominal_train_matd3": 6,
        "static_projected_matd3": 6,
        "full_maddpg": 6,
        "full_mappo": 6,
        "full_facmaddpg": 6,
    }
    assert [row["run_id"] for row in runs] == [
        row["run_id"] for row in build_submission_runs(tmp_path)
    ]


def test_every_run_carries_independent_strict_train_and_evaluation_settings(tmp_path):
    runs = build_submission_runs(tmp_path)

    for row in runs:
        train = row["training"]
        validation = row["validation"]
        test = row["test"]
        assert len({id(train), id(validation), id(test)}) == 3
        assert train["episodes"] == 1000
        assert train["eval_frequency"] == 50
        assert train["episode_points"] == 240
        assert (train["start_hour"], train["start_minute"]) == (6, 0)
        assert train["reward_weights"] == {
            "voltage": 1.0,
            "violation": 0.0,
            "reactive_power": 0.1,
            "line_loss": None,
        }
        for configuration in (train, validation, test):
            assert {
                key: configuration[key]
                for key in (
                    "voltage_weight",
                    "violation_weight",
                    "q_weight",
                    "line_weight",
                    "action_bias",
                )
            } == {
                "voltage_weight": 1.0,
                "violation_weight": 0.0,
                "q_weight": 0.1,
                "line_weight": None,
                "action_bias": 0.0,
            }
        assert train["add_noise"] is True
        assert row["data_split"] == {
            "training_start_day": 0,
            "training_stop_day_exclusive": 999,
            "guard_day": 999,
            "validation_start_day": 1000,
            "validation_days": 30,
            "test_start_day": 1030,
            "test_days": 30,
            "unused_start_day": 1060,
        }
        for evaluation, start_day in ((validation, 1000), (test, 1030)):
            assert evaluation["pv_scale"] == row["evaluation_pv_scale"]
            assert evaluation["start_day"] == start_day
            assert evaluation["days"] == 30
            assert evaluation["episode_points"] == 240
            assert (evaluation["start_hour"], evaluation["start_minute"]) == (6, 0)
            assert evaluation["reset_action"] is False
            assert evaluation["add_noise"] is False
            assert evaluation["action_mapping"] == row["action_mapping"]

        assert train["pv_scale"] == row["train_pv_scale"]
        assert train["action_mapping"] == row["action_mapping"]
        assert row["action_scale"] == (
            0.8 if row["scenario"] == "case33_3min_final" else 0.6
        )
        assert train["action_scale"] == row["action_scale"]
        assert validation["action_scale"] == row["action_scale"]
        assert test["action_scale"] == row["action_scale"]
        assert row["expected_artifacts"] == {
            "best_checkpoint": "best_model.pt",
            "final_checkpoint": "final_model.pt",
            "validation_history": "validation_history.json",
            "best_result": "seed_result_best.json",
            "final_result": "seed_result_final.json",
            "provenance": "provenance.json",
            "best_fixed_test": "test_best/fixed_test_summary.json",
            "final_fixed_test": "test_final/fixed_test_summary.json"
            if row["evaluate_terminal"]
            else None,
            "completion_manifest": "completion_manifest.json",
        }


def test_nominal_and_static_ablations_change_only_the_declared_dimension(tmp_path):
    runs = build_submission_runs(tmp_path)
    sample = {
        row["variant_id"]: row
        for row in runs
        if row["scenario"] == "case33_3min_final" and row["seed"] == 0
    }

    assert sample["nominal_train_matd3"]["training"]["pv_scale"] == 1.0
    assert sample["nominal_train_matd3"]["validation"]["pv_scale"] == 1.5
    assert sample["nominal_train_matd3"]["test"]["pv_scale"] == 1.5
    assert {
        row["variant_id"]
        for row in runs
        if row["action_mapping"] == "static_projected"
    } == {"static_projected_matd3"}


def test_plan_hash_and_canonical_content_do_not_depend_on_checkout_location(tmp_path):
    plan_a = build_submission_plan(tmp_path / "checkout-a" / FORMAL_OUTPUT_RELATIVE)
    plan_b = build_submission_plan(tmp_path / "checkout-b" / FORMAL_OUTPUT_RELATIVE)

    assert plan_a == plan_b
    assert len(plan_a["plan_sha256"]) == 64
    payload = {key: value for key, value in plan_a.items() if key != "plan_sha256"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert plan_a["plan_sha256"] == hashlib.sha256(encoded).hexdigest()
    serialized = json.dumps(plan_a, ensure_ascii=False)
    assert str(tmp_path) not in serialized


def test_execution_rejects_requested_plan_content_tampered_under_old_hash(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    tampered = deepcopy(plan)
    tampered["runs"][0]["seed"] = 99

    with pytest.raises(ValueError, match="requested plan hash"):
        validate_execution_root(
            root, tampered, resume=False, repo_root=repo_root
        )

    assert not root.exists()


def test_resume_rejects_stored_plan_content_tampered_under_old_hash(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    stored = deepcopy(plan)
    stored["runs"][0]["seed"] = 99
    root.mkdir(parents=True)
    (root / "run_plan.json").write_text(json.dumps(stored), encoding="utf-8")
    (root / "resume_metadata.json").write_text(
        json.dumps(
            {"plan_sha256": plan["plan_sha256"], "completed_runs": []}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stored plan hash"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


@pytest.mark.parametrize(
    "candidate",
    (
        Path("outside"),
        Path("exp/q1_old_result"),
        Path("论文手动修改包_20260813/exp/q1_submission_v2_strict"),
        Path("exp/q1_submission_v2_strict/nested"),
    ),
)
def test_formal_root_validation_rejects_unsafe_locations_before_writes(
    tmp_path, candidate
):
    repo_root = tmp_path / "repo"
    output = repo_root / candidate

    with pytest.raises(ValueError):
        validate_formal_output_root(output, repo_root=repo_root)

    assert not output.exists()


def test_formal_root_validation_accepts_only_the_exact_new_root(tmp_path):
    repo_root = tmp_path / "repo"
    expected = repo_root / FORMAL_OUTPUT_RELATIVE

    assert validate_formal_output_root(expected, repo_root=repo_root) == expected.resolve()
    assert not expected.exists()


FIXED_TEST_FILES = (
    "fixed_test_step_metrics.csv",
    "fixed_test_daily_metrics.csv",
    "fixed_test_node_voltages.npz",
    "fixed_test_trajectory_schema.json",
    "fixed_test_summary.json",
)


def _required_completion_artifacts(run: dict) -> tuple[str, ...]:
    artifacts = [
        "best_model.pt",
        "final_model.pt",
        "provenance.json",
        *(f"test_best/{name}" for name in FIXED_TEST_FILES),
    ]
    if run["evaluate_terminal"]:
        artifacts.extend(f"test_final/{name}" for name in FIXED_TEST_FILES)
    return tuple(artifacts)


def _rewrite_completion_manifest(
    root: Path, manifest_path: Path, manifest: dict
) -> None:
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_runs"][0]["completion_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _write_resume_metadata(
    root: Path,
    plan: dict,
    *,
    run_index: int = 0,
    stored_plan_hash: str | None = None,
    manifest_hash: str | None = None,
) -> tuple[Path, Path]:
    run = plan["runs"][run_index]
    run_id = run["run_id"]
    run_dir = root / run["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    for relative in _required_completion_artifacts(run):
        artifact_path = run_dir / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{run_id}:{relative}".encode("utf-8")
        artifact_path.write_bytes(content)
        artifact_hashes[relative] = hashlib.sha256(content).hexdigest()
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "plan_sha256": plan["plan_sha256"],
        "artifacts": artifact_hashes,
    }
    manifest_path = run_dir / "completion_manifest.json"
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    stored_plan = deepcopy(plan)
    stored_plan["plan_sha256"] = stored_plan_hash or plan["plan_sha256"]
    (root / "run_plan.json").write_text(json.dumps(stored_plan), encoding="utf-8")
    (root / "resume_metadata.json").write_text(
        json.dumps(
            {
                "plan_sha256": stored_plan_hash or plan["plan_sha256"],
                "completed_runs": [
                    {
                        "run_id": run_id,
                        "completion_manifest": str(
                            manifest_path.relative_to(root).as_posix()
                        ),
                        "completion_manifest_sha256": manifest_hash or digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_dir, manifest_path


def test_nonempty_formal_root_requires_an_exact_verified_resume(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    root.mkdir(parents=True)
    (root / "unrelated.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        validate_execution_root(root, plan, resume=False, repo_root=repo_root)
    with pytest.raises(ValueError, match="stored run plan|resume metadata"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_plan_mismatch_and_invalid_completion_hash(tmp_path):
    mismatch_repo = tmp_path / "mismatch_repo"
    mismatch_root = mismatch_repo / FORMAL_OUTPUT_RELATIVE
    mismatch_plan = build_submission_plan(mismatch_root)
    _write_resume_metadata(
        mismatch_root, mismatch_plan, stored_plan_hash="0" * 64
    )

    with pytest.raises(ValueError, match="plan hash"):
        validate_execution_root(
            mismatch_root,
            mismatch_plan,
            resume=True,
            repo_root=mismatch_repo,
        )

    manifest_repo = tmp_path / "manifest_repo"
    manifest_root = manifest_repo / FORMAL_OUTPUT_RELATIVE
    manifest_plan = build_submission_plan(manifest_root)
    _write_resume_metadata(manifest_root, manifest_plan, manifest_hash="f" * 64)
    with pytest.raises(ValueError, match="completion manifest"):
        validate_execution_root(
            manifest_root,
            manifest_plan,
            resume=True,
            repo_root=manifest_repo,
        )


def test_resume_accepts_only_compatible_completed_entries(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    _write_resume_metadata(root, plan)

    assert validate_execution_root(
        root, plan, resume=True, repo_root=repo_root
    ) == root.resolve()

    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_runs"][0]["run_id"] = "not-in-plan"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="completed run"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_manifest_outside_the_corresponding_run_directory(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    _write_resume_metadata(root, plan)
    metadata_path = root / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_runs"][0]["completion_manifest"] = (
        f'{plan["runs"][1]["output_dir"]}/completion_manifest.json'
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="corresponding run output"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_completion_manifest_without_completed_status(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    _, manifest_path = _write_resume_metadata(root, plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("status")
    _rewrite_completion_manifest(root, manifest_path, manifest)

    with pytest.raises(ValueError, match="status"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_empty_completion_manifest_shell(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    _, manifest_path = _write_resume_metadata(root, plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {}
    _rewrite_completion_manifest(root, manifest_path, manifest)

    with pytest.raises(ValueError, match="required artifact"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_missing_required_completion_artifact(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    run_dir, _ = _write_resume_metadata(root, plan)
    (run_dir / "best_model.pt").unlink()

    with pytest.raises(ValueError, match="required artifact"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_resume_rejects_completion_artifact_hash_mismatch(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    run_dir, _ = _write_resume_metadata(root, plan)
    (run_dir / "provenance.json").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact hash"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_terminal_evaluation_requires_the_complete_test_final_artifact_set(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    run_dir, _ = _write_resume_metadata(root, plan, run_index=0)
    (run_dir / "test_final" / FIXED_TEST_FILES[-1]).unlink()

    with pytest.raises(ValueError, match="required artifact"):
        validate_execution_root(root, plan, resume=True, repo_root=repo_root)


def test_nonterminal_run_does_not_require_test_final_artifacts(tmp_path):
    repo_root = tmp_path / "repo"
    root = repo_root / FORMAL_OUTPUT_RELATIVE
    plan = build_submission_plan(root)
    run_dir, _ = _write_resume_metadata(root, plan, run_index=6)

    assert not (run_dir / "test_final").exists()
    assert validate_execution_root(
        root, plan, resume=True, repo_root=repo_root
    ) == root.resolve()


def test_dry_run_prints_plan_without_writing_and_plan_out_is_exclusive(
    tmp_path, capsys
):
    repo_root = tmp_path / "repo"
    formal_root = repo_root / FORMAL_OUTPUT_RELATIVE

    assert main(["--dry-run"], repo_root=repo_root) == 0
    stdout_plan = json.loads(capsys.readouterr().out)
    assert len(stdout_plan["runs"]) == 36
    assert not formal_root.exists()

    non_dry_plan_out = tmp_path / ".tmp" / "non_dry_plan.json"
    non_dry_plan_out.parent.mkdir()
    with pytest.raises(ValueError, match="dry-run"):
        main(["--plan-out", str(non_dry_plan_out)], repo_root=repo_root)
    assert not non_dry_plan_out.exists()

    plan_out = tmp_path / ".tmp" / "submission_v2_run_plan.json"
    assert main(["--dry-run", "--plan-out", str(plan_out)], repo_root=repo_root) == 0
    capsys.readouterr()
    assert json.loads(plan_out.read_text(encoding="utf-8")) == stdout_plan

    with pytest.raises(FileExistsError):
        main(["--dry-run", "--plan-out", str(plan_out)], repo_root=repo_root)
    with pytest.raises(ValueError, match="formal output root"):
        main(
            [
                "--dry-run",
                "--plan-out",
                str(formal_root / "submission_v2_run_plan.json"),
            ],
            repo_root=repo_root,
        )
    assert not formal_root.exists()
