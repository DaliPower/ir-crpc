from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import experiments.submission_provenance as provenance_module
from experiments.submission_provenance import (
    build_provenance,
    write_provenance_atomic,
)


def _make_sources(tmp_path):
    repo_root = tmp_path / "repo"
    run_dir = tmp_path / "formal_run"
    data_a = repo_root / "data" / "a.csv"
    data_z = repo_root / "data" / "z.csv"
    config = repo_root / "args" / "config.yaml"
    best = run_dir / "best_model.pt"
    final = run_dir / "final_model.pt"
    for path, content in (
        (data_a, b"a-data"),
        (data_z, b"z-data"),
        (config, b"config"),
        (best, b"best"),
        (final, b"final"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return repo_root, run_dir, data_a, data_z, config, best, final


def _build(tmp_path):
    repo_root, run_dir, data_a, data_z, config, best, final = _make_sources(
        tmp_path
    )
    plan = {"runs": [{"seed": 0, "variant_id": "full_matd3"}], "version": 2}
    manifest = build_provenance(
        repo_root,
        data_paths=[data_z, data_a],
        config_paths=[config],
        weight_paths={"best": best, "final": final},
        run_dir=run_dir,
        run_plan_content=plan,
        best_episode=50,
        final_episode=1000,
        git_revision="abc1234",
        git_dirty=True,
    )
    return manifest, (repo_root, run_dir, data_a, data_z, config, best, final), plan


def _entry_hash(manifest, category, suffix):
    return next(
        row["sha256"]
        for row in manifest["files"][category]
        if row["path"].endswith(suffix)
    )


def test_provenance_records_relative_sorted_files_protocol_and_runtime(tmp_path):
    manifest, _, plan = _build(tmp_path)

    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert manifest["schema_version"] == "1.0"
    assert manifest["protocol_id"] == "ir-crpc-submission-v2-strict"
    assert manifest["generated_at_utc"].endswith("Z")
    assert manifest["git"] == {"revision": "abc1234", "dirty": True}
    assert manifest["run_plan_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert set(manifest["software"]) == {
        "python",
        "torch",
        "numpy",
        "pandapower",
    }
    assert isinstance(manifest["cuda"]["available"], bool)
    assert "device_name" in manifest["cuda"]
    assert manifest["checkpoints"] == {
        "best": {"episode": 50, "role": "validation-selected"},
        "final": {"episode": 1000, "role": "terminal-training-state"},
    }
    for category in ("data", "config", "weights"):
        rows = manifest["files"][category]
        assert rows == sorted(rows, key=lambda row: (row["relative_to"], row["path"]))
        assert all(not Path(row["path"]).is_absolute() for row in rows)
        assert all(row["bytes"] > 0 for row in rows)
        assert all(len(row["sha256"]) == 64 for row in rows)
    assert [row["path"] for row in manifest["files"]["data"]] == [
        "data/a.csv",
        "data/z.csv",
    ]
    assert {row["relative_to"] for row in manifest["files"]["weights"]} == {
        "run_dir"
    }


def test_embedded_official_plan_hash_is_verified_and_preserved(tmp_path):
    _, sources, _ = _build(tmp_path)
    payload = {"schema_version": 1, "runs": [{"run_id": "formal-0"}]}
    official_hash = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    plan = {**payload, "plan_sha256": official_hash}

    manifest = build_provenance(
        sources[0],
        data_paths=[sources[2]],
        config_paths=[sources[4]],
        weight_paths={"best": sources[5], "final": sources[6]},
        run_dir=sources[1],
        run_plan_content=plan,
        best_episode=50,
        final_episode=1000,
        git_revision="abc1234",
        git_dirty=False,
    )

    assert manifest["run_plan_sha256"] == official_hash

    tampered = json.loads(json.dumps(plan))
    tampered["runs"][0]["run_id"] = "tampered"
    with pytest.raises(ValueError, match="plan_sha256 mismatch"):
        build_provenance(
            sources[0],
            data_paths=[sources[2]],
            config_paths=[sources[4]],
            weight_paths={"best": sources[5], "final": sources[6]},
            run_dir=sources[1],
            run_plan_content=tampered,
            best_episode=50,
            final_episode=1000,
            git_revision="abc1234",
            git_dirty=False,
        )


@pytest.mark.parametrize(
    ("category", "source_index", "suffix"),
    [
        ("data", 2, "data/a.csv"),
        ("config", 4, "args/config.yaml"),
        ("weights", 5, "best_model.pt"),
    ],
)
def test_provenance_hash_changes_after_one_byte_change(
    tmp_path, category, source_index, suffix
):
    first, sources, plan = _build(tmp_path)
    target = sources[source_index]
    old_hash = _entry_hash(first, category, suffix)
    target.write_bytes(target.read_bytes() + b"x")

    second = build_provenance(
        sources[0],
        data_paths=[sources[3], sources[2]],
        config_paths=[sources[4]],
        weight_paths={"best": sources[5], "final": sources[6]},
        run_dir=sources[1],
        run_plan_content=plan,
        best_episode=50,
        final_episode=1000,
        git_revision="abc1234",
        git_dirty=True,
    )

    assert _entry_hash(second, category, suffix) != old_hash


def test_provenance_rejects_missing_and_outside_paths(tmp_path):
    _, sources, plan = _build(tmp_path)
    repo_root, run_dir, data_a, _data_z, config, best, final = sources
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside repo_root and run_dir"):
        build_provenance(
            repo_root,
            data_paths=[outside],
            config_paths=[config],
            weight_paths={"best": best, "final": final},
            run_dir=run_dir,
            run_plan_content=plan,
            best_episode=50,
            final_episode=1000,
            git_revision="abc1234",
            git_dirty=False,
        )

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        build_provenance(
            repo_root,
            data_paths=[data_a],
            config_paths=[repo_root / "missing.yaml"],
            weight_paths={"best": best, "final": final},
            run_dir=run_dir,
            run_plan_content=plan,
            best_episode=50,
            final_episode=1000,
            git_revision="abc1234",
            git_dirty=False,
        )


def test_preflight_rejects_in_repo_alias_that_physically_resolves_outside(
    tmp_path, monkeypatch
):
    _, sources, plan = _build(tmp_path)
    repo_root, run_dir, _data_a, _data_z, config, best, final = sources
    alias = repo_root / "data" / "junction_source.csv"
    alias.write_bytes(b"lexically inside")
    outside = tmp_path / "external-data" / "junction_source.csv"
    outside.parent.mkdir()
    outside.write_bytes(b"physical target")
    original_resolve = Path.resolve
    alias_resolved = original_resolve(alias)
    outside_resolved = original_resolve(outside)

    def resolve_alias(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        return outside_resolved if resolved == alias_resolved else resolved

    monkeypatch.setattr(Path, "resolve", resolve_alias)

    with pytest.raises(ValueError, match="outside repo_root and run_dir"):
        build_provenance(
            repo_root,
            data_paths=[alias],
            config_paths=[config],
            weight_paths={"best": best, "final": final},
            run_dir=run_dir,
            run_plan_content=plan,
            best_episode=50,
            final_episode=1000,
            git_revision="abc1234",
            git_dirty=False,
        )

def test_atomic_provenance_write_leaves_no_temporary_file_on_success(tmp_path):
    manifest, sources, _ = _build(tmp_path)
    destination = sources[1] / "provenance.json"

    returned = write_provenance_atomic(manifest, destination)

    assert returned == destination
    assert json.loads(destination.read_text("utf-8")) == manifest
    assert not list(destination.parent.glob("provenance.json.tmp*"))


def test_atomic_provenance_write_preserves_destination_on_promotion_failure(
    tmp_path, monkeypatch
):
    manifest, sources, _ = _build(tmp_path)
    destination = sources[1] / "provenance.json"
    destination.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("forced provenance promotion failure")

    monkeypatch.setattr(provenance_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced provenance promotion failure"):
        write_provenance_atomic(manifest, destination)

    assert json.loads(destination.read_text("utf-8")) == {"old": True}
    assert not list(destination.parent.glob("provenance.json.tmp*"))
