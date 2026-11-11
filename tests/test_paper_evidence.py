from copy import deepcopy
from pathlib import Path
from statistics import stdev

import pytest

from experiments.paper_evidence import build_evidence_manifest, validate_evidence


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    return build_evidence_manifest(ROOT)


def test_selected_checkpoint_results_match_source_data(evidence):
    assert evidence["selected_checkpoints"]["case33"]["episodes"] == 200
    assert evidence["selected_checkpoints"]["case33"]["violation_pct"] == pytest.approx(1.0384, abs=1e-4)
    assert evidence["selected_checkpoints"]["case141"]["episodes"] == 500
    assert evidence["selected_checkpoints"]["case141"]["violation_pct"] == pytest.approx(0.0377, abs=1e-4)


def test_unified_500_episode_results_report_unfavorable_case33_result(evidence):
    case33 = evidence["unified_500ep"]["case33"]
    assert case33["matd3"]["violation_pct"] == pytest.approx(5.8273, abs=1e-4)
    assert case33["maddpg"]["violation_pct"] == pytest.approx(0.5046, abs=1e-4)
    assert case33["maddpg"]["violation_pct"] < case33["matd3"]["violation_pct"]


def test_best_stage_1000_episode_results_make_matd3_the_safest_marl_policy(evidence):
    case33 = evidence["best_stage_1000ep"]["case33"]
    case141 = evidence["best_stage_1000ep"]["case141"]
    assert case33["matd3"]["best_episode"] == 600
    assert case141["matd3"]["best_episode"] == 600
    assert case33["matd3"]["violation_pct"] == pytest.approx(0.3411, abs=1e-4)
    assert case141["matd3"]["violation_pct"] == pytest.approx(0.2694, abs=1e-4)
    for case in (case33, case141):
        assert case["matd3"]["violation_pct"] < case["maddpg"]["violation_pct"]
        assert case["matd3"]["violation_pct"] < case["mappo"]["violation_pct"]
        assert case["matd3"]["violation_pct"] < case["facmaddpg"]["violation_pct"]


def test_network_metadata_matches_serialized_models(evidence):
    assert evidence["networks"]["case33"]["buses"] == 33
    assert evidence["networks"]["case33"]["lines"] == 32
    assert evidence["networks"]["case33"]["loads"] == 32
    assert evidence["networks"]["case33"]["pv_agents"] == 6
    assert evidence["networks"]["case33"]["slack_buses"] == [0]
    assert evidence["networks"]["case33"]["pv_buses"] == [12, 17, 21, 24, 28, 32]
    assert evidence["networks"]["case33"]["nominal_voltage_kv"] == [12.66]
    assert evidence["networks"]["case33"]["zones"] == 4
    assert evidence["networks"]["case141"]["buses"] == 141
    assert evidence["networks"]["case141"]["lines"] == 140
    assert evidence["networks"]["case141"]["loads"] == 84
    assert evidence["networks"]["case141"]["pv_agents"] == 22
    assert evidence["networks"]["case141"]["nominal_voltage_kv"] == [12.5]
    assert evidence["networks"]["case141"]["zones"] == 9


def test_reward_definition_matches_environment_configuration(evidence):
    reward = evidence["reward"]
    assert reward["voltage_barrier"] == "l1"
    assert reward["voltage_weight"] == 1.0
    assert reward["q_weight"] == 0.1
    assert reward["line_weight"] is None
    assert reward["terms_in_training_reward"] == ["mean_l1_voltage_deviation", "mean_absolute_reactive_power"]
    assert reward["evaluation_only_terms"] == ["total_line_loss", "voltage_violation_rate"]


def test_manifest_validation_has_no_errors(evidence):
    assert validate_evidence(evidence) == []


def test_multiseed_1000ep_protocol_and_key_results(evidence):
    final = evidence["multiseed_1000ep"]
    assert final["protocol"] == {
        "train_episodes": 1000,
        "eval_freq": 50,
        "validation_points": 20,
        "validation_days": 30,
        "test_days": 30,
        "seeds": [0, 1, 2],
        "checkpoint_rule": (
            "selected on an independent validation set; reported once on a fixed test set"
        ),
        "dispersion": "sample standard deviation (ddof=1)",
    }
    assert final["case33"]["matd3"]["violation_pct_mean"] == pytest.approx(
        0.0164828, abs=1e-6
    )
    assert final["case141"]["matd3"]["violation_pct_mean"] == pytest.approx(
        13.0199709, abs=1e-6
    )
    assert final["case33"]["matd3"]["selected_episodes"] == [750, 200, 850]
    assert final["case33"]["maddpg"]["selected_episodes"] == [550, 350, 550]
    assert final["case141"]["matd3"]["selected_episodes"] == [500, 700, 750]
    assert final["case141"]["maddpg"]["selected_episodes"] == [250, 200, 500]


def test_multiseed_1000ep_seed_records_have_complete_validation_histories(evidence):
    final = evidence["multiseed_1000ep"]
    expected_episodes = list(range(50, 1001, 50))
    source_paths = []
    for case in ("case33", "case141"):
        for method in ("matd3", "maddpg"):
            record = final[case][method]
            assert [item["seed"] for item in record["seed_records"]] == [0, 1, 2]
            assert [item["selected_episode"] for item in record["seed_records"]] == record[
                "selected_episodes"
            ]
            for seed_record in record["seed_records"]:
                assert seed_record["validation_episodes"] == expected_episodes
                assert seed_record["selected_episode"] in seed_record["validation_episodes"]
                source_paths.extend(
                    [
                        seed_record["seed_result_source"],
                        seed_record["validation_history_source"],
                    ]
                )
            assert record["summary_source"].startswith(
                f"exp/q1_multiseed_1000ep_{case}/"
            )
    assert len(source_paths) == 24
    assert len(set(source_paths)) == 24
    assert all("q1_multiseed_1000ep_" in source for source in source_paths)
    assert not any("200ep" in source or "100ep" in source for source in source_paths)


def test_multiseed_1000ep_ratios_and_sample_standard_deviations(evidence):
    final = evidence["multiseed_1000ep"]
    manifest_metrics = (
        "violation_pct",
        "average_voltage_deviation",
        "q_loss",
        "total_line_loss",
        "controllable_pct",
    )
    for case in ("case33", "case141"):
        for method in ("matd3", "maddpg"):
            record = final[case][method]
            assert "percentage_of_v_out_of_control_mean" not in record
            assert "totally_controllable_ratio_mean" not in record
            for metric in manifest_metrics:
                seed_values = [item[metric] for item in record["seed_records"]]
                assert record[f"{metric}_std"] == pytest.approx(
                    stdev(seed_values), abs=1e-12
                )
            assert all(0.0 <= item["violation_pct"] <= 100.0 for item in record["seed_records"])
            assert all(0.0 <= item["controllable_pct"] <= 100.0 for item in record["seed_records"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("history", "validation history"),
        ("selected_episode", "selected episode"),
        ("seed_metric", "seed test metric"),
        ("sample_std", "sample standard deviation"),
    ],
)
def test_validate_evidence_rejects_corrupt_multiseed_evidence(
    evidence, mutation, expected_error
):
    corrupt = deepcopy(evidence)
    record = corrupt["multiseed_1000ep"]["case33"]["matd3"]
    if mutation == "history":
        record["seed_records"][0]["validation_episodes"][-1] = 999
    elif mutation == "selected_episode":
        record["seed_records"][0]["selected_episode"] = 700
    elif mutation == "seed_metric":
        record["seed_records"][0]["q_loss"] += 1.0
    else:
        record["q_loss_std"] += 1.0

    errors = validate_evidence(corrupt)
    assert any(expected_error in error.lower() for error in errors), errors
