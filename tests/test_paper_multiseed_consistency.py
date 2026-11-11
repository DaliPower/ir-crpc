"""Consistency checks for the final 1000-episode multiseed paper evidence."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "exp" / "q1_paper" / "论文全文-剥洋葱逻辑完善版.md"
MANIFEST = ROOT / "exp" / "q1_paper" / "paper_evidence_manifest.json"
SCRIPT = ROOT / "experiments" / "update_paper_multiseed_table.py"
LAUNCHER = ROOT / "experiments" / "run_q1_1000ep_multiseed_final.ps1"
POWERSHELL = Path(
    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
)
SUPPORTING_DOCS = {
    "handoff": ROOT / "exp" / "q1_paper" / "CODEX-项目交接与复现指南.md",
    "ieee": ROOT / "exp" / "q1_paper" / "实验分析-IEEE丰富版.md",
    "structure": ROOT / "exp" / "q1_paper" / "实验分析-论文结构.md",
    "final": ROOT / "exp" / "q1_paper" / "实验分析-最终版.md",
}
FINAL_DIRS = (
    "exp/q1_multiseed_1000ep_case33/",
    "exp/q1_multiseed_1000ep_case141/",
)
SHORT_BUDGET_DIRS = (
    "exp/q1_multiseed_200ep_case33/",
    "exp/q1_multiseed_100ep_case141/",
)
SELECTED_EPISODE_GROUPS = (
    "33/MATD3 [750, 200, 850]",
    "33/MADDPG [550, 350, 550]",
    "141/MATD3 [500, 700, 750]",
    "141/MADDPG [250, 200, 500]",
)
OLD_MULTISEED_MARKERS = (
    "exp/q1_multiseed_200ep_case33/",
    "exp/q1_multiseed_100ep_case141/",
    "33 节点（200 episode）和 141 节点（100 episode）",
    "33 节点 200 episode 下",
    "141 节点 100 episode 下",
    "每 20 episode",
    "9.51±10.91",
    "6.24±4.96",
    "24.42±14.91",
    "46.34±30.68",
)


def paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


def manifest_multiseed() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["multiseed_1000ep"]


def supporting_text(name: str) -> str:
    return SUPPORTING_DOCS[name].read_text(encoding="utf-8")


def assert_ieee_main_table_semantics(text: str) -> None:
    assert text.count("**TABLE VII**") == 1
    assert text.count("**TABLE VIII**") == 1
    table_vii = text.split("**TABLE VII**", 1)[1].split("**TABLE VIII**", 1)[0]
    table_viii = text.split("**TABLE VIII**", 1)[1].split(
        "## V-A.", 1
    )[0]
    assert "33-BUS FINAL 1000-EPISODE MULTISEED FIXED-TEST RESULTS" in table_vii
    assert "0.0165±0.0285" in table_vii
    assert "750, 200, 850" in table_vii
    assert "141-BUS FINAL 1000-EPISODE MULTISEED FIXED-TEST RESULTS" in table_viii
    assert "13.0200±6.3243" in table_viii
    assert "500, 700, 750" in table_viii
    for number in ("IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"):
        assert f"**TABLE {number}**" not in text
    for marker in (
        "**HISTORICAL TABLE H1**",
        "**HISTORICAL TABLE H10**",
        "SUMMARY OF PRIMARY OUTCOMES (PROTOCOL B)",
        "TABLE VII | Best-checkpoint results (1000-ep, main)",
        "main comparison",
    ):
        if marker.startswith(("SUMMARY", "TABLE VII", "main")):
            assert marker not in text
        else:
            assert marker in text


def assert_structure_main_conclusion_semantics(text: str) -> None:
    conclusion = text.split("## 5.6 本节综合结论", 1)[1].split(
        "## 附录 A", 1
    )[0]
    assert "第 0 节" in conclusion
    assert "表 VII–VIII" in conclusion
    assert "0.0165%/0.6572%" in conclusion
    assert "13.0200%/18.3790%" in conclusion
    assert "历史单种子训练脚本**仅保存末轮 checkpoint**" in text
    assert "当前训练脚本**仅保存末轮 checkpoint**" not in text
    assert (
        "主结论应以第 0 节最终公平多种子结果和表 VII–VIII 为依据"
        in text
    )
    assert "应结合 §5.2 最优 checkpoint 表陈述主结论" not in text


def assert_ieee_checkpoint_persistence_semantics(text: str) -> None:
    assert "历史单种子训练脚本主要保存末轮 checkpoint" in text
    assert "最终公平多种子运行保存独立验证集所选的 `best_model.pt`" in text
    assert "当前训练脚本仅持久化末轮 checkpoint" not in text


def assert_final_metric_and_history_semantics(text: str) -> None:
    metrics = text.split("### 1.3 评价指标", 1)[1].split("### 1.4", 1)[0]
    for metric in (
        "average_voltage_deviation",
        "q_loss",
        "total_line_loss",
    ):
        assert f"`{metric}`（越低越好）" in metrics
    assert "`totally_controllable_ratio`（越高越好）" in metrics
    assert "辅助指标" not in metrics or "（**越高越好**）" not in metrics

    history = text.split("### 4.1 早期对比", 1)[1].split("### 4.1b", 1)[0]
    assert "在该历史 200 轮 MARL 批次中最低" in history
    assert "| Droop |" not in history
    assert "| OPF |" not in history
    assert "IR-CRPC（200 轮）在本档预算下越限率最低" not in history


def assert_handoff_uses_executable_powershell_commands(text: str) -> None:
    base = (
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
        ".\\experiments\\run_q1_1000ep_multiseed_final.ps1"
    )
    assert f"{base} -DryRun" in text
    assert base in text
    assert "安全检查" in text
    assert "& ./experiments/run_q1_1000ep_multiseed_final.ps1" not in text


def snapshot_tree(path: Path) -> dict[str, tuple]:
    snapshot: dict[str, tuple] = {}
    if not path.exists():
        return snapshot
    for item in (path, *sorted(path.rglob("*"))):
        stat = item.stat()
        key = item.relative_to(path.parent).as_posix()
        if item.is_dir():
            snapshot[key] = ("dir", stat.st_mtime_ns)
        else:
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            snapshot[key] = ("file", stat.st_size, stat.st_mtime_ns, digest)
    return snapshot


def load_table_module():
    spec = importlib.util.spec_from_file_location("update_paper_multiseed_table", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_directories_replace_old_multiseed_protocol() -> None:
    text = paper_text()
    for directory in FINAL_DIRS:
        assert directory in text
    for marker in OLD_MULTISEED_MARKERS:
        assert marker not in text


def test_abstract_reports_final_vrate_sample_statistics() -> None:
    abstract = paper_text().split("## I. 引言", 1)[0]
    for value in (
        "0.016±0.029%",
        "0.657±1.037%",
        "13.02±6.32%",
        "18.38±5.59%",
    ):
        assert value in abstract
    assert "1000 episode" in abstract
    assert "seeds 0、1、2" in abstract
    assert "独立验证集" in abstract
    assert "固定测试集" in abstract


def test_protocol_and_descriptive_scope_are_explicit() -> None:
    text = paper_text()
    for phrase in (
        "训练日 0–999 与 1060–1096",
        "验证日 1000–1029（30 天）",
        "测试日 1030–1059（30 天）",
        "每 50 episode",
        "20 个验证点",
        "n=3",
        "样本标准差",
        "不构成统计显著性",
        "普适优越性",
        "固定测试集只评估所选 checkpoint",
    ):
        assert phrase in text


def test_tables_include_all_metrics_and_selected_checkpoints() -> None:
    text = paper_text()
    assert "| 方法 | VRate/% (均值±样本标准差) | 平均电压偏差 | 无功调节代价 | 总网损 | CR/% | 所选 episodes |" in text
    expected_rows = (
        "| MATD3 | 0.0165±0.0285 | 0.01135±0.00240 | 0.7169±0.2535 | 0.1502±0.0402 | 99.51±0.85 | 750, 200, 850 |",
        "| MADDPG | 0.6572±1.0368 | 0.01568±0.00322 | 0.4754±0.0794 | 0.1401±0.0275 | 91.54±13.10 | 550, 350, 550 |",
        "| MATD3 | 13.0200±6.3243 | 0.02549±0.00555 | 2.6224±0.7722 | 2.1320±0.2413 | 38.15±25.19 | 500, 700, 750 |",
        "| MADDPG | 18.3790±5.5902 | 0.02939±0.00639 | 1.7497±0.5809 | 1.9959±1.3710 | 42.93±6.83 | 250, 200, 500 |",
    )
    for row in expected_rows:
        assert row in text


def test_checkpoint_selection_is_not_misreported_as_episode_1000() -> None:
    text = paper_text()
    expected = manifest_multiseed()
    for case, algorithm in (
        ("case33", "matd3"),
        ("case33", "maddpg"),
        ("case141", "matd3"),
        ("case141", "maddpg"),
    ):
        episodes = expected[case][algorithm]["selected_episodes"]
        assert all(200 <= episode <= 850 for episode in episodes)
        assert 1000 not in episodes
        rendered = ", ".join(str(episode) for episode in episodes)
        assert rendered in text
    assert "报告的不是第 1000 episode 模型" in text
    assert "没有任何一个所选 checkpoint 为 episode 1000" in text


def test_tradeoffs_and_vrate_effect_sizes_are_preserved() -> None:
    text = paper_text()
    for phrase in (
        "约 97.5%",
        "约 29.2%",
        "两个网架的无功调节代价均高于 MADDPG",
        "总网损均略高于 MADDPG",
        "141 节点 CR 低于 MADDPG",
    ):
        assert phrase in text


def test_figure_captions_match_final_artifacts() -> None:
    text = paper_text()
    captions = {
        "**图 10.**": ("逐种子", "均值±样本标准差"),
        "**图 11.**": ("2×2", "VRate", "CR"),
        "**图 12.**": ("2×2", "每个面板", "3 个 seed", "完整验证轨迹", "最终选中 checkpoint"),
        "**图 13.**": ("两个网架", "3 个 seed", "淡色", "均值", "Min–Max", "每 50 episode"),
    }
    for label, phrases in captions.items():
        caption = text.split(label, 1)[1].split("\n\n", 1)[0]
        for phrase in phrases:
            assert phrase in caption


def test_historical_stage_results_do_not_claim_component_causality_or_scalability() -> None:
    text = paper_text()
    assert "### G. MATD3与MADDPG的结构级对照（非组件消融）" in text
    assert "历史单种子阶段值只能作描述性比较" in text
    assert "不能将该差异归因于双 Critic、目标策略平滑或延迟更新" in text
    assert "141 节点结果仅证明本文设置可运行，不构成可扩展性验证" in text
    assert "支持其在当前测试规模内的可扩展性" not in text


def test_figures_12_and_13_have_distinct_checkpoint_evidence_roles() -> None:
    text = paper_text()
    assert "图 12 显示 12 个所选 checkpoint 均位于 episode 200–850" in text
    assert "图 13 显示 episode 1000 的轨迹终点不等于最终报告模型" in text


def test_table_script_defaults_and_output_match_manifest() -> None:
    module = load_table_module()
    manifest = manifest_multiseed()
    expected_paths = {
        ROOT / "exp" / "q1_multiseed_1000ep_case33" / "multiseed_summary.json",
        ROOT / "exp" / "q1_multiseed_1000ep_case141" / "multiseed_summary.json",
    }
    assert set(module.DEFAULT_SUMMARIES.values()) == expected_paths

    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = completed.stdout
    for field in (
        "VRate/%",
        "平均电压偏差",
        "无功调节代价",
        "总网损",
        "CR/%",
        "所选 episodes",
    ):
        assert field in output

    metric_fields = (
        ("violation_pct_mean", "violation_pct_std", 4),
        ("average_voltage_deviation_mean", "average_voltage_deviation_std", 5),
        ("q_loss_mean", "q_loss_std", 4),
        ("total_line_loss_mean", "total_line_loss_std", 4),
        ("controllable_pct_mean", "controllable_pct_std", 2),
    )
    paper = paper_text()
    checked_cells = 0
    checked_checkpoints = 0
    for case in ("case33", "case141"):
        for algorithm in ("matd3", "maddpg"):
            evidence = manifest[case][algorithm]
            cells = [
                f"{evidence[mean_key]:.{digits}f}±{evidence[std_key]:.{digits}f}"
                for mean_key, std_key, digits in metric_fields
            ]
            checkpoints = ", ".join(
                str(episode) for episode in evidence["selected_episodes"]
            )
            expected_row = (
                f"| {algorithm.upper()} | {' | '.join(cells)} | {checkpoints} |"
            )
            assert expected_row in output
            assert expected_row in paper
            checked_cells += len(cells)
            checked_checkpoints += 1
    assert checked_cells == 20
    assert checked_checkpoints == 4


def test_percentage_conversion_is_key_driven() -> None:
    module = load_table_module()
    summary = {
        "matd3": {
            "metrics": {
                "percentage_of_v_out_of_control": {"mean": 1.25, "std": 0.5},
                "totally_controllable_ratio": {"mean": 0.5, "std": 0.25},
            }
        }
    }
    assert module.format_metric(summary, "matd3", "percentage_of_v_out_of_control") == "125.0000±50.0000"
    assert module.format_metric(summary, "matd3", "totally_controllable_ratio") == "50.00±25.00"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "<= 1" not in source


@pytest.mark.parametrize("name", SUPPORTING_DOCS)
def test_supporting_docs_use_final_fair_multiseed_protocol(name: str) -> None:
    text = supporting_text(name)
    for directory in FINAL_DIRS:
        assert directory in text
    for phrase in (
        "两网架均为 1000 episode",
        "seeds 0/1/2",
        "每 50 episode",
        "验证 30 天",
        "测试 30 天",
        "独立验证集选择 checkpoint",
        "固定测试集最终报告",
        "不报告第 1000 episode 模型",
    ):
        assert phrase in text


@pytest.mark.parametrize("name", SUPPORTING_DOCS)
def test_supporting_docs_list_all_selected_episodes(name: str) -> None:
    text = supporting_text(name)
    for group in SELECTED_EPISODE_GROUPS:
        assert group in text


@pytest.mark.parametrize("name", SUPPORTING_DOCS)
def test_supporting_docs_preserve_tradeoffs_and_descriptive_scope(name: str) -> None:
    text = supporting_text(name)
    for phrase in (
        "MATD3 在两个网架的平均 VRate 均较低",
        "q_loss 均更高",
        "总网损略高",
        "141 节点 CR 不占优",
        "n=3 仅作描述统计",
    ):
        assert phrase in text


@pytest.mark.parametrize("name", SUPPORTING_DOCS)
def test_supporting_docs_retain_short_budget_history_as_supplement(name: str) -> None:
    text = supporting_text(name)
    for directory in SHORT_BUDGET_DIRS:
        assert directory in text
    for phrase in (
        "short-budget sensitivity / 历史补充",
        "不再支持正文主结论",
        "exp/q1_paper/supplementary_figures/short_budget/",
    ):
        assert phrase in text


@pytest.mark.parametrize("name", SUPPORTING_DOCS)
def test_supporting_docs_include_generation_and_integrity_commands(name: str) -> None:
    text = supporting_text(name)
    for command in (
        "experiments/plot_multiseed_paper_figures.py",
        "experiments/paper_evidence.py",
        "-m pytest tests/test_paper_multiseed_consistency.py -q",
    ):
        assert command in text


def test_handoff_updates_sources_and_run_recovery_commands() -> None:
    text = supporting_text("handoff")
    for source in (
        "exp/q1_multiseed_1000ep_case33/multiseed_summary.json",
        "exp/q1_multiseed_1000ep_case141/multiseed_summary.json",
    ):
        assert source in text
    assert "experiments\\run_q1_1000ep_multiseed_final.ps1" in text
    assert "--skip-existing" in text
    assert "运行/恢复" in text


def test_ieee_rich_uses_table_vii_viii_only_for_final_multiseed() -> None:
    assert_ieee_main_table_semantics(supporting_text("ieee"))


def test_structure_conclusion_points_to_final_tables_and_qualifies_legacy_script() -> None:
    assert_structure_main_conclusion_semantics(supporting_text("structure"))


def test_ieee_distinguishes_historical_terminal_and_final_selected_checkpoints() -> None:
    assert_ieee_checkpoint_persistence_semantics(supporting_text("ieee"))


def test_final_analysis_metric_directions_and_historical_ranking_are_correct() -> None:
    assert_final_metric_and_history_semantics(supporting_text("final"))


def test_handoff_has_real_executable_powershell_commands_and_paths() -> None:
    assert POWERSHELL.is_file()
    assert LAUNCHER.is_file()
    for path in (
        ROOT / "experiments" / "plot_multiseed_paper_figures.py",
        ROOT / "experiments" / "paper_evidence.py",
        ROOT / "tests" / "test_paper_multiseed_consistency.py",
        ROOT / "exp" / "q1_multiseed_1000ep_case33",
        ROOT / "exp" / "q1_multiseed_1000ep_case141",
    ):
        assert path.exists()
    assert_handoff_uses_executable_powershell_commands(supporting_text("handoff"))


@pytest.mark.parametrize(
    ("validator", "counterexample"),
    (
        (
            assert_ieee_main_table_semantics,
            "**TABLE VII**\nBEST-CHECKPOINT RESULTS (main comparison)",
        ),
        (
            assert_structure_main_conclusion_semantics,
            "## 5.6 本节综合结论\n主结论来自阶段最优。\n## 附录 A",
        ),
        (
            assert_final_metric_and_history_semantics,
            "### 1.3 评价指标\n辅助指标全部越高越好。\n### 1.4\n",
        ),
        (
            assert_handoff_uses_executable_powershell_commands,
            "& ./experiments/run_q1_1000ep_multiseed_final.ps1 -DryRun",
        ),
    ),
)
def test_semantic_validators_reject_counterexample_documents(
    validator, counterexample: str
) -> None:
    with pytest.raises((AssertionError, IndexError)):
        validator(counterexample)


@pytest.mark.parametrize(
    ("validator", "counterexample"),
    (
        (
            assert_structure_main_conclusion_semantics,
            "## 5.6 本节综合结论\n第 0 节、表 VII–VIII\n"
            "0.0165%/0.6572%，13.0200%/18.3790%\n## 附录 A\n"
            "历史单种子训练脚本**仅保存末轮 checkpoint**。\n"
            "应结合 §5.2 最优 checkpoint 表陈述主结论。",
        ),
        (
            assert_ieee_checkpoint_persistence_semantics,
            "当前训练脚本仅持久化末轮 checkpoint。",
        ),
    ),
)
def test_second_review_semantic_guards_reject_stale_wording(
    validator, counterexample: str
) -> None:
    with pytest.raises((AssertionError, IndexError)):
        validator(counterexample)


def test_launcher_dry_run_preserves_result_tree_log_hashes_and_mtimes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "dryrun-project"
    experiments = project / "experiments"
    exp = project / "exp"
    experiments.mkdir(parents=True)
    exp.mkdir()
    launcher = experiments / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)
    (experiments / "run_risk_aware_multiseed.py").write_text(
        "print('fake runner must not execute during launcher DryRun')\n",
        encoding="utf-8",
    )
    (exp / "q1_multiseed_1000ep_master.log").write_text(
        "master sentinel\n", encoding="utf-8"
    )
    for case in ("case33", "case141"):
        result_dir = exp / f"q1_multiseed_1000ep_{case}"
        result_dir.mkdir()
        (result_dir / "training.log").write_text(
            f"{case} training sentinel\n", encoding="utf-8"
        )
        (result_dir / "checkpoint.sentinel").write_bytes(case.encode("ascii"))

    before = snapshot_tree(exp)
    completed = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-DryRun",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    after = snapshot_tree(exp)

    assert after == before
    assert "DRY RUN" in completed.stdout
    assert "no files will be created or modified" in completed.stdout
    assert "case33_3min_final" in completed.stdout
    assert "case141_3min_final" in completed.stdout
    assert "--train-episodes 1000" in completed.stdout
