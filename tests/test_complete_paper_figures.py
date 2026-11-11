from pathlib import Path
import subprocess
import sys

import pytest

from experiments.plot_complete_paper_figures import (
    FIGURE_BASENAMES,
    generate_all,
    load_network_model,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def figure_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper_figures")
    generate_all(ROOT, out)
    return out


def test_expected_figure_set_is_stable():
    assert FIGURE_BASENAMES == [
        "fig1_ir_crpc_framework",
        "fig2_network_topologies",
        "fig3_penetration_risk",
        "fig4_ir_crpc_convergence",
        "fig5_unified_1000ep_comparison",
        "fig6_training_violation_curves",
        "fig7_inverter_capacity",
        "fig8_risk_formation_path",
    ]


def test_serialized_networks_have_expected_topology_counts():
    case33 = load_network_model(
        ROOT / "environments/var_voltage_control/data/case33_3min_final/model.p"
    )
    case141 = load_network_model(
        ROOT / "environments/var_voltage_control/data/case141_3min_final/model.p"
    )
    assert (len(case33["nodes"]), len(case33["edges"])) == (33, 32)
    assert (len(case141["nodes"]), len(case141["edges"])) == (141, 140)
    assert len(case33["pv_buses"]) == 6
    assert len(case141["pv_buses"]) == 22


@pytest.mark.parametrize("basename", FIGURE_BASENAMES)
def test_each_figure_has_png_svg_and_pdf_exports(figure_dir, basename):
    for suffix in (".png", ".svg", ".pdf"):
        path = figure_dir / f"{basename}{suffix}"
        assert path.exists()
        assert path.stat().st_size > 1000


@pytest.mark.parametrize("basename", FIGURE_BASENAMES)
def test_svg_keeps_editable_text(figure_dir, basename):
    svg = (figure_dir / f"{basename}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "font-family" in svg


def test_penetration_risk_figure_contains_all_uncontrolled_metrics(figure_dir):
    svg = (figure_dir / "fig3_penetration_risk.svg").read_text(encoding="utf-8")
    for label in ["电压越限率", "平均电压偏差", "总网损", "全网可控时段比例"]:
        assert label in svg
    for network in ["IEEE 33 节点", "IEEE 141 节点"]:
        assert network in svg
    for case141_value in [
        "27.88",
        "28.49",
        "39.26",
        "0.0379",
        "0.0362",
        "0.0462",
        "0.5913",
        "0.8975",
        "1.6625",
        "48.20",
        "47.36",
        "32.97",
    ]:
        assert case141_value in svg
    assert "20 轮" not in svg
    assert "200 轮" not in svg


def test_training_figures_do_not_claim_an_unjustified_five_percent_threshold(figure_dir):
    for basename in ("fig4_ir_crpc_convergence", "fig6_training_violation_curves"):
        svg = (figure_dir / f"{basename}.svg").read_text(encoding="utf-8")
        assert "5% 参考线" not in svg


def test_inverter_capacity_figure_is_pure_flowchart_without_pq_geometry(figure_dir):
    svg = (figure_dir / "fig7_inverter_capacity.svg").read_text(encoding="utf-8")
    for required in [
        "动作映射流程",
        "物理可用无功上界",
        "映射为无功指令",
        "由构造自动满足",
        "高出力",
    ]:
        assert required in svg
    for forbidden in [
        "感性",
        "容性",
        "Power Factor",
        "功率因数",
        "容量可行域",
        "归一化动作到无功指令的映射",
    ]:
        assert forbidden not in svg


def test_risk_formation_path_figure_is_uncontrolled_mechanism_flowchart(figure_dir):
    svg = (figure_dir / "fig8_risk_formation_path.svg").read_text(encoding="utf-8")
    for required in [
        "电压风险形成路径",
        "光伏缩放",
        "反向潮流",
        "末端电压抬升",
        "条件调制",
    ]:
        assert required in svg
    for forbidden in ["MATD3", "MADDPG", "感性", "容性", "功率因数"]:
        assert forbidden not in svg


def test_plotting_script_runs_directly_from_repository_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/plot_complete_paper_figures.py",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fig1_ir_crpc_framework.svg").exists()
