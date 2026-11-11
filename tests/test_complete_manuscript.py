import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "exp/q1_paper/计及电压越限风险的高渗透光伏配电网多智能体协同无功控制-完整稿.md"


def _text() -> str:
    return MANUSCRIPT.read_text(encoding="utf-8")


def test_manuscript_has_complete_section_structure():
    text = _text()
    required = [
        "## 摘要",
        "## Abstract",
        "## 1 引言",
        "## 2 系统模型与问题定义",
        "## 3 IR-CRPC 多智能体协同无功控制方法",
        "## 4 实验设置",
        "## 5 结果与分析",
        "## 6 讨论",
        "## 7 结论",
        "## 数据与代码可用性",
        "## 参考文献",
    ]
    for heading in required:
        assert heading in text


def test_all_figure_links_resolve():
    text = _text()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(links) == 6
    for relative in links:
        assert (MANUSCRIPT.parent / relative).exists(), relative


def test_no_placeholders_or_editorial_annotations_remain():
    text = _text()
    forbidden = [
        "<font",
        "TODO",
        "TBD",
        "待补充",
        "需要扩写",
        "114 节点",
        "114节点",
        "压倒性",
        "彻底失效",
        "维度呈指数级",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_reward_formula_matches_implemented_objective():
    text = _text()
    start = text.index("实际单步奖励为")
    end = text.index("该奖励在安全区间内", start)
    formula_section = text[start:end]
    assert "|v_i(t)-1|" in formula_section
    assert "0.1" in formula_section
    assert "|q_k^{PV}(t)|" in formula_section
    assert "L_{\\mathrm{line}}" not in formula_section


def test_key_results_and_unfavorable_comparison_are_reported():
    text = _text()
    for statement in [
        "从 1.98% 上升至 6.53%",
        "降至 0.34%",
        "降至 0.27%",
        "按阶段测试电压越限率选择最优 checkpoint",
        "不应理解为“只采用第 1000 轮模型”",
    ]:
        assert statement in text


def test_reference_list_is_contiguous_and_includes_primary_algorithm_sources():
    text = _text()
    references = re.findall(r"^\[(\d+)\] ", text, flags=re.MULTILINE)
    assert references == [str(i) for i in range(1, 21)]
    assert "arXiv: 2110.14300" in text
    assert "Addressing function approximation error in actor-critic methods" in text
    assert "Multi-agent actor-critic for mixed cooperative-competitive environments" in text


def test_risk_aware_retraining_protocol_is_disclosed_without_new_result_claims():
    text = _text()
    required = [
        "平方越限惩罚",
        "violation_weight=100.0",
        "独立验证集",
        "seed=0,1,2,3,4",
        "风险感知重训练结果尚未生成",
        "l1-cmp1000",
    ]
    for phrase in required:
        assert phrase in text
    assert "风险感知重训练将越限率降至" not in text
