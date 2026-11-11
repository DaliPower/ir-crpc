"""Regression contract for the authoritative IEEE Access manuscript.

These assertions deliberately encode the pre-submission requirements that
Tasks 4--6 must satisfy.  They read the manual package only: no experiment is
run and no legacy manuscript is accepted as an alternative source of truth.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "论文手动修改包_20260813" / "论文全文-问题描述重整版.md"


def _text() -> str:
    return MANUSCRIPT.read_text(encoding="utf-8")


def _reference(text: str, number: int) -> str:
    match = re.search(
        rf"^\[{number}\]\s+(.+?)(?=^\[\d+\]\s+|\Z)",
        text.split("## 参考文献", maxsplit=1)[1],
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"reference [{number}] is missing"
    return match.group(1).replace("\n", " ")


def _numbered_citations(body: str) -> set[int]:
    citations: set[int] = set()
    citation_group = re.compile(
        r"(?<![!\\])\["
        r"(\d+(?:\s*[–—-]\s*\d+)?"
        r"(?:\s*[,，;；]\s*\d+(?:\s*[–—-]\s*\d+)?)*)"
        r"\]"
    )
    for match in citation_group.finditer(body):
        for item in re.split(r"\s*[,，;；]\s*", match.group(1)):
            bounds = re.split(r"\s*[–—-]\s*", item)
            start = int(bounds[0])
            end = int(bounds[1]) if len(bounds) == 2 else start
            assert start <= end, f"descending citation range [{item}]"
            citations.update(range(start, end + 1))
    return citations


def _table_block(text: str, number: str) -> str:
    match = re.search(
        rf"^\*\*表 {number}\*\*.*?(?=^\*\*表 [IVXLCDM]+\*\*|^###?\s|^!\[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"Table {number} is missing"
    return match.group(0)


def _to_roman(number: int) -> str:
    values = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    result = []
    for value, numeral in values:
        count, number = divmod(number, value)
        result.append(numeral * count)
    return "".join(result)


def _unbounded_matd3_narrative_lines(text: str) -> list[tuple[int, str]]:
    """Return prose lines that use MATD3 without a local implementation boundary."""

    explicit_boundary = re.compile(
        r"(?:本文|本研究)采用的\s*MATD3\s*(?:衍生|派生)"
        r"(?:实现|求解器(?:配置)?)|MATD3[-‑– ]derived solver configuration",
        flags=re.IGNORECASE,
    )
    definition_context = re.compile(
        r"(?:经典|标准)\s*MATD3|"
        r"MATD3\s*[（(][^）)]+[）)]\s*(?:是|指|表示|全称为|即)",
        flags=re.IGNORECASE,
    )
    failures = []
    in_table = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            in_table = False
            continue
        if line.startswith("|"):
            in_table = True
            continue
        if in_table:
            continue
        if "MATD3" not in line:
            continue
        if (
            line.startswith("#")
            or line.startswith("![")
            or re.match(r"^\*\*(?:图|表)\s", line)
        ):
            continue
        if re.search(r"第\s*[IVXLCDM]+(?:-[A-Z])?\s*节.*MATD3.*(?:过程|机制|架构)", line):
            continue
        if (
            "经典 MATD3" in line
            and re.search(r"本文(?:采用|所称)的?\s*MATD3\s*(?:实现|配置)", line)
            and re.search(r"(?:不同|区别|而非|不用于评价经典)", line)
        ):
            continue
        without_definitions = definition_context.sub("", line)
        if "MATD3" not in without_definitions:
            continue
        if not explicit_boundary.search(line):
            failures.append((line_number, line))
    return failures


def test_authoritative_manual_package_is_the_only_manuscript_input() -> None:
    assert MANUSCRIPT == ROOT / "论文手动修改包_20260813" / "论文全文-问题描述重整版.md"
    assert MANUSCRIPT.is_file()


def test_corrected_reference_6_matches_its_doi_record() -> None:
    reference = _reference(_text(), 6)
    for field in (
        "T. N. Pham",
        "R. Shah",
        "M. N. Dao",
        "N. Sultanova",
        "S. Islam",
        "Low and medium voltage distribution network planning with distributed energy resources: a survey",
        "_Electr. Eng._",
        "vol. 107, no. 2, pp. 1797–1828, Feb. 2025",
        "doi: 10.1007/s00202-024-02535-0",
    ):
        assert field in reference


def test_corrected_reference_21_matches_its_doi_record() -> None:
    reference = _reference(_text(), 21)
    for field in (
        "D. Cao",
        "J. Zhao",
        "W. Hu",
        "F. Ding",
        "N. Yu",
        "Q. Huang",
        "Z. Chen",
        "Model-free voltage control of active distribution system with PVs using surrogate model-based deep reinforcement learning",
        "_Appl. Energy_",
        "vol. 306, Art. no. 117982, 2022",
        "doi: 10.1016/j.apenergy.2021.117982",
    ):
        assert field in reference


def test_solver_identity_is_bounded_without_the_rejected_name() -> None:
    text = _text()
    assert "SCQ-MATD3" not in text
    failures = _unbounded_matd3_narrative_lines(text)
    assert not failures, "unbounded narrative MATD3 usage:\n" + "\n".join(
        f"line {line_number}: {line}" for line_number, line in failures
    )


def test_matd3_boundary_audit_is_local_and_has_explicit_exceptions() -> None:
    fixture = """\
### MATD3 求解器标题
**图 4.** MATD3 求解器图注。
| 求解器 | MATD3 |

经典 MATD3 采用两个独立集中式 Critic。
经典 MATD3 使用独立 Critic；本文采用的 MATD3 实现使用共享 Critic，因此二者结构不同。
本文采用的 MATD3 衍生实现使用共享条件化 Critic。
另一段把 MATD3 无条件等同于经典算法。
"""
    assert _unbounded_matd3_narrative_lines(fixture) == [
        (8, "另一段把 MATD3 无条件等同于经典算法。")
    ]


def test_table_i_uses_the_normalized_action_scaling_term() -> None:
    table_i = _table_block(_text(), "I")
    assert re.search(r"^\|\s*归一化动作缩放区间\s*\|", table_i, re.MULTILINE)
    assert not re.search(r"^\|\s*动作范围\s*\|", table_i, re.MULTILINE)


def test_figure_6_uses_submission_metric_names() -> None:
    figure_six = re.search(r"\*\*图 6\.\*\*(.+)", _text())
    assert figure_six, "Figure 6 caption is missing"
    for metric in (
        "Mean total active line loss (MW)",
        "Violation-free period ratio (%)",
    ):
        assert metric in figure_six.group(1)


def test_equation_figure_table_and_reference_numbers_are_contiguous() -> None:
    text = _text()
    body, references = text.split("## 参考文献", maxsplit=1)
    equations = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", body)]
    figures = [int(value) for value in re.findall(r"^\*\*图 (\d+)\.\*\*", body, re.MULTILINE)]
    tables = re.findall(r"^\*\*表 ([IVXLCDM]+)\*\*", body, re.MULTILINE)
    reference_numbers = [int(value) for value in re.findall(r"^\[(\d+)\] ", references, re.MULTILINE)]

    assert equations == list(range(1, len(equations) + 1))
    assert figures == list(range(1, len(figures) + 1))
    assert tables == [_to_roman(number) for number in range(1, len(tables) + 1)]
    assert reference_numbers == list(range(1, len(reference_numbers) + 1))


def test_every_reference_is_cited_in_the_manuscript_body() -> None:
    text = _text()
    body, references = text.split("## 参考文献", maxsplit=1)
    listed = {int(value) for value in re.findall(r"^\[(\d+)\] ", references, re.MULTILINE)}
    assert _numbered_citations(body) == listed


def test_citation_parser_supports_lists_and_ranges() -> None:
    fixture = "文献 [1, 3–5, 7-8，10；12—13] 与单项 [15]。"
    assert _numbered_citations(fixture) == {1, 3, 4, 5, 7, 8, 10, 12, 13, 15}


def test_every_linked_manuscript_image_exists() -> None:
    image_paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", _text())
    assert image_paths, "manuscript must retain linked figures"
    missing = [path for path in image_paths if not (MANUSCRIPT.parent / path).is_file()]
    assert not missing, f"missing manuscript images: {missing}"
