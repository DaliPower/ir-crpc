import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiments.plot_multiseed_paper_figures as figures
from experiments.plot_multiseed_paper_figures import (
    EXPERIMENTS,
    TRAJECTORY_EXPERIMENTS,
    aggregate_validation_histories,
)


def _write_history(path: Path, values: list[float]) -> None:
    history = [
        {
            "episode": episode,
            "validation_metrics": {"percentage_of_v_out_of_control": value},
            "selected": False,
        }
        for episode, value in zip((20, 40, 60), values)
    ]
    path.write_text(json.dumps(history), encoding="utf-8")


def test_aggregate_validation_histories_returns_seed_mean_min_and_max(tmp_path: Path) -> None:
    paths = [tmp_path / f"seed_{seed}.json" for seed in range(3)]
    _write_history(paths[0], [0.10, 0.20, 0.30])
    _write_history(paths[1], [0.20, 0.30, 0.40])
    _write_history(paths[2], [0.30, 0.40, 0.50])

    result = aggregate_validation_histories(paths)

    np.testing.assert_array_equal(result["episodes"], [20, 40, 60])
    np.testing.assert_allclose(result["seeds"], [[10, 20, 30], [20, 30, 40], [30, 40, 50]])
    np.testing.assert_allclose(result["mean"], [20, 30, 40])
    np.testing.assert_allclose(result["min"], [10, 20, 30])
    np.testing.assert_allclose(result["max"], [30, 40, 50])


def test_experiments_use_only_final_1000_episode_summaries() -> None:
    paths = {experiment["path"] for experiment in EXPERIMENTS}

    assert paths == {
        Path(__file__).resolve().parents[1]
        / "exp/q1_multiseed_1000ep_case33/multiseed_summary.json",
        Path(__file__).resolve().parents[1]
        / "exp/q1_multiseed_1000ep_case141/multiseed_summary.json",
    }
    assert all("1000 episode" in experiment["title"] for experiment in EXPERIMENTS)


def test_trajectory_experiments_use_only_final_1000_episode_roots() -> None:
    roots = {experiment["root"] for experiment in TRAJECTORY_EXPERIMENTS}

    assert roots == {
        Path(__file__).resolve().parents[1]
        / "exp/q1_multiseed_1000ep_case33/case33_3min_final",
        Path(__file__).resolve().parents[1]
        / "exp/q1_multiseed_1000ep_case141/case141_3min_final",
    }
    assert all("1000 episode" in experiment["title"] for experiment in TRAJECTORY_EXPERIMENTS)


def test_validation_selection_panels_use_final_1000_episode_roots() -> None:
    panels = figures.VALIDATION_SELECTION_PANELS

    assert len(panels) == 4
    assert {
        (panel["root"], panel["algorithm"])
        for panel in panels
    } == {
        (
            Path(__file__).resolve().parents[1]
            / "exp/q1_multiseed_1000ep_case33/case33_3min_final",
            "matd3",
        ),
        (
            Path(__file__).resolve().parents[1]
            / "exp/q1_multiseed_1000ep_case33/case33_3min_final",
            "maddpg",
        ),
        (
            Path(__file__).resolve().parents[1]
            / "exp/q1_multiseed_1000ep_case141/case141_3min_final",
            "matd3",
        ),
        (
            Path(__file__).resolve().parents[1]
            / "exp/q1_multiseed_1000ep_case141/case141_3min_final",
            "maddpg",
        ),
    }
    assert all("1000 episode" in panel["title"] for panel in panels)


def test_load_seed_results_returns_three_seeds_in_order_and_preserves_selected_episode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "case33_3min_final"
    for seed in (2, 0, 1):
        seed_dir = root / "matd3" / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "seed_result.json").write_text(
            json.dumps({"seed": seed, "selected_episode": 250 + seed}),
            encoding="utf-8",
        )

    results = figures.load_seed_results(root, "matd3")

    assert [result["seed"] for result in results] == [0, 1, 2]
    assert [result["selected_episode"] for result in results] == [250, 251, 252]


def test_figure_basenames_are_fixed() -> None:
    assert figures.FIGURE_BASENAMES == (
        "fig10_multiseed_per_seed_violation",
        "fig11_multiseed_mean_summary",
        "fig12_multiseed_validation_selection",
        "fig13_multiseed_mean_minmax_curves",
    )


def test_generate_all_exports_large_editable_bundles_without_bars(
    tmp_path: Path, monkeypatch,
) -> None:
    def reject_bar(*args, **kwargs):
        raise AssertionError("Figure 10/11 must not create bar patches")

    monkeypatch.setattr("matplotlib.axes.Axes.bar", reject_bar)
    figures.generate_all(tmp_path)

    for basename in figures.FIGURE_BASENAMES:
        for extension in ("png", "pdf", "svg"):
            output = tmp_path / f"{basename}.{extension}"
            assert output.exists()
            assert output.stat().st_size > 1000

        image = plt.imread(tmp_path / f"{basename}.png")
        assert image.shape[1] >= 2000
        assert image.shape[0] >= 700
        svg = (tmp_path / f"{basename}.svg").read_text(encoding="utf-8")
        assert "<text" in svg


def test_validation_selection_has_three_seed_trajectories_and_selected_checkpoints(
    tmp_path: Path, monkeypatch,
) -> None:
    captured: dict[str, plt.Figure] = {}

    def capture(fig: plt.Figure, out_dir: Path, basename: str) -> None:
        captured[basename] = fig

    monkeypatch.setattr(figures, "save_bundle", capture)
    figures.plot_validation_selection(tmp_path)

    fig = captured["fig12_multiseed_validation_selection"]
    assert len(fig.axes) == 4
    for ax, panel in zip(fig.axes, figures.VALIDATION_SELECTION_PANELS):
        trajectory_lines = [line for line in ax.lines if line.get_label().startswith("Seed ")]
        assert len(trajectory_lines) == 3
        assert all(len(line.get_xdata()) == 20 for line in trajectory_lines)

        results = figures.load_seed_results(panel["root"], panel["algorithm"])
        expected = {result["selected_episode"] for result in results}
        selected_x = {
            int(offset[0])
            for collection in ax.collections
            for offset in collection.get_offsets()
        }
        assert selected_x == expected
        assert 1000 not in selected_x
        summaries = [
            text.get_text()
            for text in ax.texts
            if text.get_text().startswith("Selected checkpoints:")
        ]
        assert summaries == [
            "Selected checkpoints: "
            + " | ".join(
                f"s{result['seed']} {result['selected_episode']}" for result in results
            )
        ]

    plt.close(fig)


def test_seed_and_checkpoint_annotations_do_not_overlap(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, plt.Figure] = {}

    def capture(fig: plt.Figure, out_dir: Path, basename: str) -> None:
        captured[basename] = fig

    monkeypatch.setattr(figures, "save_bundle", capture)
    figures.apply_style()
    figures.plot_seed_bars(tmp_path)
    figures.plot_validation_selection(tmp_path)

    checks = (
        (captured["fig10_multiseed_per_seed_violation"], "  "),
        (captured["fig12_multiseed_validation_selection"], ":"),
    )
    for fig, marker in checks:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            labels = [text for text in ax.texts if marker in text.get_text()]
            boxes = [text.get_window_extent(renderer).expanded(1.03, 1.12) for text in labels]
            assert all(
                not first.overlaps(second)
                for index, first in enumerate(boxes)
                for second in boxes[index + 1 :]
            )
        plt.close(fig)
