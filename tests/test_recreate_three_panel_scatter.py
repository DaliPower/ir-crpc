from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.recreate_three_panel_scatter import (
    build_reconstructed_data,
    create_combined_figure,
)


def _centres_by_label(xy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([xy[labels == label].mean(axis=0) for label in ("blue", "cyan", "red", "green")])


def test_last_two_panels_draw_black_diamonds_at_label_centres() -> None:
    data = build_reconstructed_data()
    fig = create_combined_figure(data)
    try:
        for ax, label_key in zip(fig.axes[1:], ("middle_label", "right_label")):
            centre_collection = ax.collections[-1]
            actual = np.asarray(centre_collection.get_offsets(), dtype=float)
            expected = _centres_by_label(data["xy"], data[label_key])

            assert actual == pytest.approx(expected)
            assert centre_collection.get_facecolors()[0, :3] == pytest.approx(
                (29 / 255, 41 / 255, 44 / 255), abs=0.01
            )
    finally:
        plt.close(fig)


def test_axes_use_times_new_roman() -> None:
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        fig = create_combined_figure(build_reconstructed_data())
        try:
            assert fig.axes[0].xaxis.label.get_fontfamily()[0] == "Times New Roman"
            assert fig.axes[0].yaxis.label.get_fontfamily()[0] == "Times New Roman"
        finally:
            plt.close(fig)


def test_axes_plotting_box_has_reference_width_height_ratio() -> None:
    fig = create_combined_figure(build_reconstructed_data())
    try:
        for ax in fig.axes:
            assert ax.get_box_aspect() == pytest.approx(1 / 1.65)
    finally:
        plt.close(fig)
