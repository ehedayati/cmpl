# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
import numpy as np


def test_side_by_side_view_runs(monkeypatch):
    import matplotlib.pyplot as plt

    from cmpl.visualization import side_by_side_view

    shown = {"called": False}

    def fake_show():
        shown["called"] = True

    monkeypatch.setattr(plt, "show", fake_show)

    image1 = np.arange(64, dtype=float).reshape(8, 8)
    image2 = image1 * 2

    side_by_side_view(
        image1,
        image2,
        titles=["Image 1", "Image 2"],
        show_colorbar=False,
    )

    assert shown["called"]

    plt.close("all")


def test_visualize_segmentation_slice_runs(monkeypatch):
    import matplotlib.pyplot as plt

    from cmpl.visualization import visualize_segmentation_slice

    monkeypatch.setattr(plt, "show", lambda: None)

    image = np.zeros((8, 8, 8), dtype=np.float32)

    segmentation = np.zeros(
        (8, 8, 8),
        dtype=np.uint8,
    )

    image[3:6, 3:6, :] = 100.0
    segmentation[3:6, 3:6, :] = 1

    visualize_segmentation_slice(
        image,
        segmentation,
        slice_number=4,
        dimension="axial",
        target_shape=(8, 8),
    )

    plt.close("all")


def test_plot_3d_mri_static_slice_runs(monkeypatch):
    import matplotlib.pyplot as plt

    from cmpl.visualization import plot_3D_mri

    monkeypatch.setattr(plt, "show", lambda: None)

    image = np.random.default_rng(42).normal(
        size=(8, 8, 8)
    )

    segmentation = np.zeros(
        (8, 8, 8),
        dtype=np.uint8,
    )

    segmentation[2:5, 2:5, 4] = 1

    plot_3D_mri(
        image,
        slice_number=4,
        direction="sagittal",
        segmentation=segmentation,
    )

    plt.close("all")