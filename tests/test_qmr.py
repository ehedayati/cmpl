# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
import numpy as np


def test_reconstruct_images_matches_exponential_model():
    from mrif.quantitative_MRI import reconstruct_images

    t2_star = np.full((2, 3, 2), 20.0, dtype=np.float32)
    s0 = np.full((2, 3, 2), 100.0, dtype=np.float32)
    te = np.array([0.0, 10.0, 20.0], dtype=np.float32)

    result = reconstruct_images(
        t2_star,
        s0,
        te,
        device="cpu",
        return_numpy=True,
    )

    expected = (
        s0[..., None]
        * np.exp(-te / t2_star[..., None])
    )

    assert result.shape == (2, 3, 2, 3)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rmse_is_zero_for_perfect_reconstruction():
    from mrif.quantitative_MRI import (
        calculate_rmse_percentage_s0,
        reconstruct_images,
    )

    t2_star = np.full((2, 2, 2), 20.0, dtype=np.float32)
    s0 = np.full((2, 2, 2), 100.0, dtype=np.float32)
    te = np.array([0.0, 10.0, 20.0], dtype=np.float32)

    images = reconstruct_images(
        t2_star,
        s0,
        te,
        device="cpu",
        return_numpy=True,
    )

    rmse_pct, rse_pct = calculate_rmse_percentage_s0(
        images,
        images.copy(),
        s0,
        device="cpu",
        return_numpy=True,
    )

    np.testing.assert_allclose(rmse_pct, 0.0, atol=1e-6)
    np.testing.assert_allclose(rse_pct, 0.0, atol=1e-6)


def test_rmse_known_ten_percent_error():
    from mrif.quantitative_MRI import calculate_rmse_percentage_s0

    s0 = np.full((2, 2), 100.0, dtype=np.float32)

    original = np.full((2, 2, 3), 100.0, dtype=np.float32)
    reconstructed = np.full((2, 2, 3), 90.0, dtype=np.float32)

    rmse_pct, rse_pct = calculate_rmse_percentage_s0(
        original,
        reconstructed,
        s0,
        device="cpu",
        return_numpy=True,
    )

    np.testing.assert_allclose(rmse_pct, 10.0, rtol=1e-5)
    np.testing.assert_allclose(rse_pct, 10.0, rtol=1e-5)


def test_two_parameter_fit_exact_initial_model_cpu():
    """
    If synthetic data are generated with S0=100 and T2*=20,
    and the optimizer is initialized at those exact values,
    one iteration should preserve the solution.
    """
    from mrif.quantitative_MRI import t2_star_two_parametric_3D

    te = np.array([0.0, 10.0, 20.0], dtype=np.float32)

    true_s0 = 100.0
    true_t2 = 20.0

    signal = true_s0 * np.exp(-te / true_t2)

    images = np.broadcast_to(
        signal,
        (2, 2, 2, len(te)),
    ).copy()

    result = t2_star_two_parametric_3D(
        te,
        images,
        num_iterations=1,
        initial_lr=0.001,
        initial_T2_star=true_t2,
        plot_error=False,
        device="cpu",
    )

    t2_result = result["T2_star_map"].cpu().numpy()
    s0_result = result["S0_map"].cpu().numpy()

    np.testing.assert_allclose(
        t2_result,
        true_t2,
        rtol=1e-4,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        s0_result,
        true_s0,
        rtol=1e-4,
        atol=1e-4,
    )

def test_mapping_import_does_not_require_matplotlib():
    import mrif.quantitative_MRI.mapping

    assert mrif.quantitative_MRI.mapping is not None