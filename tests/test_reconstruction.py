# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
import numpy as np
import torch


def test_grappa_1d_preserves_acquired_lines():
    from mrif.reconstruction.grappa import grappa_1d_recon

    rng = np.random.default_rng(42)

    shape = (8, 8, 1, 2)

    calibration = (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)

    undersampled = calibration.copy()

    # R=2: retain even phase-encoding lines.
    undersampled[:, 1::2, :, :] = 0

    reconstructed = grappa_1d_recon(
        calibration,
        undersampled,
        reduction_factor=2,
        kx=3,
        ky=3,
    )

    assert reconstructed.shape == undersampled.shape

    # GRAPPA must not alter the originally acquired lines.
    np.testing.assert_allclose(
        reconstructed[:, ::2, :, :],
        undersampled[:, ::2, :, :],
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.isfinite(reconstructed).all()


def test_grappa_1d_weight_calculation_recovers_known_mapping():
    from mrif.reconstruction.grappa.grappa_1D import (
        calculate_reconstruction_weights,
    )

    sources = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.complex64,
    )

    true_weights = np.array(
        [
            [2.0 + 1.0j],
            [3.0 - 1.0j],
        ],
        dtype=np.complex64,
    )

    targets = sources @ true_weights

    # Expected GRAPPA target layout:
    # samples × missing-lines × coils
    targets = targets[:, None, :]

    recovered = calculate_reconstruction_weights(
        sources,
        targets,
        reduction_factor=2,
    )

    assert recovered.shape == (1, 2, 1)

    np.testing.assert_allclose(
        recovered[0],
        true_weights,
        rtol=1e-5,
        atol=1e-5,
    )


def test_grappa_2d_weight_calculation_recovers_known_mapping():
    from mrif.reconstruction.grappa.grappa_2D import (
        grappa_2D_compute_reconstruction_weights,
    )

    sources = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.complex64,
    )

    true_weights = np.array(
        [
            [2.0],
            [-1.0],
        ],
        dtype=np.complex64,
    )

    targets = sources @ true_weights

    # samples × target_phase × target_slice × coils
    targets = targets.reshape(-1, 1, 1, 1)

    recovered = grappa_2D_compute_reconstruction_weights(
        sources,
        targets,
    )

    assert recovered.shape == (1, 1, 2, 1)

    np.testing.assert_allclose(
        recovered[0, 0],
        true_weights,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cg_sense_zero_input_returns_zero():
    from mrif.reconstruction.sense.cg import CG_sense_2D

    kspace = torch.zeros(
        (4, 4, 1, 1),
        dtype=torch.complex64,
    )

    coil_sensitivity = torch.ones_like(kspace)

    result = CG_sense_2D(
        kspace,
        coil_sensitivity,
    )

    assert result.shape == (4, 4, 1)

    assert torch.isfinite(result).all()

    torch.testing.assert_close(
        result,
        torch.zeros_like(result),
    )