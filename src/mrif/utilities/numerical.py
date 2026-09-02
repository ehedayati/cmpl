# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    import torch


def resize_matrix(
    matrix: np.ndarray | torch.Tensor,
    target_shape: tuple[int, int] = (600, 600),
) -> np.ndarray | torch.Tensor:
    """
    Resize a 2D matrix to the target shape using linear interpolation.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Input 2D matrix.

    target_shape : tuple[int, int]
        Target shape as (height, width).

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Resized matrix. The input type, dtype, and device are preserved
        when the input is a torch.Tensor.
    """

    # No resizing is needed if the matrix already has the requested shape.
    if matrix.shape == target_shape:
        return matrix

    # Detect torch tensors without importing torch at module import time.
    # This keeps torch as an optional runtime dependency.
    is_torch = type(matrix).__module__.startswith("torch")

    if is_torch:
        # Torch is imported only when the function actually receives
        # a torch.Tensor.
        import torch

        # Resizing is performed through NumPy/SciPy, so this operation is
        # intentionally non-differentiable. The context manager safely
        # restores the caller's previous gradient state when it exits.
        with torch.no_grad():

            # Preserve tensor metadata so the returned tensor matches
            # the input dtype and device.
            device = matrix.device
            dtype = matrix.dtype

            # SciPy operates on NumPy arrays, so detach the tensor from
            # its computation graph and move it to CPU for conversion.
            array = matrix.detach().cpu().numpy()

            # Compute the scaling factor independently for each dimension.
            scale_factors = (
                target_shape[0] / array.shape[0],
                target_shape[1] / array.shape[1],
            )

            # Resize using first-order interpolation (linear interpolation).
            resized = ndimage.zoom(
                array,
                scale_factors,
                order=1,
            )

            # Convert the resized array back to a torch tensor while
            # restoring the original dtype and device.
            return torch.as_tensor(
                resized,
                dtype=dtype,
                device=device,
            )

    # NumPy path: ensure the input is represented as a NumPy array.
    array = np.asarray(matrix)

    # Compute the scaling factor independently for each dimension.
    scale_factors = (
        target_shape[0] / array.shape[0],
        target_shape[1] / array.shape[1],
    )

    # Resize using first-order interpolation (linear interpolation).
    resized = ndimage.zoom(
        array,
        scale_factors,
        order=1,
    )

    return resized