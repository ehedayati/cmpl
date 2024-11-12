# File created by: Eisa Hedayati
# Date: 2/15/2024
# Description: This file is developed at CMRR
import numpy as np
import torch as pt
import torch.nn as nn

import scipy as sp
from .utils import pad_if_required


def grappa_2d_recon(calibration_kspace, undersampled_kspace, kernel_size, reduction_factors):
    """
    Reconstruct a 3D k-space from undersampled data using the 2D GRAPPA algorithm.

    Parameters:
    - calibration_kspace (np.array): The calibration data extracted from the center of k-space.
    - undersampled_kspace (np.array): The k-space that has been undersampled and needs reconstruction.
    - kernel_size (tuple of ints): The size of the kernel used in the GRAPPA reconstruction. (kernel_height, kernel_width, kernel_depth).
    - reduction_factors (tuple of ints): The acceleration factors used in the undersampling, for each axis. (phase_undersampling, slice_undersampling)

    Returns:
    - torch.tensor: The reconstructed k-space.

    Note on Calibration and Reconstruction Axes Ordering must be:
    - frequency, phase, slice, coils

    you can use np.moveaxis to create this ordering
    """
    undersampled_kspace = pad_if_required(undersampled_kspace, reduction_factors)
    # Extract source and target blocks from the calibration k-space for GRAPPA weight calculation
    reconstruction_sources, reconstruction_targets = grappa_2D_source_target(calibration_kspace, kernel_size,
                                                                             reduction_factors)

    # Compute the GRAPPA reconstruction weights from the source and target blocks
    reconstruction_weights = grappa_2D_compute_reconstruction_weights(reconstruction_sources, reconstruction_targets)
    nc = calibration_kspace.shape[3]
    # Apply the computed weights to the undersampled k-space to reconstruct the missing data
    reconstructed_kspace = grappa_2D_weight_application(undersampled_kspace, reconstruction_weights, kernel_size,
                                                        reduction_factors, nc)

    return reconstructed_kspace


def grappa_2D_source_target(calibration_kspace, kernel_size, reduction_factors):
    """
    Generate sources and targets for reconstruction from calibration k-space data.

    Parameters:
    - calibration_kspace: The calibration k-space data array.
    - kernel_size: A tuple indicating the size of the kernel (height, width, depth).
    - reduction_factors: A tuple indicating the reduction factors for columns and slices.

    Returns:
    - Tuple of arrays: reconstruction_sources, reconstruction_targets
    """
    # Initialize dimensions based on calibration k-space shape
    n_rows, n_columns, n_slices = calibration_kspace.shape[:3]
    reconstruction_targets = []
    reconstruction_sources = []

    half_kernel_width = reduction_factors[0] * (kernel_size[1]) // 2
    half_kernel_depth = reduction_factors[1] * (kernel_size[2]) // 2
    # Iterate through rows, columns, and slices to generate sources and targets
    for row in range(n_rows - kernel_size[0]):
        source_row = calibration_kspace[row: row + kernel_size[0]]
        for column in range(n_columns):
            # Calculate indices for source columns based on reduction factor
            source_column_idx = column + np.arange(kernel_size[1]) * reduction_factors[0]
            if np.any(source_column_idx >= n_columns):
                break  # Skip if column indices exceed bounds

            # Adjust column start index based on reduction factor
            col_start_adjustment = -1 if reduction_factors[1] > 1 else 0
            col_start = column + half_kernel_width + col_start_adjustment

            # Select source columns and the sub-target column based on adjusted start index
            source_columns = source_row[:, source_column_idx]
            sub_target_column = source_row[kernel_size[0] // 2,
                                col_start:column + (1 + kernel_size[1] // 2) * reduction_factors[0]]

            for slice in range(n_slices):
                # Calculate indices for reconstruction slices based on reduction factor
                reconstruction_slices = slice + np.arange(kernel_size[2]) * reduction_factors[1]
                if np.any(reconstruction_slices >= n_slices):
                    break  # Skip if slice indices exceed bounds

                # Adjust slice start index based on reduction factor
                slice_start_adjustment = -1 if reduction_factors[1] > 1 else 0
                slice_start = slice + half_kernel_depth + slice_start_adjustment

                # Flatten source columns and select sub-target slice for reconstruction
                sub_source = source_columns[..., reconstruction_slices, :].flatten()
                reconstruction_sources.append(sub_source)
                sub_target = sub_target_column[...,
                             slice_start:slice + (1 + kernel_size[2] // 2) * reduction_factors[1], :]
                reconstruction_targets.append(sub_target)

    # Convert lists to numpy arrays before returning
    reconstruction_targets = np.array(reconstruction_targets)
    reconstruction_sources = np.array(reconstruction_sources)

    return reconstruction_sources, reconstruction_targets


def grappa_2D_compute_reconstruction_weights(reconstruction_sources, reconstruction_targets):
    """
    Computes the GRAPPA reconstruction weights for 3D MRI data.

    Parameters:
    - reconstruction_sources (np.ndarray): 2D array of data that will be used for extracting targets
    - reconstruction_targets (np.ndarray): 4D array of data to reconstruct.

    Returns:
    - np.ndarray: Reconstruction weights for grappa use
    """
    # Compute the pseudo-inverse of reconstruction sources
    reconstruction_sources_pinv = sp.linalg.pinv(reconstruction_sources,atol=0.01)

    # Adjust pseudo-inverse to match reconstruction_targets dimensions
    reconstruction_sources_pinv = np.repeat(np.expand_dims(reconstruction_sources_pinv, 0),
                                            reconstruction_targets.shape[2], axis=0)
    reconstruction_sources_pinv = np.repeat(np.expand_dims(reconstruction_sources_pinv, 0),
                                            reconstruction_targets.shape[1], axis=0)

    # Reshape reconstruction_targets for matrix multiplication
    reconstruction_targets_moved = np.moveaxis(reconstruction_targets, 0, -2)

    # Compute the reconstruction weights
    reconstruction_weights = np.matmul(reconstruction_sources_pinv, reconstruction_targets_moved)

    return reconstruction_weights


def grappa_2D_weight_application_line(undersampled_kspace, reconstruction_weights, kernel_size, reduction_factor, nc):
    """
    Applying 2D Grappa weights to reconstruct the undersampled kspace

    Parameters:
    - undersampled_kspace: The undersampled k-space data.
    - reconstruction_weights: Weights for the reconstruction kernel.
    - kernel_size: A tuple of three integers indicating the size of the convolution kernel.
    - reduction_factor: A tuple of two integers for the dilation and stride in the 2nd and 3rd dimensions.
    - nc: Number of channels in the input and output.
    - i: Index to select specific reconstruction weights.

    Returns:
    - output_volume: The reconstructed volume.
    """
    # Prepare the tensor from undersampled k-space
    pt.set_grad_enabled(False)
    undersampled_tensor = pt.tensor(undersampled_kspace).unsqueeze_(0).moveaxis(-1, 1)

    # Prepare the kernel weights
    kernel_weights = pt.tensor(
        reconstruction_weights.reshape(kernel_size[0], kernel_size[1], kernel_size[2], nc, nc)
    ).moveaxis(-1, 0).moveaxis(-1, 1)

    # Set convolution parameters
    dilation = (1, reduction_factor[0], reduction_factor[1])
    stride = (1, reduction_factor[0], reduction_factor[1])
    column_pad = (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - 1)
    slide_pad = (kernel_size[2] + (kernel_size[2] - 1) * (dilation[2] - 1) - 1)
    padding = (
        (kernel_size[0] - 1) // 2,
        column_pad // 2,
        slide_pad // 2
    )

    # Initialize the 3D convolution layer
    conv3d = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=kernel_size,
                       padding=padding, stride=stride, dilation=dilation, dtype=pt.complex64)

    # Set the kernel weights and biases
    conv3d.weight.data = kernel_weights
    conv3d.bias.data.fill_(0)

    # Compute the output volume through the convolution layer
    output_volume = conv3d(undersampled_tensor)
    pt.set_grad_enabled(True)
    return output_volume[0]


def grappa_2D_weight_application(undersampled_kspace, reconstruction_weights, kernel_size, reduction_factors, nc):
    """
    Applies GRAPPA weights to undersampled k-space data for 2D MRI reconstruction.

    Parameters:
    - undersampled_kspace (np.ndarray): The undersampled k-space data.
    - reconstruction_weights (np.ndarray): Precomputed weights for GRAPPA reconstruction
    - kernel_size (tuple): The size of the kernel used for GRAPPA weight computation.
    - reduction_factors (tuple): The acceleration factors in the phase-encode and frequency-encode directions,
      respectively.
    - nc (int): Number of coils.

    Returns:
    - torch.tensor: The reconstructed k-space data after applying GRAPPA weights, with shape [nc, height, width].

    The function iterates over the reconstruction weights to apply them to the undersampled k-space data,
    reconstructing missing lines. It then assembles the reconstructed lines into the final k-space data,
    adjusting for coil sensitivity and reduction factors.
    """
    pt.set_grad_enabled(False)
    reconstructed_lines = []
    for i in range(reconstruction_weights.shape[0]):
        for j in range(reconstruction_weights.shape[1]):
            reconstruction_weights[i, j].shape
            reconstructed_lines.append(
                grappa_2D_weight_application_line(undersampled_kspace, reconstruction_weights[i, j], kernel_size,
                                                  reduction_factors, nc))
    iterator = 0
    final_reconstruction = pt.tensor(undersampled_kspace).moveaxis(-1, 0)
    is_skip_enabled = False
    col_start_adder = 1
    if reconstruction_weights.shape[1] > 1:
        is_skip_enabled = True
    for i in range(reconstruction_weights.shape[0]):
        for j in range(reconstruction_weights.shape[1]):
            if is_skip_enabled:
                is_skip_enabled = False
                iterator += 1
                col_start_adder = 0
                continue
            final_reconstruction[..., i + col_start_adder::reduction_factors[0], j::reduction_factors[1]] = \
                reconstructed_lines[iterator]
            iterator += 1
    pt.set_grad_enabled(True)
    return final_reconstruction
