# File created by: Eisa Hedayati
# Date: 3/12/2024
# Description: This file is developed at CMRR
import numpy as np
import torch as pt
import torch.nn as nn

from .utils import pad_if_required

def grappa_1d_recon(calibration_kspace, undersampled_kspace, reduction_factor, kx, ky, is3D=False):
    """
        Reconstruct k-space data using 1D GRAPPA algorithm for the entire image. undersampled_kspace has to
         have data in the 0th column.

        :param calibration_kspace: The calibration k-space data.
        :param undersampled_kspace: The undersampled k-space data.
        :param reduction_factor: The reduction factor used for undersampling.
        :param kx: The kernel size in the x-dimension.
        :param ky: The kernel size in the y-dimension.
        :param is3D: True for 3D acquisition, False for 2D acquisition
        :return: The reconstructed k-space data.
        """
    pt.set_grad_enabled(False)
    ifft_ = lambda X, ax: pt.fft.fftshift(pt.fft.ifftn(pt.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax)
    undersampled_kspace = pad_if_required(undersampled_kspace, reduction_factor)

    if is3D:
        undersampled_kspace = ifft_(pt.tensor(undersampled_kspace), 2).numpy()
        calibration_kspace = ifft_(pt.tensor(calibration_kspace), 2).numpy()

    recond_kspace = np.zeros_like(undersampled_kspace, dtype=np.complex64)

    for i in range(undersampled_kspace.shape[2]):
        recond_kspace[...,i,:] = grappa_1d_recon_slice(calibration_kspace[...,i,:], undersampled_kspace[...,i,:]
                                                       , reduction_factor, kx, ky)
    pt.set_grad_enabled(True)
    return recond_kspace


def grappa_1d_recon_slice(calibration_kspace, undersampled_kspace, reduction_factor, kx, ky):
    """
    Reconstruct k-space data using 1D GRAPPA algorithm. undersampled_kspace has to have data in the 0th column

    :param calibration_kspace: The calibration k-space data.
    :param undersampled_kspace: The undersampled k-space data.
    :param reduction_factor: The reduction factor used for undersampling.
    :param kx: The kernel size in the x-dimension.
    :param ky: The kernel size in the y-dimension.
    :return: The reconstructed k-space data.
    """
    # Kernel size tuple
    kernel_size = (kx, ky)

    # Building Reconstruction Targets and Sources
    reconstruction_targets, reconstruction_sources = build_reconstruction_targets_sources(
        calibration_kspace, reduction_factor, kernel_size)

    # Number of rows and columns in the calibration k-space, and the number of coils
    n_rows, n_cols = calibration_kspace.shape[:2]
    acs_lines = calibration_kspace.shape[1]  # Auto-calibration signal lines
    nc = undersampled_kspace.shape[2]  # Number of coils

    # Calculating Reconstruction Weights
    reconstruction_weights = calculate_reconstruction_weights(
        reconstruction_sources, reconstruction_targets, reduction_factor)

    # Applying 1D Grappa
    final_reconstruction = reconstruct_kspace_1D_grappa(
        undersampled_kspace, reconstruction_weights, kernel_size, reduction_factor, nc)

    return final_reconstruction


def build_reconstruction_targets_sources(calibration_kspace, reduction_factor, kernel_size):
    """
    This function creates two matrices from the ACS lines provided. The first one is
    the sources, meaning the known values, and the second one is the target or the
    values to be reconstructed from the sources. These matrices are later used for training
    the grappa kernel weights.
    :param calibration_kspace:
    :param reduction_factor:
    :param kernel_size:
    :return:
    np.array reconstruction_targets: targetrs to be reconstructed from sources
    np.array reconstruction_sources: sources that should be used for reconstructing targets
    """
    # Precompute valid row and column indices
    n_rows, n_columns = calibration_kspace.shape[:2]

    # Initialize lists for targets and sources
    reconstruction_targets = []
    reconstruction_sources = []

    for row in range(n_rows-kernel_size[0]):
        source_row = calibration_kspace[row: row + kernel_size[0]]
        for column in range(n_columns):
            #building source matrix
            reconstruction_columns = column + np.arange(kernel_size[1]) * reduction_factor
            if np.sum(reconstruction_columns >= n_columns) != 0:
                break
            sub_source = source_row[:,reconstruction_columns].flatten()
            reconstruction_sources.append(sub_source)
            # building target matrix
            sub_target = source_row[kernel_size[0]//2,
                         column+reduction_factor*(kernel_size[1])//2:column + (1+kernel_size[1]//2)*reduction_factor]
            reconstruction_targets.append(sub_target)
    reconstruction_targets = np.array(reconstruction_targets)
    reconstruction_sources = np.array(reconstruction_sources)
    return reconstruction_targets, reconstruction_sources


def calculate_reconstruction_weights(reconstruction_sources, reconstruction_targets, reduction_factor):
    reconstruction_sources_pinv = np.linalg.pinv(reconstruction_sources)
    # for vectorizing the pseudo inverses are repeated accross R
    reconstruction_sources_pinv = np.repeat(np.expand_dims(reconstruction_sources_pinv,0), reduction_factor - 1, axis=0)
    # We need to move the second dimension to the last position to align it for matmul
    reconstruction_targets_moved = np.moveaxis(reconstruction_targets, 1, 0)
    # batch matmul
    reconstruction_weights = np.matmul(reconstruction_sources_pinv, reconstruction_targets_moved)

    return reconstruction_weights


def reconstruct_kspace_1D_grappa(undersampled_kspace, reconstruction_weights, kernel_size, reduction_factor, nc):
    reconstructed_kspace = undersampled_kspace.copy()
    for i in range(reduction_factor-1):
        reconstructed_kspace[:,i+1::reduction_factor] = weight_application_1D_grappa(
            undersampled_kspace, reconstruction_weights[i], kernel_size, reduction_factor, nc)
    return reconstructed_kspace


def weight_application_1D_grappa(undersampled_kspace, reconstruction_weights, kernel_size, reduction_factor, nc):
    """
    Applying 1D Grappa weights to reconstruct the undersampled kspace.

    Parameters:
    - undersampled_kspace: The input tensor representing undersampled k-space data.
    - reconstruction_weights: The weights to be used in the convolution.
    - kernel_size: A tuple specifying the size of the kernel (depth, height).
    - reduction_factor: The reduction factor used for dilation and stride in the height dimension.
    - nc: The number of output channels.

    Returns:
    - output_volume: The output tensor after applying the 3D convolution.
    """
    pt.set_grad_enabled(False)
    # Prepare the input tensor
    undersampled_tensor = pt.tensor(undersampled_kspace).unsqueeze_(0).unsqueeze_(0)

    # Prepare the kernel weights
    kernel_weights = pt.tensor(
        reconstruction_weights.reshape(1, kernel_size[0], kernel_size[1], nc, nc)).moveaxis(-1, 0)

    # Set convolution parameters
    dilation = (1, reduction_factor, 1)
    stride = (1, reduction_factor, 1)
    column_pad = (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1) - 1)
    padding = (
        (kernel_size[0] - 1) // 2,
        column_pad // 2,
        0
    )

    # Initialize the convolution layer
    conv3d = nn.Conv3d(in_channels=1, out_channels=nc, kernel_size=(kernel_size[0], kernel_size[1], nc),
                       padding=padding, stride=stride, dilation=dilation, dtype=pt.complex64)

    # Set the convolution layer weights and biases
    conv3d.weight.data = kernel_weights
    conv3d.bias.data.fill_(0)

    # Apply the convolution
    output_volume = conv3d(undersampled_tensor)
    pt.set_grad_enabled(True)
    return output_volume.squeeze().moveaxis(0, -1)