# File created by: Eisa Hedayati
# Date: 3/19/2024
# Description: This file is developed at CMRR
import numpy as np
import torch as pt
pt.set_grad_enabled(False)


def pad_if_required(undersampled_kspace, reduction_factors):
    # Ensure reduction_factors is of length 2 for uniformity, use dummy value for 2nd element if len is 1
    is_1D = False
    if not isinstance(reduction_factors, list):
        is_1D = True
    if is_1D:
        reduction_factors_expanded = (reduction_factors, 1)
    else:
        reduction_factors_expanded = reduction_factors

    mods = np.mod(undersampled_kspace.shape[1:3], reduction_factors_expanded)
    R = np.array(reduction_factors_expanded)
    mods[mods == 0] = R[mods == 0]
    pads = R - mods

    if is_1D:
        # Only pad along the dimension corresponding to the reduction factor, no padding for the 3rd dimension
        padding_config = ((0, 0), (0, pads[0]), (0, 0), (0, 0))
    else:
        # Pad along both dimensions as specified by the 2D reduction factors
        padding_config = ((0, 0), (0, pads[0]), (0, pads[1]), (0, 0))

    return np.pad(undersampled_kspace, padding_config)