# File created by: Eisa Hedayati
# Date: 5/21/2024
# Description: This file is developed at CMRR

import torch as pt
import numpy as np
import sys
from tqdm import tqdm


def CG_sense_2D(undersampled_image_space, coil_sensitivity, dims=[-3, -2]):
    fft_pt = lambda X, ax: pt.fft.fftshift(pt.fft.fftn(pt.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax)
    ifft_pt = lambda X, ax: pt.fft.fftshift(pt.fft.ifft2(pt.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax)
    """
    2D CG_sense application
    :param undersampled_image_space:
    :param coil_sensitivity:
    :param mask: undersampled_kspace  = fully_sampled_kspace .* mask (elementwise multiplication)
    :return:
    """

    def E(image, coil, mask):
        rep_image = image.unsqueeze(3).repeat(1, 1, 1, coil.shape[3])
        return fft_pt(rep_image * coil, dims) * mask

    def EH(kspace, coil, mask):
        return pt.sum(ifft_pt(kspace * mask, dims) * pt.conj(coil), dim=-1)

    def EHE(image, coil, mask):
        return EH(E(image, coil, mask), coil, mask)

    mask = undersampled_image_space != 0
    epsilon = sys.float_info.epsilon

    estimated_im = EH(undersampled_image_space, coil_sensitivity, mask)
    final_recon = pt.zeros_like(estimated_im)
    r = pt.clone(estimated_im)

    for i in tqdm(range(40)):
        q = EHE(estimated_im, coil_sensitivity, mask)
        rsold = r * pt.conj(r)
        alpha = pt.sum(rsold) / pt.sum(q * pt.conj(estimated_im) + epsilon)
        final_recon = final_recon + alpha * estimated_im
        r = r - alpha * q
        rsnew = r * pt.conj(r)
        estimated_im = r + (pt.sum(rsnew) / pt.sum(rsold + epsilon)) * estimated_im

    return final_recon
