# File created by: Eisa Hedayati
# Date: 8/29/2024
# Description: This file is developed at CMRR

import os
import torch as pt
import numpy as np
import nibabel as nib
import tempfile
import shutil
import dicom2nifti
from cmpl.utilities.io import load_dicom_scan_from_dir


def _center_crop_tensor(input_tensor, new_X, new_Y):
    X, Y = input_tensor.shape[:2]
    start_X = (X - new_X) // 2
    start_Y = (Y - new_Y) // 2
    return input_tensor[start_X:start_X + new_X, start_Y:start_Y + new_Y, ...]


class AutoSegmentation:
    def __init__(self, device=None, verbosity=0):
        self.__private_nifti_header = None
        self.__private_nifti_affine = None
        self.__private_mri_mat = None
        self.__private_dicom_dir = None
        self.__private_segmented = None
        self.__private_device = pt.device('cpu' if device is None else device)
        self.__private_model = None
        self.__private_verbosity = verbosity
        self.__private_echos = None

    def set_model(self, model, echos):
        self.__private_model = model
        self.__private_model.to(self.__private_device)
        self.__private_echos = echos
        if self.__private_verbosity > 0:
            print('Model set successfully!')

    def get_model(self):
        return self.__private_model

    def load_model_state_dict(self, model_path):
        # self.__private_model = self.model_init()
        assert self.__private_model is not None
        self.__private_model.load_state_dict(pt.load(model_path, map_location=self.__private_device, weights_only=True))
        if self.__private_verbosity > 0:
            print('Model loaded successfully')

    def save_nifti(self, output_file_path):
        if self.__private_segmented is None:
            print("No data to save")
            return

        if self.__private_nifti_affine is not None:
            new_nifti = nib.Nifti1Image(self.__private_segmented, affine=self.__private_nifti_affine, header=self.__private_nifti_header)
            nib.save(new_nifti, output_file_path)
            nib.save(new_nifti[..., ::-1], output_file_path + '_rev.nii')

            print(f'Saved file at: {output_file_path}')
        else:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            temp_dir = tempfile.mkdtemp(prefix="nifti_convert_", dir=script_dir)

            try:
                dicom2nifti.convert_directory(self.__private_dicom_dir, temp_dir)
                mri_nifti_path = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(('.nii', '.nii.gz'))][0]
                mri_nifti = nib.load(mri_nifti_path)

                self.__private_nifti_affine = mri_nifti.affine
                self.__private_nifti_header = mri_nifti.header

                new_nifti = nib.Nifti1Image(self.__private_segmented, affine=mri_nifti.affine, header=mri_nifti.header)
                nib.save(new_nifti, output_file_path)
                nib.save(new_nifti[..., ::-1], output_file_path + '_rev.nii')

                print(f'Saved file at: {output_file_path}')
            finally:
                shutil.rmtree(temp_dir)
                print('Removed temp files')

    def load_dicom_dir(self, directory):
        self.__private_dicom_dir = directory
        self.__private_mri_mat = load_dicom_scan_from_dir(directory, reshape=False)
        self.__private_mri_mat = np.moveaxis(self.__private_mri_mat, 0, -1)
        self.__private_mri_mat = self.__private_mri_mat.reshape(
            self.__private_mri_mat.shape[0], self.__private_mri_mat.shape[1], 7, self.__private_mri_mat.shape[2] // 7)
        self.__private_mri_mat = np.moveaxis(self.__private_mri_mat, -2, -1).transpose(1, 0, 2, 3)[..., ::-1, :]
        # self.__private_mri_mat = load_dicom_scan_from_dir(directory, reshape=True)
        self.__private_nifti_header = None
        self.__private_nifti_affine = None
        print('Finished loading dicom')

    def _process_quadrant(self, step_x, step_y, echo_indices=None):
        if echo_indices is None:
            echo_indices = [0]
        tensor = pt.tensor(self.__private_mri_mat[step_x::2, step_y::2, ..., echo_indices] / 65535.0, dtype=pt.float32).to(self.__private_device)
        cropped = _center_crop_tensor(tensor, 256, 256)
        padded_tensor = pt.zeros([cropped.shape[0], cropped.shape[1], cropped.shape[2] + 8, cropped.shape[-1]], dtype=pt.float32).to(self.__private_device)
        padded_tensor[:, :, :120] = cropped

        if len(echo_indices) > 1:
            batched_tensor = padded_tensor.unsqueeze(2).reshape(
                [padded_tensor.shape[0], padded_tensor.shape[1], padded_tensor.shape[2] // 32, -1, padded_tensor.shape[3]])
            batched_tensor = batched_tensor.moveaxis(2, 0).moveaxis(-1, 1)
        else:
            batched_tensor = padded_tensor.unsqueeze(2).reshape(
                [padded_tensor.shape[0], padded_tensor.shape[1], padded_tensor.shape[2] // 32, -1])
            batched_tensor = batched_tensor.moveaxis(2, 0).unsqueeze(1)
        return batched_tensor

    def _process_all_quadrants(self, echo_indices=None):
        if echo_indices is None:
            echo_indices = [0]
        quadrants = [
            self._process_quadrant(0, 0, echo_indices),
            self._process_quadrant(0, 1, echo_indices),
            self._process_quadrant(1, 0, echo_indices),
            self._process_quadrant(1, 1, echo_indices)
        ]
        return quadrants

    def _insert_matrix(self, small_matrix):
        target_shape = (small_matrix.shape[0], small_matrix.shape[1], self.__private_mri_mat.shape[0] // 2,
                        self.__private_mri_mat.shape[1] // 2, self.__private_mri_mat.shape[2])
        large_matrix = np.zeros(target_shape, dtype=small_matrix.dtype)

        start_idx3 = (target_shape[2] - small_matrix.shape[2]) // 2
        start_idx4 = (target_shape[3] - small_matrix.shape[3]) // 2

        large_matrix[:, :, start_idx3:start_idx3 + small_matrix.shape[2], start_idx4:start_idx4 + small_matrix.shape[3],
        :] = small_matrix[..., :self.__private_mri_mat.shape[2]]

        return large_matrix

    def auto_segment(self):

        mats = self._process_all_quadrants(self.__private_echos)
        with pt.no_grad():
            segmented = [pt.nn.functional.softmax(self.__private_model(mat), dim=1) for mat in mats]
            segmented = [seg.moveaxis(0, -2).reshape([seg.shape[1], seg.shape[2], seg.shape[3], -1]).cpu().numpy() for seg in segmented]
        segmented = np.stack(segmented, axis=0)
        segmented_large = self._insert_matrix(segmented)
        self.__private_segmented = np.zeros([segmented_large.shape[1], 2 * segmented_large.shape[2], 2 * segmented_large.shape[3],
                                             segmented_large.shape[4]])

        for i in range(4):
            self.__private_segmented[:, 0::2, 0::2] = segmented_large[0]  # Top-left
            self.__private_segmented[:, 0::2, 1::2] = segmented_large[1]  # Top-right
            self.__private_segmented[:, 1::2, 0::2] = segmented_large[2]  # Bottom-left
            self.__private_segmented[:, 1::2, 1::2] = segmented_large[3]

        seg_sum = np.zeros_like(self.__private_segmented[0])
        for i in range(1, self.__private_segmented.shape[0]):
            seg_sum += np.round(self.__private_segmented[i]) * i
        self.__private_segmented = seg_sum[..., ::-1]
        print('Segmentation completed')

    def get_segmented(self):
        assert self.__private_segmented is not None
        return self.__private_segmented

    def get_mri_matrix(self):
        assert self.__private_mri_mat is not None
        return self.__private_mri_mat

