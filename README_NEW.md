_# CMPL

CMPL is a Python toolkit for MRI processing workflows. Based on the `src` package, it provides utilities for:

- MRI reconstruction (GRAPPA and CG-SENSE)
- Quantitative MRI mapping (T2* fitting)
- Segmentation workflows and helpers
- DICOM/NIfTI/HDF5 I/O and data utilities
- Visualization helpers for MRI and segmentation overlays

## Top-level aliases exposed in `cmpl.__init__`:

- `cmpl.utilities` (alias `cmpl.utils`)
- `cmpl.visualization` (alias `cmpl.vis`)
- `cmpl.segmentation` (alias `cmpl.seg`)
- `cmpl.quantitative_MRI` (alias `cmpl.qmr`)
- `cmpl.reconstruction` (alias `cmpl.recon`)

## Installation

Requires Python `>=3.10`.

```bash
pip install cmpl
```

For local development from this repository:

```bash
pip install -e .
```

## Quantitative MRI API

File: `src/cmpl/quantitative_MRI/mapping.py`

### 2D fitting

- `t2_star_two_parametric_2D(TE_all, images, ...) -> (T2_star_map, S0_map)`
- `t2_star_three_parametric_2D(TE_all, images, ...) -> (T2_star_map, S0_map, C_map, loss_values)`

### 3D fitting

- `t2_star_two_parametric_3D(TE_all, images, ..., return_RMSE=False, loss_fn=None, device=None) -> dict`
- `t2_star_three_parametric_3D(TE_all, images, ...) -> (T2_star_map, S0_map)`

### Supporting utilities

- `reconstruct_images(T2_star_map, S0_map, TE_all, ...)`
- `calculate_rmse_percentage_s0(original_images, reconstructed_images, S0_map, ...)`

Notes:

- Several mapping functions explicitly call `.cuda()` and therefore require CUDA-compatible PyTorch.
- `t2_star_two_parametric_3D` is more device-flexible and accepts an explicit `device`.


## Reconstruction API

### 1D GRAPPA

File: `src/cmpl/reconstruction/grappa/grappa_1D.py`

- `grappa_1d_recon(calibration_kspace, undersampled_kspace, reduction_factor, kx, ky, is3D=False)`
- `grappa_1d_recon_slice(...)`

Notes:

- Expects k-space ordering: `[frequency, phase, slice, coils]`
- Undersampled data is expected to contain acquired lines at the 0th undersampling offset
- Returns NumPy complex k-space

### 2D GRAPPA

File: `src/cmpl/reconstruction/grappa/grappa_2D.py`

- `grappa_2d_recon(calibration_kspace, undersampled_kspace, kernel_size, reduction_factors)`

Notes:

- `kernel_size` is `(kx, ky, kz)`
- `reduction_factors` is `(Ry, Rz)`
- Uses convolution-based weight application
- Returns reconstructed k-space as a PyTorch tensor

### CG-SENSE

File: `src/cmpl/reconstruction/sense/cg.py`

- `CG_sense_2D(undersampled_image_space, coil_sensitivity, dims=[-3, -2])`

Notes:

- Uses iterative conjugate-gradient style updates (40 iterations)
- Infers sampling mask from `undersampled_image_space != 0`
- Works with complex PyTorch tensors



## Segmentation API

Files:

- `src/cmpl/segmentation/MRISegmentationTool.py`
- `src/cmpl/segmentation/tools.py`

### Auto segmentation workflow

Main class: `AutoSegmentation`

Important methods:

- `set_model(model, echos)`
- `load_model_state_dict(model_path)`
- `load_dicom_dir(directory)`
- `auto_segment()`
- `save_nifti(output_file_path)`
- `get_segmented()`
- `get_mri_matrix()`

Notes:

- Input preparation assumes a fixed multi-echo structure (notably 7 echoes in `load_dicom_dir`).
- Segmentation output is assembled from quadrant-wise inference.

### Segmentation helper

- `extract_extrusion(extrusion_path, seg_path, projection_value=11)`

Includes projection and filling logic for 3D mask processing.

## Utilities API

### I/O helpers

File: `src/cmpl/utilities/io.py`

- `nifti_read(file_name, re_orient=False)`
- `save_scalar_map_like(ref_img, data_in, out_path, ...)`
- `load_dicom_scan_from_dir(directory, reshape=True, verbose=False, with_spacing=False)`
- `update_nifti_data(file_path, new_data, output_path=None, dtype=np.float32)`
- `dicom_to_SimpleITK(dicom_directory)`
- `itk_to_nifti(itk_image, nifti_path, verbose=True)`
- `itk_mask_correction(img, mask, tol=1e-1, return_axis=False)`

### General helpers

File: `src/cmpl/utilities/utils.py`

- `h5_to_nifti(input_file, output_file)`
- `prepare_zipped_dicom(zip_path, extract_path)`
- `dicom_to_h5(dicom_directory, h5py_path, contrast='3D_gre_sag', num_contrasts=7, num_slices_per_contrast=120)`
- `kspace_to_image_space(kspace, fourier_dims=[0, 1, 2], coil_column_loc=-1, return_coil_images=False)`
- `apply_hamming_filter_4d_numpy(input_array, dim1, dim2)`
- `resize_complex_matrix_fft(image, target_shape)`
- `zero_pad(tensor, final_shape)`
- `resize_matrix(matrix, target_shape=(600, 600))`

### DataFrame builder

File: `src/cmpl/utilities/df_build.py`

- `build_medical_data_frame(root_dir)`

Builds a pandas DataFrame by traversing study folders and collecting DICOM, HDF5, and segmentation paths.

## Visualization API

File: `src/cmpl/visualization/visualization.py`

- `side_by_side_view(*images, color_palette='gray', dpi=100, titles=None, ...)`
- `visualize_segmentation_slice(grayscale_image, segmentation_matrix, slice_number, dimension='axial', target_shape=(600, 600))`
- `plot_3D_mri(mri_image, slice_number=None, direction='sagittal', segmentation=None, ...)`

`plot_3D_mri` switches between interactive and inline plotting modes depending on the active Matplotlib backend.

## Minimal Usage Example

```python
import cmpl

print(cmpl.__version__)

# Example: access modules through aliases
io = cmpl.utils.io
vis = cmpl.vis
qmr = cmpl.qmr
recon = cmpl.recon
seg = cmpl.seg
```

## Notes

- This document was created from the current `src` tree and reflects the present module/function layout.
- See `LICENSE` for license terms.
