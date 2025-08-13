# CMPL — CMRR MRI Processing Libraries

CMPL (CMRR MRI Processing Libraries) is a research-oriented Python toolkit for MRI data I/O, reconstruction, quantitative mapping, segmentation utilities, and visualization. It focuses on practical building blocks that can be composed into MRI processing workflows.

- Project homepage: this repository
- License: see LICENSE
- Python: >= 3.10

## Installation
CMPL is published as a Python package. A standard installation will also install the required dependencies listed in pyproject.toml.

```bash
pip install cmpl
```

Notes
- Some features require optional system libraries (e.g., SimpleITK) and a working CUDA-enabled PyTorch if you plan to use GPU-accelerated routines (quantitative mapping functions currently expect CUDA tensors).
- If you use Jupyter for visualization widgets, ensure ipywidgets is enabled in your environment.

## Package overview
CMPL exposes the following top-level namespaces for convenience (via cmpl.__init__):
- cmpl.utilities (alias cmpl.utils)
- cmpl.visualization (alias cmpl.vis)
- cmpl.segmentation (alias cmpl.seg)
- cmpl.quantitative_MRI (alias cmpl.qmr)
- cmpl.reconstruction (alias cmpl.recon)

You can also access the package version at runtime:
```python
import cmpl
print(cmpl.__version__)
```

## Key features and APIs

### 1) MRI k-space reconstruction (GRAPPA, SENSE)
Location: src/cmpl/reconstruction

GRAPPA (Generalized Autocalibrating Partially Parallel Acquisitions)
- 1D GRAPPA: cmpl.reconstruction.grappa.grappa_1D.grappa_1d_recon
- 2D GRAPPA: cmpl.reconstruction.grappa.grappa_2D.grappa_2d_recon

Axis ordering for GRAPPA
- Expected k-space shape: [frequency, phase, slice, coils]
- For 1D GRAPPA, a 3D variant is supported via is3D flag; internally, slices are handled along the third dimension.
- The undersampled k-space must contain acquired data in the 0th column of the undersampled positions.

Example — 1D GRAPPA (slice-wise)
```python
import numpy as np
from cmpl.reconstruction.grappa.grappa_1D import grappa_1d_recon

# calibration_kspace, undersampled_kspace: complex64 numpy arrays of shape [freq, phase, slices, coils]
R = 2                    # reduction factor along phase-encode
kx, ky = 5, 3            # kernel size (height, width)
recon_kspace = grappa_1d_recon(calibration_kspace, undersampled_kspace, R, kx, ky, is3D=False)
```

Example — 2D GRAPPA (accelerated in two phase directions)
```python
import numpy as np
from cmpl.reconstruction.grappa.grappa_2D import grappa_2d_recon

# calibration_kspace, undersampled_kspace: complex64 numpy arrays of shape [freq, phase, slice, coils]
kernel_size = (5, 3, 3)            # (height, width, depth) in k-space blocks
reduction_factors = (2, 2)         # (phase_undersampling, slice_undersampling)
recon_kspace = grappa_2d_recon(calibration_kspace, undersampled_kspace, kernel_size, reduction_factors)
```

SENSE (CG-SENSE 2D)
- cmpl.reconstruction.sense.cg.CG_sense_2D(undersampled_image_space, coil_sensitivity, dims=[-3, -2])
- Input/Output are torch.complex tensors. A binary mask is inferred as (undersampled_image_space != 0).

Example — CG-SENSE (2D)
```python
import torch as pt
from cmpl.reconstruction.sense.cg import CG_sense_2D

# undersampled_image_space: complex tensor [..., x, y, coils]
# coil_sensitivity: complex tensor [..., x, y, coils]
final_recon = CG_sense_2D(undersampled_image_space, coil_sensitivity, dims=[-3, -2])
```

Utility — Convert k-space to image space
```python
import numpy as np
from cmpl.utilities.utils import kspace_to_image_space

# kspace: shape [..., coils] (coil dimension may be last or specified via coil_column_loc)
combined_image, coil_images = kspace_to_image_space(kspace, fourier_dims=[0,1,2], coil_column_loc=-1, return_coil_images=True)
```

### 2) Quantitative MRI — T2* mapping
Location: src/cmpl/quantitative_MRI/mapping.py

Functions (GPU expected; uses PyTorch CUDA internally)
- t2_star_two_parametric_2D(TE_all, images, ...)
  - images shape: (x, y, TE)
  - returns (T2_star_map, S0_map)
- t2_star_three_parametric_2D(TE_all, images, ...)
  - images shape: (x, y, TE); includes an offset C parameter
  - returns (T2_star_map, S0_map, C_map)
- t2_star_two_parametric_3D(TE_all, images, ...)
  - images shape: (x, y, z, TE)
- t2_star_three_parametric_3D(TE_all, images, ...)
  - images shape: (x, y, z, TE)
- reconstruct_images(T2_star_map, S0_map, TE_all)
- calculate_rmse_percentage_s0(original_images, reconstructed_images, S0_map)

Example — 2D two-parameter T2*
```python
import numpy as np
from cmpl.quantitative_MRI.mapping import t2_star_two_parametric_2D

TE_all = np.array([3.5, 8.0, 12.5, 17.0], dtype=np.float32)
images = np.random.rand(256, 256, len(TE_all)).astype(np.float32)
T2_star_map, S0_map = t2_star_two_parametric_2D(TE_all, images, num_iterations=2000, initial_lr=0.01)
```

GPU requirement
- These functions call .cuda() on tensors. Ensure a CUDA-enabled PyTorch installation and a supported GPU.

### 3) Segmentation utilities
Location: src/cmpl/segmentation

- tools.py: Projection helpers for 3D label volumes; reading/writing NIfTI through cmpl.utilities.io.
  - project_3d_matrix(matrix, value, axis)
  - extract_extrusion(extrusion_path, seg_path, projection_value=11)

- MRISegmentationTool.py: AutoSegmentation helper around a user-provided PyTorch model.
  - AutoSegmentation.set_model(model, echos)
  - AutoSegmentation.load_model_state_dict(model_path)
  - AutoSegmentation.load_dicom_dir(directory)
  - AutoSegmentation.auto_segment()
  - AutoSegmentation.save_nifti(output_file_path)

Example — AutoSegmentation workflow
```python
import torch as pt
from cmpl.segmentation.MRISegmentationTool import AutoSegmentation

model = ...                # your torch.nn.Module
my_echos = [0,1,2,3,4,5,6] # indices of echoes the model expects
seg = AutoSegmentation(device='cuda', verbosity=1)
seg.set_model(model, my_echos)
seg.load_model_state_dict('path/to/model_weights.pth')
seg.load_dicom_dir('path/to/Dicoms')
seg.auto_segment()
seg.save_nifti('path/to/output_seg.nii.gz')
```

### 4) I/O and format conversion
Location: src/cmpl/utilities/io.py

- nifti_read(path, re_orient=True) -> (nifti, data)
- load_dicom_scan_from_dir(directory, reshape=True, verbose=False, with_spacing=False)
  - Returns numpy array with shape [x, y, z] or [x, y, z, echo]; optional (origin, spacing, orientation)
- update_nifti_data(file_path, new_data, output_path=None)
- dicom_to_SimpleITK(dicom_directory) -> sitk.Image (3D or 4D when multi-echo)
- itk_to_nifti(itk_image, nifti_path, verbose=True)
- itk_mask_correction(img_nifti, mask_nifti, tol=1e-1, return_axis=False)

Example — Load DICOM series and save a NIfTI copy
```python
from cmpl.utilities.io import load_dicom_scan_from_dir, update_nifti_data
imgs = load_dicom_scan_from_dir('path/to/dicoms', reshape=True)
# ... process imgs ...
update_nifti_data('template.nii.gz', imgs, 'processed.nii.gz')
```

### 5) General utilities
Location: src/cmpl/utilities/utils.py and df_build.py

- h5_to_nifti(input_file, output_file)
- prepare_zipped_dicom(zip_path, extract_path)
- dicom_to_h5(dicom_directory, h5py_path, contrast='3D_gre_sag', num_contrasts=7, num_slices_per_contrast=120)
- kspace_to_image_space(kspace, fourier_dims=[0,1,2], coil_column_loc=-1, return_coil_images=False)
- apply_hamming_filter_4d_numpy(array4d, dim1, dim2)
- resize_complex_matrix_fft(image, target_shape)
- zero_pad(tensor_or_array, final_shape)
- resize_matrix(matrix2d, target_shape=(600,600))
- df_build.build_medical_data_frame(root_dir) -> pandas.DataFrame built from a specific folder layout (Dicoms, h5_files, Segmentations)

### 6) Visualization
Location: src/cmpl/visualization/visualization.py

- side_by_side_view(*images, color_palette='gray', dpi=100, titles=None)
- visualize_segmentation_slice(grayscale_image, segmentation_matrix, slice_number, dimension='axial', target_shape=(600,600))
- plot_3D_mri(mri_image, slice_number=None, direction='sagittal', segmentation=None, alpha=0.5, dpi=150, target_shape=None, m_cmap='gray')

Example — Overlay segmentation on an MRI slice
```python
from cmpl.visualization.visualization import visualize_segmentation_slice
visualize_segmentation_slice(mri_3d, seg_3d, slice_number=50, dimension='axial', target_shape=(600,600))
```

## Data types and conventions
- Complex arrays are represented as numpy complex64 or torch.complex64 depending on the function.
- Unless otherwise stated, GRAPPA functions expect arrays ordered as [frequency, phase, slice, coils]. Use numpy.moveaxis to adjust ordering if needed.
- Quantitative mapping routines currently allocate tensors on CUDA. If you do not have a GPU, consider adapting the code (removing .cuda()) or using a CUDA-enabled environment.

## Development notes
- The codebase uses PyTorch, NumPy/SciPy, nibabel, pydicom, and SimpleITK. See pyproject.toml for exact versions.
- Progress bars are provided via tqdm in some routines.

## How to cite
If you use CMPL in a scientific publication, please cite the toolkit and the underlying algorithms (GRAPPA, SENSE, etc.). A formal citation entry will be provided in future releases.

## License
See LICENSE in the repository.
