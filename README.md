# CMPL — CMRR MRI Processing Libraries

[![PyPI](https://img.shields.io/pypi/v/cmpl.svg)](https://pypi.org/project/cmpl/)
[![Python](https://img.shields.io/pypi/pyversions/cmpl.svg)](https://pypi.org/project/cmpl/)

CMPL is a Python package for MRI processing workflows developed at CMRR. It provides tools for MRI reconstruction, quantitative MRI, visualization, DICOM/NIfTI I/O, and supporting numerical/data utilities.

CMPL is organized as a modular library: the base installation stays lightweight, while larger or domain-specific dependencies are installed only when the corresponding functionality is needed.

## Highlights

- Parallel MRI reconstruction with 1D/2D GRAPPA and conjugate-gradient SENSE
- Quantitative MRI tools for T2* fitting, signal reconstruction, and fitting-error analysis
- MRI visualization utilities for 2D comparisons, 3D volume browsing, and segmentation overlays
- DICOM, enhanced-DICOM, NIfTI, and SimpleITK utilities
- Lightweight numerical utilities shared across CMPL
- Optional pandas-based indexing for CMPL-style medical-data directory structures
- Lazy package imports so `import cmpl` does not eagerly load large optional dependencies
- Backward-compatible convenience aliases such as `cmpl.recon`, `cmpl.qmr`, `cmpl.vis`, and `cmpl.utils`

## Requirements

CMPL requires:

- Python >= 3.10
- NumPy >= 1.26, < 3
- SciPy >= 1.13, < 2
- tqdm >= 4.66

Additional dependencies are installed through optional extras.

## Installation

Install the lightweight base package:

```bash
python -m pip install cmpl
```

Install only the functionality you need:

| Extra | Purpose |
| --- | --- |
| `io` | DICOM, NIfTI, HDF5, and SimpleITK I/O |
| `data` | pandas-based data indexing |
| `viz` | Matplotlib and Jupyter visualization |
| `torch` | PyTorch-based reconstruction and quantitative MRI |
| `all` | All optional functionality declared by CMPL |
| `dev` | Testing, linting, build, and release tools |

Examples:

```bash
python -m pip install "cmpl[io]"
python -m pip install "cmpl[viz]"
python -m pip install "cmpl[torch]"
python -m pip install "cmpl[io,data,viz,torch]"
python -m pip install "cmpl[all]"
```

Quantitative-MRI fitting currently uses both PyTorch and Matplotlib, so for qMRI workflows install:

```bash
python -m pip install "cmpl[torch,viz]"
```

## Quick start

```python
import cmpl

print(cmpl.__version__)
```

CMPL exposes convenient aliases for commonly used subpackages:

```python
cmpl.recon   # reconstruction
cmpl.qmr     # quantitative MRI
cmpl.vis     # visualization
cmpl.utils   # utilities
cmpl.io      # I/O utilities
```

The aliases are resolved lazily, so importing CMPL itself does not require every optional dependency to be loaded.

## Visualization

Install the visualization extra:

```bash
python -m pip install "cmpl[viz]"
```

### Browse or display a 3D MRI volume

```python
from cmpl.visualization import plot_3D_mri

plot_3D_mri(
    volume,
    slice_number=volume.shape[2] // 2,
    alpha=0.5,
    direction="sagittal",
    cmap="gray",
    vmin=0,
    vmax=1,
    dpi=300,
)
```

If an interactive Matplotlib backend is available, `plot_3D_mri` can use interactive controls. Otherwise it falls back to static redraw mode.


### Compare images side by side

```python
from cmpl.visualization import side_by_side_view

side_by_side_view(
    image_a,
    image_b,
    titles=["Reference", "Reconstruction"],
    color_palette="gray",
)
```

### Overlay a segmentation

```python
from cmpl.visualization import visualize_segmentation_slice

visualize_segmentation_slice(
    grayscale_image,
    segmentation,
    slice_number=20,
    dimension="axial",
)
```

## Quantitative MRI

Quantitative MRI functionality is available under:

```python
cmpl.qmr
```

Install the PyTorch and visualization dependencies:

```bash
python -m pip install "cmpl[torch,viz]"
```

### Reconstruct a multi-echo signal from T2* and S0 maps

```python
import numpy as np

from cmpl.quantitative_MRI import reconstruct_images

t2_star = np.full((64, 64, 8), 20.0, dtype=np.float32)
s0 = np.full((64, 64, 8), 100.0, dtype=np.float32)
echo_times = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float32)

images = reconstruct_images(
    t2_star,
    s0,
    echo_times,
    device="cpu",
    return_numpy=True,
)

print(images.shape)
# (64, 64, 8, 4)
```

The signal model is:

```text
S(TE) = S0 * exp(-TE / T2*)
```

### Fit a 3D two-parameter T2* model

```python
from cmpl.quantitative_MRI import t2_star_two_parametric_3D

result = t2_star_two_parametric_3D(
    echo_times,
    images,
    num_iterations=1000,
    initial_lr=0.01,
    initial_T2_star=20.0,
    plot_error=False,
    device="cpu",
)

t2_star_map = result["T2_star_map"]
s0_map = result["S0_map"]
```

If CUDA is available and no device is supplied, the function can select a CUDA device automatically. CUDA usage will increase computation speed significantly.

### Calculate normalized fitting error

```python
from cmpl.quantitative_MRI import calculate_rmse_percentage_s0

rmse_pct, rse_pct = calculate_rmse_percentage_s0(
    original_images,
    reconstructed_images,
    s0_map,
    return_numpy=True,
)
```

CMPL also contains two- and three-parameter 2D/3D T2* fitting functions.

## Reconstruction

Reconstruction functionality is available under:

```python
cmpl.recon
```

PyTorch is required for the current reconstruction implementations:

```bash
python -m pip install "cmpl[torch]"
```

### 1D GRAPPA

```python
from cmpl.reconstruction.grappa import grappa_1d_recon

reconstructed_kspace = grappa_1d_recon(
    calibration_kspace,
    undersampled_kspace,
    reduction_factor=2,
    kx=3,
    ky=3,
)
```

`calibration_kspace` and `undersampled_kspace` are expected to contain coil-resolved k-space data. The current implementation uses the ordering:

```text
frequency, phase, slice, coils
```

### 2D GRAPPA

```python
from cmpl.reconstruction.grappa import grappa_2d_recon

reconstructed_kspace = grappa_2d_recon(
    calibration_kspace,
    undersampled_kspace,
    kernel_size=(3, 3, 3),
    reduction_factors=(2, 2),
)
```

### Conjugate-gradient SENSE

```python
from cmpl.reconstruction.sense import CG_sense_2D

reconstructed_image = CG_sense_2D(
    undersampled_image_space,
    coil_sensitivity,
)
```

Inputs to the current SENSE implementation are PyTorch tensors.

## I/O

Install the I/O extra:

```bash
python -m pip install "cmpl[io]"
```

### Read a NIfTI file

```python
from cmpl.utilities.io import nifti_read

nifti_image, data = nifti_read("image.nii.gz")
```

### Replace NIfTI data while preserving geometry

```python
from cmpl.utilities.io import update_nifti_data

updated = update_nifti_data(
    "reference.nii.gz",
    new_data,
    output_path="updated.nii.gz",
)
```

### Load a DICOM directory

```python
from cmpl.utilities.io import load_dicom_scan_from_dir

volume = load_dicom_scan_from_dir(
    "/path/to/dicom_directory",
    reshape=True,
)
```

For multi-echo data, the loader can return data arranged as:

```text
x, y, z, echo
```

depending on the acquisition metadata and requested reshaping behavior.

### DICOM to SimpleITK

```python
from cmpl.utilities.io import dicom_to_SimpleITK

image = dicom_to_SimpleITK("/path/to/dicom_directory")
```

### Write a SimpleITK image as NIfTI

```python
from cmpl.utilities.io import itk_to_nifti

output_path = itk_to_nifti(
    image,
    "output.nii.gz",
)
```

### Enhanced-DICOM helpers

```python
from cmpl.dicom.enhanced_dicom import (
    get_slice_thickness,
    get_spacing_between_slices,
    voxel_sizes_detailed,
)

details = voxel_sizes_detailed(dataset)
```

## Numerical utilities

Lightweight numerical helpers are kept separate from heavier I/O modules so visualization and other numerical workflows do not require unrelated optional dependencies.

```python
from cmpl.utilities.numerical import resize_matrix

resized = resize_matrix(
    image,
    target_shape=(600, 600),
)
```

`resize_matrix` accepts NumPy arrays and PyTorch tensors. PyTorch is imported at runtime only when a Torch tensor is actually passed to the function.

For backward compatibility, older imports such as:

```python
from cmpl.utilities.utils import resize_matrix
```

continue to work.

## Data indexing

Install the data extra:

```bash
python -m pip install "cmpl[data]"
```

CMPL includes a pandas-based utility for indexing a directory tree that follows the CMPL medical-data convention:

```python
from cmpl.utilities.df_build import build_medical_data_frame

df = build_medical_data_frame("/path/to/root")
```

The utility is designed around a structure such as:

```text
root/
├── Study001/
│   ├── Dicoms/
│   │   └── <contrast>/
│   ├── h5_files/
│   │   └── <contrast>.h5
│   └── Segmentations/
│       └── <contrast>/
│           └── <group>/
│               └── <segmentation>.nii.gz
└── Study002/
    └── ...
```

This utility is convention-specific rather than a general filesystem indexer.

## Lazy loading and optional dependencies

CMPL is designed so that unrelated optional packages are not imported simply because the top-level package is imported.

For example:

```python
import cmpl
```

does not immediately require PyTorch, Matplotlib, pandas, nibabel, pydicom, SimpleITK, or h5py.

Optional functionality is loaded when its corresponding module or function is accessed. This keeps startup lightweight and allows users to install only the dependencies required for their workflow.

## Development

Clone the project and install it in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

For development across the currently tested major feature groups:

```bash
python -m pip install -e ".[dev,io,data,viz,torch]"
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

The test suite includes:

- lazy-import and dependency-boundary tests
- qMRI numerical tests
- GRAPPA and SENSE reconstruction tests
- visualization smoke tests
- synthetic NIfTI, DICOM, and SimpleITK I/O tests
- data-indexing tests

### Build the package

```bash
python -m build
python -m twine check dist/*
```

## Package layout

```text
src/cmpl/
├── dicom/
│   └── enhanced_dicom.py
├── quantitative_MRI/
│   └── mapping.py
├── reconstruction/
│   ├── grappa/
│   │   ├── grappa_1D.py
│   │   ├── grappa_2D.py
│   │   └── utils.py
│   └── sense/
│       └── cg.py
├── utilities/
│   ├── df_build.py
│   ├── io.py
│   ├── numerical.py
│   └── utils.py
└── visualization/
    └── visualization.py
```

## License

See the `LICENSE` file included with the project for licensing terms.

## Author

CMPL is developed by Eisa Hedayati at CMRR.
