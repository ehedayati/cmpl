# CMPL — CMRR MRI Processing Libraries

[![PyPI version](https://img.shields.io/pypi/v/cmpl.svg)](https://pypi.org/project/cmpl/)
[![Python versions](https://img.shields.io/pypi/pyversions/cmpl.svg)](https://pypi.org/project/cmpl/)

PyPI: https://pypi.org/project/cmpl/

GitHub: https://github.com/ehedayati/cmpl

CMPL is a Python package for MRI processing workflows developed at CMRR. It provides tools for MRI reconstruction, quantitative MRI, visualization, DICOM/NIfTI conversion and I/O, and supporting numerical and data utilities.

CMPL is organized as a modular library: the base installation stays lightweight, while larger or domain-specific dependencies are installed only when the corresponding functionality is needed.

## Highlights

- Direct, geometry-aware conventional and Enhanced DICOM to NIfTI conversion with JSON metadata sidecars
- Multi-echo DICOM support, including 4D NIfTI output ordered by echo time
- Packaged `cmpl-dicom-to-nifti` command-line converter
- DICOM geometry and acquisition-metadata utilities
- Parallel MRI reconstruction with 1D/2D GRAPPA and conjugate-gradient SENSE
- Quantitative MRI tools for T2* fitting, signal reconstruction, and fitting-error analysis
- MRI visualization utilities for 2D comparisons and 3D volume browsing
- Conventional and Enhanced DICOM, NIfTI, HDF5, and SimpleITK utilities
- Lightweight numerical utilities shared across CMPL
- Optional pandas-based indexing for CMPL-style medical-data directory structures
- Lazy package imports so `import cmpl` does not eagerly load large optional dependencies
- Convenient aliases such as `cmpl.recon`, `cmpl.qmr`, `cmpl.vis`, and `cmpl.io`

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
cmpl.io      # I/O utilities
cmpl.dicom   # DICOM metadata and geometry utilities
cmpl.utils   # utilities
```

The aliases are resolved lazily, so importing CMPL itself does not require every optional dependency to be loaded.

## DICOM and NIfTI I/O

Install the I/O extra:

```bash
python -m pip install "cmpl[io]"
```

### Convert a DICOM series directly to NIfTI

CMPL includes direct DICOM-series to NIfTI conversion for both conventional and Enhanced DICOM. The converter detects the DICOM representation automatically, preserves spatial geometry, supports single- and multi-echo acquisitions, writes the NIfTI image, and creates a matching JSON sidecar containing acquisition metadata and source geometry.

```python
import cmpl

metadata = cmpl.io.dicom_to_nifti(
    "/path/to/dicom_series",
    "output.nii.gz",
)
```

This creates:

```text
output.nii.gz
output.json
```

If the output path does not end in `.nii` or `.nii.gz`, CMPL appends `.nii.gz` automatically.

For a single echo, the output is a 3D image. When multiple echoes are present, CMPL groups volumes by DICOM `EchoTime`, orders them by echo time, and writes a 4D NIfTI image.

The JSON sidecar includes:

- selected acquisition metadata
- echo time or echo times in milliseconds
- original DICOM slice-plane geometry in LPS coordinates
- the SimpleITK image size, origin, spacing, and direction used for conversion

The returned value is the same metadata dictionary written to the JSON sidecar.

If a directory contains multiple DICOM series, a specific `SeriesInstanceUID` can be selected:

```python
metadata = cmpl.io.dicom_to_nifti(
    "/path/to/dicom_directory",
    "output.nii.gz",
    series_id="1.2.840...",
)
```

### Command-line DICOM conversion

Installing the I/O extra also installs the `cmpl-dicom-to-nifti` command:

```bash
python -m pip install "cmpl[io]"
```

Convert a DICOM series with automatic conventional/Enhanced-DICOM detection:

```bash
cmpl-dicom-to-nifti /path/to/dicom_series
```

The same CLI can also be invoked as a Python module:

```bash
python -m cmpl.cli.dicom_to_nifti /path/to/dicom_series
```

If the output path is omitted, CMPL writes the NIfTI image and JSON sidecar to the current directory using the DICOM directory name:

```text
./<series_directory_name>.nii.gz
./<series_directory_name>.json
```

An explicit output path can also be supplied:

```bash
cmpl-dicom-to-nifti \
    /path/to/dicom_series \
    /path/to/output.nii.gz
```

Progress output is enabled by default. Disable it with:

```bash
cmpl-dicom-to-nifti /path/to/dicom_series --no-verbose
```

The same CLI handles conventional single-frame DICOM series and Enhanced multi-frame DICOM series. For multi-echo data, echo volumes are ordered by EchoTime and written as a 4D NIfTI image.

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

### Load a DICOM directory as a NumPy array

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

### Read a DICOM series as SimpleITK

```python
from cmpl.utilities.io import dicom_to_SimpleITK

image = dicom_to_SimpleITK("/path/to/dicom_directory")
```

The returned image is 3D for a single echo and 4D when multiple echoes are detected.

### Write a SimpleITK image as NIfTI

```python
from cmpl.utilities.io import itk_to_nifti

output_path = itk_to_nifti(
    image,
    "output.nii.gz",
)
```

### DICOM geometry and metadata helpers

CMPL separates DICOM geometry and acquisition-metadata handling into dedicated modules under `cmpl.dicom`.

```python
from cmpl.dicom import (
    extract_slice_geometry,
    get_slice_position,
)

geometry = extract_slice_geometry("slice001.dcm")
position = get_slice_position("slice001.dcm")
```

Enhanced-DICOM helpers remain available as well:

```python
from cmpl.dicom.enhanced_dicom import (
    get_slice_thickness,
    get_spacing_between_slices,
    voxel_sizes_detailed,
)

details = voxel_sizes_detailed(dataset)
```

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
from cmpl.reconstruction.sense.cg import CG_sense_2D

reconstructed_image = CG_sense_2D(
    undersampled_image_space,
    coil_sensitivity,
)
```

Inputs to the current SENSE implementation are PyTorch tensors.

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

If CUDA is available and no device is supplied, the function can select a CUDA device automatically. CUDA usage can significantly accelerate fitting.

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
│   └── h5_files/
│       └── <contrast>.h5
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

does not immediately import PyTorch, Matplotlib, pandas, nibabel, pydicom, SimpleITK, h5py, or the Jupyter visualization stack.

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
- conventional and Enhanced multi-echo DICOM-to-NIfTI conversion and JSON-sidecar tests
- DICOM geometry and metadata tests
- data-indexing tests

### Build the package

```bash
python -m build
python -m twine check dist/*
```

## Package layout

```text
src/cmpl/
├── _version.py
├── cli/
│   └── dicom_to_nifti.py
├── dicom/
│   ├── enhanced_dicom.py
│   ├── geometry.py
│   └── metadata.py
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