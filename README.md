# MRIForge — MRI Processing Tools for Python

[![PyPI version](https://img.shields.io/pypi/v/mriforge.svg?cacheSeconds=300)](https://pypi.org/project/mriforge/)
[![Python versions](https://img.shields.io/pypi/pyversions/mriforge.svg)](https://pypi.org/project/mriforge/)

**PyPI:** https://pypi.org/project/mriforge/  
**GitHub:** https://github.com/ehedayati/mriforge

MRIForge is a modular Python toolkit for MRI processing workflows. It provides tools for DICOM/NIfTI conversion and I/O, quantitative MRI, MRI reconstruction, visualization, and supporting numerical and data utilities.

The base installation is intentionally lightweight. Larger or domain-specific dependencies are installed only when the corresponding functionality is needed.

> **Project rename:** MRIForge is the continuation of CMPL. Beginning with MRIForge 0.3.0, the PyPI distribution is `mriforge` and the Python import namespace is `mrif`.

## Highlights

- Geometry-aware conventional and Enhanced DICOM to NIfTI conversion
- JSON metadata sidecars with acquisition and source-geometry information
- Multi-echo DICOM support with 4D NIfTI output ordered by echo time
- `mriforge-dicom-to-nifti` command-line converter
- `mriforge-t2star` command-line T2* and S0 mapper
- DICOM geometry and acquisition-metadata utilities
- NIfTI data replacement while preserving image geometry and metadata
- Parallel MRI reconstruction with 1D/2D GRAPPA and conjugate-gradient SENSE
- Quantitative MRI tools for T2* fitting, signal reconstruction, and fitting-error analysis
- MRI visualization utilities for 2D comparisons and 3D volume browsing
- Conventional and Enhanced DICOM, NIfTI, HDF5, and SimpleITK utilities
- Lightweight numerical utilities shared across MRIForge
- Optional pandas-based indexing for medical-data directory structures
- Lazy imports so unrelated optional dependencies are not loaded unnecessarily
- Convenient aliases such as `mrif.recon`, `mrif.qmr`, `mrif.vis`, and `mrif.io`

## Requirements

MRIForge requires:

- Python >= 3.10
- NumPy >= 1.26, < 3
- SciPy >= 1.13, < 2
- tqdm >= 4.66

Additional functionality is provided through optional dependency groups.

## Installation

Install the lightweight base package:

```bash
python -m pip install mriforge
```

Install only the functionality you need:

| Extra | Purpose |
| --- | --- |
| `io` | DICOM, NIfTI, HDF5, and SimpleITK I/O |
| `data` | pandas-based data indexing |
| `viz` | Matplotlib and Jupyter visualization |
| `torch` | PyTorch-based reconstruction and quantitative MRI |
| `all` | Complete optional MRIForge environment |
| `dev` | Testing, linting, build, and release tools |

Examples:

```bash
python -m pip install "mriforge[io]"
python -m pip install "mriforge[torch]"
python -m pip install "mriforge[viz]"
python -m pip install "mriforge[io,torch]"
python -m pip install "mriforge[all]"
```

For NIfTI-based T2* mapping:

```bash
python -m pip install "mriforge[io,torch]"
```

Matplotlib is only required when plotting is explicitly requested:

```bash
python -m pip install "mriforge[viz]"
```

## Quick start

```python
import mrif

print(mrif.__version__)
```

MRIForge exposes convenient aliases for commonly used subpackages:

```python
mrif.recon   # reconstruction
mrif.qmr     # quantitative MRI
mrif.vis     # visualization
mrif.io      # I/O utilities
mrif.dicom   # DICOM metadata and geometry utilities
mrif.utils   # utilities
```

These aliases are resolved lazily, so `import mrif` does not require every optional dependency to be installed or imported.

## Migration from CMPL

MRIForge 0.3.0 introduces a new distribution name and Python namespace:

```text
Old PyPI package:    cmpl
New PyPI package:    mriforge

Old Python import:   cmpl
New Python import:   mrif
```

For example:

```python
# Before
import cmpl
from cmpl.utilities.io import nifti_read

# MRIForge
import mrif
from mrif.utilities.io import nifti_read
```

CLI commands also use the MRIForge name:

```text
cmpl-dicom-to-nifti  ->  mriforge-dicom-to-nifti
cmpl-t2star           ->  mriforge-t2star
```

---

## DICOM and NIfTI I/O

Install the I/O extra:

```bash
python -m pip install "mriforge[io]"
```

### Convert a DICOM series to NIfTI

MRIForge supports direct conversion of both conventional and Enhanced DICOM series to NIfTI.

The converter:

- detects the DICOM representation automatically
- preserves spatial geometry
- supports single-echo and multi-echo acquisitions
- writes a NIfTI image
- writes a matching JSON metadata sidecar
- orders multi-echo volumes by echo time

```python
import mrif

metadata = mrif.io.dicom_to_nifti(
    "/path/to/dicom_series",
    "output.nii.gz",
)
```

This creates:

```text
output.nii.gz
output.json
```

If the output path does not end in `.nii` or `.nii.gz`, MRIForge appends `.nii.gz`.

For a single echo, the output is a 3D image. For a multi-echo acquisition, MRIForge writes a 4D NIfTI with echoes in the last dimension:

```text
x, y, z, echo
```

The matching JSON sidecar contains acquisition metadata and source-geometry information. For multi-echo data, echo times are stored in milliseconds under:

```json
{
  "Acquisition": {
    "EchoTimes": [2.5, 5.0, 7.5, 10.0],
    "TimeUnit": "ms"
  }
}
```

If a directory contains multiple DICOM series, select a specific `SeriesInstanceUID`:

```python
metadata = mrif.io.dicom_to_nifti(
    "/path/to/dicom_directory",
    "output.nii.gz",
    series_id="1.2.840...",
)
```

### Command-line DICOM conversion

The I/O extra installs:

```text
mriforge-dicom-to-nifti
```

Convert a DICOM series with automatic conventional/Enhanced-DICOM detection:

```bash
mriforge-dicom-to-nifti /path/to/dicom_series
```

The same CLI can be invoked as a Python module:

```bash
python -m mrif.cli.dicom_to_nifti /path/to/dicom_series
```

If the output path is omitted, MRIForge writes the NIfTI and JSON sidecar to the current directory using the DICOM directory name:

```text
./<series_directory_name>.nii.gz
./<series_directory_name>.json
```

Specify an explicit output path if needed:

```bash
mriforge-dicom-to-nifti \
    /path/to/dicom_series \
    /path/to/output.nii.gz
```

Progress output is enabled by default. Disable it with:

```bash
mriforge-dicom-to-nifti /path/to/dicom_series --no-verbose
```

The same command handles conventional single-frame DICOM and Enhanced multi-frame DICOM.

### Read a NIfTI file

```python
from mrif.utilities.io import nifti_read

nifti_image, data = nifti_read("image.nii.gz")
```

### Replace NIfTI data while preserving the reference image

```python
from mrif.utilities.io import update_nifti_data

updated = update_nifti_data(
    "reference.nii.gz",
    new_data,
    output_path="updated.nii.gz",
)
```

By default, `update_nifti_data` preserves the reference image geometry, metadata, NIfTI image type, and source data type. The replacement array must have the same shape as the reference image.

A different output dtype can be requested explicitly:

```python
import numpy as np

updated = update_nifti_data(
    "reference.nii.gz",
    new_data,
    output_path="updated.nii.gz",
    dtype=np.float32,
)
```

For `.nii.gz` output, the gzip compression level can also be selected:

```python
updated = update_nifti_data(
    "reference.nii.gz",
    new_data,
    output_path="updated.nii.gz",
    compression_level=6,
)
```

### Save a scalar map using reference NIfTI geometry

```python
from mrif.utilities.io import save_scalar_map_like

save_scalar_map_like(
    reference_image,
    scalar_map,
    "map.nii.gz",
)
```

This is useful for quantitative maps such as T2* and S0 because the spatial geometry of the source NIfTI is preserved.

### Load a DICOM directory as a NumPy array

```python
from mrif.utilities.io import load_dicom_scan_from_dir

volume = load_dicom_scan_from_dir(
    "/path/to/dicom_directory",
    reshape=True,
)
```

For multi-echo data, the loader can return:

```text
x, y, z, echo
```

depending on the acquisition metadata and requested reshaping behavior.

### Read a DICOM series as SimpleITK

```python
from mrif.utilities.io import dicom_to_SimpleITK

image = dicom_to_SimpleITK("/path/to/dicom_directory")
```

The returned image is 3D for single-echo data and 4D when multiple echoes are detected.

### Write a SimpleITK image as NIfTI

```python
from mrif.utilities.io import itk_to_nifti

output_path = itk_to_nifti(
    image,
    "output.nii.gz",
)
```

### DICOM geometry and metadata helpers

MRIForge separates DICOM geometry and acquisition-metadata handling into dedicated modules under `mrif.dicom`.

```python
from mrif.dicom import (
    extract_slice_geometry,
    get_slice_position,
)

geometry = extract_slice_geometry("slice001.dcm")
position = get_slice_position("slice001.dcm")
```

Enhanced-DICOM helpers are also available:

```python
from mrif.dicom.enhanced_dicom import (
    get_slice_thickness,
    get_spacing_between_slices,
    voxel_sizes_detailed,
)

details = voxel_sizes_detailed(dataset)
```

---

## Quantitative MRI

Quantitative MRI functionality is available under:

```python
mrif.qmr
```

Install PyTorch support:

```bash
python -m pip install "mriforge[torch]"
```

For NIfTI-based quantitative MRI workflows:

```bash
python -m pip install "mriforge[io,torch]"
```

### Signal model

The current two-parameter T2* implementation uses the mono-exponential signal model:

```text
S(TE) = S0 * exp(-TE / T2*)
```

where:

- `S0` is the extrapolated signal at TE = 0
- `T2*` is the transverse relaxation time
- TE and T2* must use the same time unit

MRIForge conventionally uses milliseconds for T2* workflows.

### Fit a 3D two-parameter T2* model

```python
from mrif.quantitative_MRI import t2_star_two_parametric_3D

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

The expected image layout is:

```text
x, y, z, echo
```

If CUDA is available and no device is supplied, the fitter can select CUDA automatically.

### Command-line 3D T2* mapping

MRIForge includes a command-line interface for calculating T2* and S0 maps directly from a 4D multi-echo NIfTI file and its JSON metadata sidecar.

Install the required dependencies:

```bash
python -m pip install "mriforge[io,torch]"
```

Given:

```text
multi_echo.nii.gz
multi_echo.json
```

run:

```bash
mriforge-t2star multi_echo.nii.gz
```

The JSON sidecar is detected automatically when it has the same basename as the NIfTI file.

The CLI reads echo times from:

```json
{
  "Acquisition": {
    "EchoTimes": [2.5, 5.0, 7.5, 10.0],
    "TimeUnit": "ms"
  }
}
```

The input NIfTI must be 4D:

```text
x, y, z, echo
```

and the number of entries in `EchoTimes` must match the number of volumes in the fourth dimension.

The command writes:

```text
multi_echo_T2star.nii.gz
multi_echo_S0.nii.gz
```

The T2* map is written in milliseconds. The S0 map retains the signal-intensity units of the input data. Output maps preserve the spatial geometry of the source NIfTI.

Specify a different JSON file:

```bash
mriforge-t2star multi_echo.nii.gz \
    --json metadata.json
```

Specify a custom output prefix:

```bash
mriforge-t2star multi_echo.nii.gz \
    -o results/subject01
```

This creates:

```text
results/subject01_T2star.nii.gz
results/subject01_S0.nii.gz
```

Request CUDA explicitly:

```bash
mriforge-t2star multi_echo.nii.gz --device cuda
```

If no device is specified, MRIForge uses CUDA when available and otherwise uses CPU.

Optimization settings can also be adjusted:

```bash
mriforge-t2star multi_echo.nii.gz \
    --device cuda \
    --iterations 10000 \
    --lr 0.01 \
    --initial-t2star 20
```

If echo times are present but not ordered, the CLI sorts the echo times and their corresponding NIfTI volumes together before fitting.

### Reconstruct a multi-echo signal from T2* and S0 maps

```python
import numpy as np

from mrif.quantitative_MRI import reconstruct_images

t2_star = np.full((64, 64, 8), 20.0, dtype=np.float32)
s0 = np.full((64, 64, 8), 100.0, dtype=np.float32)
echo_times = np.array(
    [0.0, 5.0, 10.0, 15.0],
    dtype=np.float32,
)

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

### Calculate normalized fitting error

```python
from mrif.quantitative_MRI import calculate_rmse_percentage_s0

rmse_pct, rse_pct = calculate_rmse_percentage_s0(
    original_images,
    reconstructed_images,
    s0_map,
    return_numpy=True,
)
```

MRIForge also contains additional 2D/3D T2* fitting functions.

Plotting is optional. Matplotlib is imported only when plotting is requested.

---

## Reconstruction

Reconstruction functionality is available under:

```python
mrif.recon
```

Install PyTorch support:

```bash
python -m pip install "mriforge[torch]"
```

### 1D GRAPPA

```python
from mrif.reconstruction.grappa import grappa_1d_recon

reconstructed_kspace = grappa_1d_recon(
    calibration_kspace,
    undersampled_kspace,
    reduction_factor=2,
    kx=3,
    ky=3,
)
```

The current implementation expects coil-resolved k-space in the order:

```text
frequency, phase, slice, coils
```

### 2D GRAPPA

```python
from mrif.reconstruction.grappa import grappa_2d_recon

reconstructed_kspace = grappa_2d_recon(
    calibration_kspace,
    undersampled_kspace,
    kernel_size=(3, 3, 3),
    reduction_factors=(2, 2),
)
```

### Conjugate-gradient SENSE

```python
from mrif.reconstruction.sense.cg import CG_sense_2D

reconstructed_image = CG_sense_2D(
    undersampled_image_space,
    coil_sensitivity,
)
```

Inputs to the current SENSE implementation are PyTorch tensors.

---

## Visualization

Install the visualization extra:

```bash
python -m pip install "mriforge[viz]"
```

### Browse or display a 3D MRI volume

```python
from mrif.visualization import plot_3D_mri

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
from mrif.visualization import side_by_side_view

side_by_side_view(
    image_a,
    image_b,
    titles=["Reference", "Reconstruction"],
    color_palette="gray",
)
```

---

## Numerical utilities

Lightweight numerical helpers are kept separate from heavier I/O modules.

```python
from mrif.utilities.numerical import resize_matrix

resized = resize_matrix(
    image,
    target_shape=(600, 600),
)
```

`resize_matrix` accepts NumPy arrays and PyTorch tensors. PyTorch is imported only when a Torch tensor is actually passed.

For backward compatibility within the utilities package:

```python
from mrif.utilities.utils import resize_matrix
```

continues to work.

---

## Data indexing

Install the data extra:

```bash
python -m pip install "mriforge[data]"
```

MRIForge includes a pandas-based utility for indexing directory trees that follow the package's medical-data convention:

```python
from mrif.utilities.df_build import build_medical_data_frame

df = build_medical_data_frame("/path/to/root")
```

The expected structure is similar to:

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

This utility is convention-specific rather than a general-purpose filesystem indexer.

---

## Lazy loading and optional dependencies

MRIForge is designed so unrelated optional packages are not imported simply because the top-level package is imported.

For example:

```python
import mrif
```

does not immediately import PyTorch, Matplotlib, pandas, nibabel, pydicom, SimpleITK, h5py, or the Jupyter visualization stack.

Optional functionality is loaded only when the corresponding module or function is accessed.

Within quantitative MRI, Matplotlib is also loaded lazily: non-plotting T2* workflows do not require Matplotlib.

---

## Development

Clone the project and install it in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

For development across the main optional feature groups:

```bash
python -m pip install -e ".[dev,io,data,viz,torch]"
```

Run the full test suite:

```bash
python -m pytest tests/ -v
```

The test suite covers:

- lazy-import and dependency-boundary behavior
- qMRI numerical fitting
- T2* CLI validation and NIfTI output
- GRAPPA and SENSE reconstruction
- visualization smoke tests
- synthetic NIfTI, DICOM, and SimpleITK I/O
- conventional and Enhanced multi-echo DICOM-to-NIfTI conversion
- JSON sidecar generation
- DICOM geometry and metadata
- data indexing

### Build the package

Clean previous build artifacts and build the distributions:

```bash
rm -rf build dist *.egg-info src/*.egg-info
python -m build
```

Validate them:

```bash
python -m twine check dist/*
```

Before publishing, install the built wheel into a clean environment and verify the packaged CLI commands:

```bash
mriforge-dicom-to-nifti --help
mriforge-t2star --help
```

---

## Package layout

```text
src/mrif/
├── _version.py
├── cli/
│   ├── __init__.py
│   ├── dicom_to_nifti.py
│   └── t2star.py
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

---

## License

See the `LICENSE` file included with the project for licensing terms.

## Author

MRIForge is developed by Eisa Hedayati at CMRR.
