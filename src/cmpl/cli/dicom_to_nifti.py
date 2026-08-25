"""
Command-line interface for CMPL DICOM to NIfTI conversion.
"""

import argparse
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="cmpl-dicom-to-nifti",
        description=(
            "Convert a DICOM series to NIfTI and create a "
            "matching JSON metadata sidecar."
        ),
    )

    parser.add_argument(
        "dicom_directory",
        type=Path,
        help="Path to the directory containing the DICOM series.",
    )

    parser.add_argument(
        "nifti_file",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Output NIfTI file. If omitted, the output is written "
            "to ./<series_directory_name>.nii.gz."
        ),
    )

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show conversion progress. "
            "Enabled by default; use --no-verbose to disable."
        ),
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    dicom_directory = args.dicom_directory.expanduser()

    if not dicom_directory.exists():
        parser.error(
            f"DICOM directory does not exist: "
            f"{dicom_directory}"
        )

    if not dicom_directory.is_dir():
        parser.error(
            f"DICOM input must be a directory: "
            f"{dicom_directory}"
        )

    if args.nifti_file is None:
        series_name = dicom_directory.resolve().name

        nifti_file = (
            Path.cwd()
            / f"{series_name}.nii.gz"
        )
    else:
        nifti_file = args.nifti_file.expanduser()

    # Import lazily so importing the CLI itself does not load
    # the heavier DICOM/NIfTI dependencies.
    try:
        from cmpl.utilities.io import dicom_to_nifti

    except ModuleNotFoundError as exc:
        if exc.name in {
            "nibabel",
            "pydicom",
            "SimpleITK",
            "h5py",
        }:
            parser.error(
                "DICOM/NIfTI support requires the CMPL I/O "
                "dependencies. Install them with:\n"
                "    pip install 'cmpl[io]'"
            )

        raise

    if args.verbose:
        print("CMPL DICOM to NIfTI conversion")
        print(f"DICOM series: {dicom_directory}")
        print(f"NIfTI output: {nifti_file}")
        print("Reading DICOM series...")

    dicom_to_nifti(
        dicom_directory,
        nifti_file,
        verbose=args.verbose,
    )

    if args.verbose:
        print("Conversion complete.")


if __name__ == "__main__":
    main()