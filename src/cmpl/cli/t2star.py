# File created by: Eisa Hedayati
# Date: 8/26/2026
# Description: This file is developed at CMRR
"""
Command-line interface for 3D T2* calculation.

Input:
    4D multi-echo NIfTI: X x Y x Z x echoes
    JSON sidecar containing Acquisition.EchoTimes

Output:
    3D T2* map
    3D S0 map
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _nifti_stem(path: Path) -> str:
    """Return filename without .nii or .nii.gz."""

    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    raise ValueError(
        "Input must have a .nii or .nii.gz extension."
    )


def _build_parser():

    parser = argparse.ArgumentParser(
        prog="cmpl-t2star",
        description=(
            "Calculate 3D T2* and S0 maps from a 4D "
            "multi-echo NIfTI and JSON metadata sidecar."
        ),
    )

    parser.add_argument(
        "nifti_file",
        type=Path,
        help=(
            "4D multi-echo NIfTI. "
            "Echoes must be in the last dimension."
        ),
    )

    parser.add_argument(
        "--json",
        dest="json_file",
        type=Path,
        default=None,
        help=(
            "JSON metadata file. If omitted, a JSON file "
            "with the same basename as the NIfTI is used."
        ),
    )

    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix. Default: input NIfTI basename."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "PyTorch device: auto, cpu, cuda, cuda:0, etc. "
            "Default: auto."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of fitting iterations. Default: 10000.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial Adam learning rate. Default: 0.01.",
    )

    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.1,
        help="Learning-rate decay factor. Default: 0.1.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Learning-rate scheduler patience. Default: 100.",
    )

    parser.add_argument(
        "--initial-t2star",
        type=float,
        default=20.0,
        help="Initial T2* value in ms. Default: 20.",
    )

    return parser


def main():

    parser = _build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Input NIfTI
    # ---------------------------------------------------------

    nifti_file = args.nifti_file.expanduser()

    if not nifti_file.exists():
        parser.error(
            f"NIfTI file does not exist: {nifti_file}"
        )

    try:
        stem = _nifti_stem(nifti_file)
    except ValueError as exc:
        parser.error(str(exc))

    # ---------------------------------------------------------
    # JSON sidecar
    # ---------------------------------------------------------

    if args.json_file is None:

        json_file = nifti_file.with_name(
            f"{stem}.json"
        )

    else:

        json_file = args.json_file.expanduser()

    if not json_file.exists():

        parser.error(
            f"JSON sidecar does not exist: {json_file}"
        )

    with json_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    acquisition = metadata.get(
        "Acquisition",
        {},
    )

    echo_times = acquisition.get(
        "EchoTimes"
    )

    if echo_times is None:

        parser.error(
            "JSON does not contain "
            "Acquisition.EchoTimes."
        )

    echo_times = np.asarray(
        echo_times,
        dtype=np.float32,
    )

    if echo_times.ndim != 1:

        parser.error(
            "Acquisition.EchoTimes must be a 1D list."
        )

    if len(echo_times) < 2:

        parser.error(
            "At least two echo times are required."
        )

    if not np.all(np.isfinite(echo_times)):

        parser.error(
            "Echo times contain NaN or infinite values."
        )

    if np.unique(echo_times).size != echo_times.size:

        parser.error(
            "Echo times must be unique."
        )

    # ---------------------------------------------------------
    # Time units
    # ---------------------------------------------------------

    time_unit = acquisition.get(
        "TimeUnit"
    )

    if time_unit is None:

        parser.error(
            "JSON does not contain "
            "Acquisition.TimeUnit."
        )

    time_unit = str(
        time_unit
    ).lower()

    if time_unit in {
        "ms",
        "millisecond",
        "milliseconds",
    }:

        pass

    elif time_unit in {
        "s",
        "sec",
        "second",
        "seconds",
    }:

        echo_times *= 1000.0

    else:

        parser.error(
            f"Unsupported TimeUnit: {time_unit}"
        )

    # ---------------------------------------------------------
    # Heavy imports
    # ---------------------------------------------------------

    try:

        import nibabel as nib
        import torch

        from cmpl.quantitative_MRI.mapping import (
            t2_star_two_parametric_3D,
        )

        from cmpl.utilities.io import (
            save_scalar_map_like,
        )

    except ModuleNotFoundError as exc:

        parser.error(
            f"Missing dependency: {exc.name}"
        )

    # ---------------------------------------------------------
    # Load NIfTI
    # ---------------------------------------------------------

    image = nib.load(
        str(nifti_file)
    )

    if image.ndim != 4:

        parser.error(
            "Input must be a 4D multi-echo NIfTI. "
            f"Got shape {image.shape}."
        )

    if image.shape[-1] != len(echo_times):

        parser.error(
            "Echo count mismatch: "
            f"NIfTI contains {image.shape[-1]} volumes "
            f"but JSON contains {len(echo_times)} echo times."
        )

    data = image.get_fdata(
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # Sort echoes
    #
    # This is important:
    # if TE order changes, the 4D volumes must change with it.
    # ---------------------------------------------------------

    order = np.argsort(
        echo_times
    )

    if not np.array_equal(
        order,
        np.arange(len(echo_times)),
    ):

        print(
            "Echo times are not sorted. "
            "Reordering echoes and volumes together."
        )

        echo_times = echo_times[
            order
        ]

        data = data[
            ...,
            order
        ]

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if args.device == "auto":

        device = None

        display_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = args.device
        display_device = args.device

    if (
        str(display_device).startswith("cuda")
        and not torch.cuda.is_available()
    ):

        parser.error(
            "CUDA requested, but PyTorch "
            "does not detect a CUDA device."
        )

    # ---------------------------------------------------------
    # Fit
    # ---------------------------------------------------------

    print("CMPL 3D T2* calculation")
    print(f"NIfTI: {nifti_file}")
    print(f"JSON:  {json_file}")
    print(
        f"Echo times (ms): "
        f"{echo_times.tolist()}"
    )
    print(f"Device: {display_device}")
    print("Calculating T2* and S0...")

    result = t2_star_two_parametric_3D(
        echo_times,
        data,
        num_iterations=args.iterations,
        initial_lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        patience=args.patience,
        initial_T2_star=args.initial_t2star,
        plot_error=False,
        return_RMSE=False,
        device=device,
    )

    # Current fitter returns torch tensors.
    t2star = (
        result["T2_star_map"]
        .detach()
        .cpu()
        .numpy()
    )

    s0 = (
        result["S0_map"]
        .detach()
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # Output names
    # ---------------------------------------------------------

    if args.output_prefix is None:

        output_prefix = nifti_file.with_name(
            stem
        )

    else:

        output_prefix = (
            args.output_prefix.expanduser()
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    t2star_file = Path(
        f"{output_prefix}_T2star.nii.gz"
    )

    s0_file = Path(
        f"{output_prefix}_S0.nii.gz"
    )

    # ---------------------------------------------------------
    # Save maps
    # ---------------------------------------------------------

    save_scalar_map_like(
        image,
        t2star,
        str(t2star_file),
        dtype=np.float32,
        descrip="CMPL T2* map (ms)",
        intent_name="T2star",
    )

    save_scalar_map_like(
        image,
        s0,
        str(s0_file),
        dtype=np.float32,
        descrip="CMPL S0 map",
        intent_name="S0",
    )

    print("Done.")
    print(f"T2*: {t2star_file}")
    print(f"S0:   {s0_file}")


if __name__ == "__main__":
    main()