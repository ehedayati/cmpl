# File created by: Eisa Hedayati
# Date: 8/24/2026
# Description: This file is developed at CMRR
import numpy as np
import SimpleITK as sitk

from .metadata import (
    get_metadata_float,
    get_metadata_vector,
)


__all__ = [
    "compute_slice_normal",
    "compute_slice_position",
    "get_slice_position",
    "extract_slice_geometry",
]


def compute_slice_normal(image_orientation_patient):
    """
    Compute the DICOM slice normal from ImageOrientationPatient.

    Parameters
    ----------
    image_orientation_patient : sequence of float
        Six ImageOrientationPatient values:
        first three define the row direction and the last three
        define the column direction.

    Returns
    -------
    numpy.ndarray
        Three-element slice-normal vector.
    """
    iop = np.asarray(
        image_orientation_patient,
        dtype=float,
    )

    if iop.size != 6:
        raise ValueError(
            "ImageOrientationPatient must contain 6 values."
        )

    return np.cross(
        iop[:3],
        iop[3:],
    )


def compute_slice_position(
    image_orientation_patient,
    image_position_patient,
):
    """
    Compute the physical slice position along the DICOM slice normal.
    """
    ipp = np.asarray(
        image_position_patient,
        dtype=float,
    )

    if ipp.size != 3:
        raise ValueError(
            "ImagePositionPatient must contain 3 values."
        )

    slice_normal = compute_slice_normal(
        image_orientation_patient
    )

    return float(
        np.dot(
            ipp,
            slice_normal,
        )
    )


def get_slice_position(filename):
    """
    Read one DICOM file and return its physical position along
    the DICOM slice normal.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()

    iop = get_metadata_vector(
        reader,
        "0020|0037",
    )

    ipp = get_metadata_vector(
        reader,
        "0020|0032",
    )

    if iop is None:
        raise ValueError(
            f"ImageOrientationPatient is missing: {filename}"
        )

    if ipp is None:
        raise ValueError(
            f"ImagePositionPatient is missing: {filename}"
        )

    return compute_slice_position(
        iop,
        ipp,
    )


def extract_slice_geometry(filename):
    """
    Extract the original physical DICOM plane geometry for one slice.

    DICOM patient coordinates use the LPS coordinate system.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()

    geometry = {}

    iop = get_metadata_vector(
        reader,
        "0020|0037",
    )

    ipp = get_metadata_vector(
        reader,
        "0020|0032",
    )

    pixel_spacing = get_metadata_vector(
        reader,
        "0028|0030",
    )

    if iop is not None:
        geometry["ImageOrientationPatient"] = iop

        if len(iop) == 6:
            geometry["SliceNormal"] = (
                compute_slice_normal(iop).tolist()
            )

    if ipp is not None:
        geometry["ImagePositionPatient"] = ipp

    if pixel_spacing is not None:
        geometry["PixelSpacing"] = pixel_spacing

    slice_thickness = get_metadata_float(
        reader,
        "0018|0050",
    )

    if slice_thickness is not None:
        geometry["SliceThickness"] = slice_thickness

    spacing_between_slices = get_metadata_float(
        reader,
        "0018|0088",
    )

    if spacing_between_slices is not None:
        geometry["SpacingBetweenSlices"] = (
            spacing_between_slices
        )

    return geometry