# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
import json

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    generate_uid,
)


# -------------------------------------------------------------------------
# NIfTI tests
# -------------------------------------------------------------------------

def test_nifti_read(tmp_path):
    """Verify that a NIfTI file can be read without modifying its data."""

    from cmpl.utilities.io import nifti_read

    data = np.arange(
        4 * 5 * 6,
        dtype=np.float32,
    ).reshape(4, 5, 6)

    affine = np.eye(4)

    input_path = tmp_path / "test.nii.gz"

    nib.save(
        nib.Nifti1Image(data, affine),
        input_path,
    )

    image, loaded_data = nifti_read(input_path)

    assert image.shape == data.shape

    np.testing.assert_allclose(
        loaded_data,
        data,
    )


def test_update_nifti_data_preserves_geometry(tmp_path):
    """
    Verify that replacing NIfTI data preserves the original affine and shape.
    """

    from cmpl.utilities.io import update_nifti_data

    original_data = np.zeros(
        (4, 5, 6),
        dtype=np.float32,
    )

    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    input_path = tmp_path / "input.nii.gz"
    output_path = tmp_path / "output.nii.gz"

    nib.save(
        nib.Nifti1Image(original_data, affine),
        input_path,
    )

    new_data = np.full(
        original_data.shape,
        42.0,
        dtype=np.float32,
    )

    update_nifti_data(
        input_path,
        new_data,
        output_path=output_path,
    )

    assert output_path.exists()

    # Reload from disk so the test verifies the actual written file.
    result = nib.load(output_path)

    assert result.shape == original_data.shape

    np.testing.assert_allclose(
        result.get_fdata(),
        new_data,
    )

    np.testing.assert_allclose(
        result.affine,
        affine,
    )


def test_update_nifti_data_rejects_wrong_shape(tmp_path):
    """Verify that replacing NIfTI data with a mismatched shape fails."""

    from cmpl.utilities.io import update_nifti_data

    data = np.zeros(
        (4, 5, 6),
        dtype=np.float32,
    )

    input_path = tmp_path / "input.nii.gz"

    nib.save(
        nib.Nifti1Image(data, np.eye(4)),
        input_path,
    )

    wrong_shape = np.zeros(
        (4, 5, 7),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        update_nifti_data(
            input_path,
            wrong_shape,
        )


def test_save_scalar_map_like(tmp_path):
    """
    Verify that a scalar map is saved using the reference image geometry.
    """

    from cmpl.utilities.io import save_scalar_map_like

    reference_data = np.zeros(
        (4, 5, 6),
        dtype=np.float32,
    )

    affine = np.array(
        [
            [1.5, 0.0, 0.0, 5.0],
            [0.0, 1.5, 0.0, 10.0],
            [0.0, 0.0, 2.0, 15.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    reference = nib.Nifti1Image(
        reference_data,
        affine,
    )

    scalar_map = np.full(
        (4, 5, 6),
        7.5,
        dtype=np.float32,
    )

    output_path = tmp_path / "scalar_map.nii.gz"

    save_scalar_map_like(
        reference,
        scalar_map,
        str(output_path),
    )

    assert output_path.exists()

    result = nib.load(output_path)

    assert result.shape == scalar_map.shape

    np.testing.assert_allclose(
        result.get_fdata(),
        scalar_map,
    )

    np.testing.assert_allclose(
        result.affine,
        affine,
    )


# -------------------------------------------------------------------------
# Geometry tests
# -------------------------------------------------------------------------

def test_compute_nifti_direction_identity_orientation():
    """
    Verify the direction matrix for the canonical DICOM orientation.

    Row    -> +x
    Column -> +y
    Slice  -> +z
    """

    from cmpl.utilities.io import compute_nifti_direction

    orientation = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ]

    direction = compute_nifti_direction(orientation)

    np.testing.assert_allclose(
        direction,
        np.eye(3),
    )


# -------------------------------------------------------------------------
# SimpleITK tests
# -------------------------------------------------------------------------

def test_itk_to_nifti(tmp_path):
    """Verify that a SimpleITK image can be written as a NIfTI file."""

    from cmpl.utilities.io import itk_to_nifti

    array = np.arange(
        4 * 5 * 6,
        dtype=np.float32,
    ).reshape(4, 5, 6)

    image = sitk.GetImageFromArray(array)

    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetOrigin((10.0, 20.0, 30.0))

    output_path = tmp_path / "sitk_test.nii.gz"

    result_path = itk_to_nifti(
        image,
        str(output_path),
        verbose=False,
    )

    assert output_path.exists()
    assert str(output_path) == result_path

    # Read through SimpleITK again to verify the written image.
    result = sitk.ReadImage(str(output_path))

    np.testing.assert_allclose(
        result.GetSpacing(),
        image.GetSpacing(),
    )

    np.testing.assert_allclose(
        result.GetOrigin(),
        image.GetOrigin(),
    )

    np.testing.assert_allclose(
        sitk.GetArrayFromImage(result),
        array,
    )


# -------------------------------------------------------------------------
# DICOM helper and tests
# -------------------------------------------------------------------------

def _write_test_dicom(
    path,
    pixel_data,
    instance_number,
    z_position,
    series_instance_uid=None,
    study_instance_uid=None,
    echo_time=None,
    echo_number=1,
):
    """
    Create a minimal MR DICOM file suitable for testing CMPL's DICOM loader.
    """

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    # Basic DICOM identity information.
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = (
        file_meta.MediaStorageSOPInstanceUID
    )

    ds.StudyInstanceUID = (
        study_instance_uid
        if study_instance_uid is not None
        else generate_uid()
    )

    ds.SeriesInstanceUID = (
        series_instance_uid
        if series_instance_uid is not None
        else generate_uid()
    )

    ds.Modality = "MR"

    # Slice / echo information.
    ds.InstanceNumber = instance_number
    ds.EchoNumbers = echo_number

    if echo_time is not None:
        ds.EchoTime = float(echo_time)

    # Physical geometry.
    ds.ImagePositionPatient = [
        0.0,
        0.0,
        float(z_position),
    ]

    ds.ImageOrientationPatient = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    ds.PixelSpacing = [
        1.0,
        1.0,
    ]

    ds.SliceThickness = 2.0
    ds.SpacingBetweenSlices = 2.0

    # Pixel-array metadata required by pydicom.
    ds.Rows = pixel_data.shape[0]
    ds.Columns = pixel_data.shape[1]

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    ds.PixelData = pixel_data.astype(
        np.uint16
    ).tobytes()

    ds.save_as(
        path,
        enforce_file_format=True,
    )


def test_load_dicom_scan_from_dir(tmp_path):
    """
    Verify loading a small synthetic single-echo DICOM stack.

    reshape=False is intentional here so this test focuses on DICOM reading,
    ordering, and pixel-data integrity independently of CMPL's orientation
    reshaping logic.
    """

    from cmpl.utilities.io import load_dicom_scan_from_dir

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    # Use decreasing z positions so the synthetic stack has a defined
    # orientation without triggering the loader's reversal branch.
    positions = [4.0, 2.0, 0.0]

    expected_slices = []

    for i, z_position in enumerate(
        positions,
        start=1,
    ):
        pixel_data = np.full(
            (4, 5),
            i * 10,
            dtype=np.uint16,
        )

        expected_slices.append(pixel_data)

        _write_test_dicom(
            dicom_dir / f"slice_{i}.dcm",
            pixel_data,
            instance_number=i,
            z_position=z_position,
        )

    result = load_dicom_scan_from_dir(
        str(dicom_dir),
        reshape=False,
    )

    # reshape=False should preserve rows × columns × slices.
    assert result.shape == (4, 5, 3)

    for i, expected in enumerate(expected_slices):
        np.testing.assert_array_equal(
            result[..., i],
            expected,
        )

def test_dicom_to_nifti_writes_image_and_json(tmp_path):
    """
    Verify end-to-end multi-echo DICOM to NIfTI conversion.

    The test checks:
    - multi-echo assembly,
    - physical slice ordering,
    - NIfTI pixel data,
    - JSON sidecar creation,
    - EchoTime metadata,
    - shared source geometry.
    """

    from cmpl.utilities.io import dicom_to_nifti

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    study_uid = generate_uid()
    series_uid = generate_uid()

    echo_times = [
        1.41,
        4.59,
    ]

    z_positions = [
        0.0,
        2.0,
        4.0,
    ]

    instance_number = 1

    # ------------------------------------------------------------
    # Create two echoes, each containing three slices.
    # ------------------------------------------------------------

    for echo_index, echo_time in enumerate(
        echo_times,
        start=1,
    ):
        for slice_index, z_position in enumerate(
            z_positions,
            start=1,
        ):
            pixel_value = (
                echo_index * 100
                + slice_index * 10
            )

            pixel_data = np.full(
                (4, 5),
                pixel_value,
                dtype=np.uint16,
            )

            _write_test_dicom(
                dicom_dir
                / (
                    f"echo_{echo_index}_"
                    f"slice_{slice_index}.dcm"
                ),
                pixel_data,
                instance_number=instance_number,
                z_position=z_position,
                series_instance_uid=series_uid,
                study_instance_uid=study_uid,
                echo_time=echo_time,
                echo_number=echo_index,
            )

            instance_number += 1

    # ------------------------------------------------------------
    # Convert.
    # ------------------------------------------------------------

    output_path = (
        tmp_path
        / "multi_echo.nii.gz"
    )

    metadata = dicom_to_nifti(
        dicom_dir,
        output_path,
        verbose=False,
    )

    json_path = (
        tmp_path
        / "multi_echo.json"
    )

    assert output_path.exists()
    assert json_path.exists()

    # ------------------------------------------------------------
    # Verify actual written NIfTI.
    # ------------------------------------------------------------

    image = sitk.ReadImage(
        str(output_path)
    )

    assert image.GetDimension() == 4

    assert image.GetSize() == (
        5,
        4,
        3,
        2,
    )

    np.testing.assert_allclose(
        image.GetSpacing()[:3],
        (
            1.0,
            1.0,
            2.0,
        ),
    )

    data = sitk.GetArrayFromImage(
        image
    )

    assert data.shape == (
        2,
        3,
        4,
        5,
    )

    # Echo 1.
    np.testing.assert_array_equal(
        data[0, 0],
        np.full(
            (4, 5),
            110,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[0, 1],
        np.full(
            (4, 5),
            120,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[0, 2],
        np.full(
            (4, 5),
            130,
            dtype=np.uint16,
        ),
    )

    # Echo 2.
    np.testing.assert_array_equal(
        data[1, 0],
        np.full(
            (4, 5),
            210,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[1, 1],
        np.full(
            (4, 5),
            220,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[1, 2],
        np.full(
            (4, 5),
            230,
            dtype=np.uint16,
        ),
    )

    # ------------------------------------------------------------
    # Verify metadata returned by CMPL.
    # ------------------------------------------------------------

    assert metadata["Acquisition"]["EchoTimes"] == [
        1.41,
        4.59,
    ]

    assert (
        metadata["Acquisition"]["TimeUnit"]
        == "ms"
    )

    source_geometry = metadata[
        "CMPLSourceGeometry"
    ]

    assert (
        source_geometry["SharedAcrossVolumes"]
        is True
    )

    assert len(
        source_geometry["SlicePlanes"]
    ) == 3

    # ------------------------------------------------------------
    # Verify JSON written to disk.
    # ------------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        saved_metadata = json.load(file)

    assert saved_metadata == metadata

    assert saved_metadata[
        "CMPLSimpleITKGeometry"
    ]["Dimension"] == 4

    assert saved_metadata[
        "CMPLSimpleITKGeometry"
    ]["Size"] == [
        5,
        4,
        3,
        2,
    ]

def test_dicom_to_simpleitk_preserves_public_api(tmp_path):
    """
    Verify that the public dicom_to_SimpleITK API still returns
    only a SimpleITK image after introducing metadata collection.
    """

    from cmpl.utilities.io import dicom_to_SimpleITK

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    study_uid = generate_uid()
    series_uid = generate_uid()

    for slice_index, z_position in enumerate(
        [0.0, 2.0, 4.0],
        start=1,
    ):
        pixel_data = np.full(
            (4, 5),
            slice_index * 10,
            dtype=np.uint16,
        )

        _write_test_dicom(
            dicom_dir / f"slice_{slice_index}.dcm",
            pixel_data,
            instance_number=slice_index,
            z_position=z_position,
            series_instance_uid=series_uid,
            study_instance_uid=study_uid,
            echo_time=5.0,
        )

    image = dicom_to_SimpleITK(
        dicom_dir
    )

    assert isinstance(
        image,
        sitk.Image,
    )

    assert image.GetDimension() == 3

    assert image.GetSize() == (
        5,
        4,
        3,
    )

def test_dicom_to_simpleitk_rejects_inconsistent_echo_time(
    tmp_path,
):
    """
    Verify that a DICOM series containing a mixture of present
    and missing EchoTime values is rejected.
    """

    from cmpl.utilities.io import dicom_to_SimpleITK

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    study_uid = generate_uid()
    series_uid = generate_uid()

    echo_times = [
        5.0,
        None,
        5.0,
    ]

    for slice_index, (
        z_position,
        echo_time,
    ) in enumerate(
        zip(
            [0.0, 2.0, 4.0],
            echo_times,
        ),
        start=1,
    ):
        pixel_data = np.full(
            (4, 5),
            slice_index * 10,
            dtype=np.uint16,
        )

        _write_test_dicom(
            dicom_dir / f"slice_{slice_index}.dcm",
            pixel_data,
            instance_number=slice_index,
            z_position=z_position,
            series_instance_uid=series_uid,
            study_instance_uid=study_uid,
            echo_time=echo_time,
        )

    with pytest.raises(
        ValueError,
        match="Inconsistent EchoTime metadata",
    ):
        dicom_to_SimpleITK(
            dicom_dir
        )

def test_dicom_to_simpleitk_allows_missing_echo_time(
    tmp_path,
):
    """
    Verify that a conventional DICOM series with no EchoTime
    metadata is treated as a single 3D volume.
    """

    from cmpl.utilities.io import dicom_to_SimpleITK

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    study_uid = generate_uid()
    series_uid = generate_uid()

    for slice_index, z_position in enumerate(
        [0.0, 2.0, 4.0],
        start=1,
    ):
        pixel_data = np.full(
            (4, 5),
            slice_index * 10,
            dtype=np.uint16,
        )

        _write_test_dicom(
            dicom_dir / f"slice_{slice_index}.dcm",
            pixel_data,
            instance_number=slice_index,
            z_position=z_position,
            series_instance_uid=series_uid,
            study_instance_uid=study_uid,
            echo_time=None,
        )

    image = dicom_to_SimpleITK(
        dicom_dir
    )

    assert isinstance(
        image,
        sitk.Image,
    )

    assert image.GetDimension() == 3

    assert image.GetSize() == (
        5,
        4,
        3,
    )

    data = sitk.GetArrayFromImage(
        image
    )

    assert data.shape == (
        3,
        4,
        5,
    )

    np.testing.assert_array_equal(
        data[0],
        np.full(
            (4, 5),
            10,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[1],
        np.full(
            (4, 5),
            20,
            dtype=np.uint16,
        ),
    )

    np.testing.assert_array_equal(
        data[2],
        np.full(
            (4, 5),
            30,
            dtype=np.uint16,
        ),
    )