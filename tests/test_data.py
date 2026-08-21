# File created by: Eisa Hedayati
# Date: 8/21/2026
# Description: This file is developed at CMRR
from pathlib import Path

import pandas as pd


def test_build_medical_data_frame_empty_directory(tmp_path):
    """
    An empty root directory should return an empty DataFrame.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    result = build_medical_data_frame(tmp_path)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_medical_data_frame_missing_root(tmp_path):
    """
    A nonexistent root directory should return an empty DataFrame.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    missing = tmp_path / "does_not_exist"

    result = build_medical_data_frame(missing)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_medical_data_frame_discovers_dicom_directory(tmp_path):
    """
    Verify that a study containing a DICOM contrast directory is indexed.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    study = tmp_path / "Study001"
    contrast = study / "Dicoms" / "MR-SE12-T2_STAR"

    contrast.mkdir(parents=True)

    result = build_medical_data_frame(tmp_path)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["Study"] == "Study001"
    assert row["Type"] == "Dicoms"
    assert row["Contrast"] == "T2_STAR"

    expected_path = Path("Study001") / "Dicoms" / "MR-SE12-T2_STAR"

    assert Path(row["Path"]) == expected_path


def test_build_medical_data_frame_discovers_h5_file(tmp_path):
    """
    Verify that HDF5 files are indexed correctly.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    h5_dir = tmp_path / "Study001" / "h5_files"
    h5_dir.mkdir(parents=True)

    h5_file = h5_dir / "MR-SE5-T2STAR.h5"
    h5_file.touch()

    result = build_medical_data_frame(tmp_path)

    h5_rows = result[result["Type"] == "h5_files"]

    assert len(h5_rows) == 1

    row = h5_rows.iloc[0]

    assert row["Study"] == "Study001"
    assert row["Contrast"] == "T2STAR"

    expected_path = Path("Study001") / "h5_files" / "MR-SE5-T2STAR.h5"

    assert Path(row["Path"]) == expected_path


def test_build_medical_data_frame_discovers_segmentation_file(tmp_path):
    """
    Verify that segmentation NIfTI files are indexed correctly.

    This tests DataFrame construction only; no segmentation package is used.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    seg_dir = (
        tmp_path
        / "Study001"
        / "Segmentations"
        / "MR-SE7-T2STAR"
        / "Manual"
    )

    seg_dir.mkdir(parents=True)

    seg_file = seg_dir / "liver.nii.gz"
    seg_file.touch()

    result = build_medical_data_frame(tmp_path)

    seg_rows = result[result["Type"] == "Segmentations"]

    assert len(seg_rows) == 1

    row = seg_rows.iloc[0]

    assert row["Study"] == "Study001"
    assert row["Contrast"] == "T2STAR"
    assert row["Info"] == "Manual"
    assert row["Part"] == "liver"

    expected_path = (
        Path("Study001")
        / "Segmentations"
        / "MR-SE7-T2STAR"
        / "Manual"
        / "liver.nii.gz"
    )

    assert Path(row["Path"]) == expected_path


def test_build_medical_data_frame_ignores_old_segmentation(tmp_path):
    """
    Files ending in _old.nii should not be added to the DataFrame.
    """

    from cmpl.utilities.df_build import build_medical_data_frame

    seg_dir = (
        tmp_path
        / "Study001"
        / "Segmentations"
        / "T2STAR"
        / "Manual"
    )

    seg_dir.mkdir(parents=True)

    (seg_dir / "liver_old.nii").touch()

    result = build_medical_data_frame(tmp_path)

    assert result.empty