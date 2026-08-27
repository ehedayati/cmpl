# File created by: Eisa Hedayati
# Date: 8/27/2026
# Description: This file is developed at CMRR
import json

import nibabel as nib
import numpy as np
import pytest


def _make_test_data(tmp_path):
    """Create a small synthetic multi-echo NIfTI + JSON sidecar."""

    echo_times = np.array(
        [5.0, 10.0, 15.0, 20.0],
        dtype=np.float32,
    )

    shape = (4, 4, 3)

    s0 = 100.0
    t2star = 20.0

    data = np.stack(
        [
            s0 * np.exp(-te / t2star)
            * np.ones(shape, dtype=np.float32)
            for te in echo_times
        ],
        axis=-1,
    )

    affine = np.array(
        [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 2.0, 0.0, -20.0],
            [0.0, 0.0, 3.0, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    nifti_file = tmp_path / "multi_echo.nii.gz"

    nib.save(
        nib.Nifti1Image(data, affine),
        nifti_file,
    )

    json_file = tmp_path / "multi_echo.json"

    metadata = {
        "Acquisition": {
            "EchoTimes": echo_times.tolist(),
            "TimeUnit": "ms",
        }
    }

    json_file.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return nifti_file, json_file, affine


def test_t2star_cli(tmp_path, monkeypatch):

    from cmpl.cli.t2star import main

    nifti_file, _, affine = _make_test_data(
        tmp_path
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "cmpl-t2star",
            str(nifti_file),
            "--device",
            "cpu",
            "--iterations",
            "10",
        ],
    )

    main()

    t2star_file = (
        tmp_path / "multi_echo_T2star.nii.gz"
    )

    s0_file = (
        tmp_path / "multi_echo_S0.nii.gz"
    )

    assert t2star_file.exists()
    assert s0_file.exists()

    t2star_img = nib.load(t2star_file)
    s0_img = nib.load(s0_file)

    assert t2star_img.shape == (4, 4, 3)
    assert s0_img.shape == (4, 4, 3)

    np.testing.assert_allclose(
        t2star_img.affine,
        affine,
    )

    np.testing.assert_allclose(
        s0_img.affine,
        affine,
    )

    def test_t2star_cli_echo_count_mismatch(
            tmp_path,
            monkeypatch,
    ):
        from cmpl.cli.t2star import main

        nifti_file, json_file, _ = (
            _make_test_data(tmp_path)
        )

        metadata = json.loads(
            json_file.read_text(
                encoding="utf-8"
            )
        )

        metadata["Acquisition"]["EchoTimes"] = [
            5.0,
            10.0,
            15.0,
        ]

        json_file.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "sys.argv",
            [
                "cmpl-t2star",
                str(nifti_file),
                "--device",
                "cpu",
                "--iterations",
                "1",
            ],
        )

        with pytest.raises(
                SystemExit
        ):
            main()


