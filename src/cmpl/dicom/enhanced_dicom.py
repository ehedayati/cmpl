# File created by: Eisa Hedayati
# Date: 2/19/2026
# Description: This file is developed at CMRR
from pydicom.tag import Tag
import pydicom
import inspect
import os

__all__ = [
    "voxel_sizes_detailed",
    "find_tag_anywhere",
    "get_slice_thickness",
    "get_spacing_between_slices",
    "voxel_sizes_detailed",
    "get_unique_echo_times",
    "is_enhanced_dicom",
    "get_enhanced_frame_info",
    "extract_enhanced_acquisition_metadata",
]


SLICE_THICKNESS = Tag(0x0018, 0x0050)   # Slice Thickness
PIXEL_MEASURES_SEQ = Tag(0x0028, 0x9110)
MR_FOV_GEOM_SEQ = Tag(0x0018, 0x9125)   # MR FOV/Geometry Sequence
ACQ_FREQ_STEPS  = Tag(0x0018, 0x9058)   # MR Acquisition Frequency Encoding Steps
ACQ_PHASE_STEPS = Tag(0x0018, 0x9231)   # MR Acquisition Phase Encoding Steps
PIXEL_SPACING      = Tag(0x0028, 0x0030)  # Pixel Spacing
SPACING_BETW_SLICES= Tag(0x0018, 0x0088)  # Spacing Between Slices
SPACING_BETW = Tag(0x0018, 0x0088)

PLANE_POS_SEQ      = Tag(0x0020, 0x9113)   # Plane Position (Patient) Sequence
IMAGE_POS_PATIENT  = Tag(0x0020, 0x0032)   # Image Position (Patient)


def get_unique_echo_times(dicom_dir, stop_before_pixels=True, sort_output=True):
    """
    Scan a DICOM directory and collect unique EffectiveEchoTime values
    from PerFrameFunctionalGroupsSequence / MREchoSequence.

    Parameters
    ----------
    dicom_dir : str
        Path to directory containing DICOM files.
    stop_before_pixels : bool, optional
        If True, avoid reading pixel data for faster metadata-only access.
    sort_output : bool, optional
        If True, return echo times sorted.

    Returns
    -------
    list of float
        Unique echo times found across all readable DICOM files.
    """
    tes = []

    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)

        if not os.path.isfile(fpath):
            continue

        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=stop_before_pixels)
        except Exception:
            continue

        if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            continue

        for fg in ds.PerFrameFunctionalGroupsSequence:
            if not hasattr(fg, "MREchoSequence"):
                continue

            echo_seq = fg.MREchoSequence
            if not echo_seq:
                continue

            if not hasattr(echo_seq[0], "EffectiveEchoTime"):
                continue

            try:
                te = float(echo_seq[0].EffectiveEchoTime)
                tes.append(te)
            except Exception:
                continue

    tes = list(set(tes))
    if sort_output:
        tes = sorted(tes)

    return tes

def get_spacing_between_slices(ds):
    # In your dataset, this is top-level and works
    if SPACING_BETW in ds:
        return float(ds[SPACING_BETW].value)
    if hasattr(ds, "SpacingBetweenSlices"):
        return float(ds.SpacingBetweenSlices)
    return None


def get_slice_thickness(ds):
    # Classic top-level
    if SLICE_THICKNESS in ds:
        return float(ds[SLICE_THICKNESS].value)

    # Enhanced: Shared / Per-frame functional groups
    for fg_name in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        if fg_name in ds:
            fg_seq = ds[fg_name].value  # <-- FIX: .value is the Sequence
            if fg_seq and len(fg_seq) > 0:
                fg0 = fg_seq[0]
                # Pixel Measures by tag (works even if keyword missing)
                if PIXEL_MEASURES_SEQ in fg0:
                    pm_seq = fg0[PIXEL_MEASURES_SEQ].value
                    if pm_seq and len(pm_seq) > 0:
                        pm0 = pm_seq[0]
                        if SLICE_THICKNESS in pm0:
                            return float(pm0[SLICE_THICKNESS].value)

    return None


def get_acq_steps_from_mr_fov_geometry(ds):
    """
    Returns (acq_freq_steps, acq_phase_steps) as ints if found, else (None, None).
    Searches Shared then a few PerFrame functional groups.
    """

    def extract_from_fg_item(fg_item):
        if MR_FOV_GEOM_SEQ in fg_item:
            seq = fg_item[MR_FOV_GEOM_SEQ].value
            if seq and len(seq) > 0:
                item = seq[0]
                acq_freq = int(item[ACQ_FREQ_STEPS].value) if ACQ_FREQ_STEPS in item else None
                acq_phase = int(item[ACQ_PHASE_STEPS].value) if ACQ_PHASE_STEPS in item else None
                return acq_freq, acq_phase
        return None, None

    # Shared FG
    if "SharedFunctionalGroupsSequence" in ds and len(ds.SharedFunctionalGroupsSequence) > 0:
        acq_freq, acq_phase = extract_from_fg_item(ds.SharedFunctionalGroupsSequence[0])
        if acq_freq is not None or acq_phase is not None:
            return acq_freq, acq_phase

    # Per-frame FG (check a few frames)
    if "PerFrameFunctionalGroupsSequence" in ds and len(ds.PerFrameFunctionalGroupsSequence) > 0:
        for i in range(min(5, len(ds.PerFrameFunctionalGroupsSequence))):
            acq_freq, acq_phase = extract_from_fg_item(ds.PerFrameFunctionalGroupsSequence[i])
            if acq_freq is not None or acq_phase is not None:
                return acq_freq, acq_phase

    return None, None

def voxel_sizes_detailed(ds):
    # AFTER (stored/reconstructed)
    dx_after, dy_after = get_dxdy(ds)
    dz_thk = get_slice_thickness(ds)               # may be None
    dz_spc = get_spacing_between_slices(ds)        # may be None (but yours exists: 2.2)

    rows, cols = int(ds.Rows), int(ds.Columns)

    # BEFORE (estimated from acquisition steps)
    acq_freq, acq_phase = get_acq_steps_from_mr_fov_geometry(ds)

    dx_before = dx_after * (rows / acq_freq) if acq_freq else None
    dy_before = dy_after * (cols / acq_phase) if acq_phase else None

    return {
        "interpolation": {
            "dx": dx_after,
            "dy": dy_after,
            "dz_spacing": dz_spc,
            "slice_thickness": dz_thk,
            "matrix": (rows, cols),
        },
        "acquisition": {
            "dx": dx_before,
            "dy": dy_before,
            "acq_steps": (acq_freq, acq_phase),
        },
    }

def get_dxdy(ds):
    # classic fallback
    if hasattr(ds, "PixelSpacing"):
        return tuple(float(x) for x in ds.PixelSpacing)

    where, pm = find_pixel_measures(ds)
    if pm is None or PIXEL_SPACING not in pm:
        raise ValueError("PixelSpacing not found in classic or enhanced Pixel Measures.")

    dx, dy = (float(x) for x in pm[PIXEL_SPACING].value)
    return dx, dy


def find_tag_anywhere(ds, tag, max_hits=10):
    hits = []

    def walk(d, path="ds"):
        nonlocal hits
        if len(hits) >= max_hits:
            return
        if tag in d:
            hits.append((path, d[tag].value))
        for elem in d:
            if elem.VR == "SQ" and elem.value is not None:
                for i, item in enumerate(elem.value):
                    walk(item, f"{path}.{elem.keyword or str(elem.tag)}[{i}]")

    walk(ds)
    return hits

def find_pixel_measures(ds):
    # Shared
    if "SharedFunctionalGroupsSequence" in ds and len(ds.SharedFunctionalGroupsSequence):
        fg = ds.SharedFunctionalGroupsSequence[0]
        if PIXEL_MEASURES_SEQ in fg:
            item = fg[PIXEL_MEASURES_SEQ].value[0]
            return ("shared", item)

    # Per-frame (check first few frames; usually constant)
    if "PerFrameFunctionalGroupsSequence" in ds and len(ds.PerFrameFunctionalGroupsSequence):
        for i in range(min(5, len(ds.PerFrameFunctionalGroupsSequence))):
            fg = ds.PerFrameFunctionalGroupsSequence[i]
            if PIXEL_MEASURES_SEQ in fg:
                item = fg[PIXEL_MEASURES_SEQ].value[0]
                return (f"per-frame[{i}]", item)

    return (None, None)

def is_enhanced_dicom(path):
    """
    Determine whether a DICOM file is a multi-frame Enhanced DICOM object.

    Parameters
    ----------
    path : str or Path
        Path to a DICOM file.

    Returns
    -------
    bool
        True if the file contains multiple frames with per-frame
        functional groups.
    """

    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
    )

    return (
        hasattr(
            ds,
            "PerFrameFunctionalGroupsSequence",
        )
        and int(
            getattr(ds, "NumberOfFrames", 1)
        ) > 1
    )

def get_enhanced_frame_info(path):
    """
    Extract frame-level geometry and EchoTime from an Enhanced DICOM file.
    """

    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
    )

    shared = (
        ds.SharedFunctionalGroupsSequence[0]
        if hasattr(ds, "SharedFunctionalGroupsSequence")
        else None
    )

    frames = []

    for index, fg in enumerate(
        ds.PerFrameFunctionalGroupsSequence
    ):
        # EchoTime
        echo_time = None

        if hasattr(fg, "MREchoSequence"):
            echo_time = round(
                float(
                    fg.MREchoSequence[
                        0
                    ].EffectiveEchoTime
                ),
                6,
            )

        # ImagePositionPatient
        position = None

        if hasattr(fg, "PlanePositionSequence"):
            position = [
                float(x)
                for x in fg.PlanePositionSequence[
                    0
                ].ImagePositionPatient
            ]

        # ImageOrientationPatient
        orientation = None

        if hasattr(fg, "PlaneOrientationSequence"):
            orientation = [
                float(x)
                for x in fg.PlaneOrientationSequence[
                    0
                ].ImageOrientationPatient
            ]

        elif (
            shared is not None
            and hasattr(
                shared,
                "PlaneOrientationSequence",
            )
        ):
            orientation = [
                float(x)
                for x in shared.PlaneOrientationSequence[
                    0
                ].ImageOrientationPatient
            ]

        frames.append(
            {
                "FrameIndex": index,
                "EchoTime": echo_time,
                "ImagePositionPatient": position,
                "ImageOrientationPatient": orientation,
            }
        )

    return frames

def _find_first_tag_value(ds, tag):
    """
    Find the first occurrence of a DICOM tag anywhere in a dataset,
    including nested sequences.
    """

    target = Tag(tag)

    for element in ds.iterall():
        if element.tag == target:
            value = element.value

            if value is None:
                continue

            return value

    return None


def extract_enhanced_acquisition_metadata(ds):
    """
    Extract selected acquisition metadata from an Enhanced DICOM dataset.

    Nested functional-group sequences are searched recursively.
    Patient-identifying fields are intentionally excluded.

    DICOM timing values are preserved in milliseconds.
    """

    metadata = {}

    string_tags = {
        "Manufacturer": (0x0008, 0x0070),
        "ManufacturersModelName": (0x0008, 0x1090),
        "SeriesDescription": (0x0008, 0x103E),
        "ProtocolName": (0x0018, 0x1030),
        "ScanningSequence": (0x0018, 0x0020),
        "SequenceVariant": (0x0018, 0x0021),
        "ScanOptions": (0x0018, 0x0022),
        "MRAcquisitionType": (0x0018, 0x0023),
        "SoftwareVersions": (0x0018, 0x1020),
        "ReceiveCoilName": (0x0018, 0x1250),
        "TransmitCoilName": (0x0018, 0x1251),
    }

    numeric_tags = {
        "FlipAngle": (0x0018, 0x1314),
        "MagneticFieldStrength": (0x0018, 0x0087),
        "ImagingFrequency": (0x0018, 0x0084),
        "NumberOfAverages": (0x0018, 0x0083),
        "EchoTrainLength": (0x0018, 0x0091),
        "PixelBandwidth": (0x0018, 0x0095),
        "RepetitionTime": (0x0018, 0x0080),
        "InversionTime": (0x0018, 0x0082),
    }

    for name, tag in string_tags.items():
        value = _find_first_tag_value(
            ds,
            tag,
        )

        if value is None:
            continue

        value = str(value).strip()

        if value:
            metadata[name] = value

    for name, tag in numeric_tags.items():
        value = _find_first_tag_value(
            ds,
            tag,
        )

        if value is None:
            continue

        try:
            metadata[name] = round(
                float(value),
                6,
            )
        except (TypeError, ValueError):
            continue

    return metadata