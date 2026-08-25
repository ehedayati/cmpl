# File created by: Eisa Hedayati
# Date: 8/24/2026
# Description: This file is developed at CMRR
# Description:
# Lightweight helpers for extracting selected DICOM metadata.


__all__ = [
    "get_metadata_string",
    "get_metadata_float",
    "get_metadata_vector",
    "extract_acquisition_metadata",
]


def get_metadata_string(reader, tag):
    """
    Return a stripped DICOM metadata value, or None if unavailable.
    """
    if not reader.HasMetaDataKey(tag):
        return None

    value = reader.GetMetaData(tag).strip()

    return value if value else None


def get_metadata_float(reader, tag, scale=1.0):
    """
    Return a numeric DICOM metadata value.

    Parameters
    ----------
    reader
        Object implementing HasMetaDataKey() and GetMetaData().

    tag : str
        DICOM tag in SimpleITK form, e.g. "0018|0081".

    scale : float, optional
        Multiplicative conversion factor.

    Returns
    -------
    float or None
    """
    value = get_metadata_string(reader, tag)

    if value is None:
        return None

    try:
        return float(value) * scale
    except ValueError:
        return None


def get_metadata_vector(reader, tag):
    """
    Return a multi-valued numeric DICOM tag as a list of floats.
    """
    value = get_metadata_string(reader, tag)

    if value is None:
        return None

    try:
        return [
            float(x)
            for x in value.split("\\")
        ]
    except ValueError:
        return None


def extract_acquisition_metadata(reader):
    """
    Extract an allow-listed set of DICOM acquisition metadata.

    Patient-identifying fields are intentionally excluded.

    DICOM timing values are preserved in milliseconds.
    """
    metadata = {}

    string_tags = {
        "Manufacturer": "0008|0070",
        "ManufacturersModelName": "0008|1090",
        "SeriesDescription": "0008|103e",
        "ProtocolName": "0018|1030",
        "ScanningSequence": "0018|0020",
        "SequenceVariant": "0018|0021",
        "ScanOptions": "0018|0022",
        "MRAcquisitionType": "0018|0023",
        "SoftwareVersions": "0018|1020",
        "ReceiveCoilName": "0018|1250",
        "TransmitCoilName": "0018|1251",
    }

    numeric_tags = {
        "FlipAngle": ("0018|1314", 1.0),
        "MagneticFieldStrength": ("0018|0087", 1.0),
        "ImagingFrequency": ("0018|0084", 1.0),
        "NumberOfAverages": ("0018|0083", 1.0),
        "EchoTrainLength": ("0018|0091", 1.0),
        "PixelBandwidth": ("0018|0095", 1.0),

        # Preserve DICOM timing values in milliseconds.
        "RepetitionTime": ("0018|0080", 1.0),
        "InversionTime": ("0018|0082", 1.0),
    }

    for name, tag in string_tags.items():
        value = get_metadata_string(
            reader,
            tag,
        )

        if value is not None:
            metadata[name] = value

    for name, (tag, scale) in numeric_tags.items():
        value = get_metadata_float(
            reader,
            tag,
            scale=scale,
        )

        if value is not None:
            metadata[name] = value

    return metadata