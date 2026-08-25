# File created by: Eisa Hedayati
# Date: 8/27/2024
# Description: This file is developed at CMRR
import nibabel as nib
import numpy as np
from numpy.typing import NDArray
import os
import pydicom
import SimpleITK as sitk
from collections import defaultdict
from nibabel.nifti1 import Nifti1Image
import warnings
from cmpl.dicom.geometry import (
    extract_slice_geometry,
    get_slice_position,
)

from cmpl.dicom.metadata import (
    extract_acquisition_metadata,
)

def save_scalar_map_like(ref_img: nib.Nifti1Image,
                         data_in: np.ndarray,
                         out_path: str,
                         *,
                         dtype=np.float32,
                         descrip: str = "",
                         intent_name: str = "") -> None:
    """
    Save a 3D scalar NIfTI map using ref_img geometry (affine/qform/sform).
    - Accepts (X,Y,Z) or (X,Y,Z,1[,1...]) and writes (X,Y,Z).
    - Forces scalar intent and sane scaling.
    """
    data = np.asarray(data_in)

    # Reduce trailing singleton dims only: (X,Y,Z,1,1,...) -> (X,Y,Z)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D scalar after squeezing trailing 1s, got shape {data.shape}")

    ref_shape_3d = ref_img.shape[:3]
    if data.shape != ref_shape_3d:
        raise ValueError(f"Data shape {data.shape} != reference 3D shape {ref_shape_3d}")

    data = data.astype(dtype, copy=False)

    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    hdr.set_data_shape(data.shape)  # ensures dim is 3D scalar

    # Scalar map intent
    hdr["intent_code"] = 0  # NIFTI_INTENT_NONE
    hdr["intent_p1"] = 0
    hdr["intent_p2"] = 0
    hdr["intent_p3"] = 0
    hdr["intent_name"] = (intent_name.encode("ascii", "ignore")[:15] if intent_name else b"")

    # Avoid NaN slope/inter
    hdr["scl_slope"] = 1.0
    hdr["scl_inter"] = 0.0

    if descrip:
        hdr["descrip"] = descrip.encode("ascii", "ignore")[:79]

    out_img = nib.Nifti1Image(data, ref_img.affine, header=hdr)

    # Preserve qform/sform
    qaff, qcode = ref_img.get_qform(coded=True)
    saff, scode = ref_img.get_sform(coded=True)
    if qaff is not None:
        out_img.set_qform(qaff, int(qcode))
    if saff is not None:
        out_img.set_sform(saff, int(scode))

    nib.save(out_img, out_path)
    print(f"Saved {out_path} | shape={out_img.shape} dtype={out_img.get_data_dtype()}")

NiftiImage = nib.Nifti1Image | nib.Nifti2Image

def nifti_read(
    file_name,
    re_orient: bool | None = None,
) -> tuple[NiftiImage, NDArray]:

    if re_orient is not None:
        warnings.warn(
            "'re_orient' is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    if re_orient:
        raise ValueError(
            "re_orient=True is no longer supported. "
            "NIfTI orientation is preserved as stored."
        )

    nifti = nib.load(file_name)

    if not isinstance(nifti, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError(f"Expected NIfTI image, got {type(nifti).__name__}")

    return nifti, np.asanyarray(nifti.dataobj)


def compute_nifti_direction(image_orientation_patient):
    """
    Compute the NIfTI 3x3 direction matrix from the DICOM ImageOrientationPatient.

    Parameters:
    - image_orientation_patient: List or array of 6 floats
      (row and column direction cosines in patient coordinates).

    Returns:
    - nifti_direction: Flattened list of 9 floats representing the 3x3 NIfTI direction matrix.
    """
    # Extract row and column direction cosines
    row_direction = np.array(image_orientation_patient[:3])
    column_direction = np.array(image_orientation_patient[3:])

    # Compute the slice direction as the cross product of row and column directions
    slice_direction = np.cross(row_direction, column_direction)

    # Assemble the 3x3 matrix
    nifti_direction = np.column_stack((row_direction, column_direction, slice_direction))

    return nifti_direction

def load_dicom_scan_from_dir(directory, reshape=True, verbose=False, with_spacing=False):
    """
    Load all DICOM files from the given directory and convert them into a 3D or 4D numpy array,
    depending on whether the sequence is single-echo or multi-echo.

    Args:
        directory (str): Path to the directory containing DICOM files.
        reshape (bool): If True, returns an array reshaped based on the sequence type.
        verbose (bool): If True, print additional information about the loading process.
        with_spacing (bool): If True, return a the spacing with image.

    Returns:
        numpy.ndarray: A numpy array containing the pixel data from DICOM files.
                       Shape is [x, y, z] for single-echo and [x, y, z, echo] for multi-echo.
                       if with_spacing is True, return the spacing with image.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    # Gather all .dcm files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not files:
        raise ValueError("No DICOM files found in the directory.")

    # Load DICOM files and collect relevant metadata
    dicom_info_list = []
    for file in files:
        try:
            dcm_path = os.path.join(directory, file)
            dcm = pydicom.dcmread(dcm_path)
            echo_number = getattr(dcm, 'EchoNumbers', 1)
            instance_number = getattr(dcm, 'InstanceNumber', 0)
            position = getattr(dcm, 'ImagePositionPatient', None)
            image_position = np.array(dcm.ImagePositionPatient, dtype=float)
            image_orientation = np.array(dcm.ImageOrientationPatient, dtype=float)
            if position:
                slice_location = position[2]
            else:
                slice_location = getattr(dcm, 'SliceLocation', 0)
            dicom_info_list.append({
                'dcm': dcm,
                'EchoNumber': echo_number,
                'InstanceNumber': instance_number,
                'SliceLocation': slice_location,
                'ImageOrientationPatient': image_orientation,
                'ImagePositionPatient': image_position,
            })
            if verbose:
                print(f"Loaded {file} with Instance Number: {instance_number}, Echo Number: {echo_number}")
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not dicom_info_list:
        raise ValueError("Failed to load any DICOM files.")

    # Determine unique echoes
    echo_numbers = sorted(set(info['EchoNumber'] for info in dicom_info_list))
    num_echoes = len(echo_numbers)
    is_multi_echo = num_echoes > 1
    if verbose:
        print(f"Detected {'multi-echo' if is_multi_echo else 'single-echo'} sequence with {num_echoes} echo(s).")

    # Organize DICOM files by Echo Number and Slice Location
    dicom_info_list.sort(key=lambda x: (x['EchoNumber'], x['InstanceNumber']))
    # Fix matrix orientation z direction
    filtered_list = [d for d in dicom_info_list if d['EchoNumber'] == 1]
    patient_position_difference = np.array(
        filtered_list[-1]['ImagePositionPatient'] - filtered_list[0]['ImagePositionPatient'])
    normal_vector = np.cross(np.array(filtered_list[-1]['ImageOrientationPatient'][:3]),
                             np.array(filtered_list[-1]['ImageOrientationPatient'][3:]))
    if_reverse = np.dot(normal_vector, patient_position_difference) > 0  # check direction
    slice_axis = np.argmax(np.abs(patient_position_difference))
    # print(slice_axis)
    # Extract pixel arrays
    image_data_list = [info['dcm'].pixel_array for info in dicom_info_list]
    pixel_spacing = dicom_info_list[0]['dcm'].PixelSpacing
    slice_thickness = dicom_info_list[0]['dcm'].SliceThickness
    if hasattr(dicom_info_list[0]['dcm'], 'SpacingBetweenSlices'):
        slice_spacing = dicom_info_list[0]['dcm'].SpacingBetweenSlices
    else:
        slice_spacing = slice_thickness

    origin = filtered_list[0]['ImagePositionPatient']
    spacing = list(map(float, pixel_spacing))
    # Insert slice_thickness at the position indicated by slice_axis
    spacing = spacing[:slice_axis] + [float(slice_spacing)] + spacing[slice_axis:]
    # spacing = spacing + [float(slice_spacing)]
    # Determine the number of slices
    total_images = len(image_data_list)
    num_slices = total_images // num_echoes
    if verbose:
        print(f"Number of slices: {num_slices}")

    # Stack the image data
    try:
        image_data = np.stack(image_data_list, axis=slice_axis)
    except Exception as e:
        raise RuntimeError(f"Error creating array from DICOM files: {e}")

    if reshape:
        if is_multi_echo:
            im_shape = image_data.shape
            # First step is to separate echos from each other
            target_shape = [*im_shape[0:slice_axis], num_echoes, im_shape[slice_axis] // num_echoes,
                            *im_shape[slice_axis + 1:]]
            image_data = image_data.reshape(target_shape)
            # Move axes to get [x, y, z, echo] standard
            if slice_axis == 0:
                image_data = np.moveaxis(image_data, [0, 1], [-1, -2])
                spacing = [spacing[1], spacing[2], spacing[0]]
                origin = np.array([origin[1], origin[2], origin[0]])
                if if_reverse:
                    image_data = np.flip(image_data, -2)
            elif slice_axis == 1:
                image_data = np.moveaxis(image_data, 1, -1)
            elif slice_axis == 2:
                image_data = np.moveaxis(image_data, -1, 0)
                spacing = [spacing[2], spacing[0], spacing[1]]
                origin = np.array([origin[2], origin[0], origin[1]])
                if if_reverse:
                    image_data = np.flip(image_data, 0)
        else:
            if slice_axis == 0:
                image_data = np.moveaxis(image_data, 0, -1)
                spacing = [spacing[1], spacing[2], spacing[0]]
                origin = np.array([origin[1], origin[2], origin[0]])
                if if_reverse:
                    image_data = np.flip(image_data, -1)
            elif slice_axis == 1:
                # if if_reverse:
                #     image_data = np.flip(image_data, -1)
                pass
            elif slice_axis == 2:
                image_data = np.moveaxis(image_data, -1, 0)
                spacing = [spacing[2], spacing[0], spacing[1]]
                origin = np.array([origin[2], origin[0], origin[1]])
                if if_reverse:
                    image_data = np.flip(image_data, 0)
    if with_spacing:
        orientation = filtered_list[0]["ImageOrientationPatient"]
        return image_data, (origin, spacing, orientation)
    else:
        return image_data

def update_nifti_data(file_path, new_data, output_path=None, dtype=np.float32):
    """
    Replace the data of a NIfTI file while preserving geometry and metadata safely.
    """

    # Load existing NIfTI
    nifti = nib.load(file_path)

    # Ensure numpy array
    new_data = np.asarray(new_data)

    # Shape check (recommended)
    if new_data.shape != nifti.shape:
        raise ValueError(
            f"Shape mismatch: new_data {new_data.shape} != nifti {nifti.shape}"
        )

    # Copy header to avoid mutating original
    header = nifti.header.copy()

    # Reset scaling to avoid intensity bugs
    header.set_slope_inter(None, None)

    # Set dtype explicitly
    new_data = new_data.astype(dtype)
    header.set_data_dtype(dtype)

    # Create new NIfTI
    new_nifti = nib.Nifti1Image(
        new_data,
        affine=nifti.affine,
        header=header
    )

    # Update header consistency
    new_nifti.update_header()

    # Save
    if output_path is None:
        output_path = file_path

    nib.save(new_nifti, output_path)

    return new_nifti


def _dicom_to_simpleitk(
    dicom_directory,
    series_id=None,
    collect_metadata=False,
):
    """
    Read a DICOM series into a SimpleITK image.

    Multi-echo acquisitions stored under the same SeriesInstanceUID are
    separated using EchoTime (0018|0081).

    Parameters
    ----------
    dicom_directory : str or Path
        Directory containing the DICOM files.

    series_id : str, optional
        DICOM SeriesInstanceUID to read. If omitted, the first available
        series is used. If multiple series are present, a warning is issued.

    collect_metadata : bool, default=False
        If True, also collect selected acquisition metadata and original
        per-slice DICOM geometry.

    Returns
    -------
    sitk.Image
        Returned when collect_metadata=False.

    tuple[sitk.Image, dict]
        Returned when collect_metadata=True.
    """

    dicom_directory = str(dicom_directory)

    # ------------------------------------------------------------
    # Discover DICOM series.
    # ------------------------------------------------------------

    series_reader = sitk.ImageSeriesReader()

    series_ids = series_reader.GetGDCMSeriesIDs(
        dicom_directory
    )

    if not series_ids:
        raise ValueError(
            f"No DICOM series found in directory: "
            f"{dicom_directory}"
        )

    if series_id is None:
        if len(series_ids) > 1:
            warnings.warn(
                f"Found {len(series_ids)} DICOM series in "
                f"{dicom_directory}. "
                f"Using the first series: {series_ids[0]}",
                UserWarning,
                stacklevel=2,
            )

        series_id = series_ids[0]

    elif series_id not in series_ids:
        raise ValueError(
            f"Series ID {series_id!r} was not found. "
            f"Available series IDs: {list(series_ids)}"
        )

    file_names = series_reader.GetGDCMSeriesFileNames(
        dicom_directory,
        series_id,
    )

    # ------------------------------------------------------------
    # Group files by EchoTime.
    # ------------------------------------------------------------

    echo_groups = defaultdict(list)
    missing_echo_time = []

    for filename in file_names:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(filename)
        file_reader.ReadImageInformation()

        if file_reader.HasMetaDataKey("0018|0081"):
            echo_value = file_reader.GetMetaData(
                "0018|0081"
            ).strip()

            if echo_value:
                try:
                    echo_time = float(echo_value)

                except ValueError as exc:
                    raise ValueError(
                        f"Invalid EchoTime value "
                        f"{echo_value!r} in DICOM file: "
                        f"{filename}"
                    ) from exc

                echo_groups[echo_time].append(
                    filename
                )

                continue

        missing_echo_time.append(
            filename
        )

    # EchoTime may be absent for a conventional acquisition.
    # A mixture of present and absent EchoTime values is ambiguous.
    if missing_echo_time:
        if echo_groups:
            raise ValueError(
                "Inconsistent EchoTime metadata: "
                f"{len(missing_echo_time)} of "
                f"{len(file_names)} DICOM files are "
                "missing EchoTime (0018|0081)."
            )

        echo_groups[None] = missing_echo_time

    # ------------------------------------------------------------
    # Process each echo independently.
    # ------------------------------------------------------------

    echo_images = {}

    acquisition_metadata = None
    geometry_by_echo = {}

    for echo, files in echo_groups.items():

        # Preserve physical DICOM slice ordering.
        files.sort(
            key=get_slice_position
        )

        # --------------------------------------------------------
        # Metadata from first physical slice.
        # --------------------------------------------------------

        first_file = files[0]

        meta_reader = sitk.ImageFileReader()
        meta_reader.SetFileName(first_file)
        meta_reader.ReadImageInformation()

        first_metadata = {
            key: meta_reader.GetMetaData(key)
            for key in meta_reader.GetMetaDataKeys()
        }

        # --------------------------------------------------------
        # Acquisition metadata.
        # --------------------------------------------------------

        if (
            collect_metadata
            and acquisition_metadata is None
        ):
            acquisition_metadata = (
                extract_acquisition_metadata(
                    meta_reader
                )
            )

        # --------------------------------------------------------
        # Exact source DICOM plane geometry.
        # --------------------------------------------------------

        if collect_metadata:
            volume_geometry = {
                "SlicePlanes": [
                    extract_slice_geometry(
                        filename
                    )
                    for filename in files
                ]
            }

            if echo is not None:
                # Keep EchoTime in DICOM-native milliseconds.
                volume_geometry["EchoTime"] = round(
                    float(echo),
                    6,
                )

            geometry_by_echo[echo] = (
                volume_geometry
            )

        # --------------------------------------------------------
        # Read the sorted files as a 3D image.
        # --------------------------------------------------------

        series_reader.SetFileNames(
            files
        )

        image = series_reader.Execute()

        # Preserve previous CMPL behavior: copy metadata from
        # the first DICOM slice to the resulting volume.
        for key, value in first_metadata.items():
            image.SetMetaData(
                key,
                value,
            )

        echo_images[echo] = image

    # ------------------------------------------------------------
    # Assemble the final image.
    # ------------------------------------------------------------

    if len(echo_images) == 1:
        sorted_keys = list(
            echo_images.keys()
        )

        output_image = echo_images[
            sorted_keys[0]
        ]

    else:
        sorted_keys = sorted(
            echo_images.keys()
        )

        image_list = [
            echo_images[key]
            for key in sorted_keys
        ]

        output_image = sitk.JoinSeries(
            image_list
        )

    # ------------------------------------------------------------
    # Standard behavior.
    # ------------------------------------------------------------

    if not collect_metadata:
        return output_image

    # ------------------------------------------------------------
    # Acquisition metadata.
    # ------------------------------------------------------------

    if acquisition_metadata is None:
        acquisition_metadata = {}

    valid_echoes = [
        echo
        for echo in sorted_keys
        if echo is not None
    ]

    if len(valid_echoes) == 1:
        acquisition_metadata["EchoTime"] = round(
            float(valid_echoes[0]),
            6,
        )

    elif len(valid_echoes) > 1:
        acquisition_metadata["EchoTimes"] = [
            round(
                float(echo),
                6,
            )
            for echo in valid_echoes
        ]

    acquisition_metadata["TimeUnit"] = "ms"

    # ------------------------------------------------------------
    # Source geometry.
    #
    # Avoid repeating identical slice-plane information for every
    # echo. If volume geometries differ, retain them independently.
    # ------------------------------------------------------------

    source_volumes = [
        geometry_by_echo[key]
        for key in sorted_keys
    ]

    reference_planes = (
        source_volumes[0]["SlicePlanes"]
    )

    shared_geometry = all(
        volume["SlicePlanes"] == reference_planes
        for volume in source_volumes[1:]
    )

    if shared_geometry:
        source_geometry = {
            "CoordinateSystem": "LPS",
            "LengthUnit": "mm",
            "SharedAcrossVolumes": True,
            "SlicePlanes": reference_planes,
        }

    else:
        source_geometry = {
            "CoordinateSystem": "LPS",
            "LengthUnit": "mm",
            "SharedAcrossVolumes": False,
            "Volumes": source_volumes,
        }

    # ------------------------------------------------------------
    # Final metadata object.
    # ------------------------------------------------------------

    metadata = {
        "CMPLMetadataVersion": 1,

        "Acquisition": (
            acquisition_metadata
        ),

        "CMPLSourceGeometry": (
            source_geometry
        ),

        "CMPLSimpleITKGeometry": {
            "CoordinateSystem": "LPS",

            "Dimension": (
                output_image.GetDimension()
            ),

            "Size": list(
                output_image.GetSize()
            ),

            "Origin": list(
                output_image.GetOrigin()
            ),

            "Spacing": list(
                output_image.GetSpacing()
            ),

            "Direction": list(
                output_image.GetDirection()
            ),
        },
    }

    return output_image, metadata

def dicom_to_SimpleITK(
    dicom_directory,
    series_id=None,
):
    """
    Read a DICOM series into a SimpleITK image.

    Parameters
    ----------
    dicom_directory : str or Path
        Directory containing the DICOM series.

    series_id : str, optional
        SeriesInstanceUID to read. If omitted, the first available
        series is used.

    Returns
    -------
    sitk.Image
        A 3D image for a single echo or a 4D image for multiple echoes.
    """
    return _dicom_to_simpleitk(
        dicom_directory,
        series_id=series_id,
        collect_metadata=False,
    )

def itk_to_nifti(itk_image, nifti_path, verbose=True):
    # Check if the provided path ends with valid NIfTI extensions.
    if not (nifti_path.endswith('.nii') or nifti_path.endswith('.nii.gz')):
        nifti_path += '.nii.gz'

    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(nifti_path)
        writer.Execute(itk_image)
        if verbose:
            print(f"File written successfully: {nifti_path}")
    except Exception as e:
        print(f"Error converting ITK image to NIfTI: {e}")
        raise  # re-raise the exception for further handling if needed

    return os.path.abspath(nifti_path)

def itk_mask_correction(img: Nifti1Image, mask: Nifti1Image, tol: float = 1e-1, return_axis=False) -> np.ndarray:
    """
    Automatically corrects the orientation of a segmentation mask to match a reference image.

    This function compares the affine translations of a reference image and its corresponding mask.
    It detects axes along which the mask has been flipped (i.e., where the translation difference
    corresponds to a flip) and then flips the mask data along those axes.

    Parameters:
        img (Nifti1Image): The reference image (e.g., an anatomical MRI) with the correct orientation.
        mask (Nifti1Image): The segmentation mask image whose orientation needs correction.
        tol (float): Tolerance value for comparing the expected difference in translation
                     (default is 1e-1).

    Returns:
        np.ndarray: The corrected mask data array after flipping the necessary axes.

    Notes:
        - This function assumes that the reference image and mask have the same spatial dimensions.
        - The affine matrices of the images are used to determine voxel spacing and expected translation shifts.
    """
    # Extract the affine matrices from the reference image and the mask.
    img_affine = img.affine
    mask_affine = mask.affine

    # Get the shape of the spatial dimensions (assuming the first three dimensions are x, y, z).
    shape = img.shape[:3]

    # Retrieve the mask data as a NumPy array.
    mask_data = mask.get_fdata()

    flip_axes = []  # List to store axes along which the mask is flipped.

    # Loop over each spatial axis (0, 1, 2 corresponding to x, y, z).
    for i in range(3):
        # Extract the i-th column of the reference image affine.
        # The norm of this column gives the voxel spacing along that axis.
        col = img_affine[:3, i]
        spacing = np.linalg.norm(col)

        # Calculate the expected difference in translation if the axis were flipped.
        # For a flipped axis, the translation difference should be approximately:
        # - (number of voxels in that dimension - 1) * voxel spacing.
        expected_diff = - (shape[i] - 1) * spacing

        # Calculate the actual difference in translations between the mask and the reference image.
        diff_vector = mask_affine[:3, 3] - img_affine[:3, 3]
        # Project this difference onto the axis direction.
        proj = np.dot(diff_vector, col) / spacing

        # If the projected difference is close to the expected value, we infer that the axis is flipped.
        if np.abs(proj - expected_diff) < tol:
            flip_axes.append(i)

    # Flip the mask data along each detected axis.
    for axis in flip_axes:
        mask_data = np.flip(mask_data, axis=axis)

    if return_axis:
        return mask_data.copy(), flip_axes
    else:
        return mask_data.copy()