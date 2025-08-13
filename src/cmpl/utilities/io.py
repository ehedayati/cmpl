# File created by: Eisa Hedayati
# Date: 8/27/2024
# Description: This file is developed at CMRR
import nibabel as nib
import numpy as np
import os
import pydicom
import SimpleITK as sitk
from collections import defaultdict


def nifti_read(file_name, re_orient=True):
    O = lambda MAT: np.rot90(MAT[:, ::-1, :], k=1, axes=(0, 1))[:,:,::-1]
    nifti = nib.load(file_name)
    if re_orient:
        return nifti, O(nifti.get_fdata())
    else:
        return nifti, nifti.get_fdata()


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

def update_nifti_data(file_path, new_data, output_path=None):
    """
    Load a NIfTI file, replace its data with new_data, and save it.

    Args:
    file_path (str): Path to the original NIfTI file.
    new_data (numpy.ndarray): New data array to replace the existing NIfTI data.
    output_path (str, optional): Path to save the updated NIfTI file. If None, it overwrites the original file.

    Returns:
    nib.Nifti1Image: The updated NIfTI image object.
    """
    # Load the existing NIfTI file
    nifti = nib.load(file_path)

    # Validate the new data dimensions
    # if new_data.shape != nifti.shape:
    #     raise ValueError("New data must have the same shape as the original NIfTI data.")

    # Create a new NIfTI image object with the new data and the same header
    new_nifti = nib.Nifti1Image(new_data, affine=nifti.affine, header=nifti.header)

    # Save the new NIfTI image to disk
    if output_path is None:
        output_path = file_path  # Overwrite the original file if no output path is specified
    nib.save(new_nifti, output_path)

    print(f"Updated NIfTI file saved to {output_path}")
    return new_nifti

def dicom_to_SimpleITK(dicom_directory):
    """
    Reads a multi-echo DICOM series from a directory where all echoes are stored in one series
    but with different EchoTime values (DICOM tag 0018|0081) on a per-slice basis, and returns
    either a 3D image (if a single echo is found) or a merged 4D image (if multiple echoes are present).

    For each echo, the metadata from the first DICOM file in that echo group is copied to the
    resulting 3D image wherever possible.

    Parameters:
        dicom_directory (str): Path to the directory containing the DICOM files.

    Returns:
        sitk.Image: A 4D image if multiple echoes are present, or a 3D image if only one echo is found.

    Raises:
        ValueError: If no DICOM series is found in the provided directory.
    """
    # Initialize the ImageSeriesReader and get the series IDs.
    series_reader = sitk.ImageSeriesReader()
    series_IDs = series_reader.GetGDCMSeriesIDs(dicom_directory)
    if not series_IDs:
        raise ValueError("No DICOM series found in the provided directory.")

    # For this example, use the first available series ID.
    series_id = series_IDs[0]
    file_names = series_reader.GetGDCMSeriesFileNames(dicom_directory, series_id)

    # Group file names by EchoTime.
    echo_groups = defaultdict(list)
    for f in file_names:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(f)
        file_reader.ReadImageInformation()  # Read metadata only
        try:
            # Extract EchoTime as a float from the metadata (tag 0018|0081)
            echo_time = float(file_reader.GetMetaData("0018|0081"))
        except Exception:
            # If EchoTime is missing, use None as the key.
            echo_time = None
        echo_groups[echo_time].append(f)

    # Process each echo group to read as a separate 3D image.
    echo_images = {}
    for echo, files in echo_groups.items():
        # Helper function to extract the InstanceNumber for sorting slices.
        def get_instance_number(filename):
            r = sitk.ImageFileReader()
            r.SetFileName(filename)
            r.ReadImageInformation()
            try:
                return int(r.GetMetaData("0020|0013"))
            except Exception:
                return 0  # Fallback if InstanceNumber is missing

        # Sort file names by instance number.
        files.sort(key=get_instance_number)

        # Read metadata from the first DICOM file in the group.
        first_file = files[0]
        meta_reader = sitk.ImageFileReader()
        meta_reader.SetFileName(first_file)
        meta_reader.ReadImageInformation()
        first_metadata = {}
        for key in meta_reader.GetMetaDataKeys():
            first_metadata[key] = meta_reader.GetMetaData(key)

        # Read the 3D image from the sorted files.
        series_reader.SetFileNames(files)
        image = series_reader.Execute()

        # Copy metadata from the first file to the resulting image.
        for key, value in first_metadata.items():
            image.SetMetaData(key, value)

        echo_images[echo] = image

    # If only one echo exists, return that single 3D image.
    if len(echo_images) == 1:
        return list(echo_images.values())[0]

    # If multiple echoes are present, merge them into a 4D image.
    sorted_keys = sorted(echo_images.keys())
    image_list = [echo_images[key] for key in sorted_keys]
    merged_image = sitk.JoinSeries(image_list)

    return merged_image


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


from nibabel.nifti1 import Nifti1Image


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