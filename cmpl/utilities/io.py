# File created by: Eisa Hedayati
# Date: 8/27/2024
# Description: This file is developed at CMRR
import nibabel as nib
import numpy as np
import os
import pydicom

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
