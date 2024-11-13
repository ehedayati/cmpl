# File created by: Eisa Hedayati
# Date: 8/27/2024
# Description: This file is developed at CMRR
import nibabel as nib
import numpy as np
import os
import pydicom

def nifti_read(file_name, re_orient=True):
    nifti = nib.load(file_name)
    if re_orient:
        return nifti, np.rot90(nifti.get_fdata()[:,::-1,:], k=1, axes=(0,1))
    else:
        return nifti, nifti.get_fdata()


def load_dicom_scan_from_dir(directory, reshape=True, verbose=False):
    """
    Load all DICOM files from the given directory and convert them into a 3D or 4D numpy array,
    depending on whether the sequence is single-echo or multi-echo.

    Args:
        directory (str): Path to the directory containing DICOM files.
        reshape (bool): If True, returns an array reshaped based on the sequence type.
        verbose (bool): If True, print additional information about the loading process.

    Returns:
        numpy.ndarray: A numpy array containing the pixel data from DICOM files.
                       Shape is [x, y, z] for single-echo and [x, y, z, echo] for multi-echo.
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
            if position:
                slice_location = position[2]
            else:
                slice_location = getattr(dcm, 'SliceLocation', 0)
            dicom_info_list.append({
                'dcm': dcm,
                'EchoNumber': echo_number,
                'InstanceNumber': instance_number,
                'SliceLocation': slice_location,
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
    dicom_info_list.sort(key=lambda x: (x['EchoNumber'], x['SliceLocation']))

    # Extract pixel arrays
    image_data_list = [info['dcm'].pixel_array for info in dicom_info_list]

    # Determine the number of slices
    total_images = len(image_data_list)
    num_slices = total_images // num_echoes
    if verbose:
        print(f"Number of slices: {num_slices}")

    # Stack the image data
    try:
        image_data = np.stack(image_data_list)
    except Exception as e:
        raise RuntimeError(f"Error creating array from DICOM files: {e}")

    # Reshape the data appropriately
    if reshape:
        if is_multi_echo:
            # Reshape to [num_echoes, num_slices, x, y]
            image_data = image_data.reshape(num_echoes, num_slices, *image_data.shape[1:])
            # Move axes to get [x, y, z, echo]
            image_data = np.moveaxis(image_data, [0, 1], [-1, -2])
        else:
            # Reshape to [num_slices, x, y]
            image_data = image_data.reshape(num_slices, *image_data.shape[1:])
            # Move axis to get [x, y, z]
            image_data = np.moveaxis(image_data, 0, -1)

    return image_data

def load_dicom_scan_from_dir_radiological(directory, reshape=True, verbose=False):
    """
    Load all DICOM files from the given directory and convert them into a 3D or 4D numpy array,
    ensuring radiological convention (left-right flipping if needed, correct slice and echo ordering).

    Args:
        directory (str): Path to the directory containing DICOM files.
        reshape (bool): If True, returns an array reshaped based on the sequence type.
        verbose (bool): If True, print additional information about the loading process.

    Returns:
        numpy.ndarray: A numpy array containing the pixel data from DICOM files,
                       Shape is [x, y, z] for single-echo and [x, y, z, echo] for multi-echo.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    # Gather all .dcm files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not files:
        raise ValueError("No DICOM files found in the directory.")

    dicom_info_list = []
    for file in files:
        try:
            dcm_path = os.path.join(directory, file)
            dcm = pydicom.dcmread(dcm_path)
            echo_number = getattr(dcm, 'EchoNumbers', 1)
            instance_number = getattr(dcm, 'InstanceNumber', 0)
            position = getattr(dcm, 'ImagePositionPatient', None)
            if position:
                slice_location = position[2]
            else:
                slice_location = getattr(dcm, 'SliceLocation', 0)
            dicom_info_list.append({
                'dcm': dcm,
                'EchoNumber': echo_number,
                'InstanceNumber': instance_number,
                'SliceLocation': slice_location,
            })
            if verbose:
                print(f"Loaded {file} with Instance Number: {instance_number}, Echo Number: {echo_number}")
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not dicom_info_list:
        raise ValueError("Failed to load any DICOM files.")

    # Ensure sorting by EchoNumber first, then by SliceLocation
    dicom_info_list.sort(key=lambda x: (x['EchoNumber'], x['SliceLocation']))

    # Extract pixel arrays and organize by echo
    image_data_dict = {}
    for info in dicom_info_list:
        echo_number = info['EchoNumber']
        if echo_number not in image_data_dict:
            image_data_dict[echo_number] = []
        image_data_dict[echo_number].append(info['dcm'].pixel_array)

    # Stack data for each echo
    echo_arrays = []
    for echo_number in sorted(image_data_dict.keys()):
        echo_stack = np.stack(image_data_dict[echo_number])  # Stack slices for this echo
        echo_arrays.append(echo_stack)

    # Combine echos into a 4D array if multi-echo, or 3D if single-echo
    if len(echo_arrays) > 1:
        image_data = np.stack(echo_arrays, axis=-1)  # [z, x, y, echo]
    else:
        image_data = echo_arrays[0]  # Single-echo [z, x, y]

    # Apply radiological convention (flip left-right if necessary)
    first_orientation = dicom_info_list[0]['dcm'].ImageOrientationPatient
    if first_orientation[0] < 0:  # Indicates need for left-right flip
        image_data = np.flip(image_data, axis=1)  # Flip x-axis
        if verbose:
            print("Applied left-right flip for radiological convention.")

    if reshape:
        if image_data.ndim == 4:
            # Move axes for multi-echo [x, y, z, echo]
            image_data = np.moveaxis(image_data, [0, 1, 2, 3], [2, 1, 0, 3])
        else:
            # Move axes for single-echo [x, y, z]
            image_data = np.moveaxis(image_data, [0, 1, 2], [2, 1, 0])

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
