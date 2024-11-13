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
    Load all DICOM files from the given directory and convert them into a 4D numpy array,
    with the number of echoes as the last dimension and in radiological view.

    Args:
        directory (str): Path to the directory containing DICOM files.
        reshape (bool): If True, returns an array reshaped to [x, y, z, echo].
        verbose (bool): If True, print additional information about the loading process.

    Returns:
        numpy.ndarray: A numpy array containing the pixel data from DICOM files.
                       Shape is [x, y, z, echo].
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
            image_position = np.array(getattr(dcm, 'ImagePositionPatient', [0, 0, 0]), dtype=float)
            image_orientation = np.array(getattr(dcm, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0]), dtype=float)
            # Calculate the slice location using the normal vector
            row_cosines = image_orientation[:3]
            col_cosines = image_orientation[3:]
            normal_vector = np.cross(row_cosines, col_cosines)
            slice_location = np.dot(image_position, normal_vector)
            dicom_info_list.append({
                'dcm': dcm,
                'EchoNumber': echo_number,
                'InstanceNumber': instance_number,
                'SliceLocation': slice_location,
                'ImagePositionPatient': image_position,
                'ImageOrientationPatient': image_orientation,
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
    if verbose:
        print(f"Detected sequence with {num_echoes} echo(s).")

    # Organize DICOM files by Echo Number and Slice Location
    dicom_info_list.sort(key=lambda x: (x['EchoNumber'], x['SliceLocation']))

    # Extract pixel arrays and orientations
    image_data_list = []
    for info in dicom_info_list:
        # Extract the pixel_array
        pixel_array = info['dcm'].pixel_array

        # Get the image orientation
        image_orientation = info['ImageOrientationPatient']

        # Determine axis flips
        def get_axis_flip(image_orientation):
            # image_orientation is an array of 6 elements:
            # [Yx, Yy, Yz, Xx, Xy, Xz]
            # where Y is the direction cosines of the image rows (axis 1)
            # and X is the direction cosines of the image columns (axis 0)

            X = np.array(image_orientation[3:6])  # direction cosines for image x-axis (columns)
            Y = np.array(image_orientation[0:3])  # direction cosines for image y-axis (rows)

            # For image x-axis
            abs_X = np.abs(X)
            max_index_X = np.argmax(abs_X)  # index 0,1,2 corresponds to x,y,z patient axes
            sign_X = np.sign(X[max_index_X])

            # For image y-axis
            abs_Y = np.abs(Y)
            max_index_Y = np.argmax(abs_Y)
            sign_Y = np.sign(Y[max_index_Y])

            # Flip axis if direction cosine along dominant patient axis is negative
            flip_x = sign_X < 0
            flip_y = sign_Y < 0

            return flip_x, flip_y

        flip_x, flip_y = get_axis_flip(image_orientation)

        if flip_x:
            pixel_array = np.flip(pixel_array, axis=1)  # Flip columns
        if flip_y:
            pixel_array = np.flip(pixel_array, axis=0)  # Flip rows

        image_data_list.append(pixel_array)

    # Determine the number of slices
    total_images = len(image_data_list)
    num_slices = total_images // num_echoes
    if verbose:
        print(f"Total images: {total_images}, Number of slices: {num_slices}")

    # Stack the image data
    try:
        image_data = np.stack(image_data_list)
    except Exception as e:
        raise RuntimeError(f"Error creating array from DICOM files: {e}")

    # Reshape the data appropriately
    if reshape:
        # Reshape to [num_echoes, num_slices, x, y]
        image_data = image_data.reshape(num_echoes, num_slices, *image_data.shape[1:])
        # Move axes to get [x, y, z, echo]
        image_data = np.moveaxis(image_data, [0, 1], [-1, -2])
        # The data is now in shape [x, y, z, echo]
    else:
        # Do not reshape
        pass

    return image_data
