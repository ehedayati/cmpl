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
    # Extract pixel arrays
    image_data_list = [info['dcm'].pixel_array for info in dicom_info_list]
    pixel_spacing = dicom_info_list[0]['dcm'].PixelSpacing
    slice_thickness = dicom_info_list[0]['dcm'].SliceThickness
    spacing = list(map(float, pixel_spacing))
    # Insert slice_thickness at the position indicated by slice_axis
    spacing = spacing[:slice_axis] + [float(slice_thickness)] + spacing[slice_axis:]
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

    # if_reverse = False
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
                if if_reverse:
                    image_data = np.flip(image_data, -2)
            elif slice_axis == 1:
                image_data = np.moveaxis(image_data, 1, -1)
            elif slice_axis == 2:
                image_data = np.moveaxis(image_data, -1, 0)
        else:
            if slice_axis == 0:
                image_data = np.moveaxis(image_data, 0, -1)
                if if_reverse:
                    image_data = np.flip(image_data, -2)
            elif slice_axis == 1:
                pass
            elif slice_axis == 2:
                image_data = np.moveaxis(image_data, -1, 0)
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
            image_position = np.array(dcm.ImagePositionPatient, dtype=float)
            image_orientation = np.array(dcm.ImageOrientationPatient, dtype=float)
            row_cosines = image_orientation[:3]
            col_cosines = image_orientation[3:]
            normal_vector = np.cross(row_cosines, col_cosines)
            slice_location = np.dot(normal_vector, image_position)
            dicom_info_list.append({
                'dcm': dcm,
                'EchoNumber': echo_number,
                'InstanceNumber': instance_number,
                'SliceLocation': slice_location,
                'ImagePositionPatient': image_position,
                'ImageOrientationPatient': image_orientation,
                'RowCosines': row_cosines,
                'ColCosines': col_cosines,
                'NormalVector': normal_vector,
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
            # [Xx, Xy, Xz, Yx, Yy, Yz]
            # where X is the direction cosines of the image rows (axis 0)
            # and Y is the direction cosines of the image columns (axis 1)

            X = np.array(image_orientation[:3])  # direction cosines for image rows (axis 0)
            Y = np.array(image_orientation[3:6])  # direction cosines for image columns (axis 1)

            # For image rows (axis 0)
            abs_X = np.abs(X)
            max_index_X = np.argmax(abs_X)  # index 0,1,2 corresponds to x,y,z patient axes
            sign_X = np.sign(X[max_index_X])

            # For image columns (axis 1)
            abs_Y = np.abs(Y)
            max_index_Y = np.argmax(abs_Y)
            sign_Y = np.sign(Y[max_index_Y])

            # Flip axis if direction cosine along dominant patient axis is negative
            flip_row = sign_X < 0
            flip_col = sign_Y < 0

            return flip_row, flip_col

        flip_row, flip_col = get_axis_flip(image_orientation)

        if not flip_row:
            pixel_array = np.flip(pixel_array, axis=0)  # Flip rows (axis 0)
        if not flip_col:
            pixel_array = np.flip(pixel_array, axis=1)  # Flip columns (axis 1)

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

        # Flip x-axis (axis=0) to have patient's left on image right in axial and coronal views
        image_data = np.flip(image_data, axis=0)

        # Check if we need to flip the z-axis (slice axis)
        slice_positions = np.array([info['SliceLocation'] for info in dicom_info_list[::num_echoes]])
        if slice_positions[0] > slice_positions[-1]:
            # Slices are ordered from superior to inferior, flip z-axis
            image_data = np.flip(image_data, axis=2)
    else:
        # Do not reshape
        pass

    return image_data#[...,::-1,:]
