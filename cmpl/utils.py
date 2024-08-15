# File created by: Eisa Hedayati
# Date: 12/29/2023
# Description: This file is developed at CMR

import h5py
import numpy as np
import nibabel as nib
import zipfile
import os
import pydicom
import torch as pt
pt.set_grad_enabled(False)


def h5_to_nifti(input_file, output_file):
    """
    Convert MRI data from an HDF5 file to a NIfTI format file.

    This function reads MRI data stored in an HDF5 file and converts it into
    the NIfTI format, which is commonly used for MRI data. The HDF5 file must
    contain specific datasets necessary for the conversion:

    - 'dicom_images': A dataset containing the MRI image data in a format that
                      can be converted to NIfTI. This usually includes the image
                      intensity values for each voxel.
    - 'orientation': A dataset with six values representing the orientation of
                     the MRI scan in space. The first three values are the row
                     direction cosines, and the next three are the column
                     direction cosines.
    - 'position': A dataset with three values indicating the position of the
                  first voxel in the MRI data in a 3D space.
    - 'pixel_spacing': A dataset with two values, providing the pixel spacing
                       (size of a pixel) in the row and column directions.
    - 'slice_thickness': A dataset with a single value indicating the thickness
                         of each slice in the MRI data.

    Args:
        input_file (str): Path to the input HDF5 file. This file must contain
                          the datasets 'dicom_images', 'orientation', 'position',
                          'pixel_spacing', and 'slice_thickness', all structured
                          appropriately to represent MRI data.
        output_file (str): Path for the output NIfTI file.

    Returns:
        tuple: A tuple containing a boolean indicating the success of the
               conversion and a message string.
    """
    try:
        # Reading data from the HDF5 file
        with h5py.File(input_file, 'r') as hf:
            dicom_images, orientation, position, pixel_spacing, slice_thickness = \
                [np.array(hf[key]) for key in
                 ['dicom_images', 'orientation', 'position', 'pixel_spacing', 'slice_thickness']]

        # Extracting orientation vectors
        row_cosines, col_cosines = orientation[:3], orientation[3:]
        slice_normal = np.cross(-row_cosines, col_cosines)

        # Adjusting position coordinates (negating x and y components)
        position[:2] = -position[:2]

        # Constructing the affine transformation matrix
        affine = np.zeros((4, 4))
        affine[:3, 0] = -row_cosines * pixel_spacing[0]
        affine[:3, 1] = col_cosines * pixel_spacing[1]
        affine[:3, 2] = slice_normal * slice_thickness
        affine[:3, 3] = position
        affine[3, 3] = 1.0

        # Creating and saving the NIfTI image
        nifti_img = nib.Nifti1Image(dicom_images, affine)
        # nifti_img = nib.Nifti1Image(np.flip(np.rot90(dicom_images, k=-1, axes=(0, 1)), axis=0), affine)
        nib.save(nifti_img, output_file)

        return True, "Conversion successful."
    except Exception as e:
        return False, "Conversion failed: {}".format(str(e))


def prepare_zipped_dicom(zip_path, extract_path):
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Identify the parent directory
    parent_directory = next(os.walk(extract_path))[1][0]
    dicom_directory = os.path.join(extract_path, parent_directory)

    return dicom_directory


def dicom_to_h5(dicom_directory, h5py_path, contrast='3D_gre_sag',num_contrasts=7, num_slices_per_contrast=120):
    """
    Convert DICOM files in a directory to an HDF5 file.

    Parameters:
        dicom_directory (str): Path to the directory containing DICOM files.
        h5py_path (str): Path to the output HDF5 file.
        num_contrasts (int, optional): Number of contrasts. Default is 7.
        num_slices_per_contrast (int, optional): Number of slices per contrast. Default is 120.

    Raises:
        FileNotFoundError: If the input DICOM directory is not found.
        ValueError: If there are issues with DICOM files or data conversion.
    """
    try:
        # Check if the DICOM directory exists
        if not os.path.exists(dicom_directory):
            raise FileNotFoundError(f"DICOM directory '{dicom_directory}' not found.")

        # Prepare to read DICOM files
        dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
        dicom_files.sort()  # Ensure files are sorted in ascending order

        # Initialize a list to hold 3D arrays for each contrast
        all_contrasts = []

        # Process each contrast series
        for i in range(num_contrasts):
            contrast_images = []
            for j in range(num_slices_per_contrast * i, num_slices_per_contrast * (i + 1)):
                filepath = os.path.join(dicom_directory, dicom_files[j])
                try:
                    ds = pydicom.dcmread(filepath)
                    contrast_images.append(ds.pixel_array)
                except Exception as e:
                    raise ValueError("Error reading DICOM file '{}': {}".format(filepath, str(e)))

            # Convert the list of arrays to a single 3D numpy array
            contrast_array = np.stack(contrast_images, axis=2)
            all_contrasts.append(contrast_array)

        # Get information from the last DICOM file for metadata
        last_dicom_file = pydicom.dcmread(os.path.join(dicom_directory, dicom_files[-1]))
        image_orientation_patient = last_dicom_file.ImageOrientationPatient
        image_position_patient = last_dicom_file.ImagePositionPatient
        pixel_spacing = last_dicom_file.PixelSpacing  # [width spacing, height spacing]
        slice_thickness = last_dicom_file.SliceThickness

        dicom_4d_array = np.transpose(np.stack(all_contrasts, axis=3), axes=[1, 0, 2, -1])[:, :, ::-1, :]

        # Write data to h5py file
        with h5py.File(h5py_path + '/' + contrast + '.h5', 'w') as hf:
            hf.create_dataset('dicom_images', data=dicom_4d_array)
            hf.create_dataset('orientation', data=np.array(image_orientation_patient))
            hf.create_dataset('position', data=np.array(image_position_patient))
            hf.create_dataset('pixel_spacing', data=np.array(pixel_spacing))
            hf.create_dataset('slice_thickness', data=np.array(slice_thickness))
        print("DICOM data has been saved to h5py file.")
        h5_path_2 = os.path.join(h5py_path, contrast)
        os.makedirs(h5_path_2, exist_ok=True)

        # Iterate over each slice in the 4D array
        for i in range(dicom_4d_array.shape[-1]):
            slice_filename = os.path.join(h5_path_2, f'echo_{i+1}.h5')
            with h5py.File(slice_filename, 'w') as hf:
                hf.create_dataset('dicom_images', data=dicom_4d_array[..., i])
                hf.create_dataset('orientation', data=np.array(image_orientation_patient))
                hf.create_dataset('position', data=np.array(image_position_patient))
                hf.create_dataset('pixel_spacing', data=np.array(pixel_spacing))
                hf.create_dataset('slice_thickness', data=np.array(slice_thickness))
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")


def zero_pad(tensor, final_shape):
    """
    Place the small_tensor in the center of large_tensor.

    Args:
    - large_tensor (torch.Tensor): The larger tensor in which the smaller tensor will be centered.
    - small_tensor (torch.Tensor): The smaller tensor to be placed in the center of the larger tensor.

    Returns:
    - torch.Tensor: The resulting tensor with the small_tensor centered within the large_tensor.
    """
    # Ensure the small tensor can fit in the large tensor
    is_tensor = False
    if isinstance(tensor, pt.Tensor):
        is_tensor = True

    if not is_tensor:
        tensor = pt.tensor(tensor)

    large_tensor = pt.zeros(final_shape, dtype=pt.complex64)
    for i in range(len(large_tensor.shape)):
        if tensor.shape[i] > large_tensor.shape[i]:
            raise ValueError("The small tensor is larger than the large tensor in dimension {}.".format(i))

    # Calculate start indices for small_tensor to be centered
    start_indices = [(large_dim - small_dim) // 2 for large_dim, small_dim in zip(large_tensor.shape, tensor.shape)]

    # Create a slice object for each dimension
    slices = tuple(slice(start_idx, start_idx + small_dim) for start_idx, small_dim in zip(start_indices, tensor.shape))

    # Place the small_tensor in the center of large_tensor
    large_tensor[slices] = tensor
    if is_tensor:
        return large_tensor
    else:
        return large_tensor.numpy()

def nifti_read(file_name):
    nifti = nib.load(file_name)
    return nifti,  np.rot90(nifti.get_fdata()[:,::-1,:], k=1, axes=(0,1))

def load_dicom_scan_from_dir(directory, verbose=False):
    """
    Load all DICOM files from the given directory and convert them into a 3D numpy array.

    Args:
        directory (str): Path to the directory containing DICOM files.
        verbose (bool): If True, print additional information about the loading process.

    Returns:
        numpy.ndarray: A 3D numpy array containing the pixel data from DICOM files.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    # Gather all .dcm files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not files:
        raise ValueError("No DICOM files found in the directory.")

    # Load and sort DICOM files by instance number
    dicom_files = []
    for file in files:
        try:
            dcm_path = os.path.join(directory, file)
            dcm = pydicom.dcmread(dcm_path)
            dicom_files.append(dcm)
            if verbose:
                print(f"Loaded {file} with Instance Number: {dcm.InstanceNumber}")
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not dicom_files:
        raise ValueError("Failed to load any DICOM files.")

    dicom_files.sort(key=lambda x: int(x.InstanceNumber))

    # Convert pixel data to a 3D numpy array
    try:
        image_data = np.stack([s.pixel_array for s in dicom_files])
    except Exception as e:
        raise RuntimeError(f"Error creating 3D array from DICOM files: {e}")

    return image_data