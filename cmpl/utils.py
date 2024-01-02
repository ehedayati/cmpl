# File created by: Eisa Hedayati
# Date: 12/29/2023
# Description: This file is developed at CMR

import h5py
import numpy as np
import nibabel as nib
import zipfile
import os
import pydicom


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
        nifti_img = nib.Nifti1Image(np.transpose(dicom_images,axes=[1,0,2,-1])[:,:,::-1,:], affine)
        # nifti_img = nib.Nifti1Image(np.flip(np.rot90(dicom_images, k=-1, axes=(0, 1)), axis=0), affine)
        nib.save(nifti_img, output_file)

        return True, "Conversion successful."
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"


# Example usage:
# success, message = h5_to_nifti('input.h5', 'output.nii')
# print(message)

def prepare_zipped_dicom(zip_path, extract_path):
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Identify the parent directory
    parent_directory = next(os.walk(extract_path))[1][0]
    dicom_directory = os.path.join(extract_path, parent_directory)

    return dicom_directory


def dicom_to_h5(dicom_directory, h5py_name, num_contrasts=7, num_slices_per_contrast=120):
    """
    Convert DICOM files in a directory to an HDF5 file.

    Parameters:
        dicom_directory (str): Path to the directory containing DICOM files.
        h5py_name (str): Path to the output HDF5 file.
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
                    raise ValueError(f"Error reading DICOM file '{filepath}': {str(e)}")

            # Convert the list of arrays to a single 3D numpy array
            contrast_array = np.stack(contrast_images, axis=2)
            all_contrasts.append(contrast_array)

        # Get information from the last DICOM file for metadata
        last_dicom_file = pydicom.dcmread(os.path.join(dicom_directory, dicom_files[-1]))
        image_orientation_patient = last_dicom_file.ImageOrientationPatient
        image_position_patient = last_dicom_file.ImagePositionPatient
        pixel_spacing = last_dicom_file.PixelSpacing  # [width spacing, height spacing]
        slice_thickness = last_dicom_file.SliceThickness

        dicom_4d_array = np.stack(all_contrasts, axis=3)

        # Write data to h5py file
        with h5py.File(h5py_name, 'w') as hf:
            hf.create_dataset('dicom_images', data=dicom_4d_array)
            hf.create_dataset('orientation', data=np.array(image_orientation_patient))
            hf.create_dataset('position', data=np.array(image_position_patient))
            hf.create_dataset('pixel_spacing', data=np.array(pixel_spacing))
            hf.create_dataset('slice_thickness', data=np.array(slice_thickness))
        print("DICOM data has been saved to h5py file.")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")


# Example Usage:
# zip_path = 'MR-SE012-3D_GRE_T2__Sag.zip'
# extract_path = 'extract'
# h5py_path = 'output_file.h5'
#
# dicom_directory = prepare_zipped_dicom(zip_path, extract_path)
# dicom_to_h5(dicom_directory, h5py_path)
