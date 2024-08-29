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
