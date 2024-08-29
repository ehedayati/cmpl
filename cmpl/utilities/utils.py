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
from scipy import ndimage


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
        contrast (str): acquired contrast
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


def kspace_to_image_space(kspace, fourier_dims=[0, 1, 2], coil_column_loc=-1, return_coil_images=False):
    """
    Inverse fourier transform to extract image space from the given MRI K-space.

    Args:
    - undersampled_kspace (numpy.ndarray or torch.tensor): The k-space
    - fourier_dims (list of ints): The dimensions of the inverse fourier
    - column_loc (int): coil column if not the last column

    Returns:
    - numpy.ndarray: The reconstructed volume using the square root of the sum of squared magnitudes of the coil images.
    """
    nc = kspace.shape[coil_column_loc]

    is_tensor = False
    if isinstance(kspace, pt.Tensor):
        is_tensor = True

    if not is_tensor:
        kspace = pt.tensor(kspace)

    if coil_column_loc != -1:
        kspace = kspace.moveaxis(coil_column_loc, -1)
    # Apply 3D IFFT on the entire k-space data at once
    image_space_before_shift = pt.fft.ifftn(pt.fft.ifftshift(kspace, dim=fourier_dims), dim=fourier_dims,
                                            norm="ortho")

    # Shift the zero frequency components to the center for the entire set
    image_space = pt.fft.fftshift(image_space_before_shift, dim=fourier_dims)

    # Compute the combined volume directly from the shifted_volumes array
    combined_volume = pt.sqrt(pt.sum(pt.abs(image_space) ** 2, axis=-1))

    if is_tensor:
        if return_coil_images:
            return combined_volume, image_space
        return combined_volume
    else:
        if return_coil_images:
            return combined_volume.numpy(), image_space.numpy()
        return combined_volume.numpy()


def apply_hamming_filter_4d_numpy(input_array, dim1, dim2):
    """
    Applies a Hamming filter to the specified dimensions of a 4D input array in NumPy.

    Parameters:
    - input_array: A 4D NumPy array, potentially with complex numbers.
    - dim1: The first dimension to apply the Hamming filter on.
    - dim2: The second dimension to apply the Hamming filter on.

    Returns:
    - The filtered 4D array.
    """
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if input_array.ndim != 4:
        raise ValueError("Input array must be 4D")

    # Generate the Hamming windows for the specified dimensions
    size1 = input_array.shape[dim1]
    size2 = input_array.shape[dim2]
    hamming_window_dim1 = np.hamming(size1)
    hamming_window_dim2 = np.hamming(size2)

    # Generate a 2D Hamming window for the specified dimensions
    hamming_window_2d = np.outer(hamming_window_dim1, hamming_window_dim2)

    # Reshape the 2D window to match the input dimensions
    shape = [1, 1, 1, 1]  # Default shape for a 4D array
    shape[dim1] = size1
    shape[dim2] = size2
    hamming_window_4d = np.reshape(hamming_window_2d, shape)

    # Expand the Hamming window to match the input array's shape
    expanded_window = np.broadcast_to(hamming_window_4d, input_array.shape)

    # Apply the Hamming window to the input array
    filtered_array = input_array * expanded_window

    return filtered_array


def resize_complex_matrix_fft(image, target_shape):
    """
    Resize a complex matrix using FFT and IFFT to achieve the target shape.

    This function resizes an image (or any 2D matrix) represented as a complex matrix using the Fast Fourier Transform (FFT)
    and its inverse (IFFT). The resizing process involves padding or cropping the frequency domain representation of the image
    to adjust its spatial dimensions. This method is particularly useful for applications where preserving the frequency
    characteristics of the image during resizing is important.

    Parameters:
    - image (pt.Tensor or compatible format): The input image as a complex matrix. If not a PyTorch tensor, it will be converted.
    - target_shape (tuple of int): The target dimensions (height, width) for the resized image.

    Returns:
    - pt.Tensor: The resized matrix as a complex matrix, represented in a PyTorch tensor.

    Note:
    - Padding is applied symmetrically if the target shape is larger than the original shape.
    - Cropping is centered if the target shape is smaller than the original shape.
    """

    fft_pt = lambda X, ax: pt.fft.fftshift(pt.fft.fftn(pt.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax)
    ifft_pt = lambda X, ax: pt.fft.fftshift(pt.fft.ifft2(pt.fft.ifftshift(X, dim=ax), dim=ax, norm='ortho'), dim=ax)

    if image.shape == target_shape:
        return image  # No need to resize if it's already the target shape
    if not isinstance(image, pt.Tensor):
        image = pt.tensor(image, dtype=pt.complex64)

    # Compute the FFT of the original image
    # fft_image = pt.fft.fftshift(pt.fft.fftn(image))
    ifft_image = fft_pt(image,[i for i in range(len(image.shape))])

    # Determine the difference in shape
    current_shape = pt.tensor(image.shape)
    target_shape = pt.tensor(target_shape)
    padding = target_shape - current_shape

    # Apply padding or cropping
    if (padding < 0).any():
        # Cropping
        crop_slices = tuple(slice(-p // 2, None if p // 2 == 0 else p // 2) for p in padding)
        resized_fft = ifft_image[crop_slices]
    else:
        # Padding
        # We need to pad manually since PyTorch doesn't support complex padding directly
        padding = tuple((p // 2, p - p // 2) for p in padding.tolist())  # Convert to list for iterating
        target_shape = tuple(target_shape)
        resized_fft = pt.zeros(target_shape, dtype=pt.complex64)
        start_indices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in padding)
        resized_fft[start_indices] = ifft_image

    # Compute the IFFT of the resized FFT image
    # resized_image = pt.fft.ifftn(pt.fft.ifftshift(resized_fft))
    resized_image = ifft_pt(resized_fft,[i for i in range(len(resized_fft.shape))])

    return resized_image


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
        pt.set_grad_enabled(False)

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
    pt.set_grad_enabled(True)
    if is_tensor:
        return large_tensor
    else:
        return large_tensor.numpy()


def resize_matrix(matrix, target_shape=(600, 600)):
    """
    Resize a 2D matrix to the target shape using interpolation.

    Args:
        matrix (numpy.ndarray): The input 2D matrix to be resized.
        target_shape (tuple): The target shape (height, width) for the output matrix.

    Returns:
        numpy.ndarray: The resized matrix.
    """
    pt.set_grad_enabled(False)
    if matrix.shape == target_shape:
        return matrix  # No need to resize if it's already the target shape

    # Compute the scaling factors
    scale_factors = (target_shape[0] / matrix.shape[0], target_shape[1] / matrix.shape[1])

    # Use scipy.ndimage.zoom for interpolation
    resized_matrix = ndimage.zoom(matrix, scale_factors, order=1)
    pt.set_grad_enabled(True)
    if isinstance(matrix, pt.Tensor):
        return pt.tensor(resized_matrix)
    return resized_matrix