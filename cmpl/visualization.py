# File created by: Eisa Hedayati
# Date: 1/3/2024
# Description: This file is developed at CMRR

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch as pt
pt.set_grad_enabled(False)


def resize_matrix(matrix, target_shape=(600, 600)):
    """
    Resize a 2D matrix to the target shape using interpolation.

    Args:
        matrix (numpy.ndarray): The input 2D matrix to be resized.
        target_shape (tuple): The target shape (height, width) for the output matrix.

    Returns:
        numpy.ndarray: The resized matrix.
    """

    if matrix.shape == target_shape:
        return matrix  # No need to resize if it's already the target shape

    # Compute the scaling factors
    scale_factors = (target_shape[0] / matrix.shape[0], target_shape[1] / matrix.shape[1])

    # Use scipy.ndimage.zoom for interpolation
    resized_matrix = ndimage.zoom(matrix, scale_factors, order=1)
    if isinstance(matrix, pt.Tensor):
        return pt.tensor(resized_matrix)
    return resized_matrix


def side_by_side_view(image1, image2, color_palette='gray', dpi=100):
    """
    Display two images side by side.

    Args:
        image1 (numpy.ndarray): The first image to display. It should be a 2D or 3D array.
                                If 3D, the shape should be (height, width, channels).
        image2 (numpy.ndarray): The second image to display, with the same requirements as image1.
        color_palette (str, optional): The color palette to use for displaying the images.
                                       Defaults to 'gray'. If 'gray', images are displayed in grayscale.
                                       Other color palettes can be specified to suit the images.

    Note:
        - If the images are not square (height != width), they will be resized to a square shape
          with dimensions 600x600 pixels before being displayed.
        - The function uses Matplotlib to create a subplot with 1 row and 2 columns, displaying
          each image in its own subplot.
    """

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.dpi = dpi
    # Check if images are square, if not, resize them
    if image1.shape[0] != image1.shape[1]:
        image1 = resize_matrix(image1)
    if image2.shape[0] != image2.shape[1]:
        image2 = resize_matrix(image2)

    # Set color mapping based on the specified color palette
    if color_palette == 'gray':
        cmap1, cmap2 = 'gray', 'gray'
    else:
        cmap1, cmap2 = color_palette, color_palette

    # Display the first image
    axs[0].imshow(image1, cmap=cmap1)
    axs[0].axis('off')  # Hide axis
    axs[0].set_title('Image 1')  # Set title for the first image

    # Display the second image
    axs[1].imshow(image2, cmap=cmap2)
    axs[1].axis('off')  # Hide axis
    axs[1].set_title('Image 2')  # Set title for the second image
    plt.tight_layout()
    plt.show()  # Display the plot

    
def extract_slice(matrix, slice_number, dimension):
    """
    Extract a specific slice from a 3D matrix based on the specified dimension.

    Args:
        matrix (numpy.ndarray): The input 3D matrix.
        slice_number (int): The specific slice to extract.
        dimension (str): The dimension along which to slice ('axial', 'coronal', 'sagittal').

    Returns:
        numpy.ndarray: The extracted 2D slice.
    """
    if dimension == 'coronal':
        return matrix[:, slice_number, :]
    elif dimension == 'axial':
        return matrix[slice_number, :, :]
    else:  # 'sagittal' or default
        return matrix[:, :, slice_number]

    
def visualize_segmentation_slice(grayscale_image, segmentation_matrix, slice_number, dimension='axial', target_shape=(600, 600)):
    """
    Visualize a specific slice of the 3D segmentation matrix on top of the corresponding grayscale image slice.

    Args:
        grayscale_image (numpy.ndarray): 3D grayscale image matrix.
        segmentation_matrix (numpy.ndarray): 3D segmentation matrix.
        slice_number (int): The specific slice to visualize.
        dimension (str): The dimension along which to slice ('axial', 'coronal', 'sagittal').
        target_shape (tuple): The target shape for resizing the slices.
    """
    # Define a color map for 10 distinct colors
    colors = np.array([
        [0, 0, 0],       # Color for 0
        [255, 0, 0],     # Color for 1
        [0, 255, 0],     # Color for 2
        [0, 0, 255],     # Color for 3
        [255, 255, 0],   # Color for 4
        [255, 0, 255],   # Color for 5
        [0, 255, 255],   # Color for 6
        [128, 0, 0],     # Color for 7
        [0, 128, 0],     # Color for 8
        [0, 0, 128]      # Color for 9
    ])

    # Extract and resize the specific slices
    grayscale_slice = extract_slice(grayscale_image, slice_number, dimension)
    segmentation_slice = extract_slice(segmentation_matrix, slice_number, dimension)

    grayscale_slice_resized = resize_matrix(grayscale_slice, target_shape)
    segmentation_slice_resized = resize_matrix(segmentation_slice, target_shape)

    # Apply color map
    segmentation_colored = np.zeros((*segmentation_slice_resized.shape, 3), dtype=np.uint8)
    for label in range(10):
        segmentation_colored[segmentation_slice_resized == label] = colors[label]
    plt.figure(dpi=100)
    # Overlay the segmentation on the grayscale image
    plt.imshow(grayscale_slice_resized, cmap='gray')
    plt.imshow(segmentation_colored, alpha=0.5)  # Adjust alpha for transparency
    
    # Display the result
    plt.axis('off')
    plt.show()


def kspace_to_image_space(kspace, fourier_dims=[0, 1, 2], coil_column_loc=-1):
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
        return combined_volume
    else:
        return combined_volume.numpy()


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
    if image.shape == target_shape:
        return image  # No need to resize if it's already the target shape
    if not isinstance(image, pt.Tensor):
        image = pt.tensor(image, dtype=pt.complex64)

    # Compute the FFT of the original image
    fft_image = pt.fft.fftshift(pt.fft.fftn(image))

    # Determine the difference in shape
    current_shape = pt.tensor(image.shape)
    target_shape = pt.tensor(target_shape)
    padding = target_shape - current_shape

    # Apply padding or cropping
    if (padding < 0).any():
        # Cropping
        crop_slices = tuple(slice(-p // 2, None if p // 2 == 0 else p // 2) for p in padding)
        resized_fft = fft_image[crop_slices]
    else:
        # Padding
        # We need to pad manually since PyTorch doesn't support complex padding directly
        padding = tuple((p // 2, p - p // 2) for p in padding.tolist())  # Convert to list for iterating
        target_shape = tuple(target_shape)
        resized_fft = pt.zeros(target_shape, dtype=pt.complex64)
        start_indices = tuple(slice(p[0], -p[1] if p[1] > 0 else None) for p in padding)
        resized_fft[start_indices] = fft_image

    # Compute the IFFT of the resized FFT image
    resized_image = pt.fft.ifftn(pt.fft.ifftshift(resized_fft))

    return resized_image

