# File created by: Eisa Hedayati
# Date: 1/3/2024
# Description: This file is developed at CMRR

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


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

    return resized_matrix


def side_by_side_view(image1, image2, color_palette='gray'):
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

    plt.show()  # Display the plot
