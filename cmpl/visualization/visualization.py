# File created by: Eisa Hedayati
# Date: 1/3/2024
# Description: This file is developed at CMRR

import matplotlib.pyplot as plt
import numpy
import numpy as np
from ipywidgets import widgets, HBox, VBox, interactive
from IPython.display import display
from matplotlib.colors import ListedColormap
from cmpl.utilities.utils import resize_matrix
from typing import List, Optional

def side_by_side_view(image1: numpy.ndarray, image2: numpy.ndarray,
                      color_palette: str = 'gray', dpi: int = 100, titles: Optional[List[str]] = None):
    """
    Display two images side by side.

    Args:
        image1 (numpy.ndarray): The first image to display. It should be a 2D or 3D array.
                                If 3D, the shape should be (height, width, channels).
        image2 (numpy.ndarray): The second image to display, with the same requirements as image1.
        color_palette (str, optional): The color palette to use for displaying the images.
                                       Defaults to 'gray'. If 'gray', images are displayed in grayscale.
                                       Other color palettes can be specified to suit the images.
        dpi: set dpi of the plots
        titles (Optional[List[str]], optional): A list of titles to display in the plots.
        Defaults to ['image1', 'image2'].

    Note:
        - If the images are not square (height != width), they will be resized to a square shape
          with dimensions 600x600 pixels before being displayed.
        - The function uses Matplotlib to create a subplot with 1 row and 2 columns, displaying
          each image in its own subplot.
    """
    assert image1.ndim == 2 and image2.ndim == 2
    if titles is None:
        titles = ['image1', 'image2']
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
    axs[0].set_title(titles[0])  # Set title for the first image

    # Display the second image
    axs[1].imshow(image2, cmap=cmap2)
    axs[1].axis('off')  # Hide axis
    axs[1].set_title(titles[1])  # Set title for the second image
    plt.tight_layout()
    plt.show()  # Display the plot

    
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

    def extract_slice(matrix, slice_num, dim):
        """
        Extract a specific slice from a 3D matrix based on the specified dimension.

        Args:
            matrix (numpy.ndarray): The input 3D matrix.
            slice_num (int): The specific slice to extract.
            dim (str): The dimension along which to slice ('axial', 'coronal', 'sagittal').

        Returns:
            numpy.ndarray: The extracted 2D slice.
        """
        if dim == 'coronal':
            return matrix[:, slice_num, :]
        elif dim == 'axial':
            return matrix[slice_num, :, :]
        else:  # 'sagittal' or default
            return matrix[:, :, slice_num]

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


def plot_3D_mri(mri_image, slice_number=None, direction='axial', segmentation=None, alpha=0.5, dpi=150, target_shape=None):
    """
    Plots the MRI slices with optional segmentation overlay, either as a single slice or with a slider to navigate through slices.

    Parameters:
    mri_image (numpy array): The MRI image data (3D numpy array).
    slice_number (int): The specific slice to visualize. If None, an interactive slider is used to navigate through slices.
    direction (str): The direction of slicing. Options are 'axial', 'coronal', 'sagittal'.
    segmentation (numpy array): The segmentation data (3D numpy array, same shape as mri_image). Optional.
    alpha (float): The transparency level of the segmentation overlay (0=transparent, 1=opaque).
    dpi (int): The DPI setting for the plot. Higher values yield higher resolution.
    target_shape (tuple): The target shape for resizing the slices. Optional. If None, no resizing is applied.
    """

    # Determine the number of slices and the appropriate slicing function
    if direction == 'axial':
        max_slices = mri_image.shape[0]
        slice_func = lambda i: (mri_image[i, :, :], segmentation[i, :, :] if segmentation is not None else None)
    elif direction == 'coronal':
        max_slices = mri_image.shape[1]
        slice_func = lambda i: (mri_image[:, i, :], segmentation[:, i, :] if segmentation is not None else None)
    elif direction == 'sagittal':
        max_slices = mri_image.shape[2]
        slice_func = lambda i: (mri_image[:, :, i], segmentation[:, :, i] if segmentation is not None else None)
    else:
        raise ValueError("Direction must be one of 'axial', 'coronal', or 'sagittal'.")

    # Define a color map for segmentation
    colors = [(0, 0, 0, 0)]  # transparent color for label 0
    colors += [(1, 0, 0, alpha),  # red with adjustable transparency
               (0, 1, 0, alpha),  # green with adjustable transparency
               (0, 0, 1, alpha),  # blue with adjustable transparency
               (1, 1, 0, alpha),  # yellow with adjustable transparency
               (1, 0, 1, alpha),  # magenta with adjustable transparency
               (0, 1, 1, alpha),  # cyan with adjustable transparency
               (1, 0.5, 0, alpha),  # orange with adjustable transparency
               (0.5, 0, 1, alpha),  # purple with adjustable transparency
               (0, 1, 0.5, alpha),  # teal with adjustable transparency
               (1, 0.5, 0.5, alpha)]  # pink with adjustable transparency
    cmap = ListedColormap(colors)

    # Function to plot a specific slice
    def plot_slice(slice_index):
        mri_slice, seg_slice = slice_func(slice_index)
        
        # Resize if target_shape is provided
        if target_shape:
            mri_slice = resize_matrix(mri_slice, target_shape)
            if seg_slice is not None:
                seg_slice = resize_matrix(seg_slice, target_shape)

        plt.figure(figsize=(6, 6), dpi=dpi)
        plt.imshow(mri_slice, cmap='gray')
        if seg_slice is not None:
            plt.imshow(seg_slice, cmap=cmap, alpha=alpha)  # Overlay segmentation
        plt.axis('off')
        plt.show()

    # Check if a specific slice is to be visualized or if we are using an interactive slider
    if slice_number is not None:
        # Static visualization of a specific slice
        plot_slice(slice_number)
    else:
        # Interactive visualization with a slider
        slider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=max_slices // 2)

        # Define functions for button actions
        def next_slice(b):
            slider.value = min(slider.value + 1, max_slices - 1)

        def prev_slice(b):
            slider.value = max(slider.value - 1, 0)

        # Create buttons for finer control
        button_next = widgets.Button(description="Next")
        button_prev = widgets.Button(description="Previous")
        button_next.on_click(next_slice)
        button_prev.on_click(prev_slice)

        # Display the slider and buttons together
        controls = HBox([button_prev, button_next])
        interactive_plot = interactive(plot_slice, slice_index=slider)
        display(VBox([controls, interactive_plot]))