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

def side_by_side_view(*images: np.ndarray,
                      color_palette: str = 'gray',
                      dpi: int = 100,
                      titles: Optional[List[str]] = None):
    """
    Display multiple images side by side.

    Args:
        *images (numpy.ndarray): A variable number of images to display. Each image should be a 2D array
                                 (or 3D if needed, with shape (height, width, channels)).
        color_palette (str, optional): The color palette to use for displaying the images.
                                       Defaults to 'gray'. If 'gray', images are displayed in grayscale.
                                       Other color palettes can be specified.
        dpi (int, optional): Dots per inch for the plot.
        titles (Optional[List[str]], optional): A list of titles for the images.
                                                If not provided or if its length doesn't match the number of images,
                                                default titles will be generated.

    Note:
        - If an image is not square (height != width), it will be resized to a square shape
          using the provided 'resize_matrix' function before being displayed.
        - The function creates a subplot with 1 row and as many columns as images, displaying each image in its own subplot.
    """
    # Convert the tuple of images to a list for easier handling.
    images_list = list(images)
    n_images = len(images_list)

    if n_images == 0:
        raise ValueError("At least one image must be provided.")

    # Generate default titles if none provided or if the list length doesn't match the number of images.
    if titles is None or len(titles) != n_images:
        titles = [f"image {i+1}" for i in range(n_images)]

    # Create a figure with 1 row and n_images columns.
    fig, axs = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    fig.dpi = dpi

    # If there's only one image, make axs a list to simplify looping.
    if n_images == 1:
        axs = [axs]

    # Set the colormap.
    cmap = 'gray' if color_palette == 'gray' else color_palette

    for idx, (img, title) in enumerate(zip(images_list, titles)):
        # If the image is not square, resize it.
        if img.shape[0] != img.shape[1]:
            img = resize_matrix(img)

        axs[idx].imshow(img, cmap=cmap)
        axs[idx].axis('off')
        axs[idx].set_title(title)

    plt.tight_layout()
    plt.show()

    
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


def plot_3D_mri(mri_image, slice_number=None, direction='sagittal', segmentation=None,
                alpha=0.5, dpi=150, target_shape=None, m_cmap='gray', vmax=1000, vmin=0):
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
    if len(mri_image.shape) == 4:
        mri_image = mri_image[..., 0]
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
    colors += [
        (1, 0, 0, alpha),  # 1. red
        (0, 1, 0, alpha),  # 2. green
        (0, 0, 1, alpha),  # 3. blue
        (1, 1, 0, alpha),  # 4. yellow
        (1, 0, 1, alpha),  # 5. magenta
        (0, 1, 1, alpha),  # 6. cyan
        (1, 0.5, 0, alpha),  # 7. orange
        (0.5, 0, 1, alpha),  # 8. purple
        (0, 1, 0.5, alpha),  # 9. teal
        (1, 0.5, 0.5, alpha),  # 10. pink

        # additional colors up to label 19
        (0.6, 0.4, 0.2, alpha),  # 11. brown
        (0, 0, 0.5, alpha),  # 12. navy
        (0.5, 0.5, 0, alpha),  # 13. olive
        (0.5, 0.5, 0.5, alpha),  # 14. gray
        (0.5, 0.5, 1, alpha),  # 15. light blue
        (1, 0.84, 0, alpha),  # 16. gold
        (1, 0.6, 0.6, alpha),  # 17. salmon
        (0.93, 0.51, 0.93, alpha),  # 18. violet
        (0.75, 1, 0, alpha)  # 19. lime
    ]

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
        plt.imshow(mri_slice, cmap=m_cmap, vmin=vmin, vmax=vmax)
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