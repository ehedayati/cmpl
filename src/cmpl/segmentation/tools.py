# File created by: Eisa Hedayati
# Date: 9/19/2024
# Description: This file is developed at CMRR

import numpy as np
import cmpl

def project_3d_matrix(matrix, value, axis):
    """
    Projects the specified value through the planes of a 3D matrix along a specified axis.

    Parameters:
    matrix (3D array): The 3D matrix.
    value (int): The value to project between planes.
    axis (int): The axis along which the projection occurs (0, 1, or 2).

    Returns:
    np.ndarray: The updated 3D matrix after projection.
    """
    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)

    # Get the shape of the matrix
    shape = matrix.shape

    # Determine the number of planes along the chosen axis
    num_planes = shape[axis]

    # Top to bottom propagation
    for i in range(num_planes - 1):
        if axis == 0:
            matrix[i + 1] = project_matrix(matrix[i], matrix[i + 1], value)
        elif axis == 1:
            matrix[:, i + 1] = project_matrix(matrix[:, i], matrix[:, i + 1], value)
        elif axis == 2:
            matrix[:, :, i + 1] = project_matrix(matrix[:, :, i], matrix[:, :, i + 1], value)

    # Bottom to top propagation
    for i in range(num_planes - 1, 0, -1):
        if axis == 0:
            matrix[i - 1] = project_matrix(matrix[i], matrix[i - 1], value)
        elif axis == 1:
            matrix[:, i - 1] = project_matrix(matrix[:, i], matrix[:, i - 1], value)
        elif axis == 2:
            matrix[:, :, i - 1] = project_matrix(matrix[:, :, i], matrix[:, :, i - 1], value)
    return matrix


def project_matrix(matrix1, matrix2, value):
    """
    Projects all occurrences of 'value' from matrix1 to matrix2.

    Parameters:
    matrix1 (2D array): The matrix with the values to be projected.
    matrix2 (2D array): The matrix where values from matrix1 will be projected.
    value (int): The value to be projected from matrix1 to matrix2.

    Returns:
    np.ndarray: The updated matrix2 after projection.
    """
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    mask = (matrix1 == value)
    matrix2[mask] = matrix1[mask]
    return matrix2


def extract_extrusion(extrusion_path, seg_path, projection_value=11):
    def apply_transformation(matrix):
        # Create a copy of the original matrix
        filled_matrix = matrix.copy()
        # Iterate over the 2nd and 3rd dimensions
        for i in range(matrix.shape[2]):  # Loop over the 3rd dimension (index i)
            for j in range(matrix.shape[1]):  # Loop over the 2nd dimension (index j)
                one_dimensional_slice = matrix[:, j, i]
                if np.any(one_dimensional_slice != 0):  # Check if there's any non-zero element
                    # Get the first and last non-zero indices
                    first_non_zero = np.argmax(one_dimensional_slice != 0)
                    last_non_zero = len(one_dimensional_slice) - np.argmax(one_dimensional_slice[::-1] != 0) - 1
                    # Set values between first and last to 11
                    filled_matrix[first_non_zero:last_non_zero + 1, j, i] = 11

        return filled_matrix

    def zero_elements_where_b_nonzero(b, a):
        # Create a mask where elements in b are non-zero
        mask = b != 0
        # Set corresponding elements in a to 0 where mask is True
        a[mask] = 0
        return a

    # Step 1: Read the NIfTI files
    _, seg_f_no_rot = cmpl.utilities.io.nifti_read(seg_path, re_orient=False)
    _, extrusion_cut_no_rot = cmpl.utilities.io.nifti_read(extrusion_path, re_orient=False)

    # Step 2: Apply the projection transformation
    extrusion_projected_no_rot = project_3d_matrix(extrusion_cut_no_rot, projection_value, 1)

    # Step 3: Apply the transformation to set values between first and last non-zero to 11 (or any number)
    transformed_matrix = apply_transformation(extrusion_projected_no_rot)

    # Step 4: Set elements of transformed_matrix to 0 where seg_f_no_rot has non-zero elements
    final_matrix = zero_elements_where_b_nonzero(transformed_matrix, seg_f_no_rot)
    return final_matrix