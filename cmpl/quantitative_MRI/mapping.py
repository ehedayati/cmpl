import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
torch.set_grad_enabled(True)

def t2_star_two_parametric_2D(TE_all, images, num_iterations=10000, initial_lr=0.01, lr_decay_factor=0.1, patience=100, initial_T2_star=20.0):
    """
    Computes the T2* and S0 maps from MRI images using an exponential decay __private_model.
    Also tracks and plots the loss during optimization, with learning rate adjustment.

    Parameters:
    - TE_all: A list or numpy array of echo times (TE) in milliseconds.
    - images: A numpy array of shape (x, y, TE) containing the MRI images.
    - num_iterations: Number of iterations for the optimizer (default: 10000).
    - initial_lr: Initial learning rate for the optimizer (default: 0.01).
    - lr_decay_factor: Factor by which the learning rate will be reduced (default: 0.1).
    - patience: Number of iterations to wait before reducing the learning rate (default: 100).
    - initial_T2_star: Initial guess for T2* for all voxels (default: 20.0).

    Returns:
    - T2_star_map: A numpy array containing the T2* values for each voxel.
    - S0_map: A numpy array containing the S0 values for each voxel.
    """
    torch.set_grad_enabled(True)
    # Convert echo times to a torch tensor and move to GPU
    TE = torch.tensor(TE_all, dtype=torch.float32).cuda()

    # Convert images to torch tensor and move to GPU
    images = torch.tensor(images, dtype=torch.float32).cuda()

    # Define the exponential decay function
    def exp_decay(TE, S0, T2_star):
        return S0[..., None] * torch.exp(-TE[None, None, :] / T2_star[..., None])

    # Prepare initial guesses for S0 and T2* for all voxels
    S0_init = images[..., 0]
    T2_star_init = torch.full(S0_init.shape, initial_T2_star, dtype=torch.float32).cuda()

    # Parameters to be optimized: S0 and T2* for all voxels
    params = torch.stack([S0_init, T2_star_init], dim=-1)
    params = params.view(-1, 2)
    params.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([params], lr=initial_lr)

    # Learning rate scheduler that reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience, verbose=True)

    # Loss function
    def loss_function(params, TE, signal):
        S0, T2_star = params[:, 0], params[:, 1]
        predicted_signal = exp_decay(TE, S0, T2_star)
        return torch.mean((signal - predicted_signal) ** 2)

    # Flatten the images to match the flattened params
    signal = images.view(-1, images.shape[-1])

    # List to store loss values for each iteration
    loss_values = []

    # Optimization loop
    for _ in tqdm(range(num_iterations)):  # Adjust the number of iterations as needed
        optimizer.zero_grad()
        loss = loss_function(params, TE, signal)
        loss.backward()
        optimizer.step()
        # Store the current loss value
        loss_values.append(loss.item())
        # Step the scheduler with the current loss
        scheduler.step(loss)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].view(images.shape[:-1]), params[:, 1].view(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Optimization with Learning Rate Adjustment')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(loss_values[-1])
    return T2_star_map, S0_map

def t2_star_three_parametric_2D(TE_all, images, num_iterations=10000, initial_lr=0.01, lr_decay_factor=0.1,
                                patience=100, initial_T2_star=20.0, initial_C=0.0):
    """
    Computes the T2*, S0, and C (noise) maps from MRI images using an exponential decay __private_model.
    Also tracks and plots the loss during optimization, with learning rate adjustment.

    Parameters:
    - TE_all: A list or numpy array of echo times (TE) in milliseconds.
    - images: A numpy array of shape (x, y, TE) containing the MRI images.
    - num_iterations: Number of iterations for the optimizer (default: 10000).
    - initial_lr: Initial learning rate for the optimizer (default: 0.01).
    - lr_decay_factor: Factor by which the learning rate will be reduced (default: 0.1).
    - patience: Number of iterations to wait before reducing the learning rate (default: 100).
    - initial_T2_star: Initial guess for T2* for all voxels (default: 20.0).
    - initial_C: Initial guess for the noise parameter C for all voxels (default: 0.0).

    Returns:
    - T2_star_map: A numpy array containing the T2* values for each voxel.
    - S0_map: A numpy array containing the S0 values for each voxel.
    - C_map: A numpy array containing the C (noise) values for each voxel.
    """
    torch.set_grad_enabled(True)
    # Convert echo times to a torch tensor and move to GPU
    TE = torch.tensor(TE_all, dtype=torch.float32).cuda()

    # Convert images to torch tensor and move to GPU
    images = torch.tensor(images, dtype=torch.float32).cuda()

    def exp_decay(TE, S0, T2_star, C_prime):
        # Reparameterize C as the square of C_prime to ensure non-negativity
        C = torch.abs(C_prime)
        return S0[..., None] * torch.exp(-TE[None, None, :] / (T2_star[..., None] + 1e-6)) + C[..., None]

    # Prepare initial guesses for S0, T2*, and C_prime for all voxels
    S0_init = images[..., 0]
    T2_star_init = torch.full(S0_init.shape, initial_T2_star, dtype=torch.float32).cuda()
    C_prime_init = torch.sqrt(torch.full(S0_init.shape, initial_C, dtype=torch.float32) + 1.0).cuda()

    # Parameters to be optimized: S0, T2*, and C_prime for all voxels
    params = torch.stack([S0_init, T2_star_init, C_prime_init], dim=-1)
    params = params.view(-1, 3)
    params.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([params], lr=initial_lr)

    # Learning rate scheduler that reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience,
                                                           verbose=True)

    def loss_function(params, TE, signal):
        S0, T2_star, C = params[:, 0], params[:, 1], params[:, 2]
        predicted_signal = exp_decay(TE, S0, T2_star, C)
        return torch.mean((signal - predicted_signal) ** 2)

        # Flatten the images to match the flattened params

    signal = images.view(-1, images.shape[-1])

    # List to store loss values for each iteration
    loss_values = []
    S0_values = []
    T2_star_values = []
    C_values = []

    # Optimization loop
    for _ in tqdm(range(num_iterations)):  # Adjust the number of iterations as needed
        optimizer.zero_grad()
        loss = loss_function(params, TE, signal)
        # loss = loss_function(params, TE, signal, optimizer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        # Store the current loss value
        loss_values.append(loss.item())

        S0, T2_star, C = params[:, 0].clone().detach().cpu().numpy(), params[:,
                                                                      1].clone().detach().cpu().numpy(), params[:,
                                                                                                         2].clone().detach().cpu().numpy()
        S0_values.append(S0.mean())
        T2_star_values.append(T2_star.mean())
        C_values.append(C.mean())
        # Step the scheduler with the current loss
        scheduler.step(loss)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map, C_map = params[:, 0].view(images.shape[:-1]), params[:, 1].view(images.shape[:-1]), params[:,
                                                                                                             2].view(
        images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()
    C_map = C_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Optimization with Learning Rate Adjustment')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure(figsize=(14, 7))

    plt.subplot(3, 1, 1)
    plt.plot(S0_values, label='S0')
    plt.xlabel('Iteration')
    plt.ylabel('Mean S0 Value')
    plt.title('Mean S0 Value During Training')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(T2_star_values, label='T2*')
    plt.xlabel('Iteration')
    plt.ylabel('Mean T2* Value')
    plt.title('Mean T2* Value During Training')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(C_values, label='C')
    plt.xlabel('Iteration')
    plt.ylabel('Mean C Value')
    plt.title('Mean C Value During Training')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    _print_last_non_nan(loss_values)

    return T2_star_map, S0_map, C_map, loss_values

def t2_star_two_parametric_3D(TE_all, images, num_iterations=10000, initial_lr=0.01,
                              lr_decay_factor=0.1, patience=100, initial_T2_star=20.0, plot_error=True):
    """
    Computes the T2* and S0 maps from MRI images using an exponential decay __private_model.
    Also tracks and plots the loss during optimization, with learning rate adjustment.

    Parameters:
    - TE_all: A list or numpy array of echo times (TE) in milliseconds.
    - images: A numpy array of shape (x, y, z, TE) containing the MRI images.
    - num_iterations: Number of iterations for the optimizer (default: 10000).
    - initial_lr: Initial learning rate for the optimizer (default: 0.01).
    - lr_decay_factor: Factor by which the learning rate will be reduced (default: 0.1).
    - patience: Number of iterations to wait before reducing the learning rate (default: 100).
    - initial_T2_star: Initial guess for T2* for all voxels (default: 20.0).

    Returns:
    - T2_star_map: A numpy array containing the T2* values for each voxel (x, y, z).
    - S0_map: A numpy array containing the S0 values for each voxel (x, y, z).
    """
    torch.set_grad_enabled(True)
    # Convert echo times to a torch tensor and move to GPU
    TE = torch.tensor(TE_all, dtype=torch.float32).cuda()

    # Convert images to torch tensor and move to GPU
    images = torch.tensor(images, dtype=torch.float32).cuda()

    # Define the exponential decay function
    def exp_decay(TE, S0, T2_star):
        return S0[..., None] * torch.exp(-TE[None, None, None, :] / T2_star[..., None])

    # Prepare initial guesses for S0 and T2* for all voxels
    S0_init = images[..., 0]
    T2_star_init = torch.full(S0_init.shape, initial_T2_star, dtype=torch.float32).cuda()

    # Parameters to be optimized: S0 and T2* for all voxels
    params = torch.stack([S0_init, T2_star_init], dim=-1)
    params = params.reshape(-1, 2)
    params.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([params], lr=initial_lr)

    # Learning rate scheduler that reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience, verbose=True)

    # Loss function
    def loss_function(params, TE, signal):
        S0, T2_star = params[:, 0], params[:, 1]
        predicted_signal = exp_decay(TE, S0, T2_star)
        return torch.mean((signal - predicted_signal) ** 2)

    # Flatten the images to match the flattened params
    signal = images.reshape(-1, images.shape[-1])

    # List to store loss values for each iteration
    loss_values = []

    # Optimization loop
    for _ in tqdm(range(num_iterations)):  # Adjust the number of iterations as needed
        optimizer.zero_grad()
        loss = loss_function(params, TE, signal)
        loss.backward()
        optimizer.step()
        # Store the current loss value
        loss_values.append(loss.item())
        # Step the scheduler with the current loss
        scheduler.step(loss)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].reshape(images.shape[:-1]), params[:, 1].reshape(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    if plot_error:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss During Optimization with Learning Rate Adjustment')
        plt.grid(True)
        plt.legend()
        plt.show()

    print(f"Final loss: {loss_values[-1]}")

    return T2_star_map, S0_map

def t2_star_three_parametric_3D(TE_all, images, num_iterations=10000, initial_lr=0.01, lr_decay_factor=0.1, patience=100, initial_T2_star=20.0):
    """
    Computes the T2* and S0 maps from MRI images using an exponential decay __private_model.
    Also tracks and plots the loss during optimization, with learning rate adjustment.

    Parameters:
    - TE_all: A list or numpy array of echo times (TE) in milliseconds.
    - images: A numpy array of shape (x, y, z, TE) containing the MRI images.
    - num_iterations: Number of iterations for the optimizer (default: 10000).
    - initial_lr: Initial learning rate for the optimizer (default: 0.01).
    - lr_decay_factor: Factor by which the learning rate will be reduced (default: 0.1).
    - patience: Number of iterations to wait before reducing the learning rate (default: 100).
    - initial_T2_star: Initial guess for T2* for all voxels (default: 20.0).

    Returns:
    - T2_star_map: A numpy array containing the T2* values for each voxel (x, y, z).
    - S0_map: A numpy array containing the S0 values for each voxel (x, y, z).
    """
    torch.set_grad_enabled(True)
    # Convert echo times to a torch tensor and move to GPU
    TE = torch.tensor(TE_all, dtype=torch.float32).cuda()

    # Convert images to torch tensor and move to GPU
    images = torch.tensor(images, dtype=torch.float32).cuda()

    # Define the exponential decay function
    def exp_decay(TE, S0, T2_star):
        return S0[..., None] * torch.exp(-TE[None, None, None, :] / T2_star[..., None])

    # Prepare initial guesses for S0 and T2* for all voxels
    S0_init = images[..., 0]
    T2_star_init = torch.full(S0_init.shape, initial_T2_star, dtype=torch.float32).cuda()

    # Parameters to be optimized: S0 and T2* for all voxels
    params = torch.stack([S0_init, T2_star_init], dim=-1)
    params = params.reshape(-1, 2)
    params.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([params], lr=initial_lr)

    # Learning rate scheduler that reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience, verbose=True)

    # Loss function
    def loss_function(params, TE, signal):
        S0, T2_star = params[:, 0], params[:, 1]
        predicted_signal = exp_decay(TE, S0, T2_star)
        return torch.mean((signal - predicted_signal) ** 2)

    # Flatten the images to match the flattened params
    signal = images.reshape(-1, images.shape[-1])

    # List to store loss values for each iteration
    loss_values = []

    # Optimization loop
    for _ in tqdm(range(num_iterations)):  # Adjust the number of iterations as needed
        optimizer.zero_grad()
        loss = loss_function(params, TE, signal)
        loss.backward()
        optimizer.step()
        # Store the current loss value
        loss_values.append(loss.item())
        # Step the scheduler with the current loss
        scheduler.step(loss)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].reshape(images.shape[:-1]), params[:, 1].reshape(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Optimization with Learning Rate Adjustment')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Final loss: {loss_values[-1]}")

    return T2_star_map, S0_map

def reconstruct_images(T2_star_map, S0_map, TE_all):
    """
    Reconstructs the images using the T2_star_map and S0_map.

    Parameters:
    - T2_star_map: A numpy array containing the T2* values for each voxel.
    - S0_map: A numpy array containing the S0 values for each voxel.
    - TE_all: A list or numpy array of echo times (TE) in milliseconds.

    Returns:
    - reconstructed_images: A numpy array containing the reconstructed images.
    """

    # Convert to torch tensors and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T2_star_map = torch.tensor(T2_star_map, dtype=torch.float32).to(device)
    S0_map = torch.tensor(S0_map, dtype=torch.float32).to(device)
    TE_all = torch.tensor(TE_all, dtype=torch.float32).to(device)

    # Reconstruct the images using the exponential decay __private_model
    try:
        reconstructed_images = S0_map[..., None] * torch.exp(-TE_all[None, None, :] / T2_star_map[..., None])
    except:
        return S0_map[..., None] * torch.exp(-TE_all[None, None, None, :] / T2_star_map[..., None])

    # Move back to CPU and convert to numpy array
    reconstructed_images = reconstructed_images.cpu().numpy()

    return reconstructed_images

def calculate_rmse_percentage_s0(original_images, reconstructed_images, S0_map):
    """
    Calculates the RMSE in percentage of S0 between the original and reconstructed images.

    Parameters:
    - original_images: A numpy array containing the original images.
    - reconstructed_images: A numpy array containing the reconstructed images.
    - S0_map: A numpy array containing the S0 values for each voxel.

    Returns:
    - rmse_percentage: RMSE as a percentage of S0.
    """

    # Calculate the squared error
    squared_error = (original_images - reconstructed_images) ** 2

    # Calculate the mean squared error (MSE) across the TE dimension
    mse = np.mean(squared_error, axis=-1)

    # Calculate the RMSE
    rmse = np.sqrt(mse)

    # Calculate RMSE as a percentage of S0
    rmse_percentage = 100 * (rmse / S0_map)

    return rmse_percentage

def _print_last_non_nan(lst):
    # Iterate over the list in reverse order
    for i in range(len(lst) - 1, -1, -1):
        # Check if the item is not NaN
        if not math.isnan(lst[i]):
            # Print the last non-NaN item and its index
            print(f"Last non-NaN item: {lst[i]}, at index: {i}")
            return lst[i], i  # Return the item and its index
    print("No non-NaN items found.")
    return None, None