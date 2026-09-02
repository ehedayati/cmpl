import numpy as np
import torch
import math
from tqdm import tqdm

def _get_plt():
    """Import matplotlib only when plotting is actually requested."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. "
            "Install it with: pip install 'mriforge[viz]'"
        ) from exc

    return plt


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
        loss_value = loss.detach().item()
        # Store the current loss value
        loss_values.append(loss_value)
        # Step the scheduler with the current loss
        scheduler.step(loss_value)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].view(images.shape[:-1]), params[:, 1].view(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt = _get_plt()

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
        loss_value = loss.detach().item()
        # Store the current loss value
        loss_values.append(loss_value)
        # Step the scheduler with the current loss
        scheduler.step(loss_value)

        S0, T2_star, C = params[:, 0].clone().detach().cpu().numpy(), params[:,
                                                                      1].clone().detach().cpu().numpy(), params[:,
                                                                                                         2].clone().detach().cpu().numpy()
        S0_values.append(S0.mean())
        T2_star_values.append(T2_star.mean())
        C_values.append(C.mean())

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map, C_map = params[:, 0].view(images.shape[:-1]), params[:, 1].view(images.shape[:-1]), params[:,
                                                                                                             2].view(
        images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()
    C_map = C_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt = _get_plt()

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
                              lr_decay_factor=0.1, patience=100, initial_T2_star=20.0, plot_error=True, return_RMSE=False,loss_fn=None, device=None):
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
    - plot_error: Whether to plot the loss after optimization is complete, used for num_iterations evaluation (default: True).
    -
    Returns:
    - T2_star_map: A numpy array containing the T2* values for each voxel (x, y, z).
    - S0_map: A numpy array containing the S0 values for each voxel (x, y, z).
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Default loss if user doesn't provide one
    if loss_fn is None:
        def loss_fn(signal, pred):
            return torch.mean((signal - pred) ** 2)
    torch.set_grad_enabled(True)
    # Convert echo times to a torch tensor and move to GPU
    TE = torch.tensor(TE_all, dtype=torch.float32).to(device)

    # Convert images to torch tensor and move to GPU
    images = torch.tensor(images, dtype=torch.float32).to(device)

    # Define the exponential decay function
    def exp_decay(TE, S0, T2_star):
        # Robustness: enforce strictly positive, non-zero S0 and T2* in the forward model
        eps = torch.finfo(S0.dtype).eps
        S0_safe = torch.clamp(S0, min=eps)
        T2_star_safe = torch.clamp(T2_star, min=eps)

        pred = S0_safe[..., None] * torch.exp(-TE[None, None, None, :] / T2_star_safe[..., None])

        # Robustness: prevent NaN/Inf from propagating
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred

    # Prepare initial guesses for S0 and T2* for all voxels
    S0_init = images[..., 0]
    T2_star_init = torch.full(S0_init.shape, initial_T2_star, dtype=torch.float32).to(device)

    # Parameters to be optimized: S0 and T2* for all voxels
    params = torch.stack([S0_init, T2_star_init], dim=-1)
    params = params.reshape(-1, 2)
    params.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([params], lr=initial_lr)

    # Learning rate scheduler that reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience)

    # Loss function

    def loss_function(params, TE, signal):
        S0, T2_star = params[:, 0], params[:, 1]
        predicted_signal = exp_decay(TE, S0, T2_star)

        # User-supplied fidelity term
        loss = loss_fn(signal, predicted_signal)

        # Robustness: avoid NaN/Inf loss destabilizing optimizer/scheduler
        loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)

        # (Optional but recommended) enforce scalar
        if loss.ndim != 0:
            loss = loss.mean()

        return loss

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

        # Robustness: enforce strictly positive, non-zero parameters after the update
        # (keeps the same logic/flow; prevents T2* or S0 from becoming 0/negative)
        with torch.no_grad():
            eps = torch.finfo(params.dtype).eps
            params[:, 0].clamp_(min=eps)  # S0
            params[:, 1].clamp_(min=eps)  # T2*

        loss_value = loss.detach().item()
        # Store the current loss value
        loss_values.append(loss_value)
        # Step the scheduler with the current loss
        scheduler.step(loss_value)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].reshape(images.shape[:-1]), params[:, 1].reshape(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach()#.cpu().numpy()
    S0_map = S0_map.detach()#.cpu().numpy()

    # Plot the loss values over iterations
    if plot_error:
        plt = _get_plt()

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss During Optimization with Learning Rate Adjustment')
        plt.grid(True)
        plt.legend()
        plt.show()

    print(f"Final loss: {loss_values[-1]}")
    if return_RMSE:
        recon_im = reconstruct_images(T2_star_map, S0_map, TE_all)

        rmse_pct, rse_pct = calculate_rmse_percentage_s0(
            images,
            recon_im,
            S0_map,
            return_numpy=False,
        )

        return {
            "T2_star_map": T2_star_map,
            "S0_map": S0_map,
            "RMSE_percentage": rmse_pct,
            "RSE_percentage": rse_pct,
        }

    else:
        return {
            "T2_star_map": T2_star_map,
            "S0_map": S0_map,
        }

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
        loss_value = loss.detach().item()
        # Store the current loss value
        loss_values.append(loss_value)
        # Step the scheduler with the current loss
        scheduler.step(loss_value)

    # Reshape the parameters back to the original image shape
    S0_map, T2_star_map = params[:, 0].reshape(images.shape[:-1]), params[:, 1].reshape(images.shape[:-1])

    # Convert the results back to CPU and numpy arrays for returning
    T2_star_map = T2_star_map.detach().cpu().numpy()
    S0_map = S0_map.detach().cpu().numpy()

    # Plot the loss values over iterations
    plt = _get_plt()
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

def reconstruct_images(
    T2_star_map,
    S0_map,
    TE_all,
    device=None,
    return_numpy=False,
    dtype=torch.float32,
):
    """
    Reconstruct magnitude images from T2* and S0 maps.

    Accepts:
        numpy arrays or torch tensors

    Returns:
        torch tensor (default) or numpy array if return_numpy=True

    Output shape:
        (..., TE)
    """

    # ---------- device selection ----------
    if device is None:
        if torch.is_tensor(T2_star_map):
            device = T2_star_map.device
        elif torch.is_tensor(S0_map):
            device = S0_map.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device)

    # ---------- safe conversion helper ----------
    def to_torch(x):

        if torch.is_tensor(x):

            if x.device != device:
                x = x.to(device)

            if x.dtype != dtype:
                x = x.to(dtype)

            return x

        return torch.as_tensor(x, dtype=dtype, device=device)


    T2_star_map = to_torch(T2_star_map)
    S0_map      = to_torch(S0_map)
    TE_all      = to_torch(TE_all)


    # ---------- numeric safety ----------
    eps = torch.finfo(dtype).eps

    T2_star_safe = torch.clamp(T2_star_map, min=eps)
    S0_safe      = torch.clamp(S0_map,      min=eps)


    # ---------- correct broadcasting ----------
    # Works for ANY spatial dimension count

    TE = TE_all.view(*([1] * T2_star_safe.ndim), -1)

    reconstructed = S0_safe.unsqueeze(-1) * torch.exp(-TE / T2_star_safe.unsqueeze(-1))


    # ---------- cleanup ----------
    reconstructed = torch.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


    if return_numpy:

        return reconstructed.detach().cpu().numpy()

    return reconstructed

def calculate_rmse_percentage_s0(
    original_images,
    reconstructed_images,
    S0_map,
    *,
    device=None,
    return_numpy: bool = False,
    dtype=torch.float32,
):
    """
    RMSE (across TE) normalized by S0, returned as percent of S0.

    Accepts:
      - NumPy arrays or torch tensors for inputs
      - original_images/reconstructed_images shape: (..., TE)
      - S0_map shape: (...) matching spatial dims (no TE)

    Returns:
      rmse_pct: (...), percent of S0
      rse_pct: (..., TE), per-echo sqrt squared error percent of S0

    Notes:
      - Keeps ops in torch to avoid CPU/GPU ping-pong.
      - Device selection:
          * If any input is a torch tensor, uses its device (prefers original_images, then reconstructed_images, then S0_map)
          * Else uses `device` if provided, otherwise auto-selects cuda if available.
    """

    # -------- helpers --------
    def _pick_device():
        for x in (original_images, reconstructed_images, S0_map):
            if torch.is_tensor(x):
                return x.device
        if device is not None:
            return torch.device(device) if not isinstance(device, torch.device) else device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_torch(x, dev):
        if torch.is_tensor(x):
            # Move only if needed; avoid unnecessary copies
            if x.device != dev:
                x = x.to(dev)
            if x.dtype != dtype:
                x = x.to(dtype=dtype)
            return x
        # NumPy / list -> torch (one copy onto device)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    dev = _pick_device()

    # -------- convert inputs (minimal transfers) --------
    orig = _to_torch(original_images, dev)
    recon = _to_torch(reconstructed_images, dev)
    s0   = _to_torch(S0_map, dev)

    # -------- sanitize numeric pathologies --------
    # nan_to_num exists in torch; keep on-device
    orig = torch.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
    # s0: keep as-is except for denom safety
    # (If you want to zero-out negatives instead, do: s0 = torch.clamp(s0, min=0.0))

    # -------- compute errors --------
    # squared_error: (..., TE)
    squared_error = (orig - recon) ** 2

    # mse across TE -> (...), rmse -> (...)
    mse = squared_error.mean(dim=-1)
    rmse = torch.sqrt(mse)

    # per-echo root squared error -> (..., TE)
    rse = torch.sqrt(squared_error)

    # -------- denom safety & broadcasting --------
    eps = torch.finfo(dtype).eps
    s0_safe = torch.clamp(s0, min=eps)          # (...) no TE
    rmse_pct = 100.0 * (rmse / s0_safe)         # (...) broadcasts fine
    rse_pct  = 100.0 * (rse / s0_safe.unsqueeze(-1))  # (..., TE)

    # final cleanup (should be mostly unnecessary, but cheap)
    rmse_pct = torch.nan_to_num(rmse_pct, nan=0.0, posinf=0.0, neginf=0.0)
    rse_pct  = torch.nan_to_num(rse_pct,  nan=0.0, posinf=0.0, neginf=0.0)

    if return_numpy:
        # One-time transfer at the end (still minimal)
        return rmse_pct.detach().cpu().numpy(), rse_pct.detach().cpu().numpy()

    return rmse_pct, rse_pct

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


def t2_star_two_parametric_3D_voxelmask(
    TE_all,
    images,
    num_iterations=10000,
    initial_lr=0.01,
    lr_decay_factor=0.1,
    patience=100,
    initial_T2_star=20.0,
    plot_error=True,
    return_RMSE=False,
    loss_fn=None,
    device=None,
    # --- new (Option 1: per-voxel suffix mask based on relative-to-Sref threshold) ---
    alpha=0.05,          # keep echoes while S >= alpha * Sref
    sref_num_echoes=1,   # Sref = mean of first sref_num_echoes echoes (1 => first echo)
    min_echoes=3,        # force at least this many echoes to be used (if available)
    return_mask=False,   # optionally return the boolean mask used in fitting
):
    """
    T2* / S0 fitting with per-voxel echo masking (Option 1):
      - For each voxel, compute Sref from early echoes.
      - Keep echoes while measured signal >= alpha * Sref.
      - Apply a suffix rule: once an echo is excluded, all later echoes are excluded.
      - Enforce min_echoes kept (if NTE >= min_echoes).

    Notes:
      - Mask is computed ONCE from the measured magnitude data and kept fixed during optimization.
      - Masking is applied by setting pred=signal on excluded echoes (zero residual there),
        which works for residual-based losses.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default loss if user doesn't provide one
    if loss_fn is None:
        def loss_fn(signal, pred):
            return torch.mean((signal - pred) ** 2)

    torch.set_grad_enabled(True)

    TE = torch.tensor(TE_all, dtype=torch.float32, device=device)
    images_t = torch.tensor(images, dtype=torch.float32, device=device)

    # Forward model
    def exp_decay(TE, S0, T2_star):
        eps = torch.finfo(S0.dtype).eps
        S0_safe = torch.clamp(S0, min=eps)
        T2_star_safe = torch.clamp(T2_star, min=eps)
        pred = S0_safe[..., None] * torch.exp(-TE[None, None, None, :] / T2_star_safe[..., None])
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred

    # Init params
    S0_init = images_t[..., 0]
    T2_star_init = torch.full(S0_init.shape, float(initial_T2_star), dtype=torch.float32, device=device)

    params = torch.stack([S0_init, T2_star_init], dim=-1).reshape(-1, 2)
    params.requires_grad = True

    optimizer = torch.optim.Adam([params], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_decay_factor, patience=patience
    )

    # Flatten signal (Nvox, NTE)
    signal = images_t.reshape(-1, images_t.shape[-1])
    Nvox, NTE = signal.shape

    # ----------------------------
    # Build per-voxel suffix mask
    # ----------------------------
    sref_num_echoes = int(max(1, min(sref_num_echoes, NTE)))
    Sref = signal[:, :sref_num_echoes].mean(dim=1)  # (Nvox,)

    # Robustness: avoid zero reference
    eps_ref = torch.finfo(signal.dtype).eps
    Sref_safe = torch.clamp(Sref, min=eps_ref)

    thr = alpha * Sref_safe  # (Nvox,)
    mask_raw = signal >= thr[:, None]  # (Nvox, NTE)

    # Suffix rule: once False, all later become False
    mask_suffix = mask_raw.to(torch.int32).cumprod(dim=1).bool()

    # Enforce minimum number of echoes, if possible
    if NTE > 0:
        k = int(max(0, min(min_echoes, NTE)))
        if k > 0:
            mask_suffix[:, :k] = True

    # ----------------------------
    # Loss function (with masking)
    # ----------------------------
    def loss_function(params, TE, signal, mask):
        S0, T2_star = params[:, 0], params[:, 1]
        predicted = exp_decay(TE, S0, T2_star).squeeze()  # -> (Nvox, NTE) in this flattened setup

        # Apply mask by forcing residual=0 on excluded echoes:
        # pred_eff = pred where mask True, else pred_eff = signal (so signal - pred_eff = 0)
        pred_eff = torch.where(mask, predicted, signal)

        loss = loss_fn(signal, pred_eff)
        loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
        if loss.ndim != 0:
            loss = loss.mean()
        return loss

    loss_values = []
    for _ in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss = loss_function(params, TE, signal, mask_suffix)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            eps = torch.finfo(params.dtype).eps
            params[:, 0].clamp_(min=eps)  # S0
            params[:, 1].clamp_(min=eps)  # T2*

        loss_value = loss.detach().item()
        # Store the current loss value
        loss_values.append(loss_value)
        # Step the scheduler with the current loss
        scheduler.step(loss_value)

    # Reshape back
    S0_map_t = params[:, 0].reshape(images_t.shape[:-1])
    T2_star_map_t = params[:, 1].reshape(images_t.shape[:-1])

    T2_star_map = T2_star_map_t.detach().cpu().numpy()
    S0_map = S0_map_t.detach().cpu().numpy()

    if plot_error:
        plt = _get_plt()
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss During Optimization with Learning Rate Adjustment")
        plt.grid(True)
        plt.legend()
        plt.show()

    print(f"Final loss: {loss_values[-1]}")

    # Optional outputs
    out = (T2_star_map, S0_map)

    if return_RMSE:
        recon_im = reconstruct_images(T2_star_map, S0_map, TE_all)
        RMSE = calculate_rmse_percentage_s0(images_t.detach().cpu().numpy(), recon_im, S0_map)
        out = out + (RMSE,)

    if return_mask:
        # Return mask in original (x,y,z,TE) shape for inspection
        mask_4d = mask_suffix.reshape(images_t.shape)
        out = out + (mask_4d.detach().cpu().numpy(),)

    return out
