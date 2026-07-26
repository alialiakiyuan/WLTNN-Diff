import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time


def log_prox(sigma_i, lambda_, eps=1e-6):
    """
    Non-convex log-based singular value shrinkage operator.
    Solves: prox_{lambda * log(|x|+1)}(sigma_i)
    """
    sigma = torch.zeros_like(sigma_i)

    discriminant = (sigma_i + eps) ** 2 - 4 * lambda_
    mask = discriminant > 0
    if torch.any(mask):
        sigma[mask] = (sigma_i[mask] - eps + torch.sqrt(discriminant[mask])) / 2

    return sigma


def t_svd_log_prox(Y, weights, lambd, eps=1e-6, k=2, use_weights=True):
    """
    T-SVD proximal operator with non-convex log-weighted nuclear norm.

    Supports:
        - 4D tensors [B, C, H, W] or [B, H, W, C]
        - 3D tensors [H, W, C]

    Args:
        Y: Input tensor
        weights: Weight vector [C, min(H,W)]
        lambd: Regularization parameter
        eps: Small constant for numerical stability
        k: Mode of tensor unfolding (0, 1, or 2)
        use_weights: Whether to apply adaptive weighting

    Returns:
        X: Output tensor (same shape as input)
        new_weights: Updated weights [C, min(H,W)]
    """
    input_dim = Y.dim()
    if input_dim == 4:
        batch_size, channel, height, width = Y.shape
        Y = Y.permute(0, 2, 3, 1).contiguous().squeeze(0)

    if k == 0:
        Y = Y.permute(2, 1, 0)
    if k == 1:
        Y = Y.permute(0, 2, 1)

    H, W, C = Y.shape

    # FFT along the channel dimension
    Y_bar = fft.fft(Y.to(torch.float32), dim=2, norm='ortho')

    min_dim = min(H, W)
    if weights is None:
        weights = torch.ones(C, min_dim, device=Y.device)

    new_weights = torch.ones(C, min_dim, device=Y.device)
    X_bar = torch.zeros_like(Y_bar)

    # Process first slice (DC component)
    slice_Y = Y_bar[:, :, 0].to(torch.complex64)
    U, S_vec, Vh = torch.linalg.svd(slice_Y, full_matrices=False)
    w = weights[0] if weights is not None else torch.ones(min_dim, device=Y.device)
    S_vec = S_vec.real
    S_prox = log_prox(S_vec, lambda_=lambd * w, eps=eps)
    U_scaled = U * S_prox.unsqueeze(0)
    slice_X = U_scaled @ Vh
    X_bar[:, :, 0] = slice_X

    if use_weights:
        new_weights[0] = 1 / (S_prox.detach()**0.5 + eps)
    else:
        new_weights = torch.ones(C, min_dim, device=Y.device)

    # Process conjugate symmetric pairs
    halfC = int(round(C / 2))
    for c in range(1, halfC):
        slice_Y = Y_bar[:, :, c].to(torch.complex64)

        U, S_vec, Vh = torch.linalg.svd(slice_Y, full_matrices=False)
        w = weights[c] if weights is not None else torch.ones(min_dim, device=Y.device)
        S_vec = S_vec.real
        S_prox = log_prox(S_vec, lambda_=lambd * w, eps=eps)
        U_scaled = U * S_prox.unsqueeze(0)
        slice_X = U_scaled @ Vh
        X_bar[:, :, c] = slice_X
        X_bar[:, :, C - c] = torch.conj(slice_X)

        if use_weights:
            new_weights[c] = 1 / (S_prox.detach() + eps)
            new_weights[C - c] = 1 / (S_prox.detach() + eps)

    # Handle Nyquist frequency for even C
    if C % 2 == 0:
        c = halfC
        slice_Y = Y_bar[:, :, c].to(torch.complex64)
        U, S_vec, Vh = torch.linalg.svd(slice_Y, full_matrices=False)
        w = weights[c] if weights is not None else torch.ones(min_dim, device=Y.device)
        S_vec = S_vec.real
        S_prox = log_prox(S_vec, lambda_=lambd * w, eps=eps)
        U_scaled = U * S_prox.unsqueeze(0)
        slice_X = U_scaled @ Vh
        X_bar[:, :, c] = slice_X

        if use_weights:
            new_weights[c] = 1 / (S_prox.detach() + eps)

    # Inverse FFT
    X = fft.ifft(X_bar, dim=2, norm='ortho').real

    # Restore original dimension order
    if k == 0:
        X = X.permute(2, 1, 0)
    if k == 1:
        X = X.permute(0, 2, 1)

    # Normalize weights
    row_sums = new_weights.sum(dim=1, keepdim=True)
    new_weights = (new_weights / row_sums) * new_weights.size(1)

    if input_dim == 4:
        X = X.unsqueeze(0).permute(0, 3, 1, 2)

    return X, new_weights


def prox_l1(Y, lambda_):
    """
    L1 norm proximal operator (soft thresholding).
    """
    return torch.maximum(torch.tensor(0, dtype=Y.dtype, device=Y.device),
                         Y - lambda_) + \
        torch.minimum(torch.tensor(0, dtype=Y.dtype, device=Y.device),
                      Y + lambda_)


def iterative_t_svd_denoising(noisy_img, clean_img=None, num_iters=100,
                              lambda_1=1, lambda_2=1, rho=1.2,
                              mu=1e-4, max_mu=1e10, ep=1,
                              alpha0=1/3, alpha1=1/3, alpha2=1/3,
                              beta=1e-4, max_beta=1e10,
                              use_weights=False):
    """
    Iterative T-SVD denoising with low-rank, sparse, and Gaussian priors.

    Solves: min_{X,S,N} alpha0*||X||_w + alpha1*||X||_w + alpha2*||X||_w
             + lambda_1||N||_1 + lambda_2||S||_1
             s.t. Y = X + S + N

    Args:
        noisy_img: Noisy input [H, W, C] in [0,1]
        clean_img: Ground truth for PSNR evaluation
        num_iters: Number of ADMM iterations
        lambda_1: Sparsity penalty for N (dense noise)
        lambda_2: Sparsity penalty for S (sparse noise)
        rho: Step size update factor
        mu, max_mu: Penalty parameters for low-rank terms
        ep: Epsilon for log proximal
        alpha0, alpha1, alpha2: Weights for three tensor unfoldings
        beta, max_beta: Penalty parameters for sparse terms
        use_weights: Enable adaptive weighting

    Returns:
        denoised: Clean image [H, W, C]
        psnrs: List of PSNR values per iteration
        S: Sparse component
        N: Dense noise component
    """
    noisy_img_normalized = torch.clamp(noisy_img, 0, 1)

    Y = noisy_img_normalized.clone()
    H, W, C = Y.shape
    min_dim = min(H, W)

    # ADMM variables
    S = torch.zeros_like(Y)
    N = torch.zeros_like(Y)
    X = torch.zeros_like(Y)

    M0 = torch.zeros_like(Y)
    M1 = torch.zeros_like(Y)
    M2 = torch.zeros_like(Y)
    P = torch.zeros_like(Y)

    mu0 = mu
    mu1 = mu
    mu2 = mu

    psnrs = []

    print(f"\nStarting T-SVD denoising with {num_iters} iterations...")

    for i in range(num_iters):
        start_time = time.time()

        if i == 0:
            weights0 = None
            weights1 = None
            weights2 = None

        # Low-rank proximal updates via T-SVD
        Z0, weights0 = t_svd_log_prox(
            Y=(X + M0 / mu0),
            weights=weights0,
            lambd=alpha0 / mu0,
            eps=ep,
            k=0,
            use_weights=use_weights
        )

        Z1, weights1 = t_svd_log_prox(
            Y=(X + M1 / mu1),
            weights=weights1,
            lambd=alpha1 / mu1,
            eps=ep,
            k=1,
            use_weights=use_weights
        )

        Z2, weights2 = t_svd_log_prox(
            Y=X + M2 / mu2,
            weights=weights2,
            lambd=alpha2 / mu2,
            eps=ep,
            k=2,
            use_weights=use_weights
        )

        # X update (least squares)
        X = (mu0 * Z0 + mu1 * Z1 + mu2 * Z2) - (M0 + M1 + M2) + beta * (Y - N - S + P / beta)
        X = X / (mu0 + mu1 + mu2 + beta)

        # N update (dense noise via L1)
        N = beta * (Y - X - S + P / beta) / (2 * lambda_1 + beta)

        # S update (sparse noise via L1)
        S = prox_l1(Y - X - N + P / beta, lambda_2 / beta)

        # Dual variable updates
        M0 = M0 + mu0 * (X - Z0)
        M1 = M1 + mu1 * (X - Z1)
        M2 = M2 + mu2 * (X - Z2)
        P = P + beta * (Y - X - N - S)

        # Update penalty parameters
        mu0 = min(rho * mu0, max_mu)
        mu1 = min(rho * mu1, max_mu)
        mu2 = min(rho * mu2, max_mu)
        beta = min(rho * beta, max_beta)

        if clean_img is not None:
            L_temp = torch.clamp(X, 0, 1)
            psnr = calculate_psnr(clean_img, L_temp.detach())
            psnrs.append(psnr)

        iter_time = time.time() - start_time
        print(f"Iteration {i + 1}/{num_iters} completed - Time: {iter_time:.2f}s", end='')
        if clean_img is not None:
            print(f" - PSNR: {psnr:.2f} dB")
        else:
            print()

    denoised = torch.clamp(X, 0, 1)
    S = torch.clamp(S, 0, 1)
    N = torch.clamp(N, 0, 1)

    return denoised, psnrs, S, N


def load_image(path, max_size=10e6, normalize=True):
    """Load image and resize if necessary."""
    img = Image.open(path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"Resized image to {new_w}x{new_h} (original: {w}x{h})")

    tensor = transforms.ToTensor()(img)
    tensor = tensor.permute(1, 2, 0)

    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    return tensor


def add_gaussian_noise(img, sigma=0.1):
    """Add additive Gaussian noise."""
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0, 1), noise


def calculate_psnr(img1, img2):
    """Compute PSNR between two images."""
    img1 = img1.cpu()
    img2 = img2.cpu()

    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def denoising_experiment(image_path, noise_level=0.2, num_iters=15,
                         lambda_1=1, lambda_2=1, eps=1e-5,
                         alpha0=1/2.001, alpha1=1/2.001, alpha2=0.001/2.001,
                         use_weights=False):
    """
    Run denoising experiment and visualize results.
    """
    clean_img = load_image(image_path)
    noisy_img, noise = add_gaussian_noise(clean_img, sigma=noise_level)

    print(f"\nImage shape: {clean_img.shape}")
    print(f"Added Gaussian noise with sigma={noise_level:.3f}")

    denoised, psnrs, S, N = iterative_t_svd_denoising(
        noisy_img=noisy_img,
        clean_img=clean_img,
        num_iters=num_iters,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        ep=eps,
        alpha0=alpha0,
        alpha1=alpha1,
        alpha2=alpha2,
        use_weights=use_weights
    )

    orig_psnr = calculate_psnr(clean_img, noisy_img)
    final_psnr = calculate_psnr(clean_img, denoised)

    print(f"\nSummary:")
    print(f"Original PSNR (noisy): {orig_psnr:.2f} dB")
    print(f"Final PSNR (denoised): {final_psnr:.2f} dB")
    print(f"PSNR improvement: {final_psnr - orig_psnr:.2f} dB")

    # Visualization
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(noisy_img.cpu().numpy())
    plt.title(f"Noisy (σ={noise_level:.3f}, PSNR: {orig_psnr:.2f} dB)")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(denoised.detach().cpu().numpy())
    plt.title(f"Denoised (PSNR: {final_psnr:.2f} dB)")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(S.cpu().numpy())
    plt.title("Sparse component")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(N.cpu().numpy())
    plt.title("Gaussian component")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("denoising_results.png", dpi=150, bbox_inches='tight')

    denoised_np = (denoised.detach().cpu().numpy() * 255).astype(np.uint8)
    if denoised_np.ndim == 3 and denoised_np.shape[2] == 3:
        pil_img = Image.fromarray(denoised_np)
        pil_img.save("denoised_fixed_size.png")

    print("Saved denoised image to 'denoised_fixed_size.png'")

    if psnrs:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(psnrs) + 1), psnrs, 'o-')
        plt.axhline(y=orig_psnr, color='r', linestyle='--', label='Noisy PSNR')
        plt.axhline(y=final_psnr, color='g', linestyle='--', label='Final PSNR')
        plt.xlabel('Iteration')
        plt.ylabel('PSNR (dB)')
        plt.title(f'PSNR Convergence (σ={noise_level:.3f})')
        plt.legend()
        plt.grid(True)
        plt.savefig("psnr_progress.png", dpi=150, bbox_inches='tight')
        print("Saved PSNR progress to 'psnr_progress.png'")

    plt.show()
    return denoised


if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    noise_level = 0.1
    num_iterations = 100
    lambda_1 = 0.8
    lambda_2 = 0.08
    ep = 80
    alpha0 = 1 / 2.001
    alpha1 = 1 / 2.001
    alpha2 = 0.001 / 2.001
    use_weights = True

    print("Starting T-SVD Denoising Experiment")
    print("===================================")
    print(f"Noise Level: {noise_level:.3f}")
    print(f"Number of Iterations: {num_iterations}")
    print(f"Lambda_1: {lambda_1:.3f}")
    print(f"Lambda_2: {lambda_2:.3f}")

    denoised_image = denoising_experiment(
        image_path=image_path,
        noise_level=noise_level,
        num_iters=num_iterations,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        use_weights=use_weights
    )

    print("\nDenoising completed successfully!")
