import os.path
import cv2
import logging
import numpy as np
import torch

from datetime import datetime

import lpips

from utils import utils_model
from utils import utils_logger

from utils import utils_image as util

from trpca_l1 import t_svd_log_prox, prox_l1
from skimage.metrics import structural_similarity as ssim
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


# ===========================================================================
# Noise Addition Functions
# ===========================================================================

def add_gaussian_noise(img, sigma):
    """
    Add Gaussian noise to an image.

    Args:
        img: float32 array, range [0, 1], shape (H, W, C)
        sigma: standard deviation of Gaussian noise (in [0, 1] range)

    Returns:
        noisy: image with Gaussian noise, clipped to [0, 1]
    """
    noise = sigma * np.random.randn(*img.shape).astype(np.float32)
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def add_salt_pepper_noise(img, density):
    """
    Add salt-and-pepper noise (sparse noise) to an image.

    Args:
        img: float32 array, range [0, 1], shape (H, W, C)
        density: noise density in [0, 1], proportion of pixels to corrupt

    Returns:
        noisy: image with salt-and-pepper noise, clipped to [0, 1]
    """
    noisy = img.copy()
    h, w = noisy.shape[:2]

    # Generate noise mask: which pixels are corrupted
    noise_mask = np.random.rand(h, w) < density

    # Among corrupted pixels, half are salt (white=1.0), half are pepper (black=0.0)
    salt_mask = np.random.rand(h, w) < 0.5
    pepper_mask = ~salt_mask

    # Salt noise
    noisy[salt_mask & noise_mask] = 1.0
    # Pepper noise
    noisy[pepper_mask & noise_mask] = 0.0

    return noisy


# ===========================================================================
# LPIPS Calculation
# ===========================================================================

def calculate_lpips_score(img_E, img_H, loss_fn_lpips, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate LPIPS score between two images.

    Args:
        img_E: denoised image (numpy array, uint8, 0-255)
        img_H: reference image (numpy array, uint8, 0-255)
        loss_fn_lpips: initialized LPIPS model
        device: computation device

    Returns:
        lpips_value: LPIPS score
    """
    try:
        # Ensure correct data type and range
        def prepare_image(img):
            if isinstance(img, np.ndarray):
                # uint8 (0-255) -> float32 normalized to [0, 1]
                if img.dtype == np.uint8:
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    # Already float, ensure in [0, 1] range
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                return img_tensor.to(device)
            else:
                # Already a tensor, use directly
                return img.to(device) if img.device != device else img

        # Prepare image tensors
        img_E_tensor = prepare_image(img_E)
        img_H_tensor = prepare_image(img_H)

        # Normalize to [-1, 1] range as expected by LPIPS
        img_E_tensor = img_E_tensor * 2 - 1
        img_H_tensor = img_H_tensor * 2 - 1

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = loss_fn_lpips(img_E_tensor, img_H_tensor).item()

        return lpips_value

    except Exception as e:
        logging.error(f"Error computing LPIPS: {e}")
        return None


# ===========================================================================
# Main Function
# ===========================================================================

def main():
    # Fix random seed
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =======================================================================
    # Noise Parameters
    # =======================================================================
    noise_level_img = 0.07       # Gaussian noise std (in [0, 1] range)
    sparse_density = 0.05        # Salt-and-pepper noise density
    enable_sparse = True         # True: add SP noise + enable S-term update
                                 # False: Gaussian noise only, S-term disabled

    sigma = max(0.001, noise_level_img)

    # =======================================================================
    # Model Parameters
    # =======================================================================
    model_name = 'diffusion_ffhq_10m'
    testset_name = 'testset'     # Folder containing clean images
    num_train_timesteps = 1000
    iter_num = 100
    skip = num_train_timesteps // iter_num

    show_img = False
    save_E = True
    save_progressive = False
    save_noisy = True            # Whether to save noisy images

    lambda_ = 3
    nuclear_lambda = 1
    lambda_1 = 0.1
    lambda_2 = 200

    ddim_sample = False
    model_output_type = 'pred_xstart'
    generate_mode = 'DiffPIR'
    skip_type = 'quad'
    eta = 0.0
    zeta = 1.0
    guidance_scale = 1.0

    blur_mode = 'Gaussian'

    n_channels = 3
    cwd = ''
    model_zoo = os.path.join(cwd, 'model_zoo')
    testsets = os.path.join(cwd, 'testsets')
    results = os.path.join(cwd, 'results')

    # Result name includes sparse noise flag
    sparse_tag = 'withS' if enable_sparse else 'noS'
    result_name = (f'{testset_name}_gaussian_{noise_level_img}_'
                   f'sp_{sparse_density}_{sparse_tag}_'
                   f'lambda_{lambda_}_nuc_{nuclear_lambda}_lambda2_{lambda_2}')
    model_path = os.path.join(model_zoo, model_name + '.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # =======================================================================
    # Noise Schedule
    # =======================================================================
    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    noise_model_t = 10
    t_start = num_train_timesteps - 1

    # =======================================================================
    # Path Setup
    # =======================================================================
    # Clean images path (read directly from testset folder)
    L_path = os.path.join(cwd, testset_name)
    E_path = os.path.join(results, result_name)
    noisy_path = os.path.join(E_path, 'noisy')  # Saved noisy images
    util.mkdir(E_path)
    if save_noisy:
        util.mkdir(noisy_path)

    logger_name = f"{result_name}_{datetime.now().strftime('%m%d%H%M')}"
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, 'experiment.log'))
    logger = logging.getLogger(logger_name)

    # =======================================================================
    # Load Model
    # =======================================================================
    model_config = dict(
        model_path=model_path,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    ) if model_name == 'diffusion_ffhq_10m' else dict(
        model_path=model_path,
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="8,16,32",
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # Initialize LPIPS model
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.eval()
    logger.info("LPIPS model initialized")

    logger.info(f'model_name: {model_name}')
    logger.info(f'Gaussian sigma: {noise_level_img}, Sparse density: {sparse_density}')
    logger.info(f'enable_sparse (S-term): {enable_sparse}')
    logger.info(f'lambda: {lambda_:.3f}, nuclear_lambda: {nuclear_lambda:.3f}')

    # =======================================================================
    # Get Test Image List
    # =======================================================================
    if not os.path.exists(L_path):
        logger.error(f"Testset folder not found: {L_path}")
        logger.info("Please create the testset folder and place clean images in it")
        return

    try:
        clean_image_paths = util.get_image_paths(L_path)
    except AssertionError:
        logger.error(f"No valid image files found in {L_path}")
        logger.info("Please place test images (PNG/JPG/etc.) in the testset folder")
        return

    logger.info(f"Found {len(clean_image_paths)} images in {testset_name}")
    logger.info(f"Gaussian sigma={noise_level_img}, SP density={sparse_density}")
    logger.info(f"S-term (sparse): {'enabled' if enable_sparse else 'disabled'}")

    # =======================================================================
    # Create Results File
    # =======================================================================
    results_file_path = os.path.join(E_path, 'results.txt')
    with open(results_file_path, 'w') as f:
        f.write("Image Name,Gaussian Sigma,Sparse Density,S-term,PSNR (dB),LPIPS,SSIM\n")
    logger.info("Results file created")

    # =======================================================================
    # Initialize Metric Lists
    # =======================================================================
    psnr_list = []
    lpips_list = []
    ssim_list = []

    # =======================================================================
    # Process Each Image
    # =======================================================================
    for idx, clean_img_path in enumerate(clean_image_paths):
        img_name = os.path.basename(clean_img_path)
        logger.info(f'Processing image {idx + 1}/{len(clean_image_paths)}: {img_name}')

        # ---------------------------------------------------------------
        # Load clean image
        # ---------------------------------------------------------------
        img_H_uint = util.imread_uint(clean_img_path, n_channels=n_channels)
        img_H = util.uint2single(img_H_uint)  # uint8 -> float32 [0, 1]

        # ---------------------------------------------------------------
        # Add Gaussian noise
        # ---------------------------------------------------------------
        img_L = add_gaussian_noise(img_H, noise_level_img)

        # ---------------------------------------------------------------
        # Add salt-and-pepper noise (optional)
        # ---------------------------------------------------------------
        if enable_sparse:
            img_L = add_salt_pepper_noise(img_L, sparse_density)

        # Ensure noisy image is in valid range
        img_L = np.clip(img_L, 0.0, 1.0)

        # Save noisy image
        if save_noisy:
            noisy_img_uint = util.single2uint(img_L)
            noisy_save_path = os.path.join(noisy_path, img_name)
            util.imsave(noisy_img_uint, noisy_save_path)

        # ---------------------------------------------------------------
        # Denoising (single image)
        # Both img_L and img_H are float32 in [0, 1], matching denoise_10.py
        # ---------------------------------------------------------------
        def process_single_image(img_L, img_H, lambda_=lambda_, nuclear_lambda=nuclear_lambda,
                                 lambda_1=lambda_1, lambda_2=lambda_2, enable_sparse=enable_sparse):

            # Compute rhos and sigmas
            sigmas = []
            sigma_ks = []
            rhos = []
            mus = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps - 1 - i].item())
                if model_output_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]).item())
                else:
                    sigma_ks.append(torch.sqrt(betas[i] / alphas[i]).item())
                rhos.append(lambda_ * (sigma ** 2) / (sigma_ks[i] ** 2))
                mus.append(lambda_ / sigma_ks[i] ** 2)

            rhos = torch.tensor(rhos).to(device)
            sigmas = torch.tensor(sigmas).to(device)
            sigma_ks = torch.tensor(sigma_ks).to(device)

            # Initialize variables - use noisy image as starting point
            y = util.single2tensor4(img_L).to(device)
            t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
            sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * (2 * y - 1) + torch.sqrt(
                sqrt_1m_alphas_cumprod[t_start] ** 2 -
                sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[t_y] ** 2
            ) * torch.randn_like(y)

            # Initialize auxiliary variables
            M = torch.zeros_like(y)
            S = torch.zeros_like(y)
            N = torch.zeros_like(y)

            # Build timestep sequence
            if skip_type == 'uniform':
                seq = [i * skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps - 1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps ** 2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1

            nuclear_weights = None

            # Main iteration loop
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                if t_i > t_start:
                    continue

                # Reverse diffusion step
                x0 = utils_model.model_fn(
                    x + M / mus[t_i],
                    noise_level=curr_sigma * 255,
                    model_out_type=model_output_type,
                    model_diffusion=model,
                    diffusion=diffusion,
                    ddim_sample=ddim_sample,
                    alphas_cumprod=alphas_cumprod
                )
                V = x0 / 2 + 0.5

                if model_output_type == 'pred_xstart':
                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)

                    if i < num_train_timesteps - noise_model_t:

                        current_nuclear_lambda = nuclear_lambda
                        current_lambda_1 = lambda_1

                        K = tau / (1 + tau)
                        yz = K * ((y - S - N) / tau + V)

                        x0_p, nuclear_weights = t_svd_log_prox(
                            Y=yz - sigma ** 2 * M / (1 + tau),
                            weights=nuclear_weights,
                            lambd=current_nuclear_lambda * sigma ** 2 / (1 + rhos[t_i])
                        )

                        # N = (y - x0_p - S) / (1 + 2 * lambda_2 * sigma ** 2)

                        # ---------------------------------------------------
                        # S-term update: only enabled when sparse noise exists
                        # ---------------------------------------------------
                        if enable_sparse:
                            S = prox_l1(Y=y - N - x0_p, lambda_=current_lambda_1 * sigma ** 2)

                        x0_p = x0_p * 2 - 1
                        # M = M + mus[t_i] * (x0_p - x0)
                        # Effective x0
                        x0 = x0 + guidance_scale * (x0_p - x0)
                    else:
                        model_out_type = 'pred_x_prev'
                        x0 = utils_model.model_fn(
                            x,
                            noise_level=curr_sigma * 255,
                            model_out_type=model_out_type,
                            model_diffusion=model,
                            diffusion=diffusion,
                            ddim_sample=ddim_sample,
                            alphas_cumprod=alphas_cumprod
                        )

                # Update x
                if (generate_mode == 'DiffPIR' and model_output_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                    t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy())
                    eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(
                        betas[t_i])
                    x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                            torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma ** 2) * eps +
                            eta_sigma * torch.randn_like(x)
                    ) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

            # Final image
            x_0 = (x / 2 + 0.5)
            img_E = util.tensor2uint(x_0)
            final_psnr = util.calculate_psnr(img_E, util.single2uint(img_H), border=0)
            logger.info(f'Final PSNR: {final_psnr:.4f} dB')

            # Compute LPIPS
            final_lpips = calculate_lpips_score(img_E, util.single2uint(img_H), loss_fn_lpips, device)
            logger.info(f'Final LPIPS: {final_lpips:.4f}')

            # Compute SSIM
            img_E_uint = img_E
            img_H_uint = util.single2uint(img_H)
            # For color images, use channel_axis=2 (HWC format)
            final_ssim = ssim(img_E_uint, img_H_uint, data_range=255, channel_axis=2, win_size=11,
                              gaussian_weights=True)
            logger.info(f'Final SSIM: {final_ssim:.4f}')

            if save_E:
                sparse_label = 'withS' if enable_sparse else 'noS'
                output_name = (f"{img_name.split('.')[0]}_"
                               f"gaussian_{noise_level_img}_"
                               f"sp_{sparse_density}_"
                               f"{sparse_label}.png")
                util.imsave(img_E, os.path.join(E_path, output_name))

            return final_psnr, final_lpips, final_ssim

        # ---------------------------------------------------------------
        # Run denoising
        # ---------------------------------------------------------------
        final_psnr, final_lpips, final_ssim = process_single_image(img_L, img_H)

        # Record metrics to lists
        psnr_list.append(final_psnr)
        lpips_list.append(final_lpips)
        ssim_list.append(final_ssim)

        # Write per-image result to file
        with open(results_file_path, 'a') as f:
            f.write(f"{img_name},{noise_level_img},{sparse_density},"
                    f"{'with_S' if enable_sparse else 'no_S'},"
                    f"{final_psnr:.4f},{final_lpips:.4f},{final_ssim:.4f}\n")

    # =======================================================================
    # Summary Statistics
    # =======================================================================
    if psnr_list and lpips_list and ssim_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_lpips = sum(lpips_list) / len(lpips_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete! Total images: {len(psnr_list)}")
        logger.info(f"Average PSNR:  {avg_psnr:.4f} dB")
        logger.info(f"Average LPIPS: {avg_lpips:.4f}")
        logger.info(f"Average SSIM:  {avg_ssim:.4f}")
        logger.info(f"Gaussian sigma: {noise_level_img}")
        logger.info(f"SP density: {sparse_density}")
        logger.info(f"S-term (sparse): {'enabled' if enable_sparse else 'disabled'}")
        logger.info(f"{'='*60}")

        # Write average to results file
        with open(results_file_path, 'a') as f:
            f.write(f"Average,{noise_level_img},{sparse_density},"
                    f"{'with_S' if enable_sparse else 'no_S'},"
                    f"{avg_psnr:.4f},{avg_lpips:.4f},{avg_ssim:.4f}\n")

    logger.info(f"Results saved to {results_file_path}")
    if save_noisy:
        logger.info(f"Noisy images saved to {noisy_path}")
    logger.info(f"Denoised images saved to {E_path}")


if __name__ == '__main__':
    main()
