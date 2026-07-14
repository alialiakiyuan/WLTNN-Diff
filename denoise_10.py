import os.path
import torch
import lpips  # 确保已安装lpips库
import cv2
import logging
import numpy as np

import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict


from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from scipy import ndimage
from trpca_l1 import t_svd_log_prox
from skimage.metrics import structural_similarity as ssim
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
import trpca_l1

def calculate_lpips_score(img_E, img_H, loss_fn_lpips, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算两张图像之间的LPIPS值（修复版本）

    参数:
    img_E: 去噪后图像（numpy数组，uint8格式，0-255范围）
    img_H: 参考图像（numpy数组，uint8格式，0-255范围）
    loss_fn_lpips: 已初始化的LPIPS模型
    device: 计算设备

    返回:
    lpips_value: LPIPS值
    """
    try:
        # 确保图像为正确的数据类型和范围
        def prepare_image(img):
            if isinstance(img, np.ndarray):
                # 如果图像是uint8格式(0-255)，转换为float并归一化到[0,1]
                if img.dtype == np.uint8:
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:
                    # 如果已经是float，确保在[0,1]范围内
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                return img_tensor.to(device)
            else:
                # 如果是tensor，直接使用
                return img.to(device) if img.device != device else img

        # 准备图像张量
        img_E_tensor = prepare_image(img_E)
        img_H_tensor = prepare_image(img_H)

        # 将图像归一化到[-1, 1]范围，LPIPS期望此范围[6,8](@ref)
        img_E_tensor = img_E_tensor * 2 - 1
        img_H_tensor = img_H_tensor * 2 - 1

        # 计算LPIPS[2,5](@ref)
        with torch.no_grad():
            lpips_value = loss_fn_lpips(img_E_tensor, img_H_tensor).item()

        return lpips_value

    except Exception as e:
        logging.error(f"计算LPIPS时出错: {e}")
        return None

def main():
    # 固定随机种子
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 参数设置
    noise_level_img = 0.03
    sigma = max(0.001, noise_level_img)
    model_name = 'diffusion_ffhq_10m'
    testset_name = 'CBSD68'
    num_train_timesteps = 1000
    iter_num = 100
    skip = num_train_timesteps // iter_num

    show_img = False
    save_E = True
    save_progressive = False

    lambda_ = 3
    nuclear_lambda = 14
    lambda_1 = 10
    lambda_2 = 200

    ddim_sample = False
    model_output_type = 'pred_xstart'
    generate_mode = 'DiffPIR'
    skip_type = 'quad'
    eta = 0.0
    zeta = 1.0
    guidance_scale = 1.0


    n_channels = 3
    cwd = ''
    model_zoo = os.path.join(cwd, 'model_zoo')
    testsets = os.path.join(cwd, 'testsets')
    results = os.path.join(cwd, 'results')

    # 修改结果名称以包含参数
    name_list=['5','10','15','25','35','50']
    folder_order=1
    result_name = f'{testset_name}_{name_list[folder_order]}_{noise_level_img}_lambda_{lambda_}_nuc_{nuclear_lambda}_lambda2_{lambda_2}'
    model_path = os.path.join(model_zoo, model_name + '.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # 噪声计划
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
    noise_inti_img = 50 / 255
    t_start = num_train_timesteps - 1

    # 路径设置
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)
    util.mkdir(E_path)

    logger_name = f"{result_name}_{datetime.now().strftime('%m%d%H%M')}"
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, 'experiment.log'))
    logger = logging.getLogger(logger_name)

    # 加载模型
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

    # 初始化LPIPS模型
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.eval()
    logger.info("LPIPS模型初始化完成")

    logger.info(f'model_name:{model_name}')
    logger.info(f'lambda:{lambda_:.3f}, nuclear_lambda:{nuclear_lambda:.3f}')

    # 获取CBSD68数据集中的噪声水平文件夹
    cbsd68_path = L_path
    noisy_folders = [f for f in os.listdir(cbsd68_path)
                     if f.startswith('noisy') and os.path.isdir(os.path.join(cbsd68_path, f))]

    # 按数字大小排序
    noisy_folders = sorted(noisy_folders, key=lambda x: int(x.replace('noisy', '')))
    #只处理第folder_order个文件夹
    noisy_folders=[noisy_folders[folder_order]]
    logger.info(f"排序后的文件夹顺序: {noisy_folders}")
    original_folder = os.path.join(cbsd68_path, 'original_png')

    # 检查original文件夹是否存在
    if not os.path.exists(original_folder):
        logger.error(f"Original folder not found: {original_folder}")
        return

    # 创建进度文件路径
    progress_file = os.path.join(E_path, 'progress.txt')
    processed_levels = []

    # 读取进度文件（如果存在）
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_levels = [line.strip() for line in f.readlines()]
        logger.info(f"已处理的噪声水平: {processed_levels}")

    # 确定待处理的噪声水平文件夹（只处理未处理的）
    noisy_folders_to_process = [folder for folder in noisy_folders if folder not in processed_levels]

    if not noisy_folders_to_process:
        logger.info("所有噪声水平均已处理完成！")
        return

    logger.info(f"待处理的噪声水平: {noisy_folders_to_process}")

    # 创建结果文件（如果不存在则创建并写入标题行）
    results_file_path = os.path.join(E_path, 'results.txt')
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as f:
            f.write("Noise Level,Image Name,PSNR (dB),LPIPS,SSIM\n")
        logger.info("创建新的结果文件")
    else:
        logger.info("使用现有的结果文件，将以追加模式写入新结果")

    # 处理每个噪声水平（只处理未处理的）
    for noisy_folder in noisy_folders_to_process:

        noise_level = noisy_folder.replace('noisy', '')
        current_noisy_path = os.path.join(cbsd68_path, noisy_folder)
        logger.info(f"Processing noise level: {noise_level}")

        # 获取当前噪声文件夹中的所有图像
        noisy_image_paths = util.get_image_paths(current_noisy_path)
        logger.info(f"Found {len(noisy_image_paths)} images in {noisy_folder}")

        # 初始化当前噪声水平的PSNR和LPIPS列表
        psnr_list = []
        lpips_list = []
        ssim_list = []

        # 处理当前噪声水平下的所有图像
        for idx, noisy_img_path in enumerate(noisy_image_paths):
            img_name = os.path.basename(noisy_img_path)
            # 构建原始图像路径
            original_img_path = os.path.join(original_folder, img_name)
            if not os.path.exists(original_img_path):
                logger.warning(f"Original image not found: {original_img_path}, skipping")
                continue

            logger.info(
                f'Processing image {idx + 1}/{len(noisy_image_paths)}: {img_name} for noise level {noise_level}')

            # 加载有噪声图像和原始图像
            img_L = util.imread_uint(noisy_img_path, n_channels=n_channels)
            img_H = util.imread_uint(original_img_path, n_channels=n_channels)

            # 确保图像尺寸一致
            if img_L.shape != img_H.shape:
                img_L = cv2.resize(img_L, (img_H.shape[1], img_H.shape[0]))
                logger.warning("Resized noisy image to match original image dimensions.")

            # 将图像转换为float类型，并归一化到[0,1]
            img_L = util.uint2single(img_L)
            img_H = util.uint2single(img_H)

            # 修改后的处理单张图片的函数
            def process_single_image(img_L, img_H, lambda_=lambda_, nuclear_lambda=nuclear_lambda,lambda_2=lambda_2):

                # 计算rhos和sigmas
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

                # 初始化变量 - 使用有噪声图像作为起点
                y = util.single2tensor4(img_L).to(device)
                t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
                sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
                x = sqrt_alpha_effective * (2 * y - 1) + torch.sqrt(
                    sqrt_1m_alphas_cumprod[t_start] ** 2 -
                    sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[t_y] ** 2
                ) * torch.randn_like(y)

                # 初始化其他变量
                M = torch.zeros_like(y)
                S = torch.zeros_like(y)
                N=torch.zeros_like(y)

                # 创建序列
                if skip_type == 'uniform':
                    seq = [i * skip for i in range(iter_num)]
                    if skip > 1:
                        seq.append(num_train_timesteps - 1)
                elif skip_type == "quad":
                    seq = np.sqrt(np.linspace(0, num_train_timesteps ** 2, iter_num))
                    seq = [int(s) for s in list(seq)]
                    seq[-1] = seq[-1] - 1

                nuclear_weights = None

                # 主迭代循环
                for i in range(len(seq)):
                    curr_sigma = sigmas[seq[i]].cpu().numpy()
                    t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                    if t_i > t_start:
                        continue

                    # 反向扩散步骤
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
                            current_nuclear_lambda, current_lambda_1, lambda_2 = nuclear_lambda, lambda_1,lambda_2
                            K = tau / (1 + tau)
                            yz = K * ((y - S-N) / tau + V)

                            x0_p, nuclear_weights = t_svd_log_prox(
                                Y=yz - sigma ** 2 * M / (1 + tau),
                                weights=nuclear_weights,
                                lambd=current_nuclear_lambda * sigma ** 2 / (1 + rhos[t_i])
                            )
                            # N = (y - x0_p - S) / (1 + 2 * lambda_2 * sigma ** 2)
                            # S=trpca_l1.prox_l1(Y=y-N-x0_p,lambda_=current_lambda_1*sigma**2)
                            x0_p = x0_p * 2 - 1
                            # M = M + mus[t_i] * (x0_p - x0)
                            # 有效x0
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

                    # 更新x
                    if (generate_mode == 'DiffPIR' and model_output_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy())
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(
                            betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                                torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma ** 2) * eps +
                                eta_sigma * torch.randn_like(x)
                        ) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

                # 最终图像
                x_0 = (x / 2 + 0.5)
                img_E = util.tensor2uint(x_0)
                final_psnr = util.calculate_psnr(img_E, util.single2uint(img_H), border=0)
                logger.info(f'Final PSNR: {final_psnr:.4f} dB')

                # 计算LPIPS
                final_lpips = calculate_lpips_score(img_E, util.single2uint(img_H), loss_fn_lpips, device)
                logger.info(f'Final LPIPS: {final_lpips:.4f}')

                # 计算SSIM: 将图像转换为uint8格式，并确保维度正确
                img_E_uint = img_E  # img_E已经是uint8来自util.tensor2uint
                img_H_uint = util.single2uint(img_H)  # 原始图像转换为uint8
                # 对于彩色图像，使用channel_axis=2（假设图像格式为HWC）
                final_ssim = ssim(img_E_uint, img_H_uint, data_range=255, channel_axis=2, win_size=11,
                                  gaussian_weights=True)
                logger.info(f'Final SSIM: {final_ssim:.4f}')
                if save_E:
                    output_name = f"{img_name.split('.')[0]}noise_{noisy_folders}_noise_{noise_level}_lambda_{lambda_}_nuc_{nuclear_lambda}.png"
                    util.imsave(img_E, os.path.join(E_path, output_name))

                return final_psnr, final_lpips,final_ssim

            # 处理单张图片并获取最终PSNR和LPIPS
            final_psnr, final_lpips ,final_ssim = process_single_image(img_L, img_H)

            # 记录PSNR和LPIPS到列表
            psnr_list.append(final_psnr)
            lpips_list.append(final_lpips)
            ssim_list.append(final_ssim)
            # 记录PSNR和LPIPS到文件（使用追加模式）
            with open(results_file_path, 'a') as f:
                f.write(f"{noise_level},{img_name},{final_psnr:.4f},{final_lpips:.4f},{final_ssim}\n")

        # 计算当前噪声水平的平均PSNR和LPIPS
        if psnr_list and lpips_list:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_lpips = sum(lpips_list) / len(lpips_list)
            logger.info(
                f"Noise level {noise_level} completed. Average PSNR: {avg_psnr:.4f} dB, Average LPIPS: {avg_lpips:.4f}")

            # 将平均值写入结果文件
            with open(results_file_path, 'a') as f:
                f.write(f"{noise_level},Average,{avg_psnr:.4f},{avg_lpips:.4f},{final_ssim:.4f}\n")

        # 更新进度文件
        with open(progress_file, 'a') as f:
            f.write(f"{noisy_folder}\n")
        logger.info(f"已完成噪声水平 {noise_level} 的处理，进度已更新")

    logger.info(f"所有图像处理完成。PSNR和LPIPS结果保存至 {results_file_path}")

if __name__ == '__main__':
    main()