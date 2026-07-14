import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time



def log_prox(sigma_i, lambda_, eps=1e-6):
    """
    修正的非凸Log奇异值收缩算子
    """
    # 初始化输出为0
    sigma = torch.zeros_like(sigma_i)

    # 计算判别式
    discriminant = (sigma_i + eps) ** 2 - 4 * lambda_
    mask = discriminant > 0
    # 检查判别式非负
    if torch.any(mask):
        # 计算非零解
        sigma[mask] = (sigma_i[mask] - eps + torch.sqrt(discriminant[mask])) / 2

    return sigma


def t_svd_log_prox(Y, weights, lambd, eps=1e-6, k=2, use_weights=True):
    """
    单张图片的 T-SVD 非凸加权核范数临近点算子
    - 支持4阶张量输入 [B, C, H, W] 或 [B, H, W, C]
    - 支持3阶张量输入 [H, W, C]
    Args:
        Y: torch.Tensor, 输入张量
        weights: torch.Tensor, 权重向量 [C, min(H,W)]
        lambd: float, 正则化参数
        eps: float, 防止除零的小常数
    Returns:
        X: torch.Tensor, 输出张量（与输入同维度）
        new_weights: torch.Tensor, 更新后的权重 [C, min(H,W)]
    """
    # 处理4阶张量输入
    input_dim = Y.dim()
    if input_dim == 4:
        # 提取第一个batch并转换为[H,W,C]
        batch_size, channel, height, width = Y.shape
        Y = Y.permute(0, 2, 3, 1).contiguous().squeeze(0)  # [B,H,W,C] -> [H,W,C]

    # 原始维度处理逻辑
    if k == 0:
        Y = Y.permute(2, 1, 0)
    if k == 1:
        Y = Y.permute(0, 2, 1)

    H, W, C = Y.shape

    # 1. 沿通道维进行FFT
    Y_bar = fft.fft(Y.to(torch.float32), dim=2, norm='ortho')  # [H, W, C]


    # 准备存储更新后的权重
    min_dim = min(H, W)
    if weights is None:
        weights = torch.ones(C, min_dim, device=Y.device)

    new_weights = torch.ones(C, min_dim, device=Y.device)
    X_bar = torch.zeros_like(Y_bar)

    ##### 更新第一个切片 #####
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

    ##### 遍历每个通道切片 #####
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

    # 处理中间频率（当通道数为偶数时）
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

    # 5. 逆FFT恢复空间域张量
    X = fft.ifft(X_bar, dim=2, norm='ortho').real

    # 恢复原始维度顺序
    if k == 0:
        X = X.permute(2, 1, 0)
    if k == 1:
        X = X.permute(0, 2, 1)

    # 归一化权重
    row_sums = new_weights.sum(dim=1, keepdim=True)
    new_weights = (new_weights / row_sums) * new_weights.size(1)

    # 如果输入是4阶，恢复为4阶格式
    if input_dim == 4:
        # 添加batch维度并恢复为[B,C,H,W]
        X = X.unsqueeze(0).permute(0, 3, 1, 2)  # [H,W,C] -> [B,C,H,W]

    return X, new_weights

def prox_log_l1(Y, w, lam, eps):
    abs_Y = np.abs(Y)
    T = lam * w / eps
    Delta = (abs_Y + eps) ** 2 - 4 * lam * w

    # 次梯度条件满足 → 置零
    if abs_Y <= T:
        return 0.0

    # 判别式条件
    if Delta >= 0:
        sqrt_Delta = np.sqrt(Delta)
        x_val = (abs_Y - eps + sqrt_Delta) / 2
        return np.sign(Y) * x_val

    return 0.0

def prox_l1(Y, lambda_):
    """
    L1范数近端算子，与NumPy版本完全一致
    """
    return torch.maximum(torch.tensor(0, dtype=Y.dtype, device=Y.device),
                         Y - lambda_) + \
        torch.minimum(torch.tensor(0, dtype=Y.dtype, device=Y.device),
                      Y + lambda_)

def iterative_t_svd_denoising(noisy_img, clean_img=None, num_iters=100,lambda_1=1,lambda_2=1,rho=1.2,
                              mu=1e-4,max_mu=1e10,ep=1,
                              alpha0=1/3,
                              alpha1=1/3,
                              alpha2=1/3,
                              beta=1e-4,
                              max_beta=1e10,
                              use_weights=False):
    """
    完整的迭代T-SVD去噪流程 (加入亮度对齐和归一化)
    Args:
        noisy_img: torch.Tensor, 噪声图像 [H, W, C] (0-1范围)
        clean_img: torch.Tensor, 用于PSNR计算的干净图像 [H, W, C]
        num_iters: int, 迭代次数
        lambd: float, 正则化参数
        eps: float, 防止除零的小常数
    Returns:
        denoised: torch.Tensor, 去噪后图像 [H, W, C]
        psnrs: list, 每次迭代的PSNR值 (如有干净图像)
    """
    # 确保输入在合理范围内
    noisy_img_normalized = torch.clamp(noisy_img, 0, 1)


    # 归一化预处理：将图像归一化到零均值
    # 初始化变量

    Y=noisy_img_normalized.clone()
    H, W, C = Y.shape
    min_dim = min(H, W)
    S=torch.zeros_like(Y)
    M0=torch.zeros_like(Y)
    mu0 = mu
    M1 = torch.zeros_like(Y)
    mu1 = mu
    M2 = torch.zeros_like(Y)
    mu2=mu
    P=torch.zeros_like(Y)
    X=torch.zeros_like(Y)
    N = torch.zeros_like(Y)







    # 存储PSNR变化 (用于分析)
    psnrs = []

    print(f"\nStarting T-SVD denoising with {num_iters} iterations...")

    for i in range(num_iters):
        # 记录开始时间
        start_time = time.time()

        if i==0:
            weights0=None
            weights1=None
            weights2=None
            # 应用T-SVD临近点算子
        Z0, weights0 = t_svd_log_prox(
                Y=(X + M0/mu0),
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


        X=(mu0*Z0+mu1*Z1+mu2*Z2)-(M0+M1+M2)+beta*(Y-N-S+P/beta)
        X=X/(mu0+mu1+mu2+beta)

        N=beta*(Y-X-S+P/beta)/(2*lambda_1+beta)
        S=prox_l1(Y-X-N+P/beta,lambda_2/beta)
        M0=M0+mu0*(X-Z0)
        M1 = M1 + mu1 * (X - Z1)
        M2=M2+mu2*(X-Z2)
        P=P+beta*(Y-X-N-S)


        mu0 = min(rho * mu0, max_mu)
        mu1 = min(rho * mu1, max_mu)
        mu2 = min(rho * mu2, max_mu)
        beta=min(rho * beta, max_beta)


        # 计算本次迭代的PSNR (如果有干净图像)
        if clean_img is not None:
            # 临时恢复亮度以便准确计算PSNR
            L_temp = torch.clamp(X, 0, 1)
            psnr = calculate_psnr(clean_img, L_temp.detach())
            psnrs.append(psnr)

        # 打印进度
        iter_time = time.time() - start_time
        print(f"Iteration {i + 1}/{num_iters} completed - Time: {iter_time:.2f}s", end='')
        if clean_img is not None:
            print(f" - PSNR: {psnr:.2f} dB")
        else:
            print()
    # 后处理：恢复原始亮度和数值范围
    denoised = torch.clamp(X, 0, 1)
    S=torch.clamp(S, 0, 1)
    N=torch.clamp(N, 0, 1)
    return denoised, psnrs,S,N


# 图像加载和处理工具函数 (加入归一化选项)
def load_image(path, max_size=10e6, normalize=True):
    """加载图像并调整为合适尺寸"""
    img = Image.open(path).convert("RGB")

    # 调整大小保持比例
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 确保尺寸是8的倍数（更好的处理效果）
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"Resized image to {new_w}x{new_h} (original: {w}x{h})")

    # 转换为Tensor
    tensor = transforms.ToTensor()(img)  # [C, H, W]
    tensor = tensor.permute(1, 2, 0)  # [H, W, C]

    # 归一化到[0,1]范围
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    return tensor


def add_gaussian_noise(img, sigma=0.1):
    """添加高斯噪声"""
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    # 裁剪到[0,1]范围
    return torch.clamp(noisy_img, 0, 1), noise


def calculate_psnr(img1, img2):
    """计算PSNR"""
    # 确保在CPU上计算
    img1 = img1.cpu()
    img2 = img2.cpu()

    # 确保数值在0-1范围内
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


# 实验主函数 (加入亮度对齐)
def denoising_experiment(image_path, noise_level=0.2, num_iters=15, lambda_1=1,lambda_2=1,eps=1e-5,
                         alpha0=1/2.001,alpha1=1/2.001,alpha2=0.001/2.001,use_weights=False):
    """
    执行去噪实验并显示结果
    """
    # 1. 加载并预处理图像
    clean_img = load_image(image_path)  # [H, W, C]
    noisy_img, noise = add_gaussian_noise(clean_img, sigma=noise_level)
    # noisy_img=clean_img
    print(f"\nImage shape: {clean_img.shape}")
    print(f"Added Gaussian noise with sigma={noise_level:.3f}")

    # 2. 执行T-SVD去噪 - 传递干净图像用于PSNR计算
    denoised, psnrs ,S,N= iterative_t_svd_denoising(
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



    # 4. 计算性能指标
    orig_psnr = calculate_psnr(clean_img, noisy_img)
    final_psnr = calculate_psnr(clean_img, denoised)

    print(f"\nSummary:")
    print(f"Original PSNR (noisy): {orig_psnr:.2f} dB")
    print(f"Final PSNR (denoised): {final_psnr:.2f} dB")
    print(f"PSNR improvement: {final_psnr - orig_psnr:.2f} dB")

    # 5. 可视化结果（包含指标）
    plt.figure(figsize=(18, 6))

    # 噪声图像
    plt.subplot(1, 4, 1)
    plt.imshow(noisy_img.cpu().numpy())
    plt.title(f"Noisy Image (σ={noise_level:.3f}, PSNR: {orig_psnr:.2f} dB)")
    plt.axis('off')

    # 去噪后图像
    plt.subplot(1, 4, 2)
    plt.imshow(denoised.detach().cpu().numpy())
    plt.title(f"Denoised Image (PSNR: {final_psnr:.2f} dB)")
    plt.axis('off')

    # 原始图像
    plt.subplot(1, 4, 3)
    plt.imshow(S.cpu().numpy())
    plt.title("S")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(N.cpu().numpy())
    plt.title("N")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("denoising_results.png", dpi=150, bbox_inches='tight')

    # 新增：单独保存去噪后的图像（不包含任何指标）
    denoised_np = (denoised.detach().cpu().numpy() * 255).astype(np.uint8)
    if denoised_np.ndim == 3 and denoised_np.shape[2] == 3:  # 确保是RGB
        from PIL import Image
        pil_img = Image.fromarray(denoised_np)
        pil_img.save("denoised_fixed_size.png")  # 直接保存，不设置DPI

    print("Saved denoised image without metrics to 'denoised_image_only.png'")

    # 6. 绘制PSNR变化曲线
    if psnrs:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(psnrs) + 1), psnrs, 'o-')
        plt.axhline(y=orig_psnr, color='r', linestyle='--', label='Noisy PSNR')
        plt.axhline(y=final_psnr, color='g', linestyle='--', label='Final PSNR')
        plt.xlabel('Iteration')
        plt.ylabel('PSNR (dB)')
        plt.title(f'PSNR Improvement during Denoising (σ={noise_level:.3f})')
        plt.legend()
        plt.grid(True)
        plt.savefig("psnr_progress.png", dpi=150, bbox_inches='tight')
        print("Saved PSNR progress plot to 'psnr_progress.png'")

    plt.show()

    return denoised


# 执行实验
if __name__ == "__main__":
    # 参数设置
    image_path = "path/to/your/image.png"  # Change to your image path
    noise_level = 0.1  # 噪声水平 (0-1)
    num_iterations = 100  # 迭代次数
    lambda_1 = 0.8 # 正则化参数
    lambda_2 = 0.08
    ep=80
    alpha0 = 1 / 2.001
    alpha1 = 1 / 2.001
    alpha2 = 0.001 / 2.001
    use_weights=True

    print("Starting T-SVD Denoising Experiment")
    print("===================================")
    print(f"Noise Level: {noise_level:.3f}")
    print(f"Number of Iterations: {num_iterations}")
    print(f"Lambda_1 Value: {lambda_1:.3f}")
    print(f"Lambda_2 Value: {lambda_2:.3f}")

    # 执行去噪实验
    denoised_image = denoising_experiment(
        image_path=image_path,
        noise_level=noise_level,
        num_iters=num_iterations,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        use_weights=use_weights
    )

    print("\nDenoising completed successfully!")