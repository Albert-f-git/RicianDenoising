import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 导入你的模块
from data.dataset import MRIDenoisingDataset
from models.dncnn import DnCNN
from models.unet import UNet
from models.RicianNet import RicianNet

# ==========================================
# 1. 测试配置区 (请修改为你真实的模型路径)
# ==========================================
sigma = 0.1  # 固定噪声水平，保持与训练一致
CONFIG = {
    # ⚠️ 替换为具体权重路径 (比如 experiments/DnCNN_Baseline_Adam_2026xxxx_xxxx/model_weights.pth)
    # "model_weights_path": "experiments\DnCNN_Baseline_Adam_20260302_190727\model_weights.pth", # DnCNN_Adam
    # "model_weights_path": "experiments\DnCNN_Baseline_AdamW_20260302_200103\model_weights.pth", # DnCNN_AdamW
    # "model_weights_path": "experiments\DnCNN_Sliding_Patch64_Stride14_AdamW_20260302_230306\model_weights.pth", # DnCNN_Sliding_Patch64_Stride14_AdamW
    # "model_weights_path": "experiments\DnCNN_Sliding_Patch64_Stride32_AdamW_20260306_094004\model_weights.pth", # DnCNN_Sliding_Patch64_Stride32_AdamW
    # "model_weights_path": "experiments\DnCNN_Random_Patch64_AdamW_20260303_073545\model_weights.pth", # DnCNN_Random_Patch64_AdamW_Epoch50
    # "model_weights_path": "experiments\DnCNN_Random_Patch64_AdamW_20260303_091959\model_weights.pth", # DnCNN_Random_Patch64_AdamW_Epoch200
    # "model_weights_path": "experiments\DnCNN_Random_Patch64_AdamW_20260303_095649\model_weights.pth", # DnCNN_Random_Patch64_AdamW_Epoch500
    # "model_weights_path": "experiments\DnCNN_Random_Patch64_Epoch1500_20260310_123758\model_weights.pth", # DnCNN_Random_Patch64_Epoch1500
    "model_weights_path": "experiments\DnCNN_Random_Patch41_Epoch3000_20260310_152419\model_weights.pth", # DnCNN_Random_Patch41_Epoch3000
    # "model_weights_path": "experiments\Unet_Baseline_AdamW_1e-4_20260305_125730\model_weights.pth", # Unet_AdamW_1e-4
    # "model_weights_path": "experiments\\Unet_Baseline_AdamW_1e-3_20260306_114041\model_weights.pth", # Unet_AdamW_1e-3
    # "model_weights_path": "experiments\\Unet_Sliding_Patch64_Stride14_AdamW_20260305_133805\model_weights.pth", # Unet_Sliding_Patch64_Stride14_AdamW
    # "model_weights_path": "experiments\\Unet_Sliding_Patch64_Stride32_AdamW_20260306_121908\model_weights.pth", # Unet_Sliding_Patch64_Stride32_AdamW
    # "model_weights_path": "experiments\\Unet_Random_Patch64_AdamW_20260306_135850\model_weights.pth", # Unet_Random_Patch64_AdamW_Epoch800
    # "model_weights_path": "experiments\RicianNet_baseline_AdamW_20260306_161247\model_weights.pth", # RicianNet_baseline_AdamW
    # "model_weights_path": "experiments\RicianNet_pedding_reflect_AdamW_20260306_210919\model_weights.pth", # RicianNet_pedding_reflect_AdamW    
    # "model_weights_path": "experiments\RicianNet_padding_zeros_cleaned_AdamW_20260307_171156\model_weights.pth", # RicianNet_padding_zeros_cleaned_AdamW
    # "model_weights_path": "experiments\RicianNet_random_Patch128_AdamW_20260307_203419\model_weights.pth", # RicianNet_random_Patch128
    # "model_weights_path": "experiments\RicianNet_random_Patch64_20260309_100743\model_weights.pth", # RicianNet_random_Patch64
    # "model_weights_path": "experiments\RicianNet_random_Patch64_Epoch400_20260309_104728\model_weights.pth", # RicianNet_random_Patch64_Epoch400
    # "model_weights_path": "experiments\RicianNet_sliding_Patch64_20260309_121904\model_weights.pth", # RicianNet_sliding_Patch64
    "dataset_mode": "full",                  # 使用全图零填充模式
    "noise_range": (sigma, sigma),             # 保持与训练一致的噪声水平
    "num_visualize": 5,                       # 测试结束后，画几张对比图来看看
    "test_data_dir": "data/processed/test", # 测试集路径
}

def calculate_metrics(clean, denoised):
    """
    计算单张图片的 PSNR 和 SSIM
    由于模型输出可能会有微小的越界，严格限制在 [0, 1] 区间内计算
    """
    clean = np.clip(clean, 0, 1)
    denoised = np.clip(denoised, 0, 1)
    
    # data_range=1.0 是因为我们的图像经过 Min-Max 归一化，最大值是 1
    p_val = psnr(clean, denoised, data_range=1.0)
    s_val = ssim(clean, denoised, data_range=1.0)
    return p_val, s_val

def test():
    np.random.seed(45)  # 固定随机种子，确保测试结果可复现
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动测试引擎: {device}")

    # ==========================================
    # 2. 初始化数据集与模型
    # ==========================================
    # 测试时 batch_size 强制为 1，逐张严格评估
    test_dataset = MRIDenoisingDataset(
        data_dir=CONFIG["test_data_dir"],
        noise_level_range=CONFIG["noise_range"],
        mode=CONFIG["dataset_mode"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 通过权重路径的名称，自动判断该用哪个模型架构
    if "Unet" in CONFIG["model_weights_path"]:
        print("🏗️ 检测到 U-Net 权重，正在实例化 U-Net 模型...")
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif "RicianNet" in CONFIG["model_weights_path"]:
        print("🏗️ 检测到 RicianNet 权重，正在实例化 RicianNet 模型...")
        model = RicianNet().to(device)
    else:
        print("🏗️ 正在实例化 DnCNN 模型...")
        model = DnCNN().to(device)
    
    # 加载预训练权重
    if not os.path.exists(CONFIG["model_weights_path"]):
        raise FileNotFoundError(f"找不到权重文件: {CONFIG['model_weights_path']}，请检查路径！")
        
    model.load_state_dict(torch.load(CONFIG["model_weights_path"], map_location=device, weights_only=True))
    model.eval()  # 🚨 极其重要：关闭 Dropout 和 BN 的动态更新，进入评估模式

    # ==========================================
    # 3. 核心推理循环
    # ==========================================
    psnr_noisy_list, ssim_noisy_list = [], []
    psnr_denoised_list, ssim_denoised_list = [], []
    
    visual_samples = []

    print("\n🔬 正在对测试集进行逐切片量化评估...")
    
    # torch.no_grad() 告诉 PyTorch 不需要计算梯度，极大节省显存并加速推理
    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(tqdm(test_loader, desc="Testing")):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 前向推理
            denoised_imgs = model(noisy_imgs)

            # 提取张量转为 numpy (维度: H, W)
            clean_np = clean_imgs.squeeze().cpu().numpy()
            noisy_np = noisy_imgs.squeeze().cpu().numpy()
            denoised_np = denoised_imgs.squeeze().cpu().numpy()

            # 计算 加噪图 vs 原图 的指标 (作为 Baseline Reference)
            p_noisy, s_noisy = calculate_metrics(clean_np, noisy_np)
            psnr_noisy_list.append(p_noisy)
            ssim_noisy_list.append(s_noisy)

            # 计算 去噪图 vs 原图 的指标 (验证模型的硬实力)
            p_denoised, s_denoised = calculate_metrics(clean_np, denoised_np)
            psnr_denoised_list.append(p_denoised)
            ssim_denoised_list.append(s_denoised)

            # 收集几张样本用于最后画图
            if i < CONFIG["num_visualize"]:
                visual_samples.append((clean_np, noisy_np, denoised_np, p_denoised, s_denoised))

    # ==========================================
    # 4. 汇总统计与生成报告
    # ==========================================
    exp_dir = os.path.dirname(CONFIG["model_weights_path"])
    
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_ssim_noisy = np.mean(ssim_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_ssim_denoised = np.mean(ssim_denoised_list)

    print("\n" + "="*40)
    print("🏆 测试集评估结果 (Test Metrics)")
    print("="*40)
    print(f"加噪图像平均 PSNR: {avg_psnr_noisy:.2f} dB  | SSIM: {avg_ssim_noisy:.4f}")
    print(f"去噪图像平均 PSNR: {avg_psnr_denoised:.2f} dB  | SSIM: {avg_ssim_denoised:.4f}")
    print(f"🚀 PSNR 提升: +{(avg_psnr_denoised - avg_psnr_noisy):.2f} dB")
    print("="*40)

    # 结果存入 JSON
    results = {
        "Test_Samples": len(test_loader),
        "Metrics_Noisy": {"PSNR": avg_psnr_noisy, "SSIM": avg_ssim_noisy},
        "Metrics_Denoised": {"PSNR": avg_psnr_denoised, "SSIM": avg_ssim_denoised}
    }
    with open(os.path.join(exp_dir, f"test_results_sigma{sigma}.json"), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    test()