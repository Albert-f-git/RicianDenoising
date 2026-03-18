import os
import time
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 导入你的模型
from models.unet import UNet
from models.dncnn import DnCNN

# ==========================================
# 1. 核心评测配置台 (请在这里填入你要对比的模型！)
# ==========================================
CONFIG = {
    "test_data_dir": "data/processed/test", # 你的测试集 .npy 路径
    "noise_levels": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "evaluation_results"
}

# 🌟 核心：你要对比的模型列表
MODELS_TO_TEST = [
    {
        "display_name": "UNet (Left-Attention)",
        "model_type": "UNet",
        "use_attention": True,
        "weight_path": "experiments\\Unet_LeftAttention_Random_Patch128_epoch600_GC_SGDR_20260318_151922\model_weights_final.pth", 
        "inference_mode": "full_image", 
        "color": "#e63946", "marker": "o"
    },
    {
        "display_name": "UNet (Baseline)",
        "model_type": "UNet",
        "use_attention": False,
        "weight_path": "experiments\\Unet_Random_Patch128_epoch600_GC_SGDR_20260317_180641\model_weights_final.pth", 
        "inference_mode": "full_image",
        "color": "#457b9d", "marker": "s"
    },
    {
        "display_name": "DnCNN",
        "model_type": "DnCNN",
        "weight_path": "experiments\DnCNN_Sliding_Patch64_Stride14_AdamW_20260302_230306\model_weights.pth", 
        "inference_mode": "full_image", 
        "color": "#2a9d8f", "marker": "^"
    },
    {
        "display_name": "UNet (Sliding)", 
        "model_type": "UNet",
        "use_attention": False,
        "weight_path": "experiments\\Unet_Sliding_Patch64_Stride14_SSIMLoss_20260312_195618\model_weights_final.pth",
        "inference_mode": "full_image", 
        "color": "#f4a261", "marker": "D"
    }
]

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. 基础工具函数 (与之前一致，针对 NPY 优化)
# ==========================================
def add_rician_noise(img, sigma):
    if sigma == 0: return img
    noise1 = np.random.normal(0, sigma, img.shape)
    noise2 = np.random.normal(0, sigma, img.shape)
    return np.clip(np.sqrt((img + noise1)**2 + noise2**2), 0.0, 1.0)

def get_foreground_mask(clean_img):
    img_8u = (clean_img * 255).astype(np.uint8)
    _, mask_thresh = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pad = 30
    mask_padded = cv2.copyMakeBorder(mask_thresh, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask_bridged = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel_bridge)
    contours_ext, _ = cv2.findContours(mask_bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_ext: return np.zeros_like(clean_img, dtype=bool)
    largest_contour = max(contours_ext, key=cv2.contourArea)
    mask_filled_padded = np.zeros_like(mask_padded)
    cv2.drawContours(mask_filled_padded, [largest_contour], -1, 255, thickness=-1)
    return (mask_filled_padded[pad:pad+mask_thresh.shape[0], pad:pad+mask_thresh.shape[1]] > 0)

def pad_to_multiple(img_tensor, multiple=16):
    h, w = img_tensor.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)
    return torch.nn.functional.pad(img_tensor, padding, mode='constant', value=0), padding

def unpad(img_tensor, padding):
    _, pad_w, _, pad_h = padding
    h, w = img_tensor.shape[-2:]
    return img_tensor[..., :h-pad_h, :w-pad_w]

# ==========================================
# 3. 推理引擎 (包含全图直出与滑动窗口)
# ==========================================
def run_inference(model, noisy_tensor, config):
    """根据配置动态选择推理模式"""
    if config["inference_mode"] == "full_image":
        # 兼容 U-Net 的 16倍数要求 (DnCNN其实不需要，但pad了也没副作用)
        padded_tensor, pad_info = pad_to_multiple(noisy_tensor, multiple=16)
        out_padded = model(padded_tensor)
        return unpad(out_padded, pad_info)
        
    elif config["inference_mode"] == "sliding_window":
        # 极其耗时的滑动窗口推理模拟
        b, c, h, w = noisy_tensor.shape
        patch_size = config["patch_size"]
        stride = config["stride"]
        
        out_img = torch.zeros_like(noisy_tensor)
        weight_map = torch.zeros_like(noisy_tensor)
        
        # 简单的网格滑动逻辑
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = noisy_tensor[:, :, y:y+patch_size, x:x+patch_size]
                out_patch = model(patch)
                out_img[:, :, y:y+patch_size, x:x+patch_size] += out_patch
                weight_map[:, :, y:y+patch_size, x:x+patch_size] += 1.0
                
        # 处理边缘无法被 stride 完美覆盖的残留区域 (简单兜底)
        weight_map[weight_map == 0] = 1.0 
        return out_img / weight_map

# ==========================================
# 4. 精准测速引擎 (Latency Benchmark)
# ==========================================
def measure_speed(model, model_cfg, device):
    """精准测量单张图片平均耗时 (毫秒)"""
    dummy_input = torch.rand(1, 1, 181, 217).to(device) # 使用标准 MRI 大小
    
    # 1. 显卡预热 (Warmup): 防止初次分配显存导致时间不准
    with torch.no_grad():
        for _ in range(5):
            run_inference(model, dummy_input, model_cfg)
            
    # 2. 正式测时 (同步 CUDA 保证精准)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    num_iters = 30 # 测 30 次取平均
    with torch.no_grad():
        for _ in range(num_iters):
            run_inference(model, dummy_input, model_cfg)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    avg_time_ms = ((time.time() - start_time) / num_iters) * 1000
    return avg_time_ms

# ==========================================
# 5. 主控程序：遍历测试并绘图
# ==========================================
def main():
    img_paths = sorted(glob.glob(os.path.join(CONFIG["test_data_dir"], "*.npy")))
    if not img_paths:
        raise ValueError("找不到 .npy 测试集！")

    # 存储最终所有模型的数据
    FINAL_DATA = {}

    for model_cfg in MODELS_TO_TEST:
        name = model_cfg["display_name"]
        print(f"\n" + "="*50)
        print(f"正在测试模型: {name}")
        print("="*50)
        
        # 1. 动态实例化模型
        if model_cfg["model_type"] == "UNet":
            model = UNet(in_channels=1, out_channels=1, use_attention=model_cfg.get("use_attention", False)).to(CONFIG["device"])
        elif model_cfg["model_type"] == "DnCNN":
            model = DnCNN().to(CONFIG["device"])
            
        # 加载权重
        if os.path.exists(model_cfg["weight_path"]):
            model.load_state_dict(torch.load(model_cfg["weight_path"], map_location=CONFIG["device"], weights_only=True))
        else:
            print(f"找不到权重 {model_cfg['weight_path']}，跳过此模型！")
            continue
            
        model.eval()
        
        # 2. 精准测速
        print(f"正在进行 CUDA 推理测速...")
        latency_ms = measure_speed(model, model_cfg, CONFIG["device"])
        print(f"平均单张耗时: {latency_ms:.2f} ms")
        
        # 3. 图像去噪与指标评估
        psnr_list, ssim_list = [], []
        
        for sigma in CONFIG["noise_levels"]:
            cur_psnr, cur_ssim = [], []
            
            for img_path in tqdm(img_paths, desc=f"Sigma {sigma:.2f}", leave=False):
                raw_data = np.load(img_path)
                clean_img = raw_data.astype(np.float32)
                if clean_img.ndim == 3: clean_img = np.squeeze(clean_img)
                if clean_img.max() > 1.0 or clean_img.min() < 0.0:
                    clean_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min() + 1e-8)
                    
                mask = get_foreground_mask(clean_img)
                noisy_img = add_rician_noise(clean_img, sigma)
                
                with torch.no_grad():
                    noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0).to(CONFIG['device'])
                    # 调用统一推理引擎
                    denoised_tensor = run_inference(model, noisy_tensor, model_cfg)
                    denoised_img = denoised_tensor.squeeze().cpu().numpy()
                    denoised_img = np.clip(denoised_img, 0.0, 1.0)
                
                fg_clean = clean_img[mask]
                fg_denoised = denoised_img[mask]
                
                if len(fg_clean) > 0:
                    mse = np.mean((fg_clean - fg_denoised) ** 2)
                    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 50.0
                    ssim_val = compare_ssim(clean_img, denoised_img, data_range=1.0)
                    cur_psnr.append(psnr)
                    cur_ssim.append(ssim_val)
                    
            psnr_list.append(np.mean(cur_psnr))
            ssim_list.append(np.mean(cur_ssim))
            
        # 保存该模型结果
        FINAL_DATA[name] = {
            "psnr": psnr_list,
            "ssim": ssim_list,
            "time_ms": latency_ms,
            "color": model_cfg["color"],
            "marker": model_cfg["marker"]
        }

    # ==========================================
    # 6. 一键渲染顶级三联图
    # ==========================================
    print("\n数据收集完毕，正在渲染最终对比大图...")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 图 1: PSNR
    for name, metrics in FINAL_DATA.items():
        axes[0].plot(CONFIG["noise_levels"], metrics["psnr"], label=name, color=metrics["color"], marker=metrics["marker"], linewidth=2.5, markersize=8)
    axes[0].set_title("(a) PSNR vs. Noise Level", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Rician Noise Level ($\sigma$)", fontsize=13)
    axes[0].set_ylabel("Foreground PSNR (dB)", fontsize=13)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # 图 2: SSIM
    for name, metrics in FINAL_DATA.items():
        axes[1].plot(CONFIG["noise_levels"], metrics["ssim"], label=name, color=metrics["color"], marker=metrics["marker"], linewidth=2.5, markersize=8)
    axes[1].set_title("(b) SSIM vs. Noise Level", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("Rician Noise Level ($\sigma$)", fontsize=13)
    axes[1].set_ylabel("Structural Similarity (SSIM)", fontsize=13)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    # 图 3: 推理耗时
    names = list(FINAL_DATA.keys())
    times = [FINAL_DATA[n]["time_ms"] for n in names]
    colors = [FINAL_DATA[n]["color"] for n in names]
    bars = axes[2].bar(names, times, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    
    for bar in bars:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + (max(times)*0.02), f"{yval:.1f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    axes[2].set_title("(c) Average Inference Time per Slice", fontsize=15, fontweight='bold')
    axes[2].set_ylabel("Latency (ms)", fontsize=13)
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=11)
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(CONFIG["save_dir"], "final_multimodel_benchmark.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"评测结束！图片已保存至: {save_path}")

if __name__ == '__main__':
    main()