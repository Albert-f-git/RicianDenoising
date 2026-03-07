import os
import cv2  
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  
from skimage.metrics import structural_similarity as ssim

# 导入你的模型
from models.dncnn import DnCNN
from models.unet import UNet
from models.RicianNet import RicianNet

# ==========================================
# 1. 批量测试配置区
# ==========================================
CONFIG = {
    "models": {
        # "Adam_Baseline": "experiments\DnCNN_Baseline_Patch64_Adam_20260302_190727\model_weights.pth",
        # "DnCNN_Optimized": "experiments\DnCNN_Baseline_AdamW_20260302_200103\model_weights.pth",
        # "DnCNN_Random_Patch64": "experiments\DnCNN_Random_Patch64_AdamW_20260303_095649\model_weights.pth",
        "DnCNN_Sliding_Patch64_Slide14": "experiments\DnCNN_Sliding_Patch64_Stride14_AdamW_20260302_230306\model_weights.pth",
        # "DnCNN_Sliding_Patch64_Slide32": "experiments\DnCNN_Sliding_Patch64_Stride32_AdamW_20260306_094004\model_weights.pth",
        # "UNet_Baseline1e-4": "experiments\\Unet_Baseline_AdamW_1e-4_20260305_125730\model_weights.pth",
        # "UNet_Baseline1e-3": "experiments\\Unet_Baseline_AdamW_1e-3_20260306_114041\model_weights.pth",
        "UNet_Sliding_Patch64_Slide14": "experiments\\Unet_Sliding_Patch64_Stride14_AdamW_20260305_133805\model_weights.pth",
        # "UNet_Sliding_Patch64_Slide32": "experiments\\Unet_Sliding_Patch64_Stride32_AdamW_20260306_121908\model_weights.pth",
        # "UNet_Random_Patch64": "experiments\\Unet_Random_Patch64_AdamW_20260306_135850\model_weights.pth", # Unet_Random_Patch64_AdamW_Epoch800
        "RicianNet_Baseline": "experiments\RicianNet_baseline_AdamW_20260306_161247\model_weights.pth", # RicianNet_baseline_AdamW
        # "RicianNet_Padded_Reflect": "experiments\RicianNet_pedding_reflect_AdamW_20260306_210919\model_weights.pth", # RicianNet_pedding_reflect_AdamW
        # "RicianNet_padding_zeros_cleaned": "experiments\RicianNet_padding_zeros_cleaned_AdamW_20260307_171156\model_weights.pth", # RicianNet_padding_zeros_cleaned_AdamW
        "RicianNet_Random_Patch128": "experiments\RicianNet_random_Patch128_AdamW_20260307_203419\model_weights.pth" # RicianNet_random_Patch128_AdamW_Epoch100
    },
    "images": [
        {"path": "data\special_test\\brain1_simulate.png", "type": "simulated"}, 
        {"path": "data\special_test\\brain2_simulate.png", "type": "simulated"},
        {"path": "data\special_test\\T1_axial_047.png", "type": "simulated"}, 
        {"path": "data\special_test\\T1_sagittal_079.png", "type": "simulated"}, 
        {"path": "data\special_test\\T2_sagittal_092.png", "type": "simulated"}  
               
    ],
    # "images": [
    #     {"path": "data\special_test\\brain1_real.png", "type": "real"},
    #     {"path": "data\special_test\\brain2_real.png", "type": "real"},
    #     {"path": "data\special_test\\brain3_real.png", "type": "real"},
    #     {"path": "data\special_test\\brain4_real.png", "type": "real"},
    #     {"path": "data\special_test\\brain5_real.png", "type": "real"}
    # ],
    "noise_level": 0.1,          
    "save_dir": "special_results\\3models_comparison" 
}

def add_rician_noise(clean_img, sigma):
    np.random.seed(45)  
    n1 = np.random.normal(0, sigma, clean_img.shape)
    n2 = np.random.normal(0, sigma, clean_img.shape)
    return np.sqrt((clean_img + n1)**2 + n2**2)

# ================= 新增：前景掩模生成 =================
def get_foreground_mask(clean_img, threshold=0.05):
    """根据干净图像生成前景（脑组织）掩模"""
    mask = (clean_img > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)
# ======================================================

# ================= 更新：支持掩模的指标计算 =================
def calculate_metrics(clean, denoised, mask=None):
    if mask is not None:
        mse = np.mean((clean[mask] - denoised[mask])**2)
        p_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        # 计算前景 SSIM：先算全图，再提取前景区域求均值
        _, ssim_map = ssim(clean, denoised, data_range=1.0, full=True)
        s_val = np.mean(ssim_map[mask])
    else:
        mse = np.mean((clean - denoised)**2)
        p_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        s_val = ssim(clean, denoised, data_range=1.0)
    
    return p_val, s_val
# ======================================================

def multi_model_inference(sigma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动多模型批量测试引擎: {device}")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 预加载模型
    loaded_models = {}
    print("📦 正在加载模型字典...")
    for model_name, weight_path in CONFIG["models"].items():
        if not os.path.exists(weight_path):
            print(f"⚠️ 警告: 找不到权重 {weight_path}，已跳过模型 [{model_name}]")
            continue
            
        if "UNet" in model_name:
            model = UNet(in_channels=1, out_channels=1).to(device)
        elif "RicianNet" in model_name:
            model = RicianNet   ().to(device)
        else:
            model = DnCNN().to(device)
            
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        loaded_models[model_name] = model
        print(f"   ✅ 模型 [{model_name}] 加载成功！")

    if not loaded_models:
        raise RuntimeError("没有成功加载任何模型！")

    # 逐图处理
    for img_info in CONFIG["images"]:
        img_path = img_info["path"]
        img_type = img_info["type"]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        if not os.path.exists(img_path):
            continue
            
        print(f"\n🖼️ 正在处理图片: {img_path} ({img_type} 模式)")
        
        input_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        noisy_metrics = {}
        if img_type == "simulated":
            clean_np = input_np
            noisy_np = add_rician_noise(clean_np, CONFIG["noise_level"])
            
            # 提取前景掩模并计算含噪图的全图/前景指标
            fg_mask = get_foreground_mask(clean_np)
            n_all_p, n_all_s = calculate_metrics(clean_np, noisy_np)
            n_fg_p, n_fg_s = calculate_metrics(clean_np, noisy_np, mask=fg_mask)
            noisy_metrics = {'all_p': n_all_p, 'all_s': n_all_s, 'fg_p': n_fg_p, 'fg_s': n_fg_s}
        else:
            clean_np = None
            noisy_np = input_np
            fg_mask = None
            
        noisy_tensor = torch.tensor(noisy_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        results_dict = {}
        
        with torch.no_grad():
            for model_name, model in loaded_models.items():
                denoised_tensor = model(noisy_tensor)
                denoised_np = np.clip(denoised_tensor.squeeze().cpu().numpy(), 0.0, 1.0)
                
                if img_type == "simulated":
                    # 计算模型输出的全图和前景指标
                    c_all_p, c_all_s = calculate_metrics(clean_np, denoised_np)
                    c_fg_p, c_fg_s = calculate_metrics(clean_np, denoised_np, mask=fg_mask)
                    metrics = {'all_p': c_all_p, 'all_s': c_all_s, 'fg_p': c_fg_p, 'fg_s': c_fg_s}
                    results_dict[model_name] = {"img": denoised_np, "metrics": metrics}
                else:
                    results_dict[model_name] = {"img": denoised_np, "metrics": None}

        # ==========================================
        # 4. 动态绘图 (两行排版：图相 + 残差图)
        # ==========================================
        num_models = len(loaded_models)
        num_cols = num_models + 2 if img_type == "simulated" else num_models + 1
        
        # 适当拉高图表，以适应更长的多行标题
        fig, axes = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 11))
        RESIDUAL_VMAX = 0.3 

        if img_type == "simulated":
            # --- 第一行：解剖图像 ---
            axes[0, 0].imshow(clean_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title("Ground Truth\n(Clean Reference)\n", fontsize=12, pad=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f"Noisy (Sigma={CONFIG['noise_level']})\nALL: {noisy_metrics['all_p']:.2f}dB | {noisy_metrics['all_s']:.4f}\nFG: {noisy_metrics['fg_p']:.2f}dB | {noisy_metrics['fg_s']:.4f}", fontsize=11, pad=12)
            axes[0, 1].axis('off')
            
            for idx, (model_name, res) in enumerate(results_dict.items()):
                m = res["metrics"]
                axes[0, idx + 2].imshow(res["img"], cmap='gray', vmin=0, vmax=1)
                axes[0, idx + 2].set_title(f"Model: {model_name}\nALL: {m['all_p']:.2f}dB | {m['all_s']:.4f}\nFG: {m['fg_p']:.2f}dB | {m['fg_s']:.4f}", fontsize=11, pad=12)
                axes[0, idx + 2].axis('off')

            # --- 第二行：残差图 (Error Map) ---
            axes[1, 0].imshow(np.abs(clean_np - clean_np), cmap='gray', vmin=0, vmax=RESIDUAL_VMAX)
            axes[1, 0].set_title("Residual: 0 (Reference)", fontsize=11, pad=12)
            axes[1, 0].axis('off')

            res_noisy = np.abs(noisy_np - clean_np)
            axes[1, 1].imshow(res_noisy, cmap='gray', vmin=0, vmax=RESIDUAL_VMAX)
            axes[1, 1].set_title("Absolute Error\n(|Noisy - GT|)", fontsize=11, pad=12)
            axes[1, 1].axis('off')

            for idx, (model_name, res) in enumerate(results_dict.items()):
                res_model = np.abs(res["img"] - clean_np)
                axes[1, idx + 2].imshow(res_model, cmap='gray', vmin=0, vmax=RESIDUAL_VMAX)
                axes[1, idx + 2].set_title(f"Absolute Error\n(|{model_name} - GT|)", fontsize=11, pad=12)
                axes[1, idx + 2].axis('off')

        else:
            # --- 盲降噪 第一行：解剖图像 ---
            axes[0, 0].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title("Real Noisy Input\n(Ground Truth Unavailable)\n", fontsize=11, pad=12)
            axes[0, 0].axis('off')
            
            for idx, (model_name, res) in enumerate(results_dict.items()):
                axes[0, idx + 1].imshow(res["img"], cmap='gray', vmin=0, vmax=1)
                axes[0, idx + 1].set_title(f"Model: {model_name}\nALL: N/A | N/A\nFG: N/A | N/A", fontsize=11, pad=12)
                axes[0, idx + 1].axis('off')

            # --- 盲降噪 第二行：提取的噪声 (Extracted Noise) ---
            axes[1, 0].imshow(np.abs(noisy_np - noisy_np), cmap='gray', vmin=0, vmax=RESIDUAL_VMAX)
            axes[1, 0].set_title("Residual: 0 (Reference)", fontsize=11, pad=12)
            axes[1, 0].axis('off')

            for idx, (model_name, res) in enumerate(results_dict.items()):
                extracted_noise = np.abs(noisy_np - res["img"])
                axes[1, idx + 1].imshow(extracted_noise, cmap='gray', vmin=0, vmax=RESIDUAL_VMAX)
                axes[1, idx + 1].set_title(f"Extracted Method Noise\n(|Input - {model_name}|)", fontsize=11, pad=12)
                axes[1, idx + 1].axis('off')

        plt.tight_layout()
        # 加大 hspace，防止长标题遮挡上一行的图片
        fig.subplots_adjust(top=0.92, hspace=0.4) 
        
        save_path = os.path.join(CONFIG["save_dir"], f"{base_name}_sigma{sigma}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   ✅ 双行残差对比图已生成: {save_path}")
        plt.close(fig)

    print("\n🎉 所有图像及残差分析处理完毕！")

if __name__ == '__main__':
    multi_model_inference(sigma=CONFIG["noise_level"])