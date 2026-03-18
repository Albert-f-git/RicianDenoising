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
        # "DnCNN_Optimized": "experiments\DnCNN_Baseline_AdamW_20260302_200103\model_weights.pth",
        # "DnCNN_Random_Patch64_Epoch1500": "experiments\DnCNN_Random_Patch64_Epoch1500_20260310_123758\model_weights.pth", # DnCNN_Random_Patch64_Epoch1500
        # "DnCNN_Random_Patch41": "experiments\DnCNN_Random_Patch41_Epoch3000_20260310_152419\model_weights.pth", # DnCNN_Random_Patch41_Epoch3000  
        # "DnCNN_Random_Patch41_MSE_Warmup": "experiments\DnCNN_Random_Patch41_MSE_Warmup_Epoch3000_20260311_153329\model_weights_final.pth", # DnCNN_Random_Patch41_MSE_Warmup_Epoch3000
        # "DnCNN_Random_Patch41_MAE_NoWarmup": "experiments\DnCNN_Random_Patch41_MAE_NoWarmup_Epoch3000_20260311_180003\model_weights_final.pth", # DnCNN_Random_Patch41_MAE_NoWarmup_Epoch3000
        # "DnCNN_Random_Patch41_MAE_Warmup_Epoch1500": "experiments\DnCNN_Random_Patch41_MAE_Warmup_Epoch1500_20260311_200344\model_weights_final.pth", # DnCNN_Random_Patch41_MAE_Warmup_Epoch1500
        # "DnCNN_Random_Patch41_MAE_Warmup": "experiments\DnCNN_Random_Patch41_MAE_Warmup_Epoch3000_20260311_212010\model_weights_final.pth", # DnCNN_Random_Patch41_MAE_Warmup_Epoch3000
        # "DnCNN_Random_Patch41_SSIMLOSS": "experiments\DnCNN_Random_Patch41_SSIMLOSS_20260312_150403\model_weights_final.pth", # DnCNN_Random_Patch41_SSIMLOSS_Epoch3000
        # "DnCNN_Sliding_Patch64_Slide14": "experiments\DnCNN_Sliding_Patch64_Stride14_AdamW_20260302_230306\model_weights.pth", # DnCNN_Sliding_Patch64_Stride14_AdamW
        # "DnCNN_Sliding_Patch64_Slide32": "experiments\DnCNN_Sliding_Patch64_Stride32_AdamW_20260306_094004\model_weights.pth", # DnCNN_Sliding_Patch64_Stride32_AdamW
        # "UNet_baseline": "experiments\\Unet_Baseline_AdamW_1e-4_20260305_125730\model_weights.pth", # Unet_Baseline
        # "UNet_SSIMLoss": "experiments\\Unet_SSIMLoss_20260312_182138\model_weights_final.pth", # Unet_SSIMLoss
        # "UNet_SSIMLoss_epoch150": "experiments\\Unet_SSIMLoss_epoch150_20260313_173033\model_weights_final.pth", # Unet_SSIMLoss_epoch150
        # "UNet_Sliding_Patch64_Slide14": "experiments\\Unet_Sliding_Patch64_Stride14_AdamW_20260305_133805\model_weights.pth", # Unet_Sliding_Patch64_Stride14_AdamW
        # "UNet_Sliding_Patch64_Slide14": "experiments\\Unet_Sliding_Patch64_Stride14_SSIMLoss_20260312_195618\model_weights_final.pth", # Unet_Sliding_Patch64_Stride14_SSIMLoss
        # "UNet_Random_Patch64": "experiments\\Unet_Random_Patch64_AdamW_20260306_135850\model_weights.pth", # Unet_Random_Patch64
        # "UNet_Random_Patch64_SSIMLoss": "experiments\\Unet_Random_Patch64_SSIMLoss_20260313_123947\model_weights_final.pth", # Unet_Random_Patch64_SSIMLoss
        # "UNet_Random_Patch64_epoch1500": "experiments\\Unet_Random_Patch64_SSIMLoss_epoch1500_20260313_153944\model_weights_final.pth", # Unet_Random_Patch64_SSIMLoss_epoch1500
        # "UNet_Attention_Random_Patch64_epoch1500": "experiments\\Unet_Attention_Random_Patch64_SSIMLoss_20260316_182600\model_weights_final.pth", # Unet_Attention_Random_Patch64_SSIMLoss_epoch1500
        # "UNet_Random_Patch128_epoch1000": "experiments\\Unet_Random_Patch128_SSIMLoss_20260313_181510\model_weights_final.pth", # Unet_Random_Patch128_SSIMLoss_epoch1000
        # "UNet_Random_Patch128_epoch600_GC_SGDR": "experiments\\Unet_Random_Patch128_epoch600_GC_SGDR_20260317_180641\model_weights_final.pth", # Unet_Random_Patch128_epoch600_GC_SGDR
        # "UNet_Attention_Random_Patch128_epoch1000": "experiments\\Unet_Attention_Random_Patch128_SSIMLoss_20260316_212839\model_weights_final.pth", # Unet_Attention_Random_Patch128_SSIMLoss_epoch1000
        # "UNet_Attention_Random_Patch128_epoch600_GC_SGDR": "experiments\\Unet_Attention_Random_Patch128_epoch600_GC_SGDR_20260317_145634\model_weights_final.pth", # Unet_Attention_Random_Patch128_epoch600_GC_SGDR
        "UNet_LeftAttention_Random_Patch128_epoch600_GC_SGDR": "experiments\\Unet_LeftAttention_Random_Patch128_epoch600_GC_SGDR_20260318_151922\model_weights_final.pth", # Unet_LeftAttention_Random_Patch128_epoch600_GC_SGDR
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
    "noise_level": 0.10,          
    "save_dir": "special_results\\UNet_leftAttention_comparison", 
}

def add_rician_noise(clean_img, sigma):
    np.random.seed(45)  
    n1 = np.random.normal(0, sigma, clean_img.shape)
    n2 = np.random.normal(0, sigma, clean_img.shape)
    return np.sqrt((clean_img + n1)**2 + n2**2)

# ================= 前景掩模生成 =================
def get_foreground_mask(clean_img):
    """使用 面积过滤 + 椭圆闭运算 打造绝对纯净的头部掩模"""
    # 1. 基础二值化
    img_8u = (clean_img * 255).astype(np.uint8)
    _, mask_thresh = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ========================================================
    # 面积过滤器
    # ========================================================
    # 找出二值化图中的所有独立岛屿
    contours_raw, _ = cv2.findContours(mask_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_raw:
        return np.zeros_like(clean_img, dtype=bool)
        
    # 计算最大岛屿（大脑主体）的面积
    max_area = max([cv2.contourArea(c) for c in contours_raw])
    mask_clean = np.zeros_like(mask_thresh)
    
    # 遍历所有岛屿，只保留面积大于大脑 1% 的部分（头皮会保留，离散鬼影会被直接抛弃）
    for c in contours_raw:
        if cv2.contourArea(c) > max_area * 0.01:
            # thickness=-1 会顺便把内部的孔洞（如脑室）提前填实
            cv2.drawContours(mask_clean, [c], -1, 255, thickness=-1)
            
    # ========================================================

    # 2. 护城河填充 (Padding)
    pad = 30
    mask_padded = cv2.copyMakeBorder(mask_clean, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # ========================================================
    # 解剖学椭圆包裹 (拒绝方形核的对角线越界)
    # ========================================================
    # 使用 17x17 的椭圆核，完美缝合头皮间隙，且绝对平滑
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask_bridged = cv2.morphologyEx(mask_padded, cv2.MORPH_CLOSE, kernel_bridge)
    # ========================================================

    # 3. 寻找最外围轮廓 (保鲜膜打包)
    contours_ext, _ = cv2.findContours(mask_bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_ext:
        return np.zeros_like(clean_img, dtype=bool)
        
    # 4. 提取并填实最外围最大轮廓
    largest_contour = max(contours_ext, key=cv2.contourArea)
    mask_filled_padded = np.zeros_like(mask_padded)
    cv2.drawContours(mask_filled_padded, [largest_contour], -1, 255, thickness=-1)
    
    # 5. 裁掉黑边，恢复原状
    h, w = mask_thresh.shape
    mask_final = mask_filled_padded[pad : pad+h, pad : pad+w]
    
    return (mask_final > 0).astype(bool)
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
            # 根据模型名称判断是否使用注意力机制
            use_attention = "Attention" in model_name
            model = UNet(in_channels=1, out_channels=1, use_attention=use_attention).to(device)
        elif "RicianNet" in model_name:
            model = RicianNet().to(device)
        else:
            model = DnCNN().to(device)
            
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        loaded_models[model_name] = model
        attention_status = "✅" if ("UNet" in model_name and "Attention" in model_name) else ""
        print(f"   ✅ 模型 [{model_name}] 加载成功！{attention_status}")

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