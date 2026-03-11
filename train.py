import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 导入你写的模块
from data.dataset import MRIDenoisingDataset
from models.dncnn import DnCNN
from models.unet import UNet
from models.RicianNet import RicianNet
# ==========================================
# 1. 核心实验配置区 (Centralized Configuration)
# ==========================================
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_Epoch3000", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 裁剪大小
    "batch_size": 256,               # 显存最大占用3517.69
    "num_epochs": 3000,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}

def train():
    # ==========================================
    # 2. 环境与目录初始化
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 运算引擎启动: {device}")
    if torch.cuda.is_available():
        print(f"💻 显卡型号: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats() # 重置显存峰值记录器

    # 创建带有时间戳的专属实验文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['experiment_name']}_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)

    # ==========================================
    # 3. 加载数据与模型
    # ==========================================
    if "stride" in CONFIG:
        train_dataset = MRIDenoisingDataset(
        data_dir=CONFIG["data_dir"],
        patch_size=CONFIG["patch_size"],
        stride=CONFIG["stride"],
        noise_level_range=CONFIG["noise_range"],
        mode=CONFIG["dataset_mode"]
        )
    else:
        train_dataset = MRIDenoisingDataset(
        data_dir=CONFIG["data_dir"],
        patch_size=CONFIG["patch_size"],
        noise_level_range=CONFIG["noise_range"],
        mode=CONFIG["dataset_mode"]
        )
    # num_workers=4 可以利用你 32GB 的大内存来加速数据读取
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

    # 根据 CONFIG 动态加载模型
    if CONFIG["model_type"] == "DnCNN":
        model = DnCNN().to(device)
    elif CONFIG["model_type"] == "UNet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif CONFIG["model_type"] == "RicianNet":
        model = RicianNet().to(device)
    else:
        raise ValueError(f"未知的模型类型: {CONFIG['model_type']}")

    # 断点续训逻辑
    resume_path = CONFIG.get("resume_weight")
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 正在加载历史权重进行断点续训: {resume_path}")
        # weights_only=True 保证安全加载
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        print("✅ 历史训练已载入，准备继续训练！")

    criterion = nn.MSELoss()
    # 根据 CONFIG 动态选择优化器
    if CONFIG["optimizer_type"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    elif CONFIG["optimizer_type"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    else:
        raise ValueError("未知的优化器类型！请选择 'Adam' 或 'AdamW'")
    # ==========================================
    # 4. 训练主循环 (带进度条和显存监控)
    # ==========================================
    loss_history = []
    start_time = time.time()
    
    print(f"\n🔥 开始训练 [{CONFIG['experiment_name']}] ...")
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 包装 train_loader 实现实时进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", leave=False)
        
        for noisy_imgs, clean_imgs in progress_bar:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 前向传播与优化
            optimizer.zero_grad()
            denoised_imgs = model(noisy_imgs)
            loss = criterion(denoised_imgs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 实时更新进度条后缀（展示当前 batch 的 loss）
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        # 打印 Epoch 总结
        print(f"Epoch [{epoch+1:03d}/{CONFIG['num_epochs']:03d}] - Avg Loss: {avg_epoch_loss:.6f}")

    # ==========================================
    # 5. 训练后处理 (统计、画图、保存)
    # ==========================================
    total_time = time.time() - start_time
    
    # 获取显存占用 (转换为 MB)
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    print("\n✅ 训练完成！正在生成实验报告...")
    print(f"⏱️ 总耗时: {total_time/60:.2f} 分钟")
    print(f"💾 峰值显存占用: {peak_vram_mb:.2f} MB")

    # 5.1 保存模型权重
    model_save_path = os.path.join(exp_path, "model_weights.pth")
    torch.save(model.state_dict(), model_save_path)

    # 5.2 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, CONFIG["num_epochs"] + 1), loss_history, marker='o', color='b', label='Train Loss (MSE)')
    plt.title(f"Training Loss Curve - {CONFIG['experiment_name']}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_curve_path = os.path.join(exp_path, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()

    # 5.3 生成标准化实验记录文档 (JSON)
    report = {
        "experiment_name": CONFIG["experiment_name"],
        "timestamp": timestamp,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "peak_vram_mb": round(peak_vram_mb, 2)
        },
        "hyperparameters": CONFIG,
        "results": {
            "total_time_seconds": round(total_time, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "final_epoch_loss": loss_history[-1],
            "loss_history": loss_history
        }
    }
    
    report_path = os.path.join(exp_path, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"📁 实验结果已全部保存在: {exp_path}")

if __name__ == '__main__':
    train()