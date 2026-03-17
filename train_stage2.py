import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data.dataset import MRIDenoisingDataset
from models.dncnn import DnCNN
from models.unet import UNet

# ==========================================
# 1. 核心实验配置区
# ==========================================
CONFIG = {
    "experiment_name": "Unet_Random_Patch128_epoch600_GC_SGDR", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 128,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 64,               # 显存最大占用
    "num_epochs": 600,               # 训练轮数 
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None, # 可选: 预训练权重路径，若不使用预训练则设为 None
    "use_attention": False # 是否使用注意力机制
}

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 运算引擎启动: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ==========================================
    # 目录构建
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['experiment_name']}_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)
    
    resume_path = CONFIG.get("resume_weight")
    if resume_path and os.path.exists(resume_path):
        print(f"📂 已载入预训练权重，将在【全新目录】下记录本轮特训: {exp_path}")
    else:
        print(f"📂 将在全新目录下记录训练: {exp_path}")

    # 数据集加载
    train_dataset = MRIDenoisingDataset(
        data_dir=CONFIG["data_dir"],
        patch_size=CONFIG["patch_size"],
        noise_level_range=CONFIG["noise_range"],
        mode=CONFIG["dataset_mode"]
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

    # 模型与优化器
    if CONFIG["model_type"] == "DnCNN":
        model = DnCNN().to(device)
    elif CONFIG["model_type"] == "UNet":
        use_attention = CONFIG.get("use_attention", True)
        model = UNet(in_channels=1, out_channels=1, use_attention=use_attention).to(device)
    else:
        raise ValueError(f"未知的模型类型: {CONFIG['model_type']}")
        
    criterion_l1 = nn.L1Loss() 

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["learning_rate"], 
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # ==========================================
    # 🌟 智能断点解析器 (仅读取数据，暂不创建调度器)
    # ==========================================
    start_epoch = 0
    loss_history = []
    checkpoint = None
    
    if resume_path and os.path.exists(resume_path):
        print(f"🔍 正在解析存档文件: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss_history = checkpoint.get('loss_history', [])
            print(f"✅ 成功从 Epoch {start_epoch} 恢复模型与优化器动量！")
        else:
            model.load_state_dict(checkpoint)
            print(f"⚠️ 警告: 只加载了纯权重，将从 Epoch 0 重新开始。")

    # ==========================================
    # 🌟 调度器 (Scheduler) 构建与记忆恢复中心
    # ==========================================
    remaining_epochs = CONFIG["num_epochs"] - start_epoch
    
    if CONFIG["use_warmup"]:
        # 模式 A: 热重启 (OneCycleLR) - 专为剩余轮次定制
        if remaining_epochs > 0:
            print(f"🔥 触发热重启！将为接下来的 {remaining_epochs} 轮创建新 OneCycleLR 调度曲线。")
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=CONFIG["learning_rate"], 
                epochs=remaining_epochs,             
                steps_per_epoch=len(train_loader), 
                pct_start=0.05, 
                anneal_strategy='cos'
            )
        else:
            scheduler = None
    else:
        # 模式 B: 全局余弦退火 (CosineAnnealingLR) - 最强稳压器
        print(f"🌊 启用余弦退火 + 热重启 (CosineAnnealingWarmRestarts)！")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=len(train_loader) * 10,  # 每10个epoch重启一次
            T_mult=2,
            eta_min=1e-6
        )
        # 核心防雷：如果从断点恢复，必须唤醒调度器的记忆，否则滑梯会归零！
        if checkpoint and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"✅ 成功恢复余弦退火曲线的历史进度！")

    # ==========================================
    # 2. 训练主循环
    # ==========================================
    start_time = time.time()
    current_epoch = start_epoch
    print(f"\n🚀 开始训练 [{CONFIG['experiment_name']}] ... (按 Ctrl+C 可中断并保存)")
    
    try:
        for epoch in range(start_epoch, CONFIG["num_epochs"]):
            current_epoch = epoch
            model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", leave=False)
            
            for noisy_imgs, clean_imgs in progress_bar:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                
                optimizer.zero_grad()
                denoised_imgs = model(noisy_imgs)
                
                # 计算复合 Loss
                loss_l1 = criterion_l1(denoised_imgs, clean_imgs)
                loss_ssim = 1.0 - ssim(denoised_imgs, clean_imgs, data_range=1.0, size_average=True)
                loss = 0.84 * loss_l1 + 0.16 * loss_ssim

                # 反向传播与防爆处理
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                
                # 步进调度器 (按 Batch)
                if scheduler:
                    scheduler.step()
                
                epoch_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({'loss': f"{loss.item():.5f}", 'lr': f"{current_lr:.2e}"})
                
            avg_epoch_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1:03d}/{CONFIG['num_epochs']:03d}] - Avg Loss: {avg_epoch_loss:.5f} | LR: {current_lr:.2e}")

            # 正常 Epoch 存档
            torch.save({
                'epoch': current_epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss_history': loss_history
            }, os.path.join(exp_path, "latest_checkpoint.pth"))

    except KeyboardInterrupt:
        print("\n🛑 收到中断信号 (Ctrl+C)！正在紧急执行现场保护...")
        torch.save({
            'epoch': current_epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': loss_history
        }, os.path.join(exp_path, "latest_checkpoint.pth"))
        print(f"🔒 中断状态已封存: {os.path.join(exp_path, 'latest_checkpoint.pth')}")

    # ==========================================
    # 3. 终极后处理
    # ==========================================
    finally:
        total_time = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

        print("\n📊 正在生成实验报告与可视化图表...")
        
        torch.save(model.state_dict(), os.path.join(exp_path, "model_weights_final.pth"))

        if len(loss_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b')
            plt.title(f"Training Loss Curve - {CONFIG['experiment_name']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(exp_path, "loss_curve.png"))
            plt.close()

        report = {
            "experiment_name": CONFIG["experiment_name"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "hardware": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "peak_vram_mb": round(peak_vram_mb, 2)
            },
            "hyperparameters": CONFIG,
            "results": {
                "total_epochs_run": len(loss_history),
                "total_time_minutes": round(total_time / 60, 2),
                "final_epoch_loss": loss_history[-1] if loss_history else None,
                "loss_history": loss_history
            }
        }
        
        with open(os.path.join(exp_path, "training_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        print(f"🎉 所有结果及报告已成功落地至: {exp_path}")

if __name__ == '__main__':
    train()