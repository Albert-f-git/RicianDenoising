本阶段试验在DnCNN的基础上进行调整尝试.
# 问题与解决
## 1. 前景掩膜(foreground mask)
之前的前景掩膜算法简单但效果差，尤其是在T1高对比度的情况下。
![Old Foreground Mask](data\record\batch_mask_test_result_1.png)
修改后的算法略微复杂但能计算出比较完美的前景。
![New Foreground Mask](data\record\batch_mask_test_result.png)



# DnCNN试验记录Stage1

>## 1. DnCNN basline AdamW
```python
CONFIG = {
    "experiment_name": "DnCNN_Baseline_Patch64", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "full",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 4,               # 显存最大占用1662.32，可考虑16
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-3,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments"       # 实验结果统一保存路径
}
```
**结果**: 在测试集上的表现: AdamW平均PSNR和SSIM均优于Adam. 在特殊测试中，AdamW的PSNR指标略差于Adam，但是SSIM指标优于Adam，降噪图片中明显保留了更多细节.

>## 2. DnCNN Sliding AdamW
### 2.1 Stride=14
```python
CONFIG = {
    "experiment_name": "DnCNN_Sliding_Patch64_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 64,               # 显存最大占用2127.43，可考虑128+
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-3,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments"       # 实验结果统一保存路径
}
```
**结果**: 在测试集表现最优，训练速度较慢。特殊测试中显著最优。

### 2.2 Stride = 32
```python
CONFIG = {
    "experiment_name": "DnCNN_Sliding_Patch64_Stride32_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用4248.43
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-3,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```

## 3. patch_size=64, Epoch = 1500
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch64_Epoch1500", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 128,               # 显存最大占用4248.43
    "num_epochs": 1500,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能近似patch64,slide32的滑动窗口，这两个模型的数据吞吐量几乎一致，但是由于随机裁剪每次都需要计算随机位置，时间略慢一点。

## 4 patch_size=41, Epoch = 3000
```python
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
```
**结果**: 和patch64的性能一致，但该模型可使用更大的`batch_size`，训练效率更高。

# DnCNN试验记录Stage2
## patch41, epoch3000, MSE, Warmup
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_MSE_Warmup_Epoch3000", 
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 窗口大小
    "batch_size": 256,              # 显存最大占用
    "num_epochs": 3000,             # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),        # Rician 噪声区间
    "use_warmup": True,
    "data_dir": "data/processed/train", 
    "save_dir": "experiments",       
    "resume_weight": None           
}
```
**结果**: PSNR略降低，SSIM略升高。

## patch41, epoch3000, MAE, No Warmup
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_MAE_NoWarmup_Epoch3000", 
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 窗口大小
    "batch_size": 256,              # 显存最大占用
    "num_epochs": 3000,             # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),        # Rician 噪声区间
    "use_warmup": False,
    "data_dir": "data/processed/train", 
    "save_dir": "experiments",       
    "resume_weight": None           
}
```
**结果**: 在前景掩膜下效果最优。

## patch41, epoch1500, MAE, Warmup
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_MAE_Warmup_Epoch3000", 
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 窗口大小
    "batch_size": 256,              # 显存最大占用
    "num_epochs": 1500,             # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),        # Rician 噪声区间
    "use_warmup": Ture,
    "data_dir": "data/processed/train", 
    "save_dir": "experiments",       
    "resume_weight": None           
}
```
**结果**: 性能比patch64,epoch1500略强。

## patch41, epoch3000, MAE, Warmup
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_MAE_Warmup_Epoch3000", 
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 窗口大小
    "batch_size": 256,              # 显存最大占用
    "num_epochs": 3000,             # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),        # Rician 噪声区间
    "use_warmup": True,
    "data_dir": "data/processed/train", 
    "save_dir": "experiments",       
    "resume_weight": "experiments\DnCNN_Random_Patch41_MAE_Warmup_Epoch1500_20260311_200344\latest_checkpoint.pth",           
}
```
**结果**: 显著改善了MAE NoWarmup 低SSIM的劣势，但是去噪效果较弱。

## patch41, epoch3000, SSIMLOSS
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch41_SSIMLOSS", 
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 41,               # 窗口大小
    "batch_size": 256,              # 显存最大占用3516.05
    "num_epochs": 3000,             # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),        # Rician 噪声区间
    "use_warmup": False,
    "data_dir": "data/processed/train", 
    "save_dir": "experiments",       
    "resume_weight": None,           
}
```

# U-net试验记录
## 全图，SSIMLoss
```python
CONFIG = {
    "experiment_name": "Unet_SSIMLoss",
    "model_type": "UNet",
    "optimizer_type": "AdamW",
    "dataset_mode": "full",
    "patch_size": 64,           # 裁剪大小
    "batch_size": 32,           # 显存最大占用4509.67
    "num_epochs": 50,           # 训练轮次
    "learning_rate": 1e-4,      # 学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train",     # 训练集路径
    "save_dir": "experiments",      # 实验结果统一保存路径
    "resume_weight": None
    }
```
**结果**: 效果一般。

## 全图，SSIMLoss, epoch150
```python
CONFIG = {
    "experiment_name": "Unet_SSIMLoss_epoch150",
    "model_type": "UNet",
    "optimizer_type": "AdamW",
    "dataset_mode": "full",
    "patch_size": 64,           # 裁剪大小
    "batch_size": 32,           # 显存最大占用
    "num_epochs": 150,           # 训练轮次
    "learning_rate": 1e-4,      # 学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,         # 是否启用热重启 (Warm Restart)
    "data_dir": "data/processed/train",     # 训练集路径
    "save_dir": "experiments",      # 实验结果统一保存路径
    "resume_weight": None
    }
```
**结果**: 优于50轮训练

## Sliding, patch64, stride14, SSIMLoss
```python
CONFIG = {
    "experiment_name": "Unet_Sliding_Patch64_Stride14_SSIMLoss", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 256,               # 显存最大占用5860.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 相较于MSE Loss，性能略微提升。

---------------------------------------------------------------------------------------------------------------------------

## Random, patch64, SSIMLoss，epoch800/1500
```python
CONFIG = {
    "experiment_name": "Unet_Random_Patch64_SSIMLoss", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 512,               # 显存最大占用5850.3
    "num_epochs": 800,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 相较于MSE Loss, 在SSIM指标上有提升；epoch1500模型的性能再次提升。


## Random, patch128, SSIMLoss, epoch1000
```python
CONFIG = {
    "experiment_name": "Unet_Random_Patch128_SSIMLoss", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 128,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用5850.3
    "num_epochs": 1000,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能接近14步长64窗口的滑动窗口模型。

# UNet with Attention
## Random, patch64, SSIMLoss, epoch1500
```python
CONFIG = {
    "experiment_name": "Unet_Attention_Random_Patch64_SSIMLoss", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 256,               # 显存最大占用
    "num_epochs": 1500,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None, # 可选: 预训练权重路径，若不使用预训练则设为 None
    "use_attention": True # 是否使用注意力机制
}
```
**结果**: 性能略优于No Attention模型。

## Sliding, patch128, SSIMLoss, epoch1000
```python
CONFIG = {
    "experiment_name": "Unet_Attention_Random_Patch128_SSIMLoss", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 128,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用
    "num_epochs": 1000,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "use_warmup": False,          # 是否使用热重启 (OneCycleLR)
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None, # 可选: 预训练权重路径，若不使用预训练则设为 None
    "use_attention": True # 是否使用注意力机制
}
```
**结果**: SSIM指标大幅提升。