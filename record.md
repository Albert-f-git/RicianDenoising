# Q＆A
>1. **Q**: 随机裁剪只从一张图片中裁剪出一个图像块，数据量没有提升，如何实现防止过拟合的效果？
**A**: 虽然看起来数据量没有提升，但是随机裁剪是在线动态的，同一张图片在每个*Epoch*中裁剪出的图像块是不一样的。也就是说每次裁剪相当于从很多不同的图像块中随机选出一个，从而达到实际数据量远大于图片数量，同时又不用将所有不同剪切方法保存在本地存储。这种随即剪切搭配随机噪声使得网络学习的数据量大幅提升。

>2. **Q**: 对训练集加噪的过程应该放在裁剪前还是裁剪后？
**A**: 应该放在裁剪后。如果在裁剪前对图像进行加噪，那么在裁剪后，对图片的剩余区域加噪所进行的运算就完全浪费了。同时，根据Rician噪声的模型: 
$$ f = \sqrt{(u + n_1)^2 + n_2^2}, \quad n_1, n_2 \sim N(0, \sigma^2) $$
这个运算是针对像素进行处理的，因此裁剪后加噪没有问题。

>3. **Q**: 测试模型时未统一噪声强度和噪声随机种子，导致横向对比性能时基准线不一致。

>4. **Q**: RicianNet训练出的模型出现边缘伪影，使用复制Padding依然无法改善，反而出现四周伪影。![伪影图片](data\record\brain_test_02_comparison_sigma0.1.png)
   **解决方案1**(无效)
采用之前的办法，对数据集进行清洗，对类似下列不含组织的无效切片进行筛选(通过手动筛选出有效和无效的切片作为训练集，训练一个专门用来分类切片的CNN，准确率超过99%。)
![无效实例1](data\record\PD_axial_003.png) ![无效实例2](data\record\PD_axial_018.png) ![无效实例3](data\record\PD_axial_154.png) 
![无效实例4](data\record\PD_axial_154.png) ![无效实例5](data\record\PD_coronal_018.png)
    **最终解决**
  产生伪影的原因是全图作输入训练的过程中统一padding至尺寸$ 224\times 224 $(原因: 1. 每个batch需要同一维度，而训练集中有$ 217\times181 $和$ 181\times217 $两种尺寸；2. Unet中输入尺寸必须为$16$的倍数)，这导致*RicianNet*的训练输入中会额外添加$ 0 $边界填充，从而导致学习结果产生边缘伪影。
  解决方法: 对*RicianNet*不采用全图输入，而是采用**滑动窗口**或**随机裁剪**，既做到统一尺寸，又不影响模型效果。

------------
# DnCNN试验记录

>## 1. DnCNN basline
```python
CONFIG = {
    "experiment_name": "DnCNN_Baseline_Patch64", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "Adam",
    "dataset_mode": "full",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 4,               # 4显存最大占用1662.32，可考虑16
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-3,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments"       # 实验结果统一保存路径
}
```
**BUG1.** OutOfMemoryError
对于全图测试，`batch_size=64`超出了8GB显存的极限，重新设置`batch_size=4`。

>## 2. DnCNN basline AdamW
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

>## 3. DnCNN Sliding AdamW
### 3.1 Stride=14
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
**结果**: 在测试集表现最优（1-4），训练速度较慢。特殊测试中显著最优（1-4）。

### 3.2 Stride = 32
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

>## 4. DnCNN Random Crop AdamW
### 4.1 第一轮50Epoch
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch64_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
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
**结果**: 首轮50Epoch训练速度很快，测试集性能表现最差（1-4）。特殊测试中表现最差（1-4）。

### 4.2 第二轮150Epoch
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch64_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 64,               # 显存最大占用2127.43，可考虑128+
    "num_epochs": 150,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": "experiments\DnCNN_Random_Patch64_AdamW_20260303_073545\model_weights.pth" # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能对比第一轮显著提升，测试中PSNR追平baseline，SSIM略输于baseline。

### 4.3 第三轮300Epoch
```python
CONFIG = {
    "experiment_name": "DnCNN_Random_Patch64_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "DnCNN",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 128,               # 显存最大占用4248.43
    "num_epochs": 300,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": "experiments\DnCNN_Random_Patch64_AdamW_20260303_091959\model_weights.pth" # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能对比第二轮再次显著提升，测试中PSNR和SSIM均超越baseline。

### 4.4 patch_size=64, Epoch = 1500
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

### 4.5 patch_size=41, Epoch = 3000
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

--------------------------------------------------------
# U-net试验记录
>## 1. 全图baseline AdamW
### 1.1 lr=1e-4
```python
CONFIG = {
    "experiment_name": "Unet_Baseline_AdamW",
    "model_type": "UNet",
    "optimizer_type": "AdamW",
    "dataset_mode": "full",
    "patch_size": 64,           # 裁剪大小
    "batch_size": 16,           # 最大显存占用2309.17
    "num_epochs": 50,           # 训练轮次
    "learning_rate": 1e-4,      # 学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train",     # 训练集路径
    "save_dir": "experiments",      # 实验结果统一保存路径
    "resume_weight": None
    }
```
**结果**: 性能略弱于DnCNN baseline。

### 1.2 lr=1e-3
```python
CONFIG = {
    "experiment_name": "Unet_Baseline_AdamW",
    "model_type": "UNet",
    "optimizer_type": "AdamW",
    "dataset_mode": "full",
    "patch_size": 64,
    "batch_size": 16,           # 最大显存占用2309.17
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train",
    "save_dir": "experiments",
    "resume_weight": None
    }
```
**结果**: 出现异常横纹，效果差于lr=1e-4。![对比图](data\record\brain_test_01_comparison_sigma0.1_comp_lr.png)

>## 2. 滑动窗口 AdamW
### 2.1 Stride=14, lr=1e-4
```python
CONFIG = {
    "experiment_name": "Unet_Sliding_Patch64_Stride14_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "batch_size": 512,               # 显存最大占用5860.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: PSNR和SSIM的值都不错，但是看残差图可以看出来明显的组织结构，该模型过度平滑了。

### 2.2 Stride=32, lr=1e-4
```python
CONFIG = {
    "experiment_name": "Unet_Sliding_Patch64_Stride32_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 512,               # 显存最大占用5858.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 效果很一般。
![simulate_comp](data\record\unet_simulate_comparison_sigma0.1.png)
![real](data\record\unet_real_comparison_sigma0.1.png)

>## 3. 随机裁剪 
```python
CONFIG = {
    "experiment_name": "Unet_Random_Patch64_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "UNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 512,               # 显存最大占用5859.42
    "num_epochs": 800,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能介于步长14和32的滑动窗口之间。

------------
# RicianNet试验记录
>## 1. baseline
```python
CONFIG = {
    "experiment_name": "RicianNet_random_padding_zeros_sAdamW", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "full",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 16,               # 最大显存占用5860.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能逼近步长14的滑动窗口，但测试图片中有样例边缘出现伪影。(处理: 0 Pedding换为reflect padding；结果: 无效)

>## 2. reflect pedding
```python
CONFIG = {
    "experiment_name": "RicianNet_pedding_reflect_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "full",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 8,               # 最大显存占用5858.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 四周反而多出白色伪影，效果很差。

>## 3. Clean data / zero padding
```python
CONFIG = {
    "experiment_name": "RicianNet_padding_zeros_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "full",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 8,               # 最大显存占用5858.42
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train_valid", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 结果不理想。

>## 4. Random_patch128
```python
CONFIG = {
    "experiment_name": "RicianNet_random_Patch128_AdamW", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 128,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 16,               # 显存最大占用2123.46
    "num_epochs": 100,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能与baseline持平，并且改善了伪影问题。

------------
🚨待测试🚨
>## 5. Random_patch64
```python
CONFIG = {
    "experiment_name": "RicianNet_random_Patch64", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用4179.49
    "num_epochs": 100,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
```python
CONFIG = {
    "experiment_name": "RicianNet_random_Patch64_Epoch400", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "random",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用4179.49
    "num_epochs": 300,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": "experiments\RicianNet_random_Patch64_20260309_100743\model_weights.pth" # 可选：预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 100轮效果一般，但计算量小。再进行额外300轮训练，性能与patch128持平。

>## 6. Sliding_patch64
```python
CONFIG = {
    "experiment_name": "RicianNet_sliding_Patch64", # 实验名称，用于区分后续不同模型
    "model_type": "RicianNet",
    "optimizer_type": "AdamW",  # 'Adam' 或 'AdamW'
    "dataset_mode": "sliding",       # 'random' (随机裁剪), 'sliding' (滑动窗口), 'full' (全图填充)
    "patch_size": 64,               # 裁剪大小
    "stride": 32,                    # 滑动窗口步长
    "batch_size": 128,               # 显存最大占用4178.34
    "num_epochs": 50,               # 训练轮数
    "learning_rate": 1e-4,          # 初始学习率
    "weight_decay": 1e-4,
    "noise_range": (0, 0.3),    # Rician 噪声区间
    "data_dir": "data/processed/train", # 训练集路径
    "save_dir": "experiments",       # 实验结果统一保存路径
    "resume_weight": None # 可选: 预训练权重路径，若不使用预训练则设为 None
}
```
**结果**: 性能竟然不如全图和随机裁剪。RicianNet最深层的感受野为$ 67\times 67 $，而输入的`Patch_Size`小于这个尺寸，应该是因此影响了模型性能。