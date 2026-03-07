# 📂 项目结构 (Project Structure)
```text
MRI-Denoising/
├── data/                           # 🧠 数据集存储目录 (Dataset Directory)
│   ├── processed/                  # 预处理后的 Numpy 切片数据 (用于训练)
│   │   └── train/                  # 包含大量独立切片的训练集
│   └── special_test/               # 🎯 专门用于模型测试与评估的精选图像
│   |   ├── brain1_simulated.png    # 带有 Ground Truth 的模拟测试集 (用于量化指标)
│   |   ├── brain_test_03.png       # 真实的临床带噪图像 (用于盲降噪效果评估)
│   |   └── ...
│   └── dataset.py                  # 📦 数据加载器 (Data Loader & Preprocessing)
|                                   # -> 支持三种动态调度模式: 'sliding'(滑动窗口),
|                                   'random'(随机裁剪), 'full'(全图反射填充) 
|                                   # -> 包含底层物理级别的 Rician 噪声动态注入模块
|
├── models/                         # 🧬 核心算法与网络架构 (Neural Network Architectures)
│   ├── dncnn.py                    # 基准模型：DnCNN (纯卷积, 擅长局部特征提取)
│   ├── unet.py                     # 医学标杆模型：U-Net (带 padding=1 的改进版编解码结构)
│   └── rician_net.py               # 🌟 创新模型：RicianNet (针对莱斯噪声设计的无池化、大感受野、带反射填充的双子网结构)
│
├── experiments/                    # 💾 训练过程与权重输出 (Training Checkpoints)
│   ├── RicianNet_Random_Patch128.../ # 根据 CONFIG 动态生成的单次实验记录
│   │   ├── model_weights.pth       # 保存的最佳模型权重
│   │   └── log.txt                 # 训练日志 (可选)
│   └── ...
│
├── special_results/                # 📊 推理与可视化分析输出 (Inference & Visualization)
│                                                                 
├── train.py                        # 🚀 统一训练引擎 (Unified Training Script)
│                                   # -> 支持快速切换模型、优化器 (AdamW) 和超参数配置
│
├── special_test.py                 # 👁️ 高级推理与病理级可视化脚本 (Advanced Evaluation Engine)
│                                   # -> 支持多模型批量加载与并行推理
│                                   # -> 内置 SSIM/PSNR (支持全局与脑组织前景掩模分离计算)
│                                   # -> 搭载自动画图模块：残差分析图 (Method Noise) 
│
└── README.md                       # 📖 项目说明文档

```
## 1. 数据集
本项目采用BrainWeb上的模拟MRI图像作为数据集，选用 PN0 (0% 噪声) 与 RF0 (0% 强度不均匀性) 的理想模型作为 Ground Truth (GT)，包含T1、T2、PD三种模态，像素大小均为1×1mm，分辨率: $181 \times 217 \times 181$ 像素。

## 2. 预处理pipeline

### 2.1 多角度2D切片提取 (Multi-planar Slicing)
为了使模型能够学习到大脑在 3D 空间中的解剖结构特征，并成倍增加训练样本量，本项目采用了多平面重构 (MPR) 策略：
- **横断位 (Axial)**: 沿 Z 轴提取切片。
- **冠状位 (Coronal)**: 沿 Y 轴提取切片。
- **矢状位 (Sagittal)**: 沿 X 轴提取切片。
- **无效数据清洗**: 自动剔除平均像素强度 $< 0.05$ 的全黑背景切片，确保模型专注于脑组织特征。
### 2.2 归一化 (Normalization)
将所有原始 .mnc 数据均转换为 float32 张量，并进行全局 Min-Max 归一化，将像素值严格映射至 $[0, 1]$ 区间，以防止学习训练过程中的梯度爆炸。
### 2.3 坐标系校正 (Orientation Correction)
由于医学影像坐标系（LPS/RAS）与计算机视觉张量坐标系（Top-Left Origin）存在差异，预处理阶段通过`np.flipud`进行了上下翻转校正，确保输入神经网络的数据符合人类解剖学视觉直觉。

## 3. 数据集类 (Dataset)
按照PyTorch的规范编写了一个`Dataset`类，实现了在线添加Rician噪声，从而避免了对加噪图像额外的本地存储。
### 3.1 Rician噪声
- **数学模型**：$$ f = \sqrt{(u + n_1)^2 + n_2^2}, \quad n_1, n_2 \sim N(0, \sigma^2) $$
将噪声强度设定为$ \sigma \in [0, 0.3] $.
### 3.2 随机图块裁剪 (Random Patching)
为了抑制过拟合并提高显存利用率，采用 Patch-based Training 策略：从原始切片中随即裁剪出$ 64\times 64 $的局部图块，从而显著提升数据量，并且使模型更聚焦于局部纹理而非整体轮廓。