# 公平方法横向对比

本目录给出 GT / Noisy / DnCNN / RicianNet / Attention U-Net 在统一测试条件下的定量与定性结果。

## 统一实验条件

- 测试集：`data/processed/test` 中全部 334 张 MRI 切片。
- 噪声：Rician 噪声，$\sigma=0.05,0.10,0.15,0.20,0.25,0.30$。
- 随机性：固定种子 `20260901`；每个“切片 + 噪声等级”只对应一份确定的 Noisy 输入，所有方法使用完全相同的输入。
- 推理：完整切片推理；仅在右侧和下侧零填充到 16 的倍数，推理后裁回原尺寸。
- 数据范围：统一归一化至 $[0,1]$。
- 评价区域：PSNR 与 SSIM 均在由 GT 生成的同一前景掩模内统计。SSIM 先生成局部 SSIM map，再对掩模内像素求均值。
- 汇总方式：先逐切片计算指标，再对 334 张切片取算术平均；标准差见 `summary_metrics.csv`。
- 定性样本：`T1_axial_047.npy`；所有方法和噪声等级使用相同 ROI `(x=87, y=121, w=52, h=52)`。

> “统一实验条件”指统一的测试数据、噪声 realization、预处理、推理方式和指标实现。当前比较复用了各模型已有权重，因此各方法的原始训练 patch、优化器和训练轮数并不完全相同。Attention U-Net 使用的是与当前代码匹配的 Left-Attention U-Net 权重。

## 平均 PSNR / SSIM

| Method | σ=0.05 | σ=0.10 | σ=0.15 | σ=0.20 | σ=0.25 | σ=0.30 |
|---|---:|---:|---:|---:|---:|---:|
| Noisy | 26.09 / 0.7805 | 20.25 / 0.5800 | 16.99 / 0.4495 | 14.76 / 0.3571 | 13.09 / 0.2876 | 11.81 / 0.2334 |
| DnCNN | **33.07 / 0.9601** | **29.22 / 0.9125** | **26.66 / 0.8594** | **24.58 / 0.8027** | **22.73 / 0.7419** | **20.85 / 0.6717** |
| RicianNet | 30.71 / 0.9414 | 28.02 / 0.8900 | 26.10 / 0.8379 | 24.39 / 0.7842 | 22.66 / 0.7253 | 20.58 / 0.6532 |
| Attention U-Net | 31.41 / 0.9459 | 27.81 / 0.8902 | 25.57 / 0.8338 | 23.82 / 0.7773 | 22.10 / 0.7155 | 20.07 / 0.6393 |

每个单元格格式为 `PSNR (dB) / SSIM`。在当前已有权重和统一测试协议下，DnCNN 在全部六个噪声等级上均取得最高平均 PSNR 与 SSIM；RicianNet 在中高噪声区间通常位列第二。

## 文件说明

- `horizontal_method_comparison.csv`：适合 Excel/绘图软件的横向结果表。
- `horizontal_method_comparison.tex`：可直接插入论文的 LaTeX `table*`。
- `summary_metrics.csv`：各方法、各噪声等级的均值与样本标准差。
- `per_image_metrics.csv`：逐切片指标，便于显著性检验或误差分析。
- `psnr_ssim_vs_noise.png`：PSNR/SSIM 随噪声等级变化曲线。
- `denoising_comparison_sigma_*.png`：GT、Noisy、三种方法的整图与统一 ROI 放大图。
- `benchmark_metadata.json`：种子、权重路径、ROI、设备等复现实验元数据。

复现实验：

```powershell
& 'D:\Miniconda\envs\RicianDenoising\python.exe' 'D:\Albert\Desktop\projects\RicianDenoising\fair_method_benchmark.py'
```
