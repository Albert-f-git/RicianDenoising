import nibabel as nib
import numpy as np
import os
import random
from tqdm import tqdm

def process_and_split_mri(mnc_path, base_save_dir, mod_name, test_ratio=0.2):
    """
    读取 3D MRI 数据，多角度切片，并划分为训练集和测试集
    """
    # 创建训练集和测试集的保存目录
    train_dir = os.path.join(base_save_dir, 'train')
    test_dir = os.path.join(base_save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"📦 正在加载 {mod_name} 数据: {mnc_path}")
    img = nib.load(mnc_path)
    data = img.get_fdata()
    
    # 工业界标准：全局 Min-Max 归一化到 [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    valid_slices = []

    # 1. 提取横断位 (Axial) - 沿 Z 轴 (dim 2)
    for i in range(data.shape[2]):
        slice_2d = data[:, :, i].astype(np.float32)
        if np.mean(slice_2d) > 0.05:  # 过滤全黑背景
            valid_slices.append((f"{mod_name}_axial_{i:03d}.npy", slice_2d))

    # 2. 提取冠状位 (Coronal) - 沿 Y 轴 (dim 1)
    for i in range(data.shape[1]):
        slice_2d = data[:, i, :].astype(np.float32)
        if np.mean(slice_2d) > 0.05:
            valid_slices.append((f"{mod_name}_coronal_{i:03d}.npy", slice_2d))

    # 3. 提取矢状位 (Sagittal) - 沿 X 轴 (dim 0)
    for i in range(data.shape[0]):
        slice_2d = data[i, :, :].astype(np.float32)
        if np.mean(slice_2d) > 0.05:
            valid_slices.append((f"{mod_name}_sagittal_{i:03d}.npy", slice_2d))

    print(f"🔍 {mod_name} 共提取到 {len(valid_slices)} 张有效切片。")

    # 4. 打乱切片顺序以保证数据分布均匀
    random.seed(42) # 固定随机种子，保证每次划分结果一致（科研好习惯）
    random.shuffle(valid_slices)

    # 5. 划分训练集和测试集
    test_size = int(len(valid_slices) * test_ratio)
    test_slices = valid_slices[:test_size]
    train_slices = valid_slices[test_size:]

    # 6. 保存到对应文件夹
    for name, slice_data in tqdm(train_slices, desc=f"保存 {mod_name} 训练集"):
        np.save(os.path.join(train_dir, name), slice_data)
        
    for name, slice_data in tqdm(test_slices, desc=f"保存 {mod_name} 测试集"):
        np.save(os.path.join(test_dir, name), slice_data)

    print(f"✅ {mod_name} 处理完毕: 训练集 {len(train_slices)} 张，测试集 {len(test_slices)} 张。\n")

if __name__ == "__main__":
    # 请确保你的 data/raw 文件夹下有你刚下载的 rf0 mnc 文件
    RAW_DIR = 'data/raw'
    PROCESSED_DIR = 'data/processed'
    
    # 定义你要处理的文件列表 (根据你的实际文件名修改)
    files_to_process = [
        ('t1_icbm_normal_1mm_pn0_rf0.mnc', 'T1'),
        ('t2_icbm_normal_1mm_pn0_rf0.mnc', 'T2'),
        ('pd_icbm_normal_1mm_pn0_rf0.mnc', 'PD')
    ]

    for filename, mod in files_to_process:
        file_path = os.path.join(RAW_DIR, filename)
        if os.path.exists(file_path):
            process_and_split_mri(file_path, PROCESSED_DIR, mod, test_ratio=0.2)
        else:
            print(f"⚠️ 找不到文件: {file_path}")