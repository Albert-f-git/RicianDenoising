import numpy as np
import matplotlib.pyplot as plt
import os
import random

def show_random_slices(data_dir, num_slices=3):
    # 1. 获取目录下所有的 .npy 文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not all_files:
        print(f"❌ 在 {data_dir} 中没有找到 .npy 文件，请检查路径。")
        return
        
    # 2. 随机抽取几张图像
    sample_files = random.sample(all_files, min(num_slices, len(all_files)))
    
    # 3. 设置画布大小
    plt.figure(figsize=(15, 5))
    
    for i, file_name in enumerate(sample_files):
        # 读取数据
        file_path = os.path.join(data_dir, file_name)
        img_array = np.load(file_path)
        
        # 创建子图
        plt.subplot(1, num_slices, i + 1)
        
        img_array = np.flipud(img_array)
        # 核心：使用 cmap='gray' 显示医学灰度图
        plt.imshow(img_array, cmap='gray')
        
        # 将文件名作为标题，方便确认你看到的是哪个切面
        plt.title(file_name, fontsize=12)
        plt.axis('off')  # 关闭坐标轴，让图像更干净
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 指向我们刚刚生成切片的训练集目录
    TRAIN_DIR = 'data/processed/train'
    
    print("正在加载并显示切片...")
    show_random_slices(TRAIN_DIR)