import os
import glob
import numpy as np
from PIL import Image

# ==========================================
# 1. 批量转换配置区
# ==========================================
CONFIG = {
    "source_dir": "data/processed/train",       # 🚨 替换为你当前测试集数据所在的文件夹
    "output_dir": "data/processed/train_png",   # 转换后的 PNG 图片保存位置
    "target_size": None              # 如果需要统一缩放(如 224, 224)请填入，保持原尺寸则保留 None
}

def normalize_to_uint8(img_matrix):
    """
    核心数学逻辑：将任意值域的矩阵，线性拉伸并映射到 [0, 255] 的 uint8 空间
    """
    img_matrix = img_matrix.astype(np.float32)
    min_val = np.min(img_matrix)
    max_val = np.max(img_matrix)
    
    # 防止纯色图片导致除以 0 的错误
    if max_val - min_val == 0:
        return np.zeros_like(img_matrix, dtype=np.uint8)
        
    # Min-Max 归一化到 [0.0, 1.0]
    normalized = (img_matrix - min_val) / (max_val - min_val)
    
    # 映射到 [0, 255] 并强转为 8 位无符号整数
    return (normalized * 255.0).astype(np.uint8)

def convert_dataset_to_png():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 自动检索常见的底层数据格式 (你可以根据需要增删后缀)
    supported_exts = ['.npy', '.mat', '.tiff', '.tif', '.jpg', '.jpeg']
    
    files = []
    for ext in supported_exts:
        files.extend(glob.glob(os.path.join(CONFIG["source_dir"], f"*{ext}")))
        
    if not files:
        print(f"❌ 在 [{CONFIG['source_dir']}] 中没有找到支持的文件。")
        print("💡 如果你的原始数据是医疗专用的 .nii 或 .nii.gz 格式，请告诉我，我们需要引入 nibabel 库！")
        return

    print(f"🔍 检索到 {len(files)} 个数据文件，启动批量转换引擎...\n")
    
    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        name_no_ext, ext = os.path.splitext(filename)
        
        try:
            # ----------------------------------------
            # 步骤 A: 根据后缀名，将文件读取为 Numpy 矩阵
            # ----------------------------------------
            if ext == '.npy':
                img_array = np.load(file_path)
            elif ext == '.mat':
                import scipy.io as sio
                mat_data = sio.loadmat(file_path)
                # 动态提取 mat 文件中的有效矩阵 (忽略 __header__ 等内置键)
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                img_array = mat_data[keys[0]]
            else:
                # 处理常规图像格式，强制转为单通道灰度矩阵 ('L')
                img_array = np.array(Image.open(file_path).convert('L'))

            # 鲁棒性处理：确保提取出来的是 2D 切片 (H, W)
            # 如果不小心读到了多通道数据 (比如 RGB 或者多层核磁)，强行取第一个切片
            if img_array.ndim > 2:
                img_array = np.squeeze(img_array) # 去除维度为 1 的冗余轴
                if img_array.ndim > 2:
                    img_array = img_array[:, :, 0]

            # ----------------------------------------
            # 步骤 B: 灰度映射与导出
            # ----------------------------------------
            img_array = np.flipud(img_array)  # 根据需要进行垂直翻转，确保医学图像的正确朝向
            img_uint8 = normalize_to_uint8(img_array)
            img_pil = Image.fromarray(img_uint8, mode='L')
            
            # 尺寸缩放 (按需触发)
            if CONFIG["target_size"] is not None:
                img_pil = img_pil.resize(CONFIG["target_size"], Image.Resampling.LANCZOS)
            
            save_path = os.path.join(CONFIG["output_dir"], f"{name_no_ext}.png")
            img_pil.save(save_path)
            
            # 打印进度条 (每 10 张打印一次，防止刷屏)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(files):
                print(f"✅ 进度: {idx + 1}/{len(files)} | 成功导出 -> {filename}.png")
                
        except Exception as e:
            print(f"⚠️ 转换文件 [{filename}] 时发生异常: {e}")

    print(f"\n🎉 转换任务结束！所有 PNG 已归档至: {CONFIG['output_dir']}")

if __name__ == '__main__':
    convert_dataset_to_png()