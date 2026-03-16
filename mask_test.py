import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 前景掩模核心函数 (保持不变)
# ==========================================
def get_foreground_mask(clean_img):
    """使用 面积过滤 + 椭圆闭运算 打造绝对纯净的头部掩模"""
    # 1. 基础二值化
    img_8u = (clean_img * 255).astype(np.uint8)
    _, mask_thresh = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ========================================================
    # 🌟 核心升级 1：面积过滤器 (物理斩杀悬浮鬼影)
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
    # 🌟 核心升级 2：解剖学椭圆包裹 (拒绝方形核的对角线越界)
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

# ==========================================
# 2. 多图批量测试可视化引擎
# ==========================================
def test_multiple_masks_visualization(image_paths, save_path="batch_mask_test_result.png"):
    num_images = len(image_paths)
    if num_images == 0:
        print("❌ 未找到任何测试图像，请检查路径！")
        return

    print(f"🔍 准备同时测试 {num_images} 张图像...")
    
    # 🌟 动态调整画布大小：宽度固定为 20，高度随图片数量成比例增加 (每张图分配高度 5)
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    
    # 防止传入单张图时 axes 变成一维数组而报错
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{num_images}] 正在处理: {filename}")
        
        # 1. 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   ⚠️ 警告: 无法读取图像，跳过该图。")
            continue
            
        # 2. 模拟归一化
        clean_img = img.astype(np.float32) / 255.0
        
        # 3. 计算掩模
        mask = get_foreground_mask(clean_img)
        
        # 4. 生成可视化层
        mask_visual = mask.astype(np.float32) 
        foreground_only = clean_img * mask_visual
        
        overlay = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
        red_mask = np.zeros_like(overlay)
        red_mask[mask] = [1.0, 0.0, 0.0] 
        blended = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

        # 5. 填入对应的子图位置
        ax_orig = axes[i, 0]
        ax_mask = axes[i, 1]
        ax_fg   = axes[i, 2]
        ax_blend= axes[i, 3]

        ax_orig.imshow(clean_img, cmap='gray')
        # 🌟 在第一列的左侧打上当前图片的文件名，方便你对比找图
        ax_orig.set_ylabel(filename, fontsize=12, rotation=90, labelpad=10)
        ax_orig.set_xticks([]); ax_orig.set_yticks([]) # 隐藏坐标轴刻度
        
        ax_mask.imshow(mask_visual, cmap='gray')
        ax_mask.axis('off')
        
        ax_fg.imshow(foreground_only, cmap='gray')
        ax_fg.axis('off')
        
        ax_blend.imshow(blended)
        ax_blend.axis('off')
        
        # 🌟 为了界面清爽，只有第一行才会显示顶部的列标题
        if i == 0:
            ax_orig.set_title("1. Original Image [0,1]", fontsize=16)
            ax_mask.set_title("2. Binary Mask (Otsu + Close)", fontsize=16)
            ax_fg.set_title("3. Extracted Foreground", fontsize=16)
            ax_blend.set_title("4. Mask Overlay (Red)", fontsize=16)

    # 自动紧凑布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ 批量测试图已成功保存为: {save_path}")

if __name__ == '__main__':

    image_paths = [
        "data\special_test\\brain1_simulate.png", 
        "data\special_test\\brain2_simulate.png",
        "data\special_test\\T1_axial_047.png", 
        "data\special_test\\T1_sagittal_079.png", 
        "data\special_test\\T2_sagittal_092.png"         
    ]
    test_multiple_masks_visualization(image_paths)