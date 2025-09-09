import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 手动颜色表（label=1 对应 color_list[0]）
color_list_bgr = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [128, 0, 128],
    [255, 165, 0], [255, 192, 203], [64, 224, 208], [165, 42, 42], [128, 128, 128],
    [0, 255, 255], [0, 128, 255], [0, 0, 128], [128, 0, 0], [0, 128, 0],
    [128, 128, 0], [255, 105, 180], [70, 130, 180], [139, 69, 19]
]
color_list_rgb = [[b, g, r] for [r, g, b] in color_list_bgr]
color_list = [[c[0]/255, c[1]/255, c[2]/255] for c in color_list_rgb]

def visualize_segmentation_map(segmentation, title="Segmentation", seed=42):
    """
    可视化语义分割结果：
    - label=0 显示为黑色背景
    - label=1 映射到 color_list[0]，label=2 映射到 color_list[1]，依此类推
    - 若标签超出颜色表定义，使用随机颜色替代所有颜色
    """
    if segmentation.ndim != 2:
        raise ValueError("输入必须是二维数组")

    h, w = segmentation.shape
    rgb_image = np.zeros((h, w, 3))  # 全黑背景

    # 获取非背景标签（label > 0）
    class_labels = np.unique(segmentation)
    nonzero_labels = class_labels[class_labels > 0]
    max_label = np.max(nonzero_labels) if len(nonzero_labels) > 0 else 0

    if max_label > len(color_list):
        print(f"[提示] 最大标签 {max_label} 超出颜色表上限 {len(color_list)}，使用随机颜色替代。")
        np.random.seed(seed)
        random_colors = np.random.rand(len(nonzero_labels), 3)
        label_to_color = {label: random_colors[i] for i, label in enumerate(nonzero_labels)}
    else:
        # 用 color_list 映射：label=1 -> color_list[0], label=2 -> color_list[1] ...
        label_to_color = {label: color_list[label - 1] for label in nonzero_labels}

    # 填充颜色
    for label, color in label_to_color.items():
        rgb_image[segmentation == label] = color

    # 显示
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.savefig("test.pdf")


def plot_heatmap(matrix, title="Matrix", cmap='viridis'):
    fig, ax = plt.subplots(figsize=(4, 4))  # 控制图像大小
    cax = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    fig.colorbar(cax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf
