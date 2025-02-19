import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history, title="Convergence of FJ Model"):
    """
    绘制观点最大变化的收敛曲线
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker='o', linestyle='--', color='#2c7bb6')
    plt.xlabel("Iteration Step")
    plt.ylabel("Maximum Opinion Change")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_opinion_distribution(initial_z, final_z, title="Opinion Distribution Comparison"):
    """
    绘制初始和最终观点分布的对比直方图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 计算条形宽度
    bins = 20
    bin_width = (max(initial_z) - min(initial_z)) / bins

    # 初始观点分布
    axes[0].hist(initial_z, bins=10, color='#d7191c', alpha=0.6, edgecolor='black', width=bin_width)
    axes[0].set_title("Initial Opinions")
    axes[0].set_xlabel("Opinion Value")
    axes[0].set_ylabel("Number of Nodes")
    # 最终观点分布
    axes[1].hist(final_z, bins=10, color='#2c7bb6', alpha=0.6, edgecolor='black', width=bin_width)
    axes[1].set_title("Final Opinions")
    axes[1].set_xlabel("Opinion Value")
    axes[1].set_ylabel("Number of Nodes")

    # 设置两个图的x坐标范围一致
    min_val = 0.9 * min(min(initial_z), min(final_z))
    max_val = 1.1 * max(max(initial_z), max(final_z))
    axes[0].set_xlim([min_val, max_val])
    axes[1].set_xlim([min_val, max_val])

    # 设置两个图的y坐标范围一致
    max_y = 1.5 * max(max(np.histogram(initial_z, bins=10)[0]), max(np.histogram(final_z, bins=10)[0]))
    axes[0].set_ylim([0, max_y])
    axes[1].set_ylim([0, max_y])

    plt.tight_layout()  # 自动调整子图的布局
    plt.show()
