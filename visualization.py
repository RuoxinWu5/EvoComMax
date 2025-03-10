import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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


def plot_network_comparison(G, initial_values, final_values, title="Network Structure Comparison"):
    """
    绘制初始和最终网络拓扑结构的对比

    参数:
    - G: networkx.Graph，社交网络
    - initial_values: list，初始观点值（用于着色）
    - final_values: list，最终观点值（用于着色）
    - title: str，图标题
    """
    # 生成固定布局，确保两张图的节点位置相同
    layout = nx.spring_layout(G, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 归一化颜色数据
    vmin = min(min(initial_values), min(final_values))
    vmax = max(max(initial_values), max(final_values))
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.Blues

    # 初始状态的网络拓扑
    axes[0].set_title("Initial Network Opinions")
    nx.draw(G, pos=layout, with_labels=True, node_color=initial_values, cmap=cmap, ax=axes[0], node_size=500,
            font_size=10, edge_color='gray', vmin=vmin, vmax=vmax)
    # 最终状态的网络拓扑
    axes[1].set_title("Final Network Opinions")
    nx.draw(G, pos=layout, with_labels=True, node_color=final_values, cmap=cmap, ax=axes[1], node_size=500,
            font_size=10, edge_color='gray', vmin=vmin, vmax=vmax)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.1)
    cbar.set_label("Opinion Value")

    plt.suptitle(title)
    plt.show()


def plot_network_and_community(G, values, communities, title1="Opinion Distribution", title2="Community Detection"):
    """
    绘制网络拓扑结构及其社区划分：
    - 左图：节点颜色表示观点值
    - 右图：节点颜色表示社区，外圈标记社区

    参数:
    - G: networkx.Graph，社交网络
    - values: list，观点值（用于左图着色）
    - communities: dict，社区划分结果，键为社区ID，值为节点列表（用于右图着色）
    - title1: str，左图标题（默认"Opinion Distribution"）
    - title2: str，右图标题（默认"Community Detection"）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 创建左右两个子图
    pos = nx.spring_layout(G)  # 生成固定布局，确保两张图的节点位置相同

    # 左侧：观点分布
    nx.draw(G, pos, with_labels=True, node_color=values, cmap=plt.cm.Blues,
            node_size=500, font_size=10, edge_color='gray', ax=axes[0])
    axes[0].set_title(title1)

    # 右侧：社区检测
    unique_colors = [plt.cm.Set1(i) for i in range(len(communities))]
    color_map = {}  # 存储每个节点的颜色
    for idx, (comm_id, nodes) in enumerate(communities.items()):
        color = unique_colors[idx % len(unique_colors)]
        for node in nodes:
            color_map[node] = color
    node_colors = [color_map[node] for node in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500,
            font_size=10, edge_color='gray', ax=axes[1])

    # 圈出社区
    ax = axes[1]
    for comm_id, nodes in communities.items():
        x, y = zip(*[pos[n] for n in nodes])
        center_x, center_y = np.mean(x), np.mean(y)
        radius = max(np.std(x), np.std(y)) + 0.1  # 计算合适的半径
        circle = patches.Circle((center_x, center_y), radius, fill=False,
                                edgecolor="red", linewidth=2, linestyle='dashed')
        ax.add_patch(circle)

    axes[1].set_title(title2)

    plt.show()
