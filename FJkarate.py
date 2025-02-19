"""
以空手道俱乐部网络为例，使用Friedkin-Johnsen（FJ）模型模拟社交网络中观点的演化。
"""
import numpy as np
import networkx as nx
from FJmodel import FJModel
from visualize import plot_convergence, plot_opinion_distribution
import matplotlib.pyplot as plt

# Step1: 加载空手道俱乐部网络并生成邻接矩阵
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).toarray().astype(float)  # 邻接矩阵（无向图，边权重=1）

# Step2: 随机生成每个节点的内部观点
np.random.seed(42)
s = np.random.rand(len(G.nodes))

# Step3: 初始化FJ模型并进行迭代，直到收敛或达到最大迭代次数
model = FJModel(A, s)
history = model.iterate(max_iter=100, tolerance=1e-6)

# Step4: 可视化结果
# 绘制观点最大变化的收敛曲线
plot_convergence(history, "Convergence of FJ Model")
print("最终观点分布:", model.z)

# 绘制初始和最终观点分布的对比直方图
plot_opinion_distribution(model.s, model.z)

plt.show()
