import numpy as np
import networkx as nx
from FJmodel import FJModel
from visualize import plot_network_and_community
import matplotlib.pyplot as plt

# Step1: 加载空手道俱乐部网络
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).toarray().astype(float)

# Step2: 随机生成每个节点的内部观点
np.random.seed(42)
s = np.random.rand(len(G.nodes))

# Step3: 初始化FJ模型并检测社区
model = FJModel(A, s)
model.detect_communities()

# Step4: 打印初始观点和社区结构
print("最初观点分布:", s)
print("检测到的社区：")
for comm_id, nodes in model.communities.items():
    print(f"社区 {comm_id}: 节点 {nodes}")

# Step5: 绘制网络拓扑结构和社区结构
plot_network_and_community(G, values=model.s, communities=model.communities)
plt.show()
