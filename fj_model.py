"""
Friedkin-Johnsen (FJ) 模型类，用于模拟有向图中个体意见的演变。
该模型考虑了每个个体的内部观点和表达观点，并通过迭代过程模拟观点的动态变化。
定义了 iterate() 方法模拟意见演变过程，直到达到最大迭代次数或满足收敛条件。
每次迭代计算表达观点的新值，并记录最大观点变化历史。
"""
import networkx as nx
import numpy as np
from community import community_louvain


class FJModel:
    def __init__(self, graph, s=None):
        """
        初始化FJ模型。

        :param graph: 邻接矩阵（numpy数组）
        :param s: 可选自定义内部观点向量,若不定义则随机生成
        """
        self.A = graph
        self.n = len(graph)
        self.s = s if s is not None else np.random.rand(self.n)
        self.z = self.s.copy()
        self.communities = None  # 存储社区结构

    def iterate(self, max_iter, tolerance=None):
        """
        迭代计算表达观点的平衡状态。

        :param max_iter: 最大迭代次数
        :param tolerance: 收敛条件
        :return: 每次迭代的最大观点变化历史
        """
        history = []
        for step in range(max_iter):
            z_new = np.zeros_like(self.z)
            for i in range(self.n):
                neighbors = np.where(self.A[i] > 0)[0]
                new_z_up = self.s[i] + self.A[i, neighbors] @ self.z[neighbors]
                new_z_down = 1 + self.A[i, neighbors].sum()
                z_new[i] = new_z_up / new_z_down
            delta = np.max(np.abs(z_new - self.z))  # 做差后取绝对值，选最大
            history.append(delta)
            if tolerance is not None and delta < tolerance:
                print(f"在第 {step + 1} 步收敛")
                break
            # print("new z", step, "=", z_new)
            self.z = z_new
        return history

    def detect_communities(self):
        """
        使用Louvain算法检测社区
        """
        G = nx.from_numpy_array(self.A)
        partition = community_louvain.best_partition(G)
        # 按社区ID分组节点
        self.communities = {}
        for node, comm_id in partition.items():
            if comm_id not in self.communities:
                self.communities[comm_id] = []
            self.communities[comm_id].append(node)

    def conflict_risk(self, threshold=0.5):
        """
        计算冲突风险，并归一化到 0-100% 之间。
        该方法基于节点之间的意见差异和连接（权重）来计算网络中的冲突风险，

        :param threshold: 意见差异阈值，默认0.5。
        :return: 归一化冲突风险（百分比，保留两位小数，带%）
        """
        risk = 0.0
        max_risk = np.sum(self.A)  # 计算所有边的总权重，作为最大可能风险
        for i in range(self.n):
            for j in range(self.n):
                # 条件1: 节点i和j之间有连接（边权重 > 0）
                # 条件2: 两者的意见差异超过阈值（默认0.5）
                if self.A[i][j] > 0 and abs(self.z[i] - self.z[j]) > threshold:
                    # 风险计算公式：边权重 × 意见差异
                    risk += self.A[i][j] * abs(self.z[i] - self.z[j])
        # 归一化到 0-100%
        normalized_risk = (risk / max_risk) * 100 if max_risk > 0 else 0
        return "{:.2f}%".format(normalized_risk)
