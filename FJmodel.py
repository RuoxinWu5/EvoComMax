"""
Friedkin-Johnsen (FJ) 模型类，用于模拟有向图中个体意见的演变。
该模型考虑了每个个体的内部观点和表达观点，并通过迭代过程模拟观点的动态变化。
定义了 iterate() 方法模拟意见演变过程，直到达到最大迭代次数或满足收敛条件。
每次迭代计算表达观点的新值，并记录最大观点变化历史。
"""
import numpy as np


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
            print("new z", step, "=", z_new)
            self.z = z_new
        return history
