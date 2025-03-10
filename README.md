# Friedkin-Johnsen 观点演化模型
该代码实现了一个 Friedkin-Johnsen (FJ) 模型，用于模拟有向图中节点观点的动态演化过程。模型通过迭代更新节点的表达观点，最终达到平衡状态。

## Table of contents

* [我的环境](#我的环境)
* [代码结构](#代码结构)
* [如何运行](#如何运行)

## 我的环境

* Python 3.9.6
* numpy 2.0.2
* networkx 3.2.1
* matplotlib 3.9.4
* scipy 1.13.1
* python-louvain 0.16


## 代码结构

### fj_model.py
包含了Friedkin-Johnsen模型的核心实现。定义了一个 `FJModel` 类，该类初始化模型并提供了迭代计算表达观点平衡状态的方法。 主要功能包括：
包含了 Friedkin-Johnsen 模型的核心实现。定义了一个 `FJModel` 类，该类初始化模型并提供了迭代计算表达观点平衡状态的方法。主要功能包括：
* **初始化模型**：接受邻接矩阵和可选的内部观点向量。
* **iterate**：通过迭代计算来更新表达观点，直到达到最大迭代次数或满足收敛条件（可选参数，用于提前结束迭代）。
* **detect_communities**：使用 Louvain 算法检测社区，并返回网络中的社区结构。
* **conflict_risk**：计算冲突风险，并归一化到 0-100% 之间。基于节点之间的意见差异和连接权重，返回一个百分比格式的冲突风险值。

### visualization.py
提供了可视化工具，用于绘制模型的收敛过程、最终观点分布以及社区结构。
* **plot_convergence**：绘制观点最大变化的收敛曲线。
* **plot_opinion_distribution**：绘制初始与最终观点分布的对比直方图，帮助分析节点观点的变化。
* **plot_network_comparison**：绘制初始状态和最终状态的网络拓扑结构，其中颜色表示节点观点。
* **plot_network_and_community**：绘制网络结构的双视图：
  - 左侧显示节点观点的颜色分布。
  - 右侧使用 Louvain 社区检测算法识别社区，并用不同颜色区分社区，同时用红色虚线圈出社区。

### test_fj_karate.py
展示了如何在实际的空手道俱乐部网络上应用Friedkin-Johnsen模型。主要功能包括：
* **加载网络**：加载空手道俱乐部网络，生成随机内部观点。
* **运行FJ模型**：初始化 `FJModel` 类并执行迭代计算，直到模型收敛或达到最大迭代次数。
* **输出**： 迭代过程观点变化、初始观点分布和最终观点分布。
* **可视化结果**：调用 `visualize.py` 中的函数来绘制收敛曲线、观点分布对比直方图、初始和最终网络拓扑结构的对比图。

### test_communities.py
用于测试 FJ 模型在社区检测中的表现。
* **加载网络**：加载空手道俱乐部网络，生成随机内部观点。
* **运行FJ模型**：初始化 `FJModel` 类并检测社区。
* **输出**： 初始观点分布和社区结构。
* **可视化结果**：调用 `plot_network_and_community` 展示观点分布和社区划分的对比视图。

## 如何运行
运行 `test_fj_karate.py` 来执行空手道俱乐部网络示例，或运行 `test_communities.py` 来测试社区检测功能。
```bash
python test_fj_karate.py
python test_communities.py
```