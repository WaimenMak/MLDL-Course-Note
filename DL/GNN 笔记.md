# GNN

## Spectral-based

### Preliminary

### Message Passing Paradigm

![](/Users/mmai/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/Research/Notes/GNN 笔记.assets/Screen Shot 2022-11-02 at 11.31.05.png)

#### Degree matrix

$D = diag(A1)$, where A is the adjacency matrix，if it is directed graph, then A is not symetric $D_O = diag(A1)$, $D_I = diag(A^T1)$.

#### Laplacian matrix 

[(72 封私信 / 81 条消息) 如何理解 Graph Convolutional Network（GCN）？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/54504471)

1. $L = D - A = diag(A1) - A$, 

2. and normalized laplacian: $\bar L = D^{1/2}LD^{1/2}.$

3. random walk: $D^{-1}A$

   

### GCN

Operation: 拉普拉斯矩阵作用于节点特征。

​	[一文读懂图卷积GCN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/89503068)

![](/Users/mmai/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/Research/Notes/GNN 笔记.assets/Screen Shot 2022-12-27 at 15.54.13.png)

为什么GCN是Transductive：

https://www.zhihu.com/question/363234280





#### Graph Fourier Transform



## Spatial-based

### Diffusion Convolution Network

- [(31条消息) 《Diffusion-Convolutional Neural Networks》论文理解_monster.YC的博客-CSDN博客](https://blog.csdn.net/weixin_43450885/article/details/106134104?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-3-106134104-blog-81407888.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-3-106134104-blog-81407888.pc_relevant_default&utm_relevant_index=6)

​	参数各维度。

### DCRNN

1. Adopt diffusion convolution in RNN. Simply it's just a RNN and it's trained based on the graph architecture. Therefore it's not graph structure agnostic.

### Locale GN



### GAT

<ul>
  <li> df
</ul>





## Transductive and Inductive Learning

[图神经网络-GraphSage - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/499849059#:~:text=整体实现逻辑 1 对邻居随机采样。 目的：降低计算复杂度（图中一跳邻居采样数%3D3，二跳邻居采样数%3D5）,2 生成目标节点embedding。 先聚合2跳邻居特征，生成一跳邻居embedding，再聚合一跳邻居embedding，生成目标节点embedding，从而获得二跳邻居信息 3 将embedding作为全连接层的输入，预测目标节点的标签)

## Engineering

[(31条消息) 【学习笔记】图神经网络库 DGL 入门教程（backend pytorch）_囚生CY的博客-CSDN博客](https://blog.csdn.net/CY19980216/article/details/110629996)

#### Neighborhood sampling

- Purpose：inductive learning.



# DGL

### Message Passing

`reduce_func` and `message_func` are two important component in  DGL

```python
def message_func(self, edges):
  # 公式 (3), (4)所需，传递消息用的用户定义函数
  return {'z' : edges.src['z'], 'e' : edges.data['e']}
```

Above function send the feature 'z' of each source node and the edge data 'e' to the dot node. So the shape of `edges.src` is [num_edges, features].

```python
def reduce_func(self, nodes):
    # 公式 (3), (4)所需, 归约用的用户定义函数
    # 公式 (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # 公式 (4)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h' : h}
```

> Here, the receiving message are **'e'** and **'z'**, the shapes are `[num of dst nodes, num of neighbors, features]`, so after message passing,  **dgl** directly get the destination of nodes with the same number of neighbors once at a time to do the message aggregation.
>

After we run the `update_all(message_func, reduce_func)`, the function automatically iterate all the destiatnion nodes with the same number of neighbors.





