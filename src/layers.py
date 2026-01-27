import math
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        #定义可学习的权重矩阵 weight。其维度为 (输入特征, 输出特征)。使用 Parameter 包装，确保它会被 Adam 等优化器识别并更新。
        init.xavier_uniform_(self.weight)
        #使用 Xavier 均匀分布初始化权重。这能确保在深度网络中，各层特征的方差保持稳定，防止训练初期的梯度消失或爆炸。

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            #如果开启偏置，创建一个长度等于输出特征维度的向量。
            stdv = 1. / math.sqrt(self.bias.size(0))
            #根据输出维度计算一个标准差。
            self.bias.data.uniform_(-stdv, stdv)
            #用均匀分布初始化偏置。这是一种非常经典且稳健的初始化方式。
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        #(batch, nodes, in_features) @ (in_features, out_features) → (batch, nodes, out_features)
        output = torch.matmul(adj, support)
        #(batch, nodes, nodes) @ (batch, nodes, out_features) → (batch, nodes, out_features)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 
    """
    辅助函数 __repr__
    作用：定义打印模型时的显示格式。
    效果：当你执行 print(layer) 时，它会输出类似 GraphConvLayer (16 -> 32)，让你一眼看出这一层的输入输出维度，非常方便调试。
    """

 