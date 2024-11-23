import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        print("TemporalConv - 输入X的形状:", X.shape)
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        print("TemporalConv - 输出H的形状:", H.shape)
        return H


class STConv(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        print("STConv - 输入X的形状:", X.shape)

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """

        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        print("STConv - 输出T的形状:", T.shape)
        return T





# 定义空间-时间注意力机制
class STAttention(nn.Module):
    def __init__(self, hidden_channels):
        super(STAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4)

    def forward(self, X):
        batch_size, time_steps, num_nodes, hidden_channels = X.shape
        X = X.permute(1, 0, 2, 3).reshape(time_steps, batch_size * num_nodes, hidden_channels)
        attn_output, _ = self.attention(X, X, X)
        attn_output = attn_output.permute(1, 0, 2).contiguous().reshape(batch_size, time_steps, num_nodes, hidden_channels)
        return attn_output


# 定义STGCN多任务学习模型
class STGCNMultiTask(nn.Module):
    def __init__(self, num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers=3):
        super(STGCNMultiTask, self).__init__()
        self.st_convs = nn.ModuleList([STConv(num_nodes=num_nodes, in_channels=in_channels if i == 0 else hidden_channels,
                                              hidden_channels=hidden_channels, out_channels=hidden_channels,
                                              kernel_size=kernel_size, K=K) for i in range(num_layers)])
        self.angle_fc = nn.Linear(num_angles, hidden_channels)
        self.attention = STAttention(hidden_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.key_action_detector = nn.Linear(hidden_channels, 1)
        # 在模型中添加评分预测的子网络
        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.score_attention = STAttention(hidden_channels)

    def forward(self, X, angles, edge_index, edge_weight=None):
        for st_conv in self.st_convs:
            X = st_conv(X, edge_index, edge_weight)
            X = F.relu(X)
        X = self.attention(X)
        score_features = self.score_attention(X)
        score_features = score_features.mean(dim=2).mean(dim=1)  # Pooling over nodes and time steps
        X = X.mean(dim=2).mean(dim=1)  # Pooling over nodes and time steps
        angle_features = self.angle_fc(angles)
        X = X + angle_features  # Combine with angle features
        logits = self.classifier(X)
        key_action = torch.sigmoid(self.key_action_detector(X))  # This will be a scalar output
        #outputs_angle_scores = X.mean(dim=1)  # Take the mean across all nodes to get a scalar per sample
        score_output = torch.sigmoid(self.score_predictor(score_features))

        return logits, key_action, score_output.squeeze(-1)   # Return angle scores

    def predict(self, X, angles, edge_index, edge_weight=None):
        """
        预测函数，用于在模型训练后进行推理预测
        输入:
            X: 输入数据 (batch_size, time_steps, num_nodes, in_channels)
            angles: 角度特征数据
            edge_index: 图的连接关系
        输出:
            predicted_class: 预测的类别标签张量，形状为 [batch_size]
            angle_scores: 预测的角度得分张量，形状为 [batch_size]
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():
            logits, key_action, angle_scores = self(X, angles, edge_index, edge_weight)
            predicted_class = torch.argmax(logits, dim=1)
        return predicted_class, angle_scores.squeeze()
