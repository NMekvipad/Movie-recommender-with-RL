import torch
from torch.nn import LeakyReLU, ReLU
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing


# placeholder activation function
class NoActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Placeholder activation function
        """
        return x


# message passing for edge aggregation https://github.com/pyg-team/pytorch_geometric/issues/1489
class MultiHeadEdgeAttention(MessagePassing):
    def __init__(self, num_head, in_features, activation=None):
        super(MultiHeadEdgeAttention, self).__init__(aggr='add')  # "Add" aggregation.
        self.num_head = num_head
        self.in_features = in_features

        # weighted attention take aggregation output concat with node as input
        self.target_attn = torch.nn.Linear(in_features=self.in_features, out_features=self.num_head, bias=False)
        self.metapath_attn = torch.nn.Parameter(torch.empty(size=(self.num_head, self.in_features)))
        self.leaky_relu = LeakyReLU()

        if activation is None:
            self.activation = NoActivation
        if activation == 'relu':
            self.activation = ReLU
        else:
            raise ValueError("Undefined activation function")

        # init weight
        torch.nn.init.xavier_normal_(self.target_attn.weight)
        torch.nn.init.xavier_normal_(self.metapath_attn.data)

    def forward(self, x, edge_index, edge_attr):
        # edge_index is an array of shape (2, E) when E is the number of edges
        # x is an array of shape (N, feature_dim) when N is the number of nodes
        # x_i an array of shape (E, feature_dim). Each row in this array correspond to feature vector of a target node
        # (node in the second row of edge index array)
        # x_j an array of shape (E, feature_dim). Each row in this array correspond to feature vector of a source node
        # (node in the first row of edge index array)
        # x_i and x_j are automatically generate from x by PyG
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, edge_attr, edge_index):
        # target node attention
        target_attention = self.target_attn(x_i)  # (n_edge, num_head)

        # metapath attention
        metapath_attention = edge_attr * self.metapath_attn  # (n_edge, num_head, feat_dim) * (num_head, feat_dim)
        metapath_attention = metapath_attention.sum(dim=-1)  # (n_edge, num_head)

        # calculate attention weight
        e_p = self.leaky_relu(target_attention + metapath_attention)
        # (n_edge, softmax(num_head)); softmax is applied along axis 1 for each col
        attn = softmax(e_p, edge_index.permute(1, 0)[:, -1])
        # (n_edge, num_head, feat_dim) * (n_edge, num_head, 1)
        attn_weighted_feat = self.activation(edge_attr * attn.unsqueeze(dim=-1))

        return attn_weighted_feat.view(-1, self.num_head * self.in_features)  # (n_edge, num_head * feat_dim)

    def update(self, aggr_out):
        return aggr_out

