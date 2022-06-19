from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing


# message passing for edge aggregation https://github.com/pyg-team/pytorch_geometric/issues/1489
class EdgeSoftmaxAttention(MessagePassing):
    def __init__(self):
        super(EdgeSoftmaxAttention, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, edge_attr):
        # edge_index is an array of shape (2, E) when E is the number of edges
        # x is an array of shape (N, feature_dim) when N is the number of nodes
        # x_i an array of shape (E, feature_dim). Each row in this array correspond to feature vector of a target node
        # (node in the second row of edge index array)
        # x_j an array of shape (E, feature_dim). Each row in this array correspond to feature vector of a source node
        # (node in the first row of edge index array)
        # x_i and x_j are automatically generate from x by PyG
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, edge_index):
        # calculate attention weighted feature vector for each edge
        source, destination = edge_index
        attn_val = softmax(edge_attr, index=destination)

        return attn_val * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out