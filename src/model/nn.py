import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from src.model.layers import MultiHeadEdgeAttention


#TODO: currentlt written for single head case, need to expand to multi-head
class IntraMetaPathAggregation(torch.nn.Module):
    def __init__(self, hidden_size, num_head=1, aggregator_type='mean'):
        super().__init__()

        # model hyperparameters
        self.aggregator_type = aggregator_type
        self.hidden_size = hidden_size
        self.num_head = num_head

        # model aggregator
        if aggregator_type == 'GRU':
            # single layer GRU
            # output hidden layer will be reshape into (hidden_size, num_head)
            self.aggregator = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size * num_head, batch_first=True)
        elif aggregator_type == 'mean':
            pass
        else:
            raise ValueError("aggregator_type can only be values in ['GRU', 'mean']")

        self.attention = MultiHeadEdgeAttention(num_head=num_head, in_features=hidden_size)

    def forward(self, node_features, edge_index, metapath_idx):
        #TODO:  Node feature and edge pairs to be encompassed in PyG graph class.
        #       Dim of each layer to be verified and test

        # feature look up for nodes in each metapath
        metapath_feat = F.embedding(metapath_idx, node_features)  # (n_edge, n_metapath_len, hidden_size)

        # metapath aggregation
        if self.aggregator_type == 'mean':
            agg_feat = torch.mean(metapath_feat, dim=1)  # (n_edge, hidden_size)
            agg_feat = torch.concat([metapath_feat] * self.num_head, dim=-1)  # (n_edge, hidden_size * num_head)

        elif self.aggregator_type == 'GRU':
            _, agg_feat = self.aggregator(input=metapath_feat)  # (1, n_edge, hidden_size * num_head)
            agg_feat = agg_feat.squeeze(dim=0)  # (n_edge, hidden_size * num_head)

        # reshape and swap dim
        agg_feat = agg_feat.view(-1, self.num_head, self.hidden_size)  # (n_edge, num_head, hidden_size)

        # edge attention
        # (n_edge, num_head * feat_dim)
        updated_node_feat = self.attention(x=node_features, edge_index=edge_index, edge_attr=agg_feat)

        return updated_node_feat


class InterMetaPathAggregation(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self):
        pass


class MAGNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self):
        # map node feature to the same dimension

        #####################################
        #         For each node type        #
        #####################################

        # intra metapath aggregation

        # inter metapath aggregation

        # message passing

        # update node feature

        pass
