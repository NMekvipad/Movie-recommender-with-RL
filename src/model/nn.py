import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from src.model.layers import MultiHeadEdgeAttention


#TODO: currentlt written for single head case, need to expand to multi-head
class IntraMetaPathAggregator(torch.nn.Module):
    def __init__(self, hidden_size, num_head=1, aggregator_type='mean', activation=None):
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

        self.attention = MultiHeadEdgeAttention(num_head=num_head, in_features=hidden_size, activation=activation)

    def forward(self, node_features, edge_index, metapath_idx):
        #TODO:  Node feature and edge pairs to be encompassed in PyG graph class.
        #       Dim of each layer to be verified and test

        # feature look up for nodes in each metapath
        metapath_feat = F.embedding(metapath_idx, node_features)  # (n_edge, n_metapath_len, hidden_size)

        # metapath aggregation
        if self.aggregator_type == 'mean':
            agg_feat = torch.mean(metapath_feat, dim=1)  # (n_edge, hidden_size)
            agg_feat = torch.concat([agg_feat] * self.num_head, dim=-1)  # (n_edge, hidden_size * num_head)

        elif self.aggregator_type == 'GRU':
            _, agg_feat = self.aggregator(input=metapath_feat)  # (1, n_edge, hidden_size * num_head)
            agg_feat = agg_feat.squeeze(dim=0)  # (n_edge, hidden_size * num_head)

        # reshape and swap dim
        agg_feat = agg_feat.view(-1, self.num_head, self.hidden_size)  # (n_edge, num_head, hidden_size)

        # edge attention
        # (n_edge, num_head * feat_dim)
        metapath_node_feat = self.attention(x=node_features, edge_index=edge_index, edge_attr=agg_feat)

        return metapath_node_feat


class InterMetaPathAggregator(torch.nn.Module):
    def __init__(
            self, hidden_size, metapath_map, num_head=1,
            intrapath_aggregator='mean', activation=None
    ):
        """
        :metapath_list: list metapath by node type
            {
                # for each node type
                'node_type': [(metapath 1, metapath 2)],
                # metapath 1 is a tuple of node type mask for each metapath nodes e.g. ('M', 'S', 'M')
            }

        """
        super().__init__()

        # model hyperparameters
        self.intrapath_aggregator = intrapath_aggregator
        self.metapath_map = metapath_map
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.activation_type = activation

        # layers
        self.intra_metapath_aggregator = defaultdict(dict)
        self.path_summarizer = defaultdict(dict)
        self.path_attn = defaultdict(dict)
        self.output_linear = dict()
        self.output_activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        for node_type, metapaths in self.metapath_map.items():
            for metapath in metapaths:
                self.intra_metapath_aggregator[node_type][metapath] = IntraMetaPathAggregator(
                    hidden_size=self.hidden_size,
                    num_head=self.num_head,
                    aggregator_type=self.activation_type,
                    activation=self.activation_type
                )

                self.path_summarizer[node_type][metapath] = torch.nn.Linear(
                    in_features=self.hidden_size * self.num_head, out_features=self.hidden_size, bias=True
                )

                self.path_attn[node_type][metapath] = torch.nn.Linear(
                    in_features=self.hidden_size, out_features=1, bias=False
                )

            self.output_linear[node_type] = torch.nn.Linear(
                    in_features=self.hidden_size * self.num_head, out_features=self.hidden_size, bias=True
            )

    def forward(self, node_features, meta_graphs_by_n_type):
        """
        :meta_graphs_by_n_type: list of tuples where each tuple contains information about metapath graphs for each node type
        These information are edge indices, metapath indices and node type mask
            {
                # for each node type
                'node_type': ({
                    (metapath 1): (edge_idx_metapath_1, metapath_idx_1),
                    (metapath 2): (edge_idx_metapath_2, metapath_idx_2)
                }, node_type_mask)
            }
        :node_features: Tensor of shape (no. of node, node feature dimension)
        """
        outputs = dict()
        for node_type, graph_data in meta_graphs_by_n_type.items():
            metapath_dict, node_type_mask = graph_data
            intra_path_agg_node = list()
            e_p = list()

            # intra-metapath message passing
            for metapath, metapath_data in metapath_dict.items():
                metapath_node_feat = self.intra_metapath_aggregator[node_type][metapath](
                    node_features=node_features,
                    edge_index=metapath_data[0],
                    metapath_idx=metapath_data[1]
                )

                h_p = metapath_node_feat[np.where(node_type_mask == node_type)]
                s_p = self.tanh(self.path_summarizer[node_type][metapath](h_p)).mean(dim=0)

                intra_path_agg_node.append(h_p.unsqueeze(dim=1))
                e_p.append(self.path_attn[node_type][metapath](s_p))

            # inter_metapath_aggregation
            beta = F.softmax(torch.concat(e_p, dim=0), dim=0).unsqueeze(dim=-1)
            h_p = torch.concat(intra_path_agg_node, dim=1)
            h_p_a = (h_p * beta).sum(dim=1)
            h_v = self.output_activation(self.output_linear[node_type](h_p_a))
            outputs[node_type] = h_v

        return outputs


class MAGNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self, metapath_graph):
        # map node feature to the same dimension

        #####################################
        #         For each node type        #
        #####################################

        # intra metapath aggregation

        # inter metapath aggregation

        # update node feature

        pass
