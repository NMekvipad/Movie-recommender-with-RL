import torch
import numpy as np
from scipy.sparse import csr_matrix


def edge_pairs_to_adjacency(edge_pairs):
    # max node index
    dim = edge_pairs.max() + 1
    adj_mat = np.zeros(shape=(dim, dim), dtype=np.int8)

    row_idx = edge_pairs[:, 0]
    col_idx = edge_pairs[:, 1]

    # source --> destination
    adj_mat[row_idx, col_idx] = 1

    # destination --> source (to create bi-directed/undirected graph)
    adj_mat[col_idx, row_idx] = 1

    return adj_mat


def edge_pairs_to_sparse_adjacency(edge_pairs, dim):
    edges = np.concatenate([edge_pairs, edge_pairs[:, ::-1]], axis=0)  # convert to bi-directed graph
    adjacency_matrix = csr_matrix(
        ([1] * edges.shape[0], (edges[:, 0], edges[:, 1])),
        shape=(dim, dim)
    )

    return adjacency_matrix


def extract_source_idx_list(edge_pairs):
    destination_nodes = np.unique(edge_pairs[:, 1]).tolist()
    source_idx = list()

    for node in destination_nodes:
        mask = edge_pairs[:, 1] == node
        source_idx.append(edge_pairs[mask][:, 0])

    return destination_nodes, source_idx


def aggregate_neighbour_node_feature(adjacency_matrix, features):
    denominator = adjacency_matrix.sum(axis=1)
    denominator = (1/denominator).A[:, 0]
    denominator_len = denominator.shape[0]
    diag_idx = np.arange(denominator_len)
    denominator_diag = csr_matrix((denominator, (diag_idx, diag_idx)), shape=(denominator_len, denominator_len))

    feature_weight = denominator_diag.dot(adjacency_matrix)
    aggregate_features = feature_weight.dot(features)

    return aggregate_features


def filter_edge_pairs(edge_pairs, include_type, node_type_map, return_df=False):
    """
    :param edge_pairs: edge pairs array
    :param include_type: type of node to include
    :param node_type_map: pandas df with node_ids and entity_types column.
        node_ids is an id of a node which the same set of ids as used in edge pairs.
        entity_types is a type of a given node.
    :param return_df: a boolean indicating whether to return results as pandas df (if True) or as array
    :return: an array of filtered edge pairs
    """
    edge_df = pd.DataFrame(edge_pairs, columns=['source', 'destination'])

    edge_df = edge_df.merge(
        node_type_map[['node_ids', 'entity_types']], how='left', left_on='source', right_on='node_ids'
    )
    edge_df.columns = ['source', 'destination', 'source_id', 'source_type']
    edge_df = edge_df.merge(
        node_type_map[['node_ids', 'entity_types']], how='left', left_on='destination', right_on='node_ids'
    )
    edge_df.columns = ['source', 'destination', 'source_id', 'source_type', 'destination_id', 'destination_type']
    edge_df = edge_df[['source', 'destination', 'source_type', 'destination_type']]

    filter_bool = (edge_df['source_type'].isin(include_type) & edge_df['destination_type'].isin(include_type))
    output = edge_df[filter_bool][['source', 'destination']]

    if not return_df:
        output = output[['source', 'destination']].values

    return output

