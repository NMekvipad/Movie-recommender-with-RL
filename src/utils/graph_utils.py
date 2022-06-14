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




