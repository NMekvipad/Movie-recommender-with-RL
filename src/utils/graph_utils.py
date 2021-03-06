import torch
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
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


def extract_symmetric_metapath(metapath, edge_pairs, node_type_map):
    filtered_edges = filter_edge_pairs(
        edge_pairs=edge_pairs, include_type=list(set(metapath)),
        node_type_map=node_type_map, return_df=False
    )

    entity_type_mask = node_type_map['entity_types'].values

    g = nx.Graph()
    g.add_edges_from(filtered_edges.tolist())
    include_nodes = set(list(g.nodes))

    # search until middle of the path
    source_nodes = set((entity_type_mask == metapath[0]).nonzero()[0].tolist())
    source_nodes = source_nodes.intersection(include_nodes)

    middle_node_idx = (len(metapath) - 1) // 2
    middle_node_type = metapath[middle_node_idx]

    explored_pairs = set()
    metapath_instances = defaultdict(list)

    for source in source_nodes:
        for destination in source_nodes:
            if source != destination and (
                    (source, destination) not in explored_pairs or (destination, source) not in explored_pairs
            ):
                # get all path between source and target
                paths = list()
                for p in nx.all_simple_paths(g, source=source, target=destination, cutoff=len(metapath)):
                    if len(p) == len(metapath):
                        is_correct_type = entity_type_mask[p[middle_node_idx]] == middle_node_type

                        if is_correct_type:
                            paths.append(p)

                if len(paths) > 0:
                    metapath_instances[(source, destination)].extend(paths)

                explored_pairs.add((source, destination))
                explored_pairs.add((destination, source))

    return metapath_instances


def extract_metapath(metapath, edge_pairs, node_type_map):
    # adapted from https://github.com/cynricfu/MAGNN/blob/master/utils/preprocess.py

    filtered_edges = filter_edge_pairs(
        edge_pairs=edge_pairs, include_type=list(set(metapath)),
        node_type_map=node_type_map, return_df=False
    )

    entity_type_mask = node_type_map['entity_types'].values

    g = nx.Graph()
    g.add_edges_from(filtered_edges.tolist())
    include_nodes = set(list(g.nodes))

    # search until middle of the path
    source_nodes = set((entity_type_mask == metapath[0]).nonzero()[0].tolist())
    destination_nodes = set((entity_type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0].tolist())
    source_nodes = source_nodes.intersection(include_nodes)
    destination_nodes = destination_nodes.intersection(include_nodes)

    metapath_instances = defaultdict(list)

    # get all path to middle node
    for source in source_nodes:
        for destination in destination_nodes:

            # print(source, destination)
            paths = [
                p for p in nx.all_simple_paths(
                    g, source=source, target=destination, cutoff=(len(metapath) - 1) // 2
                ) if len(p) == ((len(metapath) - 1) // 2)
            ]
            # print(paths)

            if len(paths) > 0:
                return None

            metapath_instances[destination].extend(paths)

    metapath_neighbor_paris = defaultdict(list)
    for key, value in metapath_instances.items():
        for path_left in value:
            for path_right in value:
                metapath_neighbor_paris[(path_left[0], path_right[0])].extend([path_left + path_right[-2::-1]])

    return metapath_neighbor_paris
