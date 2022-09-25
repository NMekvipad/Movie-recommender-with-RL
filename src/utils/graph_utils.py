import torch
import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from collections import defaultdict
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


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
                ) if len(p) == ((len(metapath) - 1) // 2) + 1
            ]

            metapath_instances[destination].extend(paths)

    metapath_neighbor_paris = defaultdict(list)
    for key, value in metapath_instances.items():
        for path_left in value:
            for path_right in value:
                if path_left != path_right:
                    metapath_neighbor_paris[(path_left[0], path_right[0])].extend([path_left + path_right[-2::-1]])

    edge_index_updated = list()
    metapath_indices = list()

    for key, values in metapath_neighbor_paris.items():
        for path in values:
            edge_index_updated.append(key)
            metapath_indices.append(path)

    return edge_index_updated, metapath_indices


def sample_nodes(sampling_nodes, sample_size, batch_size=1, replace=True, p=None, seed=0):
    random_state = RandomState(MT19937(SeedSequence(seed)))

    if replace:
        last_batch_idx = sample_size // batch_size
        last_batch_size = sample_size % batch_size
        num_batch = last_batch_idx + (0 if last_batch_size == 0 else 1)

        for i in range(num_batch):
            yield random_state.choice(sampling_nodes, size=batch_size, replace=True, p=p)

    else:
        sampled_nodes = list()

        while True:
            filter_bool = np.invert(np.isin(sampling_nodes, sampled_nodes))
            sampling_nodes = sampling_nodes[filter_bool]
            if p is not None:
                p = p[filter_bool]
                p = p/p.sum()  # re-normalize probability after filtering

            if sampling_nodes.shape[0] == 0:
                break
            elif sampling_nodes.shape[0] < batch_size:
                samples = sampling_nodes
                sampled_nodes.extend(samples.tolist())
                yield samples
            else:
                samples = random_state.choice(sampling_nodes, size=batch_size, replace=False, p=p)
                sampled_nodes.extend(samples.tolist())
                yield samples


# create customer data set as PyGeo cannot handle sampling at metapath level
class MetapathGraphData(Dataset):
    def __init__(self, metapath_graph, samples_nodes, depth, seed=0):  #, num_neighbors=None
        self.idx_to_node_map = {idx: node for idx, node in enumerate(samples_nodes)}
        self.metapath_graph = metapath_graph
        self.depth = depth
        self.random_state = RandomState(MT19937(SeedSequence(seed)))

        #TODO: implement filtering by num neighbors
        #if num_neighbors is None:
        #    self.num_neighbors = [None] * self.depth
        #elif len(num_neighbors) == depth:
        #    self.num_neighbors = num_neighbors
        #else:
        #    ValueError('num_neighbors should be a list like structure with length equal to sampling depth')

        # convert to array
        for node_type, metapaths in self.metapath_graph.items():
            for metapath in metapaths.keys():
                metapath_edges, metapath_members = self.metapath_graph[node_type][metapath]
                metapath_edges = np.array([list(i) for i in metapath_edges])
                metapath_members = np.array(metapath_members)
                self.metapath_graph[node_type][metapath] = (metapath_edges, metapath_members)

    def __get_nodes_on_metapath(self, metapath_graph):
        nodes = set()
        for metapaths in metapath_graph.values():
            for metapath_edges, metapath_members in metapaths.values():
                new_nodes = list(np.unique(metapath_edges)) + list(np.unique(metapath_members))
                nodes = nodes.union(new_nodes)

        return list(nodes)

    def __get_subgraph_by_destination_nodes(self, node_ids):
        metapath_subgraph = defaultdict(dict)

        for node_type, metapaths in self.metapath_graph.items():
            for metapath in metapaths.keys():
                metapath_edges, metapath_members = self.metapath_graph[node_type][metapath]
                filtering_args = np.argwhere(np.isin(metapath_edges[:, 1], node_ids)).ravel()

                metapath_subgraph[node_type][metapath] = (
                    metapath_edges[filtering_args], metapath_members[filtering_args]
                )

        return metapath_subgraph

    def __len__(self):
        return len(self.idx_node_map)

    def __getitem__(self, idx):
        node_id = self.idx_to_node_map.get(idx)
        sample = None
        source_nodes = [node_id]

        for i in range(self.depth):
            sample = self.__get_subgraph_by_destination_nodes(source_nodes)
            source_nodes = self.__get_nodes_on_metapath(sample)

        return sample


def get_metapath_graph_size(metapath_graph):
    size = 0
    for node_type, metapaths in metapath_graph.items():
        for metapath, value in metapaths.items():
            size += value[0].shape[0]

    return size

# TODO: Add function to convert graph to bi-directional graph
# TODO: Check out sampling method https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_sampler.html
