import os
import pandas as pd
import torch
import pickle
import numpy as np
import tqdm
from multiprocessing import Pool
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from src.data.movie_data_processing import (
    movielens_genre_map, load_movielens_movie_list
)
from src.utils.graph_utils import (
    extract_source_idx_list, edge_pairs_to_sparse_adjacency, aggregate_neighbour_node_feature, extract_metapath
)
from src.utils.utils import join_string
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.utils import RunTimer

included_staff_type = ['Acting', 'Directing', 'Writing', 'Production']


# Separate data processing into separate function that can be run in stepwise manner via Airflow or other scheduler
def extract_node_and_edge(data_path='dataset'):

    ##################################################
    #                specify data path               #
    ##################################################
    movielens_path = os.path.join(data_path, 'MovieLens20M')
    movielens_genome_score_path = os.path.join(movielens_path, 'genome_scores.csv')
    movielens_genome_tag_path = os.path.join(movielens_path, 'genome_tags.csv')
    tmdb_staff_data_path = os.path.join(data_path, 'staff_data.pickle')
    output_path = os.path.join(data_path, 'graph_input.pickle')

    ##################################################
    #                   movie nodes                  #
    ##################################################
    node_ids = list()
    entity_ids = list()
    entity_types = list()

    movie_df, exclude_movies = load_movielens_movie_list(data_path=movielens_path)
    movies = movie_df[['tmdbId', 'title']].drop_duplicates()
    movie_entity_id = list(movies['tmdbId'])
    entity_ids.extend(movie_entity_id)
    node_ids.extend(list(range(len(movie_entity_id))))
    entity_types.extend(['M'] * len(movie_entity_id))

    ##################################################
    #                   genre nodes                  #
    ##################################################

    genre_df = movielens_genre_map(movie_df)
    genre_df = genre_df[genre_df['genre'] != '(no genres listed)']
    genre_map = {genre: idx for idx, genre in enumerate(genre_df['genre'].unique())}
    genre_df = genre_df.assign(genre_id=genre_df['genre'].map(genre_map))  # (movie, genre) edge

    ##################################################
    #                 movie tag nodes                #
    ##################################################

    genome_score_df = pd.read_csv(movielens_genome_score_path)
    genome_tag_df = pd.read_csv(movielens_genome_tag_path)

    # filter tag with score less than 0.8 (base on histogram)
    genome_score_df = genome_score_df[genome_score_df['relevance'] >= 0.8]
    genome_score_df = genome_score_df.merge(genome_tag_df, how='left', on='tagId')

    tag_counts = genome_score_df['tagId'].value_counts().reset_index()
    tag_counts.columns = ['tagId', 'counts']

    # filter out tag with single movie (input for network)
    genome_score_df = genome_score_df.merge(tag_counts, on='tagId', how='left')
    genome_score_df = genome_score_df[genome_score_df['counts'] > 1]
    genome_score_df = genome_score_df.merge(
        movie_df[['movieId', 'tmdbId']].drop_duplicates(), how='inner', on='movieId'
    )  # (movie, tag) edge

    ##################################################
    #                   staff nodes                  #
    ##################################################

    with open(tmdb_staff_data_path, 'rb') as f:
        staff_data = pickle.load(f)

    staff_data = pd.concat(staff_data)  # (movie, staff) edge
    staff_data = staff_data[staff_data['known_for_department'].isin(included_staff_type)]

    # Acting staff
    actor_ids = list(staff_data[staff_data['known_for_department'] == 'Acting']['id'].unique())

    # Production staff
    production_staff_ids = list(staff_data[staff_data['known_for_department'] != 'Acting']['id'].unique())

    entity_ids.extend(actor_ids + production_staff_ids)
    node_ids.extend(list(range(len(node_ids), len(node_ids) + len(actor_ids + production_staff_ids))))
    entity_types.extend(['A'] * len(actor_ids))
    entity_types.extend(['S'] * len(production_staff_ids))

    if len(entity_ids) != len(node_ids) or len(entity_ids) != len(entity_types):
        raise ValueError('Unequal data shape')

    ##################################################
    #               Create network graph             #
    ##################################################

    node_entity_map = pd.DataFrame({'entity_ids': entity_ids, 'node_ids': node_ids, 'entity_types': entity_types})

    entity_map = dict()
    entities = {'M': 'movie_node', 'A': 'actor_node', 'S': 'staff_node'}
    for entity_cd, entity_type in entities.items():
        df = node_entity_map[node_entity_map['entity_types'] == entity_cd][['entity_ids', 'node_ids']]
        df.columns = ['entity_ids', entity_type]
        entity_map[entity_cd] = df

    genre_df = genre_df.merge(entity_map['M'], how='left', left_on='tmdbId', right_on='entity_ids')
    genome_score_df = genome_score_df.merge(entity_map['M'], how='left', left_on='tmdbId', right_on='entity_ids')

    # ('movie', 'has staff', 'staff')
    production_staff = staff_data.merge(entity_map['M'], how='inner', left_on='movie_id', right_on='entity_ids')
    production_staff = production_staff.merge(entity_map['S'], how='inner', left_on='id', right_on='entity_ids')
    staff_movie_edges = production_staff[['movie_node', 'staff_node']]
    staff_movie_edges = staff_movie_edges.assign(edge_type=[('M', 'has_staff', 'S')] * staff_movie_edges.shape[0])

    # ('movie', 'has actor', 'actor')
    acting_staff = staff_data.merge(entity_map['M'], how='inner', left_on='movie_id', right_on='entity_ids')
    acting_staff = acting_staff.merge(entity_map['A'], how='inner', left_on='id', right_on='entity_ids')
    actor_movie_edges = acting_staff[['movie_node', 'actor_node']]
    actor_movie_edges = actor_movie_edges.assign(edge_type=[('M', 'has_actor', 'A')] * actor_movie_edges.shape[0])

    edge_dfs = [actor_movie_edges, staff_movie_edges]
    for df in edge_dfs:
        df.columns = ['source', 'destination', 'edge_type']

    edge_df = pd.concat(edge_dfs)

    ##################################################
    #                   write output                 #
    ##################################################

    output = {
        'genre_data': genre_df,
        'tag_data': genome_score_df,
        'staff_data': staff_data,
        'edge_data': edge_df,
        'node_entity_type_map': node_entity_map
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    # Pytorch geometric heterogeneous undirected graph data ref
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToUndirected


def generate_movie_sbert_embedding():
    ##################################################
    #                specify data path               #
    ##################################################

    movie_data_path = os.path.join('dataset', 'movie_data.json')
    output_path = os.path.join('dataset', 'movie_node_embedding.pickle')
    graph_data_path = os.path.join('dataset', 'graph_input.pickle')

    if torch.cuda.is_available():
        print('CUDA available')

    with open(graph_data_path, 'rb') as f:
        graph_data = pickle.load(f)

    # load movie id to node id map
    node_entity_type_map = graph_data['node_entity_type_map']
    node_entity_type_map = node_entity_type_map[node_entity_type_map['entity_types'] == 'M']

    ##################################################
    #                Generate embedding              #
    ##################################################

    movie_data = pd.read_json(movie_data_path)
    movie_data = movie_data[~movie_data['movie_id'].duplicated()]  # remove duplicate movie id
    movie_data = node_entity_type_map.merge(movie_data, how='left', left_on='entity_ids', right_on='movie_id')
    movie_data = movie_data.assign(overview_len=movie_data['overview'].apply(lambda x: len(x) if x is not None else 0))

    # SBERT embedding for movie overview text
    overview_movie = movie_data[movie_data['overview_len'] != 0]
    model = SentenceTransformer('sentence-t5-large')
    sentences = list(overview_movie['overview'].values)
    movie_ids_w_overview = list(overview_movie['node_ids'])
    embedding = model.encode(sentences)

    # For movie with no overview text, use random guassian vector with means and variances of SBERT embedding
    # as embedding
    mean = embedding.mean()
    std = embedding.std()

    no_overview_movie = movie_data[movie_data['overview_len'] == 0]
    node_features = torch.normal(mean, std, size=(no_overview_movie.shape[0], embedding.shape[-1]))
    movie_ids_wo_overview = list(no_overview_movie['node_ids'])
    node_features = node_features.numpy()

    # concat and output data
    nodes = np.array(movie_ids_wo_overview + movie_ids_w_overview)
    sorted_args = np.argsort(nodes)
    nodes = nodes[sorted_args]
    features = np.concatenate([node_features, embedding], axis=0)
    features = features[sorted_args, :]

    output = (nodes, features)

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)


def propagate_sbert_movie_embedding():
    graph_data_path = os.path.join('dataset', 'graph_input.pickle')
    embedding_path = os.path.join('dataset', 'movie_node_embedding.pickle')
    sbert_output_path = os.path.join('dataset', 'sbert_node_embedding.npy')
    sbert_rand_output_path = os.path.join('dataset', 'sbert_rand_node_embedding.npy')

    with open(graph_data_path, 'rb') as f:
        graph_data = pickle.load(f)

    edge_data = graph_data['edge_data']
    node_type_map = graph_data['node_entity_type_map']
    edge_pairs = edge_data[['source', 'destination']].values

    ##################################################
    #             create adjacency matrix            #
    ##################################################

    adj_dim = node_type_map['node_ids'].max() + 1
    adjacency_matrix = edge_pairs_to_sparse_adjacency(edge_pairs, adj_dim)

    ##################################################
    #            propagate sbert embedding           #
    ##################################################
    # load sbert movie embedding
    with open(embedding_path, 'rb') as f:
        node_id, embedding = pickle.load(f)

    # filter adjacency for (genre + tag +staff, movie)
    non_movie_start_idx = node_type_map[node_type_map['entity_types'] == 'M']['node_ids'].max() + 1
    adjacency_matrix_filtered = adjacency_matrix[non_movie_start_idx:, :non_movie_start_idx]

    # propagate sbert embedding
    non_movie_embedding = aggregate_neighbour_node_feature(adjacency_matrix_filtered, embedding)

    node_embeddings = np.concatenate([embedding, non_movie_embedding])
    np.save(sbert_output_path, node_embeddings)

    ##################################################
    #              sbert-random embedding            #
    ##################################################

    mean = embedding.mean()
    std = embedding.std()
    rand_features = torch.normal(
        mean, std,
        size=(node_type_map[node_type_map['entity_types'] != 'M'].shape[0], embedding.shape[-1])
    ).numpy()

    sbert_rand_embedding = np.concatenate([embedding, rand_features])
    np.save(sbert_rand_output_path, sbert_rand_embedding)


def generate_genre_tag_movie_embedding():
    """
    Similar style of embedding to MAGNN paper
    """

    graph_data_path = os.path.join('dataset', 'graph_input.pickle')
    output_path = os.path.join('dataset', 'genre_tag_features.npy')

    with open(graph_data_path, 'rb') as f:
        graph_data = pickle.load(f)

    genre_data = graph_data['genre_data']
    tag_data = graph_data['tag_data']
    node_type_map = graph_data['node_entity_type_map']
    movie_nodes = node_type_map[node_type_map['entity_types'] == 'M']
    edge_data = graph_data['edge_data']
    edge_pairs = edge_data[['source', 'destination']].values

    ##################################################
    #             create genre-tag string            #
    ##################################################

    # create movie string
    movie_genre = genre_data[['movie_node', 'genre']]
    movie_genre.columns = ['movie_node', 'genre_tag']
    movie_tag = tag_data[['movie_node', 'tag']]
    movie_tag.columns = ['movie_node', 'genre_tag']
    genre_tag = pd.concat([movie_genre, movie_tag], axis=0)
    genre_tag = genre_tag.assign(genre_tag=genre_tag['genre_tag'].apply(lambda x: x.lower()))
    genre_tag = genre_tag.assign(
        genre_tag_string=genre_tag.groupby('movie_node')['genre_tag'].transform(lambda x: ' '.join(x))
    )
    genre_tag = genre_tag[['movie_node', 'genre_tag_string']].drop_duplicates().reset_index(drop=True)
    genre_tag_data = movie_nodes.merge(genre_tag, how='left', left_on='node_ids', right_on='movie_node')

    vectorizer = CountVectorizer(min_df=2)
    movie_features = vectorizer.fit_transform(genre_tag_data['genre_tag_string'].fillna('').values)
    movie_features = movie_features.toarray()

    ##################################################
    #             create adjacency matrix            #
    ##################################################

    adj_dim = node_type_map['node_ids'].max() + 1
    adjacency_matrix = edge_pairs_to_sparse_adjacency(edge_pairs, adj_dim)

    ##################################################
    #          propagate genre-tag embedding         #
    ##################################################
    # filter adjacency for (genre + tag +staff, movie)
    non_movie_start_idx = node_type_map[node_type_map['entity_types'] == 'M']['node_ids'].max() + 1
    adjacency_matrix_filtered = adjacency_matrix[non_movie_start_idx:, :non_movie_start_idx]

    # propagate genre-tag embedding
    non_movie_embedding = aggregate_neighbour_node_feature(adjacency_matrix_filtered, movie_features)

    node_embeddings = np.concatenate([movie_features, non_movie_embedding])
    np.save(output_path, node_embeddings)


def extract_metapath_wrapper(x):
    node, metapath, edge_pairs, node_type_map = x
    metapath_instances = extract_metapath(metapath, edge_pairs, node_type_map)

    return node, metapath, metapath_instances


def generate_metapath_graph(processes=4):

    graph_data_path = os.path.join('dataset', 'graph_input.pickle')
    output_path = os.path.join('dataset', 'metapath_graph.pickle')

    with open(graph_data_path, 'rb') as f:
        graph_data = pickle.load(f)

    node_type_map = graph_data['node_entity_type_map']
    edge_data = graph_data['edge_data']
    edge_pairs = edge_data[['source', 'destination']].values

    node_type_specific_metapath = {
        'A': [('A', 'M', 'A'), ('A', 'M', 'S', 'M', 'A')],
        'S': [('S', 'M', 'S'), ('S', 'M', 'A', 'M', 'S')],
        'M': [('M', 'A', 'M'), ('M', 'S', 'M')],
    }

    timer = RunTimer()
    timer.get_time_elaspe('Start run')
    output = defaultdict(dict)
    metapaths_tuples = list()

    for node, metapaths in node_type_specific_metapath.items():
        for metapath in metapaths:
            metapaths_tuples.append((node, metapath, edge_pairs, node_type_map))

    with Pool(processes) as p:
        results = list(tqdm.tqdm(p.imap(extract_metapath_wrapper, metapaths_tuples), total=len(metapaths_tuples)))
    timer.get_time_elaspe('Finish metapath extraction')

    for metapath_item in results:
        output[metapath_item[0]][metapath_item[1]] = metapath_item[2]

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    #extract_node_and_edge()
    #generate_movie_sbert_embedding()
    #propagate_sbert_movie_embedding()
    generate_metapath_graph()












