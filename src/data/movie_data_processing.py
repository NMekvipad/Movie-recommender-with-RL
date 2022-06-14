import os
import requests
import tqdm
import pandas as pd
from multiprocessing import Pool
from functools import partial


api_key = 'ce201838458dccb56d1e2f11830b7250'
api_endpoints = {
    'movie_details': 'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US',
    'staff_details': 'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=en-US'
}


def get_tmdb_movie_details(movie_id, return_json=True):
    if return_json:
        return requests.get(api_endpoints['movie_details'].format(movie_id=str(movie_id), api_key=api_key)).json()
    else:
        return requests.get(api_endpoints['movie_details'].format(movie_id=str(movie_id), api_key=api_key))


def get_tmdb_staff_details(movie_id, return_json=True):
    if return_json:
        return requests.get(api_endpoints['staff_details'].format(movie_id=str(movie_id), api_key=api_key)).json()
    else:
        return requests.get(api_endpoints['staff_details'].format(movie_id=str(movie_id), api_key=api_key))


def get_movie_data(movie_id):
    movie_data = get_tmdb_movie_details(movie_id=movie_id)

    main_production_company_id = None
    main_production_company_name = None
    production_country = None

    if movie_data.get('production_companies') is not None:
        if len(movie_data.get('production_companies')) > 0:
            main_production_company = movie_data['production_companies'][0]
            main_production_company_id = main_production_company['id']
            main_production_company_name = main_production_company['name']

    if movie_data.get('production_countries') is not None:
        if len(movie_data['production_countries']) > 0:
            production_country = movie_data['production_countries'][0]['name']

    output = {
        'title': movie_data.get('title'),
        'overview': movie_data.get('overview'),
        'movie_id': movie_id,
        'original_language': movie_data.get('original_language'),
        'runtime': movie_data.get('runtime'),
        'popularity': movie_data.get('popularity'),
        'vote_average': movie_data.get('vote_average'),
        'vote_count': movie_data.get('vote_count'),
        'main_production_company_id': main_production_company_id,
        'main_production_company_name': main_production_company_name,
        'production_country': production_country
    }

    return output


def get_staff_data(movie_id, n_cast=10):
    movie_people = get_tmdb_staff_details(movie_id=movie_id)
    cast_df = None
    crew_df = None
    output = None

    # if API return no results
    if movie_people.get('id'):
        # get top n movie casts
        if len(movie_people.get('cast')) > 0:
            cast_df = pd.DataFrame(movie_people['cast']).sort_values('order').head(n_cast)
            cast_df = cast_df.assign(movie_id=movie_id)
            cast_df = cast_df[['movie_id', 'gender', 'id', 'name', 'known_for_department']]

        # get all important crews
        if len(movie_people.get('crew')) > 0:
            # For crew data, use
            crew_df = pd.DataFrame(movie_people['crew']).sort_values('job')

            # department: Directing, job: Director
            director_df = crew_df[(crew_df['department'] == 'Directing') & (crew_df['job'] == 'Director')]

            # department: Writing, job: Novel
            # writer_df = crew_df[(crew_df['department'] == 'Writing') & (crew_df['job'] == 'Novel')]

            # department: Writing, job: Screenplay --pick the one with highest popularity (or all)
            # screenplay_df = crew_df[(crew_df['department'] == 'Writing') & (crew_df['job'] == 'Screenplay')]

            # crew_df = pd.concat([director_df, writer_df, screenplay_df])
            crew_df = director_df
            crew_df = crew_df.assign(movie_id=movie_id)
            crew_df = crew_df[['movie_id', 'gender', 'id', 'name', 'known_for_department']]

    if (crew_df is not None) or (cast_df is not None):
        output = pd.concat([cast_df, crew_df]).drop_duplicates()

    return output


def movielens_genre_map(df):
    genre_df = df.set_index(['movieId', 'imdbId', 'tmdbId'])['genres'].apply(
        lambda x: pd.Series(x.split('|'))
    ).stack().reset_index()

    genre_df.columns = ['movieId', 'imdbId', 'tmdbId', 'level_3', 'genre']

    return genre_df[['movieId', 'imdbId', 'tmdbId', 'genre']]


def load_movielens_movie_list(data_path):
    # movie nodes
    movie_df = pd.read_csv(os.path.join(data_path, 'movie.csv'), dtype=str)
    movie_id_df = pd.read_csv(os.path.join(data_path, 'link.csv'), dtype=str)
    movie_df = movie_df.merge(movie_id_df, how='left', on='movieId')

    exclude_movies = movie_df[movie_df['tmdbId'].isnull()]
    movie_df = movie_df[movie_df['tmdbId'].notnull()]
    movie_df = movie_df.assign(tmdbId=movie_df['tmdbId'].astype(int))
    movie_df = movie_df.assign(movieId=movie_df['movieId'].astype(int))

    return movie_df, exclude_movies


def request_movie_data(movie_df, processes=4, return_df=True):
    movie_id = list(movie_df['tmdbId'])

    with Pool(processes) as p:
        results = list(tqdm.tqdm(p.imap(get_movie_data, movie_id), total=len(movie_id)))

    if return_df:
        results = pd.concat(results)

    return results


def request_staff_data(movie_df, n_cast=10, processes=4, return_df=True):
    movie_id = list(movie_df['tmdbId'])
    get_staff_data_partial = partial(get_staff_data, n_cast=n_cast)

    with Pool(processes) as p:
        results = list(tqdm.tqdm(p.imap(get_staff_data_partial, movie_id), total=len(movie_id)))

    if return_df:
        results = pd.concat(results)

    return results


