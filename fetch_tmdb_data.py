import os
import pickle
import pandas as pd
from src.data.movie_data_processing import request_movie_data, request_staff_data, load_movielens_movie_list


if __name__ == '__main__':
    movie_df, exclude_movies = load_movielens_movie_list(data_path=os.path.join('dataset', 'MovieLens20M'))
    movie_data_df = request_movie_data(movie_df)
    movie_data_df.to_json(os.path.join('dataset', 'movie_data.json'))
    staff_data_df = request_staff_data(movie_df, n_cast=3, return_df=False)

    with open(os.path.join('dataset', 'staff_data.pickle'), 'wb') as f:
        pickle.dump(staff_data_df, f)

    # staff_data_df = pd.concat(staff_data_df)
    # staff_data_df.to_json("staff_data.json")

