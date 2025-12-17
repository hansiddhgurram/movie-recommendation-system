import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_genre_matrix(movies_df):
    """
    Build a movie-genre matrix.
    """
    genre_cols = movies_df.columns[1:]
    return movies_df[genre_cols]

def content_based_recommender(movie_title, movies_df, similarity_matrix, top_n=10):
    """
    Recommend movies similar to a given movie using cosine similarity.
    """
    if movie_title not in movies_df["title"].values:
        raise ValueError("Movie not found")

    idx = movies_df[movies_df["title"] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_movies = similarity_scores[1 : top_n + 1]
    indices = [i[0] for i in top_movies]

    return movies_df.iloc[indices][["title"]]