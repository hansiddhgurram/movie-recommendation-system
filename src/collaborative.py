import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_user_item_matrix(df):
    return df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

def user_based_recommender(user_id, user_item_matrix, movies_df, top_n=10):
    if user_id not in user_item_matrix.index:
        raise ValueError("User not found")

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(ascending=False).iloc[1:]

    weighted_ratings = np.dot(
        similar_users.values,
        user_item_matrix.loc[similar_users.index]
    )

    recommendations = pd.Series(
        weighted_ratings,
        index=user_item_matrix.columns
    )

    rated = user_item_matrix.loc[user_id]
    recommendations = recommendations[rated == 0]

    top_movies = recommendations.sort_values(ascending=False).head(top_n)

    return movies_df[movies_df["movie_id"].isin(top_movies.index)][["title"]]