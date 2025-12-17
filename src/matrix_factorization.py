import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def train_svd(user_item_matrix, n_components=20, random_state=42):
    """
    Train matrix factorization model using Truncated SVD.
    """
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )

    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_

    return user_factors, item_factors, svd


def svd_recommender(
    user_id,
    user_item_matrix,
    user_factors,
    item_factors,
    movies_df,
    top_n=10
):
    """
    Recommend movies using matrix factorization (SVD).
    """

    if user_id not in user_item_matrix.index:
        raise ValueError("User not found")

    user_index = user_item_matrix.index.get_loc(user_id)

    predicted_ratings = np.dot(
        user_factors[user_index],
        item_factors
    )

    predictions = pd.Series(
        predicted_ratings,
        index=user_item_matrix.columns
    )

    already_rated = user_item_matrix.loc[user_id]
    predictions = predictions[already_rated == 0]

    top_movies = predictions.sort_values(
        ascending=False
    ).head(top_n)

    return movies_df[movies_df["movie_id"].isin(top_movies.index)][
        ["title"]
    ]