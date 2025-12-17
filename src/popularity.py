import pandas as pd

def popularity_recommender(df, top_n=10, min_ratings=50):
    """
    Recommend movies based on average rating and number of ratings.
    """
    stats = (
        df.groupby("title")
        .agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        )
        .reset_index()
    )

    filtered = stats[stats["num_ratings"] >= min_ratings]

    recommendations = filtered.sort_values(
        by=["avg_rating", "num_ratings"],
        ascending=False
    )

    return recommendations.head(top_n)