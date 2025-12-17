import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.popularity import popularity_recommender
from src.content_based import build_genre_matrix, content_based_recommender
from src.collaborative import create_user_item_matrix, user_based_recommender
from src.matrix_factorization import train_svd, svd_recommender

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center;">🎬 Movie Recommendation System</h1>
    <p style="text-align:center; font-size:16px;">
    End-to-end Movie Recommendations using Machine Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/raw/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/raw/u.item",
        sep="|",
        encoding="latin-1",
        header=None
    )

    genre_columns = [
        "unknown","Action","Adventure","Animation","Children","Comedy",
        "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
        "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
    ]

    movies.columns = [
        "movie_id", "title", "release_date", "video_release",
        "imdb_url"
    ] + genre_columns

    return ratings, movies

ratings, movies = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Configuration")

recommender_type = st.sidebar.radio(
    "Choose Recommendation Method",
    [
        "🔥 Popularity-Based",
        "🎯 Content-Based",
        "👥 Collaborative Filtering",
        "🧠 Matrix Factorization (SVD)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **About**
    - Dataset: MovieLens 100K  
    - ML: Similarity + Matrix Factorization  
    - Tools: Python, Scikit-learn, Streamlit  
    """
)

# -------------------- POPULARITY --------------------
if recommender_type == "🔥 Popularity-Based":
    st.subheader("🔥 Popular Movies")

    col1, col2 = st.columns(2)
    with col1:
        min_ratings = st.slider("Minimum ratings", 10, 200, 50)
    with col2:
        top_n = st.slider("Top N movies", 5, 20, 10)

    with st.spinner("Computing popular movies..."):
        df = ratings.merge(
            movies[["movie_id", "title"]],
            on="movie_id"
        )
        recommendations = popularity_recommender(
            df,
            top_n=top_n,
            min_ratings=min_ratings
        )

    st.success("Recommendations ready!")
    st.dataframe(recommendations, use_container_width=True)

# -------------------- CONTENT BASED --------------------
elif recommender_type == "🎯 Content-Based":
    st.subheader("🎯 Similar Movies")

    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(movies["title"].unique())
    )

    if st.button("Recommend Similar Movies"):
        with st.spinner("Finding similar movies..."):
            genre_df = movies[["title"] + movies.columns[5:].tolist()]
            genre_matrix = build_genre_matrix(genre_df)
            similarity_matrix = cosine_similarity(genre_matrix)

            recommendations = content_based_recommender(
                selected_movie,
                genre_df,
                similarity_matrix,
                top_n=10
            )

        st.success(f"Movies similar to **{selected_movie}**")
        st.table(recommendations)

# -------------------- USER-BASED CF --------------------
elif recommender_type == "👥 Collaborative Filtering":
    st.subheader("👥 Personalized Recommendations (User-Based)")

    user_id = st.number_input(
        "User ID (1–943)",
        min_value=1,
        max_value=943,
        value=1,
        step=1
    )

    if st.button("Get Recommendations"):
        with st.spinner("Analyzing user preferences..."):
            user_item_matrix = create_user_item_matrix(ratings)

            recommendations = user_based_recommender(
                user_id=user_id,
                user_item_matrix=user_item_matrix,
                movies_df=movies[["movie_id", "title"]],
                top_n=10
            )

        st.success(f"Recommendations for User {user_id}")
        st.table(recommendations)

# -------------------- MATRIX FACTORIZATION (SVD) --------------------
else:
    st.subheader("🧠 Personalized Recommendations (SVD)")

    user_id = st.number_input(
        "User ID (1–943)",
        min_value=1,
        max_value=943,
        value=1,
        step=1
    )

    n_components = st.slider(
        "Number of latent factors",
        min_value=5,
        max_value=50,
        value=20
    )

    if st.button("Get SVD Recommendations"):
        with st.spinner("Training SVD model and generating recommendations..."):
            user_item_matrix = create_user_item_matrix(ratings)

            user_factors, item_factors, _ = train_svd(
                user_item_matrix,
                n_components=n_components
            )

            recommendations = svd_recommender(
                user_id=user_id,
                user_item_matrix=user_item_matrix,
                user_factors=user_factors,
                item_factors=item_factors,
                movies_df=movies[["movie_id", "title"]],
                top_n=10
            )

        st.success(f"SVD Recommendations for User {user_id}")
        st.table(recommendations)

# -------------------- FOOTER --------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Built by Hansiddh Gurram • Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True
)