import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.popularity import popularity_recommender
from src.content_based import build_genre_matrix, content_based_recommender
from src.collaborative import create_user_item_matrix, user_based_recommender

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
    Interactive movie recommendations using Machine Learning
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
        "👥 Collaborative Filtering"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **About**
    - Dataset: MovieLens 100K  
    - Techniques: Similarity-based ML  
    - Tools: Python, Scikit-learn, Streamlit  
    """
)

# -------------------- POPULARITY BASED --------------------
if recommender_type == "🔥 Popularity-Based":
    st.subheader("🔥 Popular Movies")

    st.write(
        "Recommends movies with high average ratings and sufficient number of ratings."
    )

    col1, col2 = st.columns(2)

    with col1:
        min_ratings = st.slider(
            "Minimum number of ratings",
            10, 200, 50
        )

    with col2:
        top_n = st.slider(
            "Number of recommendations",
            5, 20, 10
        )

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
    st.dataframe(
        recommendations.reset_index(drop=True),
        use_container_width=True
    )

# -------------------- CONTENT BASED --------------------
elif recommender_type == "🎯 Content-Based":
    st.subheader("🎯 Find Similar Movies")

    st.write(
        "Select a movie and get recommendations based on genre similarity."
    )

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

# -------------------- COLLABORATIVE FILTERING --------------------
else:
    st.subheader("👥 Personalized Recommendations")

    st.write(
        "Enter a user ID to get personalized recommendations based on similar users."
    )

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