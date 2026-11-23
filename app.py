
import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "audible_advanced_clean.csv"

# Page configuration
st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Generate TF-IDF feature matrices
@st.cache_resource
def load_features(df):

    title_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    description_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    genre_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    title_matrix = title_vectorizer.fit_transform(
        df["Book Name"].fillna("")
    )

    description_matrix = description_vectorizer.fit_transform(
        df["Description"].fillna("")
    )

    genre_matrix = genre_vectorizer.fit_transform(
        df["Genres"].fillna("")
    )

    return title_matrix, description_matrix, genre_matrix


df = load_data()

title_matrix, description_matrix, genre_matrix = load_features(df)

# Application header
st.title("🎧 Audible Insights")
st.subheader("Intelligent Audiobook Recommendation System")

st.write(
    "Discover audiobooks based on their titles, descriptions, "
    "and genres using content-based filtering."
)

st.divider()

# Audiobook selection
st.header("Find Your Next Audiobook")

book_list = sorted(df["Book Name"].dropna().unique())

selected_book = st.selectbox(
    "Select an audiobook",
    book_list
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# Generate recommendations
if st.button("Get Recommendations", type="primary"):

    selected_index = df.index[
        df["Book Name"] == selected_book
    ][0]

    # Calculate similarity for the selected audiobook
    title_scores = cosine_similarity(
        title_matrix[selected_index],
        title_matrix
    ).flatten()

    description_scores = cosine_similarity(
        description_matrix[selected_index],
        description_matrix
    ).flatten()

    genre_scores = cosine_similarity(
        genre_matrix[selected_index],
        genre_matrix
    ).flatten()

    # Weighted similarity
    scores = (
        0.2 * title_scores +
        0.6 * description_scores +
        0.2 * genre_scores
    )

    results = df.copy()
    results["Similarity Score"] = scores * 100

    results = results[
        (results["Book Name"] != selected_book) &
        (results["Similarity Score"] > 0)
    ]

    results = results.sort_values(
        "Similarity Score",
        ascending=False
    )

    results = results.drop_duplicates(
        subset="Book Name",
        keep="first"
    )

    results = results[
        [
            "Book Name",
            "Author",
            "Rating",
            "Genres",
            "Similarity Score"
        ]
    ].head(top_n).round({"Similarity Score": 2})

    st.subheader("Recommended Audiobooks")

    if results.empty:
        st.warning("No similar audiobooks found.")
    else:
        st.success(f"Found {len(results)} recommendations.")

        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

st.divider()

st.caption(
    "Powered by TF-IDF Vectorization and Weighted Cosine Similarity | "
    "Audible Insights 2.0"
)
