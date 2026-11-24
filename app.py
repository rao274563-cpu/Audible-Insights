
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .hero {
        padding: 45px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81);
        color: white;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 38px;
        color: white;
    }

    .hero p {
        font-size: 17px;
        color: #dbeafe;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    h2, h3 {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. LOAD DATASET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "audible_advanced_clean.csv"
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data.reset_index(drop=True)


df = load_data()


# =====================================================
# 3. LOAD TF-IDF FEATURES
# =====================================================

@st.cache_resource
def load_features(data):

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
        data["Book Name"].fillna("")
    )

    description_matrix = description_vectorizer.fit_transform(
        data["Description"].fillna("")
    )

    genre_matrix = genre_vectorizer.fit_transform(
        data["Genres"].fillna("")
    )

    return title_matrix, description_matrix, genre_matrix


title_matrix, description_matrix, genre_matrix = load_features(df)


# =====================================================
# 4. HORIZONTAL NAVIGATION
# =====================================================

st.markdown(
    '<div class="main-title">🎧 Audible Insights</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Audiobook Discovery Platform'
    '</div>',
    unsafe_allow_html=True
)

pages = [
    "🏠 Home",
    "📚 Dataset Explorer",
    "📊 EDA",
    "✨ Get Recommendation",
    "ℹ️ About"
]

# Centered horizontal navigation
left, center, right = st.columns([1.8, 5, 0.4])

with center:
    selected_page = st.radio(
        "Main Navigation",
        pages,
        horizontal=True,
        label_visibility="collapsed"
    )

st.divider()


# =====================================================
# 5. HOME PAGE
# =====================================================

if selected_page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>Discover Your Next Great Listen.</h1>
        <p>
        Explore thousands of audiobooks and find your next favourite
        through intelligent content-based recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Platform Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Audiobooks",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Unique Authors",
            f"{df['Author'].nunique():,}"
        )

    with col3:
        st.metric(
            "Recommendation Features",
            "3"
        )

    st.divider()

    st.subheader("Explore Audible Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📚 Dataset Explorer")
        st.write(
            "Explore audiobook records, search titles, "
            "and examine authors, ratings and genres."
        )

        st.markdown("### 📊 Exploratory Data Analysis")
        st.write(
            "Understand audiobook ratings, popular authors "
            "and the distribution of available data."
        )

    with c2:
        st.markdown("### ✨ Intelligent Recommendations")
        st.write(
            "Discover similar audiobooks using TF-IDF "
            "and weighted cosine similarity."
        )

        st.markdown("### ℹ️ About the Project")
        st.write(
            "Learn about the recommendation methodology, "
            "technologies and project objectives."
        )

    st.divider()

    st.subheader("Featured Audiobooks")

    featured = df.dropna(subset=["Book Name"]).head(5)

    st.dataframe(
        featured[
            ["Book Name", "Author", "Rating", "Genres"]
        ],
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 6. DATASET EXPLORER
# =====================================================

elif selected_page == "📚 Dataset Explorer":

    st.title("📚 Dataset Explorer")

    st.write(
        "Search, filter and explore the audiobook dataset."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search by audiobook title or author",
            placeholder="Enter book name or author..."
        )

    with col2:
        min_rating = st.selectbox(
            "Minimum Rating",
            ["All", "3.0", "3.5", "4.0", "4.5"]
        )

    filtered_df = df.copy()

    if search_query:
        mask = (
            filtered_df["Book Name"].fillna("").str.contains(
                search_query,
                case=False
            )
            |
            filtered_df["Author"].fillna("").str.contains(
                search_query,
                case=False
            )
        )

        filtered_df = filtered_df[mask]

    if min_rating != "All":
        filtered_df = filtered_df[
            filtered_df["Rating"] >= float(min_rating)
        ]

    st.metric(
        "Matching Audiobooks",
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df[
            ["Book Name", "Author", "Rating", "Genres"]
        ],
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 7. EXPLORATORY DATA ANALYSIS
# =====================================================

elif selected_page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the statistical characteristics "
        "and patterns in the audiobook dataset."
    )

    # Rating statistics
    ratings = pd.to_numeric(
        df["Rating"],
        errors="coerce"
    ).dropna()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Rating",
            f"{ratings.mean():.2f}" if not ratings.empty else "N/A"
        )

    with col2:
        st.metric(
            "Highest Rating",
            f"{ratings.max():.2f}" if not ratings.empty else "N/A"
        )

    with col3:
        st.metric(
            "Books with Ratings",
            f"{len(ratings):,}"
        )

    st.divider()

    # Rating distribution
    st.subheader("1. Audiobook Rating Distribution")

    if not ratings.empty:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(
            ratings,
            bins=20,
            edgecolor="black"
        )

        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Audiobooks")
        ax.set_title("Distribution of Audiobook Ratings")

        st.pyplot(fig)
        plt.close(fig)

    # Top authors
    st.subheader("2. Top 10 Authors by Audiobook Count")

    top_authors = (
        df["Author"]
        .dropna()
        .value_counts()
        .head(10)
        .sort_values()
    )

    if not top_authors.empty:
        st.bar_chart(top_authors)

    # Top-rated audiobooks
    st.subheader("3. Top Rated Audiobooks")

    top_rated = df.copy()
    top_rated["Rating"] = pd.to_numeric(
        top_rated["Rating"],
        errors="coerce"
    )

    top_rated = top_rated.dropna(
        subset=["Rating"]
    ).sort_values(
        "Rating",
        ascending=False
    ).head(10)

    st.dataframe(
        top_rated[
            ["Book Name", "Author", "Rating", "Genres"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Missing values
    st.subheader("4. Dataset Missing Value Analysis")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing_values = missing_values[
        missing_values > 0
    ]

    if not missing_values.empty:
        st.bar_chart(missing_values)
    else:
        st.success("No missing values found in the dataset.")


# =====================================================
# 8. GET RECOMMENDATION
# =====================================================

elif selected_page == "✨ Get Recommendation":

    st.title("✨ Audiobook Recommendations")

    st.write(
        "Select an audiobook and discover similar books "
        "based on content, title and genre."
    )

    book_list = sorted(
        df["Book Name"].dropna().unique()
    )

    selected_book = st.selectbox(
        "Choose an Audiobook",
        book_list
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button(
        "✨ Get Recommendations",
        type="primary",
        use_container_width=True
    ):

        selected_index = df.index[
            df["Book Name"] == selected_book
        ][0]

        # Calculate individual similarities
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
            0.2 * title_scores
            + 0.6 * description_scores
            + 0.2 * genre_scores
        )

        results = df.copy()

        results["Similarity Score"] = scores * 100

        results = results[
            (results["Book Name"] != selected_book)
            & (results["Similarity Score"] > 0)
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
        ].head(top_n).round(
            {"Similarity Score": 2}
        )

        st.divider()

        st.subheader("Recommended Audiobooks")

        if results.empty:
            st.warning("No similar audiobooks found.")

        else:
            st.success(
                f"{len(results)} recommendations generated!"
            )

            st.dataframe(
                results.reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# 9. ABOUT PAGE
# =====================================================

elif selected_page == "ℹ️ About":

    st.title("ℹ️ About Audible Insights")

    st.markdown("""
    ## Intelligent Audiobook Recommendation System

    Audible Insights 2.0 is a content-based audiobook
    recommendation platform developed to help users
    discover audiobooks based on their content.

    ### Project Objectives

    - Explore and analyze audiobook metadata.
    - Apply Natural Language Processing techniques.
    - Generate relevant audiobook recommendations.
    - Provide an interactive user experience.

    ### Recommendation Methodology

    The system uses three independent TF-IDF feature matrices:

    1. **Title Similarity – 20%**
    2. **Description Similarity – 60%**
    3. **Genre Similarity – 20%**

    The three similarity scores are combined using
    weighted cosine similarity.

    ### Technologies Used

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - TF-IDF Vectorization
    - Cosine Similarity
    - Streamlit
    - Matplotlib

    ### Project Dataset

    The application currently uses a processed dataset
    containing 4,293 audiobooks.

    ### Developer

    **Sachin Kumar Rao**

    B.Tech – Computer Science and Engineering

    GitHub: [Audible Insights Repository](https://github.com/rao274563-cpu/Audible-Insights)
    
    Linkedin: [Connect with me](https://www.linkedin.com/in/sachin-k-rao-535b0b331)
    """)
    
    


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "🎧 Audible Insights 2.0 | Intelligent Audiobook Discovery"
    "</div>",
    unsafe_allow_html=True
)
