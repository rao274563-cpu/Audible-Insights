# 🎧 Audible Insights 2.0
## Intelligent Audiobook Recommendation System

Audible Insights is a content-based audiobook recommendation system that helps users discover similar audiobooks based on their titles, descriptions, and genres.

The project uses Natural Language Processing (NLP), TF-IDF vectorization, and weighted cosine similarity to identify and recommend relevant audiobooks through an interactive Streamlit application.

## Features

- Search and select audiobooks from a dataset of 4,293 books.
- Generate personalized audiobook recommendations.
- Analyze book titles, descriptions, and genres separately.
- Combine multiple similarity scores using weighted cosine similarity.
- Display recommended books with author, rating, genre, and similarity score.
- Interactive Streamlit web interface.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Joblib
- Streamlit
- Jupyter Notebook

## Recommendation Methodology

The system follows a content-based filtering approach.

1. Dataset inspection and preprocessing.
2. Extraction and preparation of audiobook genres.
3. Separate TF-IDF vectorization of:
   - Book titles
   - Book descriptions
   - Book genres
4. Calculation of cosine similarity for each feature matrix.
5. Weighted combination of similarity scores.

The final similarity formula is:

**Final Similarity = (0.2 × Title Similarity) + (0.6 × Description Similarity) + (0.2 × Genre Similarity)**

The description receives the highest weight because it provides detailed information about audiobook content.

## Project Structure

```text
Audible-Insights/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── audible_clean.csv
│       ├── audible_advanced_clean.csv
│       └── weighted_similarity.pkl
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_advanced_data_inspection.ipynb
│   ├── 03_data_cleaning.ipynb
│   └── 04_recommendation_system.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rao274563-cpu/Audible-Insights.git
cd Audible-Insights
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

## Model Evaluation

The recommendation system was tested using five audiobooks from different categories.

The qualitative tests demonstrated relevant recommendations in areas such as personal development, mindfulness, animals, storytelling, and sports.

Recommendation relevance was not measured using a formal accuracy metric.

## Limitations

- Recommendation quality depends on the available audiobook metadata.
- Missing or incomplete descriptions and genres may affect relevance.
- Similarity scores indicate textual similarity, not recommendation accuracy or probability.
- The system currently uses content-based filtering and does not incorporate user ratings or collaborative filtering.

## Future Improvements

- Integrate audiobook cover images and additional metadata.
- Improve recommendation relevance through user feedback.
- Develop user profiles and personalized recommendations.
- Deploy the application online.

## Author

**Sachin Kumar Rao**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/rao274563-cpu

LinkedIn : www.linkedin.com/in/sachin-k-rao-535b0b331

---

*Audible Insights 2.0 – Discover your next audiobook.*
