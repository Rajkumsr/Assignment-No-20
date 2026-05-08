## Movie Matcher: Content-Based Recommendation System
Welcome to Movie Matcher, an end-to-end machine learning application that suggests movies based on plot descriptions, genres, and cast metadata. This project was built as part of an AI/ML workflow assignment, covering everything from raw data preprocessing to cloud deployment.

## Live Demo
You can access the deployed application here: https://assignment-no-20.onrender.com

## Project Overview
This system uses a Content-Based Filtering approach. Unlike collaborative filtering which relies on user ratings, this model recommends movies by analyzing the "DNA" of the film itself.

# How it Works:
Data Merging: Combines TMDB movie metadata with credits (cast/crew) datasets.

Feature Engineering: Extracts the Director and the top 3 lead actors to create a comprehensive "tags" metadata for each film.

NLP Pipeline: * Text cleaning (lowercase, punctuation removal).
TF-IDF Vectorization to convert text into numerical features.
Cosine Similarity to calculate the mathematical distance between movies.
Interface: A sleek, minimal UI built with Streamlit.

🛠️ Tech Stack
Language: Python 3.x
Libraries: Pandas, Scikit-Learn, NumPy
Web Framework: Streamlit
Deployment: Render
Environment: Git & GitHub

# Project Structure
├── app.py              # Streamlit Web Interface logic
├── model_script.py     # Data processing and ML similarity matrix generation
├── movie_list.pkl      # Pickled movie dataframe (Generated)
├── similarity.pkl      # Pickled similarity matrix (Generated)
├── requirements.txt    # List of dependencies for deployment
└── README.md   


# Installation & Setup
To run this project locally, follow these steps:

# Clone the repository:
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

# Install dependencies:
pip install -r requirements.txt

# Prepare the Data:
Make sure you have tmdb_5000_movies.csv and tmdb_5000_credits.csv in the root folder. Then run:
python model_script.py

# Launch the App:
streamlit run app.py

# Dataset Reference
This project uses the TMDB 5000 Movie Dataset available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/chaitanyasood1/tmdb-5000-movies?select=tmdb_5000_credits.csv