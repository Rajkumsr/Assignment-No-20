import streamlit as st
import pandas as pd
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

# Cache the data so it only runs ONCE when the app loads (saves memory/time)
@st.cache_resource
def load_and_process_data():
    # 1. Load Datasets
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge and keep only what we need to save RAM
    df = movies.merge(credits, on='title')[['title', 'overview', 'genres', 'cast', 'crew']]
    
    # 2. Extract features safely
    def convert_json_col(text, is_director=False):
        try:
            val_list = ast.literal_eval(text)
            if is_director:
                for i in val_list:
                    if i.get('job') == 'Director':
                        return [i['name'].replace(" ", "")]
                return []
            return [i['name'].replace(" ", "") for i in val_list[:3]]
        except:
            return []

    df['genres'] = df['genres'].apply(lambda x: [i['name'].replace(" ", "") for i in ast.literal_eval(x)] if isinstance(x, str) else [])
    df['cast'] = df['cast'].apply(lambda x: convert_json_col(x))
    df['crew'] = df['crew'].apply(lambda x: convert_json_col(x, is_director=True))
    df['overview'] = df['overview'].fillna('')

    # 3. Create tags
    df['tags'] = df['overview'] + " " + \
                 df['genres'].apply(lambda x: " ".join(x)) + " " + \
                 df['cast'].apply(lambda x: " ".join(x)) + " " + \
                 df['crew'].apply(lambda x: " ".join(x))
                 
    df['clean_text'] = df['tags'].str.lower().replace(r'[^\w\s]', '', regex=True)

    # 4. Vectorize & Similarity
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    similarity = cosine_similarity(tfidf_matrix)
    
    return df, similarity

# Run the loader
df, similarity = load_and_process_data()

# Recommendation Logic
def recommend(movie_title):
    try:
        idx = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        recommended_movies = [df.iloc[i[0]].title for i in distances[1:6]]
        return recommended_movies
    except:
        return ["Movie details not found."]

# UI
st.header("Content-Based Movie Recommender")
selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

if st.button('Get Recommendations'):
    recommendations = recommend(selected_movie)
    st.subheader("You might also like:")
    for movie in recommendations:
        st.write(f"- {movie}")