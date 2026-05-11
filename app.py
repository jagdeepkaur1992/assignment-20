import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Title
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# Task 2: Cleaning Logic for the App
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# Load Dataset
df = pd.read_csv('movies_dataset.csv')
df['clean_text'] = (df['genre'] + " " + df['description']).apply(clean_text)

# Task 3 & 4: Vectorization & Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
similarity = cosine_similarity(tfidf_matrix)

# Task 6.1: UI Item Selection (Dropdown)
selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

# Task 6.2: UI Trigger (Button)
if st.button("Recommend Similar Movies"):
    idx = df[df['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    st.subheader(f"Top Recommendations for {selected_movie}:")
    
    # Task 5: Displaying Results
    for i in distances[1:6]:
        st.success(f"🎥 {df.iloc[i[0]].title}")