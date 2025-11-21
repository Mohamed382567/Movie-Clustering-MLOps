import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests # New library for downloading files
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import nltk
from nltk.tokenize import word_tokenize

# --- Configuration for Large File Download ---
FILE_PATH = 'all_models_system.joblib'
# 🛑 IMPORTANT: REPLACE THIS URL WITH YOUR DIRECT MODEL FILE DOWNLOAD LINK!
DOWNLOAD_URL = 'https://drive.google.com/uc?export=download&id=16OLeeaQSKZJ-xPq43RzBM7U8cuZhBgBu' 


def download_file_if_missing():
    """Downloads the model file from the external URL if it doesn't exist locally."""
    if not os.path.exists(FILE_PATH):
        st.info("Downloading large model file from external host (this may take a minute)...")
        try:
            # Use requests to get the file stream
            response = requests.get(DOWNLOAD_URL, stream=True)
            if response.status_code == 200:
                with open(FILE_PATH, 'wb') as f:
                    # Write file content chunk by chunk to handle large files efficiently
                    for chunk in response.iter_content(chunk_size=8192): 
                        f.write(chunk)
                st.success("Download complete. Starting application.")
            else:
                st.error(f"Failed to download model file. Status code: {response.status_code}. Please check the DOWNLOAD_URL.")
                st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Network error during download: {e}. Check your URL or network connection.")
            st.stop()
    return FILE_PATH

# --- Page Setup ---
st.set_page_config(page_title="Movie Clustering MLOps Portfolio", layout="wide", page_icon="🎬")

# --- Load NLTK (Necessary for Doc2Vec tokenization) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Text Cleaning Utility ---
def clean_name(name):
    """Cleans categorical names for concatenation."""
    if not isinstance(name, str): return ""
    return name.lower().replace(" ", "")

# --- Load System (The merged joblib file) ---
@st.cache_resource
def load_system():
    """Loads all models and artifacts into memory once."""
    # Ensure the file is downloaded before attempting to load it
    model_path = download_file_if_missing() 
    
    # Load the system from the local file
    artifacts = joblib.load(model_path)
    
    # Load Sentence Transformer (SBERT)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    return artifacts, sbert

# Attempt to load and handle errors
try:
    artifacts, sbert_model = load_system()
except Exception as e:
    # Catch any error during the loading process after the download
    st.error(f"An error occurred during model loading (joblib.load): {e}. Check if the file path is correct or if the downloaded file is corrupted.")
    st.stop()

# --- Sidebar (Model Configuration and Guide) ---
with st.sidebar:
    st.title("⚙️ Model Settings")
    
    # Model Selection List
    model_choice = st.radio(
        "Choose Clustering Algorithm:",
        ("K-Means", "HDBSCAN", "GMM"),
        index=0
    )
    
    st.markdown("---")
    
    # Project Guide and Disclaimers (Updated Content)
    with st.expander("📖 Project Guide & Disclaimers", expanded=True):
        
        # 1. Project Scope Note (Disclaimer)
        st.warning("""
        **⚠️ Project Scope Note:**
        This is primarily an **educational MLOps project** designed to demonstrate the end-to-end pipeline (Data Engineering → Modeling → Deployment). 
        
        While functional, the results are based on a specific dataset. In a real-world business scenario, this system would be scaled with:
        - Larger, real-time datasets.
        - More computational resources.
        - Human-in-the-loop feedback for fine-tuning.
        """)
        
        # 2. Expectations Warning
        st.info("""
        **🤖 Model vs. Human Intuition:**
        Clustering finds mathematical patterns in high-dimensional space (Budget, NLP embeddings, etc.) that might not always align 100% with human genre labels. 
        *Example: A 'Comedy' might be grouped with 'Action' if it has a high budget and similar keyword embeddings.*
        """)
        
        st.markdown("---")
        
        # 3. Dynamic Cluster Definitions
        st.markdown("### 🏷️ Cluster Definitions")
        
        if model_choice == "K-Means":
            st.markdown("""
            **K-Means Groups (Balanced):**
            - **Cluster 0:** Profit-Driven Comedy/Action Mix (Commercial focus).
            - **Cluster 1:** Low-Performing Dramas (Recent, low revenue).
            - **Cluster 2:** Underperformers (Newer movies, below average ratings).
            - **Cluster 3:** **Blockbuster Hits** (High Revenue, High Quality).
            - **Cluster 4:** Niche/Cult Films (Critically panned but profitable).
            """)
            
        elif model_choice == "GMM":
            st.markdown("""
            **GMM Groups (Probabilistic):**
            - **Cluster 0:** Classic Cinema (High profit, older release years).
            - **Cluster 1:** Low-Performing Dramas.
            - **Cluster 2:** Underperformers.
            - **Cluster 3:** **Commercial Hits** (High popularity mix).
            - **Cluster 4:** Niche Profitable.
            """)
            
        elif model_choice == "HDBSCAN":
            st.markdown("""
            **HDBSCAN Groups (Density-Based):**
            - **Label -1:** **Noise / Outliers** (Unique movies that don't fit standard patterns).
            - **Cluster 0:** Standard Commercial Films.
            - **Cluster 1:** Low-Budget / Indie Projects.
            - **Cluster 2:** High-Performance Blockbusters.
            - **Cluster 3:** Genre-Specific Niche (e.g., Horror/Thriller).
            """)
    
    st.markdown("---")
    st.markdown("Developed for MLOps Portfolio Demonstration.")


# --- Main Interface ---
st.title("🎬 MLOps Movie Classifier")
st.caption(f"Current System Model: **{model_choice}**")

# Input Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numerical & Scale Inputs")
    budget = st.number_input("Budget ($)", 0.0, value=1000000.0)
    revenue = st.number_input("Revenue ($)", 0.0, value=2000000.0)
    vote_average = st.slider("Vote Average", 0.0, 10.0, 6.0)
    vote_count = st.number_input("Vote Count", 0, value=100)
    popularity = st.number_input("Popularity", 0.0, value=10.0)
    runtime = st.number_input("Runtime (min)", 0.0, value=90.0)
    release_year = st.number_input("Release Year", 1900, 2030, 2023)

with col2:
    st.subheader("Textual & Categorical Inputs")
    overview = st.text_area("Overview (Movie Description)", "A brief description of the movie's plot and themes...")
    genres = st.text_input("Genres (Comma Separated)", "Action, Adventure, Thriller")
    keywords = st.text_input("Keywords (Comma Separated)", "hero, battle, epic")
    cast = st.text_input("Cast (Comma Separated)", "Actor One, Actor Two")
    director = st.text_input("Director", "Director Name")

# Prediction Button
if st.button("🔮 Predict Cluster"):
    with st.spinner('Analyzing inputs and predicting cluster...'):
        
        # 1. Select Model Data
        model_key = model_choice.lower().replace("-", "") # e.g., "k-means" -> "kmeans"
        
        if model_key not in artifacts:
            st.error(f"Sorry, data for the {model_choice} model is not available in the artifact file.")
            st.stop()
            
        selected_model_data = artifacts[model_key]
        
        # 2. Feature Engineering & Preprocessing
        cleaned_genres = [clean_name(x) for x in genres.split(',') if x.strip()]
        cleaned_keywords = [clean_name(x) for x in keywords.split(',') if x.strip()]
        cleaned_cast = [clean_name(x) for x in cast.split(',') if x.strip()]
        cleaned_director = [clean_name(director)] if director else []
        
        all_features_str = ' '.join(cleaned_genres + cleaned_keywords + cleaned_cast + cleaned_director)
        
        profit_ratio = (revenue - budget) / (budget + 1e-6) if budget != 0 else 0.0
        profit_ratio_transformed = np.sign(profit_ratio) * np.log1p(np.abs(profit_ratio))
        
        raw_num = pd.DataFrame([{
            'vote_average': vote_average, 'vote_count': vote_count, 'popularity': popularity,
            'budget_log': np.log1p(budget), 'revenue_log': np.log1p(revenue), 'runtime': runtime,
            'profit_ratio': profit_ratio_transformed, 'release_year': release_year
        }])
        
        # 3. Transformation & Vectorization
        scaled_num = artifacts['scaler'].transform(raw_num)
        sbert_vec = sbert_model.encode([overview if overview else 'No description'])[0]
        
        tokens = word_tokenize(all_features_str) if all_features_str else []
        doc2vec_vec = artifacts['doc2vec_model'].infer_vector(tokens, epochs=20)
        
        combined_vec = np.concatenate([
            scaled_num, 
            doc2vec_vec.reshape(1, -1), 
            sbert_vec.reshape(1, -1)
        ], axis=1)
        
        final_vec = artifacts['umap_model'].transform(combined_vec)
        
        # 4. Prediction Logic (Closest Centroid)
        centroids = selected_model_data['centroids']
        label_map = selected_model_data['map']
        
        dists = euclidean_distances(final_vec, centroids)[0]
        dist_map = sorted([(d, i) for i, d in enumerate(dists)], key=lambda x: x[0])
        
        pred_idx = dist_map[0][1]
        pred_label = label_map[pred_idx]
        cluster_name = selected_model_data['names'].get(pred_label, f"Cluster {pred_label}")
        
        # --- Display Results ---
        st.success(f"🎥 Predicted Cluster ({model_choice}): **{cluster_name}**")
        
        # Display Proximity Info
        st.info(f"Proximity to Cluster Center: {dist_map[0][0]:.4f}")
        
        # Display Second closest alternative
        if len(dist_map) > 1:
            sec_idx = dist_map[1][1]
            sec_label = label_map[sec_idx]
            sec_name = selected_model_data['names'].get(sec_label, f"Cluster {sec_label}")
            st.text(f"Closest Alternative: {sec_name} (Distance: {dist_map[1][0]:.4f})")