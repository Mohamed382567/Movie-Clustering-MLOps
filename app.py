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
              Numerical Profile: Average quality, slightly positive profit, older movies on average.
              Characteristics: A large, commercially focused group. Defined by a slightly positive profit margin (despite moderate popularity) and a broad mix of popular genres like Comedy, Thriller, and Action. These are solid, mainstream films that consistently make money.
            - **Cluster 1:** Low-Performing Dramas (Recent, low revenue).
              Numerical Profile: Sub-average quality, negative profit, recent. Very low popularity.
              Characteristics: These are recent films, mainly Dramas and Romances, that struggle to gain traction (very low popularity) and are generally losing money (-0.016 profit ratio). They often represent a large chunk of newly released, quickly forgotten titles.
            - **Cluster 2:** Underperformers (Newer movies, below average ratings).
              Numerical Profile: Lower quality, negative profit, newest films on average.
              Characteristics: Represents the very newest films in the dataset (highest mean release year). They are defined by poor performance, slightly lower quality than Cluster 1, and low visibility.
            - **Cluster 3:** **Blockbuster Hits** (High Revenue, High Quality).
              Numerical Profile: Highest quality (0.501), highest popularity (0.718), relatively recent.
              Characteristics: The "Mega-Hit" category. These films combine high critical acclaim with massive public interest, often dominated by large-scale Action, Adventure, and Thriller genres. This is the goal for major studio tentpoles.
            - **Cluster 4:** Niche/Cult Films (Critically panned but profitable).
              Numerical Profile: Extremely low quality (-1.169), but surprisingly positive profit (0.027), very recent.
              Characteristics: This cluster is defined by the sharp contradiction between high negative critical perception (worst quality score) and a positive profit ratio. These are often low-budget, niche films (like some horror or direct-to-video titles) that achieve a small profit despite negative reviews.
            """)
            
        elif model_choice == "GMM":
            st.markdown("""
            **GMM Groups (Probabilistic):**
            - **Cluster 0:** Classic Cinema (High profit, older release years).
              Numerical Profile: High quality, highest profit ratio (0.107), but extremely old (-1.379). Low current popularity is expected due to age.
              Characteristics: This cluster separates older, classic films (like '42nd Street' or 'The Nun's Story') that were highly successful and critically acclaimed in their time, resulting in a high historical profit ratio. The model isolated them based on the low release year Z-score.
            - **Cluster 1:** Low-Performing Dramas.
              Numerical Profile: Sub-average quality, negative profit, recent. Very low popularity.
              Characteristics: Similar to the K-Means result. These are recent titles, mostly Dramas and Romances, that struggle commercially (negative profit ratio) and critically (sub-average rating). They are quickly forgotten releases.
            - **Cluster 2:** Underperformers.
              Numerical Profile: Lower quality, negative profit, newest films on average.
              Characteristics: Represents the newest batch of released films that performed poorly on average. Defined by lower mean quality and a moderate lack of popularity, struggling to perform in the market.
            - **Cluster 3:** **Commercial Hits** (High popularity mix).
              Numerical Profile: Above average quality, highest current popularity (0.332), recent.
              Characteristics: This is the large "Mainstream Hit" category. Defined primarily by high popularity and a strong presence across all major commercial genres (Drama, Action, Comedy, Thriller). These are the films that dominate the public conversation.
            - **Cluster 4:** Niche Profitable.
              Numerical Profile: Extremely low quality (-1.180), yet positive profit (0.027), recent.
              Characteristics: A highly contradictory cluster. Films here receive terrible reviews (worst quality score) but achieve a positive profit margin. This points to low-budget, niche, or genre-specific films (like specific horror or low-budget comedies) that are cost-effective despite poor critical reception.
            """)
            
        elif model_choice == "HDBSCAN":
            st.markdown("""
            **HDBSCAN Groups (Density-Based):**
            - **Label -1:** **Noise / Outliers** (Unique movies that don't fit standard patterns):
              Numerical Profile: vote_average (-0.416), popularity (-0.485), release_year (-0.585) - Generally below average and older.
              Description: These are points that the clustering algorithm couldn't reliably assign to any main group. They often include extremely unique or corrupted data entries.
            - **Cluster 0:** Standard Commercial Films.
              Numerical Profile: Low quality and popularity, but average profit, relatively recent.
              Characteristics: These are modern films, mostly dramas and comedies, that perform poorly in terms of popularity but maintain an average critical rating. They are generally forgettable films not capturing wide public attention.
            - **Cluster 1:** Extremely Poor Quality Outliers.
              Numerical Profile: Extremely low quality (-1.637), very low popularity, older.
              Characteristics: This cluster consists of some of the worst-rated films in the dataset, often cheap horror or generic independent films that barely register a score.
            - **Cluster 2:** High-Performance Blockbusters.
              Numerical Profile: Above average quality (0.185), high popularity (0.244), slightly positive profit.
              Characteristics: This is the largest, most successful, and most diverse group. It represents popular blockbuster and major studio releases that are well-received (above average) and widely consumed across all major genres (Action, Drama, Comedy, Thriller). This is the 'Hit' category.
            - **Cluster 3:** Genre-Specific Niche (e.g., Horror/Thriller).
              Numerical Profile: Highest quality (0.254), highest profit ratio (0.045), low popularity, but relatively recent.
              Characteristics: These are critically acclaimed films (highest mean vote average) that managed to generate good relative profit, often being niche dramas or independent comedies with strong word-of-mouth rather than mass market blockbusters.
            - **cluster 4:** Classic (Older) Cinema Mix.
              Numerical Profile: Very old (-2.086), low popularity, average quality.
              Characteristics: Defined purely by age. These are older, non-recent films (predominantly pre-2000s) that mostly belong to the Drama or Romance genres. Their current low popularity is expected due to their age.
            - **cluster 5:** New Releases, Sub-Average Drama Focus.
              Numerical Profile: Below average quality (-0.183), low popularity, very recent (0.268).
              Characteristics: This group focuses on recent releases (high Z-Score for release year) that generally fail to impress critically or popularly, often filling the market with generic dramas and romantic comedies.
            - **cluster 6:** Lowest Rated, Unpopular Duds.
              Numerical Profile: The absolute lowest quality (-4.736), lowest popularity, older.
              Characteristics: These are highly niche or poorly made films that have extremely low critic/user scores, making them outliers on the low end of the quality scale.
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

