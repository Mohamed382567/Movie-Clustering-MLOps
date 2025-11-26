# Movie Clustering MLOps Deployment

This repository is dedicated to the **deployment of the Movie Clustering models** (developed in [**https://github.com/Mohamed382567/Educational-Movie-Clustering-Project**]) as an interactive Streamlit application.

**Goal:** To provide an interactive platform showcasing the results of **advanced unsupervised learning** and to allow users to **select and compare** the different trained models.

### 🚀 Live Application & Cluster Exploration

The application enables exploration of clustering results based on a complex feature space (NLP embeddings + numerical data) and allows switching between the three trained models.

| Description | Link |
| :--- | :--- |
| **Live Streamlit App** | [https://movie-clustering-mohamed-elbaz-web-kzjzypbkjmd2wxmleuaxhw.streamlit.app/] |
| **Source Code (Modeling & Analysis)** | [https://github.com/Mohamed382567/Educational-Movie-Clustering-Project] |

### 🛠️ Key Technical Components (Deployment Assets)

| File | Purpose / MLOps Role |
| :--- | :--- |
| **`app.py`** | **The main deployment file**. Manages the UI and, importantly, the logic to handle the **dynamic loading of the large model asset**. It facilitates user selection between the **three trained models**. |
| **`requirements.txt`** | Lists necessary libraries, including specialized tools like `umap-learn` and `hdbscan`. |
| **External Link (Google Drive/Cloud)** | **CRITICAL NOTE ON FILE SIZE:** Due to the large size of the merged file containing **all three trained clustering models** and all the essential files for the models to deal with input data, this essential asset is **dynamically loaded via a Google Drive/Cloud link** embedded in the `app.py` code. This represents a practical **MLOps solution** to deployment constraints when dealing with large assets. |

### ⚙️ MLOps Highlights & Engineering Challenges

| Highlight | Description |
| :--- | :--- |
| **Continuous Integration (CI) of Features** | **All Feature Engineering and modeling steps** are integrated directly into the site's logic, ensuring the correct data processing and prediction when the models are queried by the user. |
| **Model Agility** | The application allows users to **select and switch between the three trained models** (K-Means, HDBSCAN, GMM), demonstrating the ability to deploy and serve multiple model versions simultaneously for comparison. |
| **Deployment of Large Assets** | Successfully deployed a system requiring a massive model/data file by utilizing **external cloud storage** (Google Drive/Cloud), which is a common strategy in production MLOps environments. |
| **High-Dimensional Deployment** | Successfully operationalized models reliant on complex, high-dimensional features like **UMAP-reduced NLP embeddings**. |
| **Educational Note** | This project is primarily for educational purposes, demonstrating knowledge of **MLOps and Development (Deployment)**. The core idea is scalable if more resources and data become available. |
