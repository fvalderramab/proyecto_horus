# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown] id="bd6b26a0"
# # Classical Clustering: Baseline NLP with HORUS data

# %% [markdown] id="0fa745fb"
# This notebook implements the classical NLP clustering baseline for the project. The goal is to group scientific products into thematic clusters using traditional text mining techniques.
#
# The pipeline follows these steps:
#
# 1. Load the preprocessed modeling dataset.
# 2. Select the text column prepared for classical NLP.
# 3. Build a TF-IDF matrix.
# 4. Reduce dimensionality with Latent Semantic Analysis (TruncatedSVD) to stabilize clustering.
# 5. Apply K-Means and Agglomerative Clustering.
# 6. Evaluate both models with internal clustering metrics.
# 7. Export labels and metrics for the comparative evaluation notebook.

# %% [markdown] id="dd7cd49e"
# # 1. Setup and imports

# %% colab={"base_uri": "https://localhost:8080/"} id="4c5699d8" outputId="952c4890-7a5f-4cee-e924-2ab319840151"
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("All imports successful.")

# %% [markdown] id="ea9d2eaf"
# ### 1.1. Environment configuration

# %% colab={"base_uri": "https://localhost:8080/"} id="c3ba9312" outputId="e45e21d4-d56e-4b9c-ae94-262506b1c82e"
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    base_path = Path("/content/drive/Shareddrives/Minería/proyecto_horus/data")
else:
    base_path = Path("../data")

df = pd.read_parquet(base_path / "processed/products_modeling.parquet")

# %% [markdown] id="7c16811f"
# # 2. Load dataset

# %% [markdown] id="c77d0335"
# This dataset already contains preprocessed columns for modeling. For the classical baseline, the most relevant column is expected to be `classical_text_for_modeling`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 590} id="b72d8d71" outputId="ef337e43-c870-4d61-a7e6-26a96b21b20a"
print("Dataset shape:", df.shape)
print("Available columns:")
print(df.columns.tolist())

df.head()

# %% [markdown] id="ac4e26e0"
# ### 2.1. Select rows and text column

# %% [markdown] id="ce412f37"
# Only records marked as useful for classical NLP are used when the `_useful_for_classical_nlp` flag is available. The text column is selected with the following priority:
#
# 1. `classical_text_for_modeling`
# 2. `classic_text`
# 3. `tokens_ngrams`
# 4. `embeddings_text`
# 5. `original_description`
#
# The first option should be used in the current preprocessed dataset.

# %% colab={"base_uri": "https://localhost:8080/", "height": 275} id="c05f10aa" outputId="0a331a19-faa8-41f6-f0e7-5658e4df93e6"
if "_useful_for_classical_nlp" in df.columns:
    df_classical = df[df["_useful_for_classical_nlp"] == True].copy()
else:
    df_classical = df.copy()

candidate_text_columns = [
    "classical_text_for_modeling",
    "classic_text",
    "tokens_ngrams",
    "embeddings_text",
    "original_description",
]

TEXT_COLUMN = None
for column in candidate_text_columns:
    if column in df_classical.columns:
        TEXT_COLUMN = column
        break

if TEXT_COLUMN is None:
    raise ValueError(f"No valid text column found. Available columns: {df_classical.columns.tolist()}")

def normalize_text(value):
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)

# Preserve the original row index for later merging with other notebooks.
df_classical = df_classical.copy()
df_classical["original_row_index"] = df_classical.index

df_classical["text_for_clustering"] = (
    df_classical[TEXT_COLUMN]
    .fillna("")
    .apply(normalize_text)
    .str.strip()
)

df_classical = df_classical[df_classical["text_for_clustering"].str.len() > 0].reset_index(drop=True)

texts = df_classical["text_for_clustering"]

print(f"Selected text column: {TEXT_COLUMN}")
print("Classical dataframe shape:", df_classical.shape)
texts.head()

# %% [markdown] id="9703fb35"
# # 3. TF-IDF representation

# %% [markdown] id="c2c277af"
# TF-IDF converts each document into a sparse vector that gives more weight to terms that are frequent in a document but not too common across the whole corpus.
#
# `max_features=3000` keeps the representation manageable for clustering while retaining the most informative vocabulary.

# %% colab={"base_uri": "https://localhost:8080/"} id="914b00d9" outputId="97825f31-d3ff-4a50-e185-d2d28a2d4daf"
tfidf_vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
    norm="l2"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(feature_names))

# %% [markdown] id="d99b3354"
# ### 3.1. Latent Semantic Analysis for stable clustering

# %% [markdown] id="46114f11"
# Sparse TF-IDF matrices can make K-Means inertia curves noisy because the space is very high-dimensional. To make the classical clustering baseline more stable and feasible for hierarchical clustering, TF-IDF is reduced with TruncatedSVD, also known as Latent Semantic Analysis (LSA).
#
# This is still a classical NLP approach: it uses TF-IDF term weighting plus matrix factorization, not neural embeddings.

# %% colab={"base_uri": "https://localhost:8080/"} id="ff434381" outputId="36321981-62a4-4abd-8255-6d1100605442"
n_samples, n_features = tfidf_matrix.shape
N_COMPONENTS = min(100, n_features - 1, n_samples - 1)

if N_COMPONENTS < 2:
    raise ValueError("Not enough samples or features to perform dimensionality reduction.")

lsa_model = make_pipeline(
    TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE),
    Normalizer(copy=False)
)

lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

explained_variance = lsa_model.named_steps["truncatedsvd"].explained_variance_ratio_.sum()

print("LSA matrix shape:", lsa_matrix.shape)
print(f"Explained variance ratio: {explained_variance:.4f}")

# %% [markdown] id="2ba6da9c"
# # 4. K-Means clustering

# %% [markdown] id="3c90720a"
# K-Means is used as the main classical baseline. Since the number of clusters is not known in advance, the elbow method is used as a practical heuristic.

# %% [markdown] id="1cde473d"
# ### 4.1. Elbow method

# %% [markdown] id="2da53601"
# The elbow method evaluates the K-Means inertia for different values of `k`. A good value is usually near the point where the curve stops decreasing sharply.
#
# The curve is computed on the LSA representation to avoid the noisy behavior often observed when K-Means is applied directly over sparse high-dimensional TF-IDF.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="dc6eb94b" outputId="b5f0bf7d-8ffb-40e2-fb96-172abaca74ee"
K_RANGE = range(5, 31)
MAX_ELBOW_SAMPLE_SIZE = min(8000, lsa_matrix.shape[0])

rng = np.random.default_rng(RANDOM_STATE)
if lsa_matrix.shape[0] > MAX_ELBOW_SAMPLE_SIZE:
    elbow_indices = rng.choice(lsa_matrix.shape[0], size=MAX_ELBOW_SAMPLE_SIZE, replace=False)
    elbow_matrix = lsa_matrix[elbow_indices]
else:
    elbow_matrix = lsa_matrix

inertias = []
for k in K_RANGE:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=RANDOM_STATE
    )
    kmeans.fit(elbow_matrix)
    inertias.append(kmeans.inertia_)

elbow_results = pd.DataFrame({
    "k": list(K_RANGE),
    "inertia": inertias
})

plt.figure(figsize=(10, 5))
plt.plot(elbow_results["k"], elbow_results["inertia"], marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

elbow_results


# %% [markdown] id="7ee2d2d4"
# ### 4.2. Automatic elbow suggestion

# %% [markdown] id="736bca58"
# The following helper estimates the elbow by finding the point with the largest distance to the line joining the first and last points of the inertia curve. The final value can still be adjusted manually after inspecting the plot.

# %% colab={"base_uri": "https://localhost:8080/"} id="7776a45f" outputId="796d8281-317c-4262-e8db-ee178cb5a485"
def suggest_elbow_k(k_values, inertia_values):
    points = np.column_stack([np.array(k_values, dtype=float), np.array(inertia_values, dtype=float)])
    first_point = points[0]
    last_point = points[-1]
    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return int(k_values[0])

    distances = np.abs(np.cross(line_vector, first_point - points)) / line_length
    elbow_index = int(np.argmax(distances))
    return int(k_values[elbow_index])

ELBOW_K = suggest_elbow_k(list(K_RANGE), inertias)

print(f"Suggested k according to the elbow heuristic: {ELBOW_K}")

# %% [markdown] id="8f63f968"
# ### 4.3. Select final K

# %% [markdown] id="c3e6796d"
# The selected value of `k` will also be useful for the modern clustering notebook, because the embedding-based K-Means model should use the same number of clusters for a fair comparison.
#
# If the elbow is not clear, keep a medium value that balances interpretability and topic granularity. The automatic suggestion is used by default, but it can be changed manually.

# %% colab={"base_uri": "https://localhost:8080/"} id="3675c017" outputId="857d057d-86bc-4e1f-c315-b7dbd280ecaa"
SELECTED_K = ELBOW_K

# Uncomment and edit this line if the team decides to use a fixed shared value.
# SELECTED_K = 12

print(f"Selected number of clusters: {SELECTED_K}")

# %% [markdown] id="466b6d67"
# ### 4.4. Fit final K-Means model

# %% colab={"base_uri": "https://localhost:8080/"} id="1781dab3" outputId="839a239c-7428-46d5-baba-6041610ba615"
kmeans_model = KMeans(
    n_clusters=SELECTED_K,
    init="k-means++",
    n_init=50,
    max_iter=500,
    random_state=RANDOM_STATE
)

kmeans_labels = kmeans_model.fit_predict(lsa_matrix)

df_classical["cluster_classical_kmeans"] = kmeans_labels

print("K-Means labels generated:")
print(pd.Series(kmeans_labels).value_counts().sort_index())


# %% [markdown] id="8225ef9f"
# ### 4.5. Inspect top terms by K-Means cluster

# %% [markdown] id="7255b99a"
# To interpret the clusters, the average TF-IDF weight is computed for each K-Means group. The highest-weighted terms are used as a descriptive approximation of the cluster topic.

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="3daf7920" outputId="e3a15028-b828-4e66-aba1-834031a631fe"
def get_top_terms_by_cluster(tfidf_data, labels, feature_names, top_n=10):
    rows = []
    labels = np.array(labels)

    for cluster_id in sorted(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_tfidf = tfidf_data[cluster_mask]
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        top_terms = feature_names[top_indices].tolist()
        rows.append({
            "cluster": int(cluster_id),
            "size": int(cluster_mask.sum()),
            "top_terms": ", ".join(top_terms)
        })

    return pd.DataFrame(rows)

kmeans_top_terms = get_top_terms_by_cluster(
    tfidf_matrix,
    kmeans_labels,
    feature_names,
    top_n=12
)

kmeans_top_terms

# %% [markdown] id="ce6fba64"
# # 5. Agglomerative Clustering

# %% [markdown] id="d8301f42"
# Agglomerative Clustering is the second classical algorithm. This model can be expensive with many rows, so it is run on a reproducible sample when the dataset is large.
#
# The same LSA representation and the same selected number of clusters are used to make it comparable with K-Means.

# %% colab={"base_uri": "https://localhost:8080/"} id="e87ceb11" outputId="8db8b174-420a-4588-d80d-81f578b89d47"
MAX_AGGLOMERATIVE_SAMPLE_SIZE = min(5000, lsa_matrix.shape[0])

rng = np.random.default_rng(RANDOM_STATE)
if lsa_matrix.shape[0] > MAX_AGGLOMERATIVE_SAMPLE_SIZE:
    agglomerative_indices = rng.choice(
        lsa_matrix.shape[0],
        size=MAX_AGGLOMERATIVE_SAMPLE_SIZE,
        replace=False
    )
else:
    agglomerative_indices = np.arange(lsa_matrix.shape[0])

agglomerative_matrix = lsa_matrix[agglomerative_indices]

agglomerative_model = AgglomerativeClustering(
    n_clusters=SELECTED_K,
    linkage="ward"
)

agglomerative_labels_sample = agglomerative_model.fit_predict(agglomerative_matrix)

# Store labels only for the sampled rows. Non-sampled rows remain missing by design.
df_classical["cluster_classical_agglomerative"] = pd.NA
df_classical.loc[agglomerative_indices, "cluster_classical_agglomerative"] = agglomerative_labels_sample

print("Agglomerative sample size:", len(agglomerative_indices))
print("Agglomerative labels generated on sample:")
print(pd.Series(agglomerative_labels_sample).value_counts().sort_index())

# %% [markdown] id="adbc4622"
# ### 5.1. Inspect top terms by Agglomerative cluster

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="631570e9" outputId="a17f8715-4e5d-43c9-cd3c-342e73f080d4"
agglomerative_top_terms = get_top_terms_by_cluster(
    tfidf_matrix[agglomerative_indices],
    agglomerative_labels_sample,
    feature_names,
    top_n=12
)

agglomerative_top_terms


# %% [markdown]
# # 6. Latent Dirichlet Allocation (LDA)

# %% [markdown]
# LDA mathematically requires integer word counts (Bag-of-Words), so we instantiate a `CountVectorizer` instead of using the TF-IDF matrix. We map this sparse matrix to a Gensim corpus to calculate Topic Coherence properly using `CoherenceModel`.

# %% colab={"base_uri": "https://localhost:8080/"}
count_vectorizer = CountVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.90
)

count_matrix = count_vectorizer.fit_transform(texts)

# Gensim expects a corpus where columns are documents and rows are features (words), 
# so we pass the count_matrix and set documents_columns=False since rows are documents in count_matrix.
corpus = Sparse2Corpus(count_matrix, documents_columns=False)

# Extract native sklearn vocabulary mapping
vocab_dict = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())

# Convert it to the formal Gensim Dictionary object
gensim_dictionary = Dictionary.from_corpus(corpus, id2word=vocab_dict)

print("Count matrix shape:", count_matrix.shape)
print("Vocabulary size (CountVectorizer):", len(gensim_dictionary))

# %% [markdown]
# ### 6.1. Train LDA Model and Calculate Coherence

# %% colab={"base_uri": "https://localhost:8080/"}
# We use LdaMulticore for faster training
lda_model = LdaMulticore(
    corpus=corpus,
    id2word=gensim_dictionary,
    num_topics=SELECTED_K,
    random_state=RANDOM_STATE,
    passes=10,
    workers=max(1, os.cpu_count() - 1) if os.cpu_count() else 1
)

# To evaluate the topics qualitatively, we use the Coherence Score (c_v)
# Use the internal analyzer to guarantee identical tokenization
analyzer = count_vectorizer.build_analyzer()
tokenized_texts = [analyzer(text) for text in texts]

coherence_model = CoherenceModel(
    model=lda_model, 
    texts=tokenized_texts, 
    dictionary=gensim_dictionary, 
    coherence='c_v'
)

lda_coherence = coherence_model.get_coherence()
print(f"LDA Coherence Score (c_v) for {SELECTED_K} topics: {lda_coherence:.4f}")

# Extract dominant topic per document for later comparison
# get_document_topics returns a list of (topic_id, probability) for each document
dominant_topics = []
for i, doc in enumerate(corpus):
    topic_probs = lda_model.get_document_topics(doc, minimum_probability=0.0)
    topic_probs = sorted(topic_probs, key=lambda x: x[1], reverse=True)
    dominant_topic = topic_probs[0][0]
    dominant_topics.append(dominant_topic)

df_classical["cluster_classical_lda"] = dominant_topics
print("LDA labels generated on full dataset.")

# %% [markdown]
# ### 6.2. Inspect top terms by LDA cluster

# %% colab={"base_uri": "https://localhost:8080/"}
# Extract the top 12 terms for each topic directly from the model
lda_topics = lda_model.show_topics(num_topics=SELECTED_K, num_words=12, formatted=False)
lda_rows = []

for topic_id, word_probs in lda_topics:
    top_terms = [word for word, prob in word_probs]
    lda_rows.append({
        "cluster": topic_id,
        "size": dominant_topics.count(topic_id),
        "top_terms": ", ".join(top_terms)
    })

lda_top_terms = pd.DataFrame(lda_rows)
lda_top_terms

# %% [markdown]
# ### 6.3. LDA Visualization with pyLDAvis

# %% colab={"base_uri": "https://localhost:8080/"}
import pyLDAvis
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()

# Prepare the visualization using the gensim model
vis = pyLDAvis.gensim_models.prepare(
    topic_model=lda_model, 
    corpus=corpus, 
    dictionary=gensim_dictionary
)

# Display the visualization natively
vis

# %% [markdown] id="366bd92a"
# # 7. Internal evaluation

# %% [markdown] id="f1dbf880"
# Internal metrics evaluate the compactness and separation of the clusters without using external labels.
#
# Metrics included:
#
# - **Silhouette Score:** higher is better.
# - **Calinski-Harabasz Score:** higher is better.
# - **Davies-Bouldin Score:** lower is better.
#
# For efficiency and comparability, K-Means is evaluated on the same sample used for the Agglomerative model when possible.
# Note: LDA is excluded from geometric evaluation (Silhouette, Calinski-Harabasz, Davies-Bouldin) because it is a probabilistic generative model, not a distance-based clustering algorithm. Its performance is evaluated through Coherence Score and Top Terms instead.

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="b6412fcd" outputId="5ac151a0-d2b2-4d7e-ac96-d9612e71a46d"
def safe_internal_metrics(matrix, labels, model_name):
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {
            "model": model_name,
            "n_samples": matrix.shape[0],
            "n_clusters": len(unique_labels),
            "silhouette_score": np.nan,
            "calinski_harabasz_score": np.nan,
            "davies_bouldin_score": np.nan,
        }

    return {
        "model": model_name,
        "n_samples": matrix.shape[0],
        "n_clusters": len(unique_labels),
        "silhouette_score": silhouette_score(matrix, labels, metric="euclidean"),
        "calinski_harabasz_score": calinski_harabasz_score(matrix, labels),
        "davies_bouldin_score": davies_bouldin_score(matrix, labels),
    }

kmeans_labels_on_sample = kmeans_labels[agglomerative_indices]

metrics_results = pd.DataFrame([
    safe_internal_metrics(agglomerative_matrix, kmeans_labels_on_sample, "K-Means + TF-IDF/LSA"),
    safe_internal_metrics(agglomerative_matrix, agglomerative_labels_sample, "Agglomerative + TF-IDF/LSA"),
])

metrics_results

# %% [markdown] id="3f0ec6b1"
# # 8. Export results

# %% [markdown] id="137bd6df"
# The outputs are saved in the shared Drive folder under `proyecto_horus/outputs/`. These files can be used later by the comparative evaluation notebook.
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="df248826" outputId="d02f1a0d-9ed1-48ce-c614-a91a40ad4f07"
PROJECT_DIR = base_path.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labels_output_path = OUTPUT_DIR / "04_classical_clustering_labels.parquet"
metrics_output_path = OUTPUT_DIR / "04_classical_clustering_metrics.csv"
top_terms_kmeans_path = OUTPUT_DIR / "04_classical_kmeans_top_terms.csv"
top_terms_agglomerative_path = OUTPUT_DIR / "04_classical_agglomerative_top_terms.csv"
top_terms_lda_path = OUTPUT_DIR / "04_classical_lda_top_terms.csv"
elbow_output_path = OUTPUT_DIR / "04_classical_elbow_results.csv"
lda_vis_output_path = OUTPUT_DIR / "04_classical_lda_vis.html"

export_columns = [
    "original_row_index",
    "cluster_classical_kmeans",
    "cluster_classical_agglomerative",
    "cluster_classical_lda",
]

for optional_column in ["original_title", "faculty", "type", "source", "date"]:
    if optional_column in df_classical.columns:
        export_columns.append(optional_column)

labels_output = df_classical[export_columns].copy()

labels_output.to_parquet(labels_output_path, index=False)
metrics_results.to_csv(metrics_output_path, index=False)
kmeans_top_terms.to_csv(top_terms_kmeans_path, index=False)
agglomerative_top_terms.to_csv(top_terms_agglomerative_path, index=False)
lda_top_terms.to_csv(top_terms_lda_path, index=False)
elbow_results.to_csv(elbow_output_path, index=False)

pyLDAvis.save_html(vis, str(lda_vis_output_path))

print("Saved files in shared Drive:")
print(labels_output_path)
print(metrics_output_path)
print(top_terms_kmeans_path)
print(top_terms_agglomerative_path)
print(top_terms_lda_path)
print(elbow_output_path)
print(lda_vis_output_path)


# %% [markdown] id="042ebcd1"
# # 9. Brief interpretation

# %% [markdown] id="6b066acc"
# The classical baseline incorporates three models: K-Means, Agglomerative Clustering, and Latent Dirichlet Allocation (LDA). K-Means and Agglomerative Clustering were built using TF-IDF and Latent Semantic Analysis (LSA). LDA, being a probabilistic generative model, was built directly over a Bag-of-Words (CountVectorizer) representation to properly preserve the topic-word distributions, acting as the true historical baseline of the HORUS platform.
#
# The elbow method on the LSA space was used to choose the number of clusters, and this identical `k` was applied across K-Means, Agglomerative, and LDA. This balances interpretability and granularity, and ensures consistency when comparing with the modern embedding-based clustering pipeline later on.
#
# The top terms per cluster, alongside the LDA Coherence Score and interactive pyLDAvis exploration, provide a human-readable interpretation of the discovered topics. K-Means and Agglomerative models were additionally evaluated via internal geometric metrics. All generated labels and results are exported for the final comparative evaluation.
