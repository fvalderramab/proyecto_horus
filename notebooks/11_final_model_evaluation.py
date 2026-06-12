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

# %% [markdown] id="af154d9c"
# # 11. Final Comparison and Qualitative LLM Evaluation
# This notebook represents the final phase of the academic production analysis project.
#
# **Objectives:**
# 1. **Universal Quantitative Metrics:** Compute geometric and semantic metrics universally for all methods on the same embedding space to ensure fairness.
# 2. **Qualitative Evaluation (LLM):** Use DeepSeek to evaluate the cohesion of clusters based on their most representative documents.
# 3. **Meta-Evaluation:** Ask DeepSeek to synthesize the results of all models and provide a final verdict on the state-of-the-art methodology for this project.

# %% [markdown] id="ae890b40"
# ## 11.1 Environment Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="ePRuiPI7kNeJ" outputId="258a3a3b-50d3-4e27-c2d7-a6fe447ebd43"
import os
import importlib.util
import subprocess
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations

def check_and_install(package_name, pip_name=None):
    if pip_name is None:
        pip_name = package_name
    if importlib.util.find_spec(package_name) is None:
        try:
            from google.colab import drive
            print(f"Installing {pip_name} in Colab...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except ImportError:
            print(f"Warning: '{package_name}' is missing. Make sure to install '{pip_name}' in your local .venv")

check_and_install("openai")
check_and_install("seaborn")

try:
    from google.colab import drive, userdata
    IN_COLAB = True
    drive.mount('/content/drive')
    BASE_PATH = Path("/content/drive/Shareddrives/Minería/proyecto_horus")

    deepseek_api_key = userdata.get('DEEPSEEK_API_KEY')
    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
except ImportError:
    IN_COLAB = False
    BASE_PATH = Path("..") # Assuming execution from /notebooks

if str(BASE_PATH) not in sys.path:
    sys.path.append(str(BASE_PATH))

from utils.llm_labeler import DocumentTopicLabeler, TopicMetaEvaluator

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown] id="e453f12c"
# ## 11.2 Load Datasets and Consolidate Labels

# %% colab={"base_uri": "https://localhost:8080/"} id="a225849d" outputId="672f08a3-de2e-4c8a-ecb2-e517b1e87849"
print("Loading core data...")
df_main = pd.read_parquet(BASE_PATH / "data/processed/products_modeling.parquet")
embeddings = np.load(BASE_PATH / "data/processed/final_embeddings_matrix.npy")

print(f"Main dataset shape: {df_main.shape}")
print(f"Embeddings shape: {embeddings.shape}")

# Define the methods we want to evaluate
methods_config = {
    "Classical_K-Means": {"file": "outputs/04_classical_clustering_labels.parquet", "col": "cluster_classical_kmeans"},
    "Classical_Agglomerative": {"file": "outputs/04_classical_clustering_labels.parquet", "col": "cluster_classical_agglomerative"},
    "Classical_LDA": {"file": "outputs/04_classical_clustering_labels.parquet", "col": "cluster_classical_lda"},
    "Modern_K-Means": {"file": "outputs/05_modern_clustering_labels.parquet", "col": "cluster_mod_kmeans"},
    "Modern_HDBSCAN": {"file": "outputs/05_modern_clustering_labels.parquet", "col": "cluster_mod_hdbscan"},
    "FASTopic": {"file": "outputs/07_fastopic_theta.parquet", "is_theta": True},
    "LLM-XTM": {"file": "outputs/08_llm_xtm_theta.parquet", "is_theta": True},
    "Graph_Leiden_MaxMod": {"file": "data/processed/graph_topics_max_modularity.parquet", "col": "community_id"},
    "Graph_Leiden_MinComm": {"file": "data/processed/graph_topics_min_communities.parquet", "col": "community_id"}
}

# Consolidate all labels into a single mapping dictionary
methods_labels = {}

for method, cfg in methods_config.items():
    file_path = BASE_PATH / cfg["file"]
    if not file_path.exists():
        print(f"Warning: Missing file for {method} -> {file_path}")
        continue

    try:
        if cfg.get("is_theta"):
            # It's a theta matrix (probability distribution)
            df_theta = pd.read_parquet(file_path)
            # The index of df_theta corresponds to the indices in df_main
            labels_series = pd.Series(index=df_main.index, dtype=float)
            # Convert probabilities to hard labels
            hard_labels = df_theta.values.argmax(axis=1)
            labels_series.loc[df_theta.index] = hard_labels
            methods_labels[method] = labels_series.values

        else:
            # It's a labels dataframe
            df_labels = pd.read_parquet(file_path)
            if "original_row_index" in df_labels.columns:
                # Merge based on original_row_index
                labels_series = pd.Series(index=df_main.index, dtype=float)
                labels_series.loc[df_labels["original_row_index"]] = df_labels[cfg["col"]].values
                methods_labels[method] = labels_series.values
            elif cfg["col"] in df_labels.columns:
                # Assume 1-to-1 mapping
                methods_labels[method] = df_labels[cfg["col"]].values
            else:
                print(f"Warning: Could not find {cfg['col']} in {file_path}")
    except Exception as e:
        print(f"Error loading {method}: {e}")

print(f"Successfully loaded labels for {len(methods_labels)} methods.")

# %% [markdown] id="8ca11885"
# ## 11.3 Universal Metrics Computation

# %% colab={"base_uri": "https://localhost:8080/", "height": 552} id="MgkCwxSzWTv6" outputId="e0e04d4b-17f9-45f9-8213-c39e9b2ca058"
def compute_npmi(topic_words_list, texts_list):
    cv = CountVectorizer(vocabulary=list(set([w for words in topic_words_list for w in words])))
    doc_word_matrix = cv.fit_transform(texts_list)
    doc_word_matrix = (doc_word_matrix > 0).astype(int) # Binary occurrence
    vocab_map = cv.vocabulary_

    num_docs = len(texts_list)
    npmi_scores = []

    # Precompute column sums for p_i
    col_sums = doc_word_matrix.sum(axis=0).A1
    p_probs = col_sums / num_docs

    # Convert to CSC matrix for fast column intersection
    csc_matrix = doc_word_matrix.tocsc()

    for words in topic_words_list:
        word_ids = [vocab_map[w] for w in words if w in vocab_map]
        if len(word_ids) < 2:
            npmi_scores.append(0.0)
            continue

        topic_npmi = []
        for i, j in combinations(word_ids, 2):
            p_i = p_probs[i]
            p_j = p_probs[j]

            # Fast intersection
            col_i = csc_matrix[:, i].indices
            col_j = csc_matrix[:, j].indices
            co_occurrences = len(np.intersect1d(col_i, col_j, assume_unique=True))

            p_ij = co_occurrences / num_docs

            if p_ij == 0:
                topic_npmi.append(-1.0)
            else:
                pmi = np.log(p_ij / (p_i * p_j))
                npmi = pmi / -np.log(p_ij)
                topic_npmi.append(npmi)

        if topic_npmi:
            npmi_scores.append(np.mean(topic_npmi))
        else:
            npmi_scores.append(0.0)

    return np.mean(npmi_scores)

def get_top_terms_per_cluster(labels_array, texts_series, top_k=10):
    valid_mask = (pd.notna(labels_array)) & (labels_array != -1)
    df_temp = pd.DataFrame({'cluster': labels_array[valid_mask], 'text': texts_series[valid_mask]})

    docs_per_cluster = df_temp.groupby('cluster')['text'].apply(' '.join).reset_index()

    if len(docs_per_cluster) == 0:
        return []

    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(docs_per_cluster['text'])
    words = vectorizer.get_feature_names_out()

    topics = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray()[0]
        top_indices = row.argsort()[-top_k:][::-1]
        topics.append([words[idx] for idx in top_indices])

    return topics

OUTPUTS_DIR = BASE_PATH / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSAL_METRICS_FILE = OUTPUTS_DIR / "11_universal_metrics.csv"

# Load existing metrics to avoid recomputing if source files haven't changed
existing_metrics = {}
if UNIVERSAL_METRICS_FILE.exists():
    try:
        existing_df = pd.read_csv(UNIVERSAL_METRICS_FILE)
        existing_metrics = existing_df.set_index('Method').to_dict('index')
    except Exception as e:
        print(f"Could not load existing metrics: {e}")

all_texts = df_main['embeddings_text'].fillna("").astype(str).tolist()
metrics_list = []

for method, labels in methods_labels.items():
    file_path = BASE_PATH / methods_config[method]["file"]

    # Resilience Check: If metrics exist, compare timestamps
    recompute = True
    if method in existing_metrics and file_path.exists():
        file_mtime = os.path.getmtime(file_path)
        metrics_mtime = os.path.getmtime(UNIVERSAL_METRICS_FILE)
        if file_mtime < metrics_mtime:
            print(f"Skipping {method} (No changes detected since last computation).")
            recompute = False

    if not recompute:
        row = existing_metrics[method]
        row['Method'] = method
        metrics_list.append(row)
        continue

    print(f"Computing metrics for {method}...")

    valid_mask = pd.notna(labels)
    clustered_mask = valid_mask & (labels != -1)

    if clustered_mask.sum() < 2:
        print(f"  Not enough clustered points for {method}.")
        continue

    # Sample down to max 20k for geometric metrics to prevent memory issues
    eval_embeddings = embeddings[clustered_mask]
    eval_labels = labels[clustered_mask]

    if len(eval_embeddings) > 20000:
        np.random.seed(42)
        idx = np.random.choice(len(eval_embeddings), 20000, replace=False)
        eval_embeddings = eval_embeddings[idx]
        eval_labels = eval_labels[idx]

    n_topics = len(np.unique(eval_labels))
    outlier_ratio = (labels == -1).sum() / valid_mask.sum() if -1 in labels else 0.0

    try:
        sil = silhouette_score(eval_embeddings, eval_labels)
        ch = calinski_harabasz_score(eval_embeddings, eval_labels)
        db = davies_bouldin_score(eval_embeddings, eval_labels)
    except ValueError:
        sil, ch, db = np.nan, np.nan, np.nan

    # Semantic Metrics
    try:
        top_terms = get_top_terms_per_cluster(labels, df_main['embeddings_text'].fillna("").astype(str))

        # Topic Diversity (unique words / total words)
        all_words = [w for topic in top_terms for w in topic]
        diversity = len(set(all_words)) / len(all_words) if len(all_words) > 0 else 0.0

        # NPMI
        npmi = compute_npmi(top_terms, all_texts)
    except Exception as e:
        print(f"  Semantic metric error for {method}: {e}")
        diversity, npmi = np.nan, np.nan

    metrics_list.append({
        "Method": method,
        "N_Topics": n_topics,
        "Outlier_Ratio": outlier_ratio,
        "Silhouette": sil,
        "Calinski_Harabasz": ch,
        "Davies_Bouldin": db,
        "Topic_Diversity": diversity,
        "NPMI": npmi
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(UNIVERSAL_METRICS_FILE, index=False)
print(f"\n--- Universal Metrics Saved to {UNIVERSAL_METRICS_FILE} ---")
display(metrics_df)

# %% [markdown] id="934a464d"
# ## 11.4 Visualization of Universal Metrics

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="GlH0dpcC2803" outputId="c8491a4f-c7a1-41f3-8189-5f8e993c9ce1"
geom_df = metrics_df.dropna(subset=['Silhouette']).sort_values('Silhouette', ascending=False)
if not geom_df.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=geom_df, x='Silhouette', y='Method', palette='viridis')
    plt.title("Clustering Quality Comparison (Silhouette Score)\nHigher is better")
    plt.xlabel("Silhouette Score")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.show()

sem_df = metrics_df.dropna(subset=['NPMI']).sort_values('NPMI', ascending=False)
if not sem_df.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sem_df, x='NPMI', y='Method', palette='magma')
    plt.title("Semantic Coherence Comparison (NPMI)\nHigher is better")
    plt.xlabel("Normalized Pointwise Mutual Information (NPMI)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.show()

# %% [markdown] id="55cd0a4d"
# ## 11.5 Document-Centric Qualitative LLM Evaluation
# We evaluate the models using the top representative documents for each cluster.

# %% id="bab0559c" outputId="cb9802f5-8781-4f6e-8f3c-ff4be8cf446c"
import asyncio

# Create the labeler instance
labeler = DocumentTopicLabeler(
    semaphore_limit=20,
    checkpoint_dir=str(BASE_PATH / "outputs"),
    checkpoint_filename="11_llm_document_evaluations.json" # Will be overwritten per method below
)

# We will run this sequentially for each method to collect the evaluations
all_llm_evaluations = {}

async def evaluate_all_methods():
    for method, labels in methods_labels.items():
        print(f"\n--- LLM Evaluation for: {method} ---")

        # Prepare valid subset for this method
        valid_mask = pd.notna(labels)
        if valid_mask.sum() == 0:
            continue

        df_subset = df_main[valid_mask].copy()
        df_subset['cluster_id'] = labels[valid_mask]

        embeddings_subset = embeddings[valid_mask]

        # Modify checkpoint filename temporarily to store method-specific progress cleanly
        labeler.checkpoint_path = Path(labeler.checkpoint_path.parent) / f"11_llm_evals_{method}.json"

        evals = await labeler.generate_labels(
            df=df_subset,
            embeddings=embeddings_subset,
            cluster_col='cluster_id',
            text_col='embeddings_text',
            noise_label_id=-1
        )
        all_llm_evaluations[method] = evals

# Execute evaluations (Top-level await allowed in Jupyter/Colab)
await evaluate_all_methods()
print("All document-centric LLM evaluations complete.")

# %% id="71e7374c" outputId="bc1a4dce-d1eb-4601-b85c-198fd46b1654"
# Visualizing LLM Cohesion Scores
cohesion_records = []
for method, evals in all_llm_evaluations.items():
    if not evals: continue
    for cid, res in evals.items():
        score = res.get('cohesion_score', 'Error') if isinstance(res, dict) else 'Error'
        if score not in ['High', 'Medium', 'Low']:
            score = 'Error'
        cohesion_records.append({'Method': method, 'Topic': cid, 'Cohesion': score})

df_cohesion = pd.DataFrame(cohesion_records)
if not df_cohesion.empty:
    cohesion_counts = df_cohesion.groupby(['Method', 'Cohesion']).size().unstack(fill_value=0)

    for col in ['High', 'Medium', 'Low', 'Error']:
        if col not in cohesion_counts.columns:
            cohesion_counts[col] = 0

    # Sort by number of High/Medium cohesion topics
    cohesion_counts['Score'] = cohesion_counts['High'] * 2 + cohesion_counts['Medium']
    cohesion_counts = cohesion_counts.sort_values('Score', ascending=False).drop(columns=['Score'])

    # Absolute counts plot
    cohesion_counts[['High', 'Medium', 'Low', 'Error']].plot(
        kind='bar', stacked=True, figsize=(12, 6),
        color={'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c', 'Error': '#95a5a6'}
    )
    plt.title("LLM Topic Cohesion Evaluation per Method (Absolute Counts)")
    plt.xlabel("Method")
    plt.ylabel("Number of Topics")
    plt.legend(title="Cohesion Score")
    plt.tight_layout()
    plt.show()

    # Proportional plot
    cohesion_props = cohesion_counts.div(cohesion_counts.sum(axis=1), axis=0)
    cohesion_props[['High', 'Medium', 'Low', 'Error']].plot(
        kind='bar', stacked=True, figsize=(12, 6),
        color={'High': '#2ecc71', 'Medium': '#f1c40f', 'Low': '#e74c3c', 'Error': '#95a5a6'}
    )
    plt.title("Proportion of LLM Topic Cohesion per Method")
    plt.xlabel("Method")
    plt.ylabel("Proportion of Topics")
    plt.legend(title="Cohesion Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %% [markdown] id="d40f515f"
# ## 11.6 Visualization of Final Clusters (UMAP)
# To visually inspect the final clusters (especially from methods 07, 08, and 10), we project the high-dimensional embeddings into a 2D space using UMAP and color them by their assigned cluster. To save time on repeated runs, the 2D projection is cached.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="0451e06b" outputId="0db78199-cb79-4af5-ed32-6c8b3ef99114"
check_and_install("umap", "umap-learn")
import umap
import matplotlib.pyplot as plt
import seaborn as sns

UMAP_CACHE_FILE = BASE_PATH / "data/processed/umap_2d_embeddings.npy"

if UMAP_CACHE_FILE.exists():
    print("Loading cached 2D UMAP embeddings...")
    umap_2d = np.load(UMAP_CACHE_FILE)
else:
    print("Computing 2D UMAP embeddings (this may take a few minutes)...")
    # Subsampling might be needed if memory is an issue, but UMAP can handle 100k points efficiently.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    np.save(UMAP_CACHE_FILE, umap_2d)
    print("UMAP embeddings computed and cached.")

# Methods to visualize specifically
methods_to_plot = ["FASTopic", "LLM-XTM", "Graph_Leiden_MaxMod", "Graph_Leiden_MinComm"]

for method in methods_to_plot:
    if method in methods_labels:
        labels = methods_labels[method]

        # Filter out NaN or noise (-1) for better visualization
        valid_mask = pd.notna(labels)
        plot_df = pd.DataFrame({
            'umap_x': umap_2d[valid_mask, 0],
            'umap_y': umap_2d[valid_mask, 1],
            'cluster': labels[valid_mask]
        })

        # Sort so that noise (-1) is plotted first (at the bottom)
        plot_df = plot_df.sort_values(by='cluster')

        plt.figure(figsize=(12, 10))

        # Create a custom palette: gray for -1 (noise), and a nice palette for the rest
        unique_clusters = sorted(plot_df['cluster'].unique())
        n_colors = len([c for c in unique_clusters if c != -1])
        palette = sns.color_palette("husl", n_colors)

        color_map = {}
        color_idx = 0
        for c in unique_clusters:
            if c == -1:
                color_map[c] = (0.8, 0.8, 0.8)  # Light gray for noise
            else:
                color_map[c] = palette[color_idx]
                color_idx += 1

        sns.scatterplot(
            data=plot_df,
            x='umap_x',
            y='umap_y',
            hue='cluster',
            palette=color_map,
            s=5, # Marker size
            alpha=0.6,
            linewidth=0,
            legend=False # Hide legend as it can be huge for 50+ topics
        )
        plt.title(f"2D UMAP Projection of Topics: {method}", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        # Save figure to outputs
        out_fig = OUTPUTS_DIR / f"11_umap_plot_{method}.png"
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        print(f"Saved plot for {method} to {out_fig.name}")
        plt.show()

# %% [markdown]
# The visual discrepancy between the UMAP projections primarily stems from categorical color palette limitations rather than underlying model quality. While FASTopic appears visually balanced and distinct due to its smaller, easily color-coded number of clusters (26 topics), the Graph_Leiden methods generated a much larger volume of highly granular micro-topics (105 and 1174, respectively). Since plotting libraries cannot generate hundreds of highly distinguishable categorical colors, the Leiden visualizations suffer from color recycling and blending, creating a deceptive monochromatic appearance that obscures the actual cluster boundaries.
#
# Furthermore, 2D UMAP visualizations and global separation metrics (like the Silhouette score) evaluate broad spatial distinctness, which does not necessarily correlate with semantic coherence. FASTopic may be forcing the creation of globally separated but internally heterogeneous, low-cohesion clusters. Conversely, Graph_Leiden_MinComm may sacrify broad spatial separation—resulting in high cluster overlap in the 2D plane in exchange for exceptionally high local cohesion, successfully isolating highly specific and interpretable research niches.
#
# For the development of an academic recommendation system, local semantic cohesion is vastly more critical than global structural separation. A robust recommendation engine relies on precisely matching researchers to hyper-specific conceptual neighborhoods (which Graph_Leiden_MinComm provides), rather than assigning them to visually distinct but conceptually vague macro-topics. Therefore, despite the misleading UMAP aesthetics and poor global clustering metrics, Graph_Leiden_MinComm could be delivering the granular semantic accuracy required for highly targeted literature recommendations.

# %% [markdown]
# ## 11.7 Interactive LLM Topic Graphs
# This section generates interactive PyVis networks for all methods, showing how topics relate to each other. 
# Hover over the nodes to see the LLM-generated label, cohesion score, and reasoning.
# For methods without explicit graph topologies (like K-Means or FASTopic), edges are drawn between topics whose centroids have a cosine similarity above the 80th percentile.

# %%
check_and_install("pyvis")
check_and_install("networkx")
from pyvis.network import Network
from IPython.display import display, HTML, IFrame
from sklearn.metrics.pairwise import cosine_similarity
import json

print("Generating Interactive Topic Graphs with LLM evaluations...")

for method, labels in methods_labels.items():
    print(f"\n--- Processing {method} ---")
    
    out_html = OUTPUTS_DIR / f"11_interactive_topic_graph_{method}.html"
    
    # 0. Cache Check: If HTML already exists, skip generation and render directly
    if out_html.exists():
        print(f"  Found existing interactive graph at {out_html.name}. Skipping generation.")
        with open(out_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        import base64
        b64_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        display(IFrame(src=f"data:text/html;base64,{b64_html}", width="100%", height="800px"))
        continue
    
    # Check if LLM evaluations exist for this method
    llm_eval_file = OUTPUTS_DIR / f"11_llm_evals_{method}.json"
    llm_data = {}
    if llm_eval_file.exists():
        with open(llm_eval_file, "r", encoding="utf-8") as f:
            llm_data = json.load(f)
    else:
        print(f"  No LLM evaluations found at {llm_eval_file.name}. Nodes will only show basic stats.")
        
    valid_mask = pd.notna(labels) & (labels != -1)
    if valid_mask.sum() == 0:
        continue
        
    method_labels_clean = labels[valid_mask]
    method_embeddings = embeddings[valid_mask]
    
    # 1. Compute Topic Centroids and Sizes
    unique_topics = np.unique(method_labels_clean)
    topic_centroids = {}
    topic_sizes = {}
    
    for t in unique_topics:
        idx = np.where(method_labels_clean == t)[0]
        topic_sizes[int(t)] = len(idx)
        # L2 normalize the centroid for cosine similarity
        centroid = np.mean(method_embeddings[idx], axis=0)
        topic_centroids[int(t)] = centroid / np.linalg.norm(centroid)
        
    # 2. Compute Edges based on centroid similarity
    topic_ids = list(topic_centroids.keys())
    centroid_matrix = np.array([topic_centroids[t] for t in topic_ids])
    
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(centroid_matrix)
    
    # Extract upper triangle to get unique pairs
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_tri_indices]
    
    edges_to_add = []
    if len(similarities) > 0:
        # Dynamic threshold: Hard limit on max edges to prevent graph explosion and browser crashes
        max_edges = 1500
        if len(similarities) > max_edges:
            # Pick the threshold that keeps only the top max_edges most similar pairs
            threshold = np.sort(similarities)[-max_edges]
            print(f"  Graph has many topics. Limiting to top {max_edges} edges to prevent browser crash.")
        else:
            threshold = np.percentile(similarities, 80)
            
        for i, j in zip(*upper_tri_indices):
            if sim_matrix[i, j] >= threshold:
                edges_to_add.append((topic_ids[i], topic_ids[j], float(sim_matrix[i, j])))
                
    # 3. Build PyVis Network
    net = Network(height='800px', width='100%', notebook=True, cdn_resources='remote', bgcolor='#222222', font_color='white')
    net.force_atlas_2based()
    
    for t in topic_ids:
        size = topic_sizes[t]
        t_str = str(t)
        
        # Extract LLM data if available
        node_label = f"Topic {t}"
        hover_text = f"Topic {t} | Documents: {size}"
        
        if t_str in llm_data:
            info = llm_data[t_str]
            llm_title = info.get("label", node_label)
            cohesion = info.get("cohesion_score", "N/A")
            reasoning = info.get("reasoning", "No reasoning provided.")
            
            # Truncate label if too long for display
            display_label = llm_title if len(llm_title) < 40 else llm_title[:37] + "..."
            node_label = f"[{t}] {display_label}"
            
            # Format hover text with line breaks for readability
            hover_text = f"Title: {llm_title}\n" \
                         f"Topic ID: {t} | Size: {size}\n" \
                         f"Cohesion Score: {cohesion}\n\n" \
                         f"Reasoning:\n{reasoning}"
        
        net.add_node(int(t), label=node_label, title=hover_text, value=int(size))
        
    for c1, c2, w in edges_to_add:
        # Scale weight for visual distinction
        visual_weight = float((w - threshold) / (1 - threshold) * 5 + 1) if threshold < 1 else 1.0
        net.add_edge(int(c1), int(c2), value=visual_weight, title=f"Similarity: {float(w):.3f}")
        
    net.write_html(str(out_html))
    print(f"  Saved interactive graph to {out_html.name}")
    
    # 4. Display inline using standard HTML embedding
    # We read the file content and pass it directly to HTML to ensure it renders 
    # even when downloaded from Colab, without needing the external file.
    with open(out_html, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # Inject custom CSS for tooltips to allow wrapping and prevent truncation
    custom_tooltip_style = """
<style type="text/css">
    div.vis-tooltip, div.vis-network-tooltip {
        max-width: 450px !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        font-family: Arial, sans-serif !important;
        font-size: 14px !important;
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5) !important;
    }
</style>
"""
    html_content = html_content.replace("</head>", f"{custom_tooltip_style}\n</head>")
    
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    import base64
    b64_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
    display(IFrame(src=f"data:text/html;base64,{b64_html}", width="100%", height="800px"))


# %% [markdown] id="44e71b0b"
# ## 11.8 Final LLM Meta-Evaluation Verdict

# %% id="7ce22a1c" outputId="36cbbe78-02bc-46fe-e722-2687be85512e"
print("\n--- Generating LLM Meta-Evaluation Summaries ---")
meta_evaluator = TopicMetaEvaluator()

method_summaries = {}

# We generate a summary per method first
for method, evals in all_llm_evaluations.items():
    if not evals:
        continue
    print(f"Generating summary for {method}...")

    # Get metrics for this method
    method_metrics = None
    if not metrics_df.empty and method in metrics_df['Method'].values:
        method_metrics = metrics_df[metrics_df['Method'] == method].to_dict('records')[0]

    summary = meta_evaluator.evaluate_method(method, evals, metrics=method_metrics)
    method_summaries[method] = summary
    print(f"\n[{method} Summary]:\n{summary}\n")

print("\n--- Generating Final Verdict ---")
final_verdict = meta_evaluator.generate_final_verdict(method_summaries)

# We print it directly in the cell as markdown output
from IPython.display import Markdown, display
display(Markdown("## 🏆 DeepSeek Final Verdict\n\n" + final_verdict))
