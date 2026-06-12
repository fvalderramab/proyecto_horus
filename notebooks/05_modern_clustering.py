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

# %% [markdown]
# # Project Modern Clustering: Expert Discovery System and Collaboration Network Analysis with HORUS data

# %% [markdown]
# # 1. Setup and imports

# %% id="luuM1NudYjs0"
import os
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
from tqdm.auto import tqdm
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import euclidean_distances

# %% [markdown]
# ### 1.1. Environment Configuration (Mounting Drive)

# %% colab={"base_uri": "https://localhost:8080/"} id="b6ZlWmp3i2an" outputId="41ff4800-4138-40f2-9341-05697e47ad59"
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    base_path = Path("/content/drive/Shareddrives/Minería/proyecto_horus/data")
    # base_path = Path("/content/drive/MyDrive/proyecto_horus/data")
else:
    base_path = Path("../data/")

df = pd.read_parquet(base_path / "processed/products_modeling.parquet")

# %% [markdown]
# # 2. Check of text length before embed phase

# %%
text_lengths = df['embeddings_text'].str.len()

# Samples of the longest values
print("\nTop 5 longest samples:")
longest_samples = df.loc[text_lengths.sort_values(ascending=False).index[:5], ['embeddings_text']]
longest_samples['char_count'] = longest_samples['embeddings_text'].str.len()
display(longest_samples)

# %%
# Calculate lengths
text_lengths = df['embeddings_text'].str.len()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Histogram (Distribution)
sns.histplot(text_lengths, bins=50, kde=True, ax=ax1, color='skyblue')
ax1.set_title('Distribution of embeddings_text Lengths')
ax1.set_xlabel('Character Count')
ax1.set_ylabel('Frequency')
ax1.axvline(512, color='red', linestyle='--', label='512 chars threshold')
ax1.legend()

# 2. Boxplot (Outliers/Quartiles)
sns.boxplot(x=text_lengths, ax=ax2, color='lightgreen')
ax2.set_title('Boxplot of embeddings_text Lengths')
ax2.set_xlabel('Character Count')

plt.tight_layout()
plt.show()

# %%
# 1. Calculate the character length of each row
df['text_length'] = df['embeddings_text'].astype(str).str.len()

# 2. Get detailed percentiles to see exactly where the tail gets long
stats = df['text_length'].describe(percentiles=[.25, .5, .75, .90, .95, .98, .99])

# 3. Calculate the specific proportion of your >4000 outliers
outlier_threshold = 4000
outlier_count = len(df[df['text_length'] > outlier_threshold])
total_count = len(df)
outlier_percentage = (outlier_count / total_count) * 100

print("--- Distribution Stats ---")
print(stats.round(2))
print(f"\nTotal rows > {outlier_threshold} chars: {outlier_count} ({outlier_percentage:.2f}%)")

# %%
# 1. Grab 2 random samples from the "normal" (interquartile) range
normal_samples = df[(df['text_length'] >= stats['25%']) & (df['text_length'] <= stats['75%'])]['embeddings_text'].sample(2, random_state=42)

print("--- Normal Samples ---")
for i, text in enumerate(normal_samples):
    print(f"Sample {i+1} (Len: {len(str(text))}): {text}")

# 2. Grab 2 random samples from the >4000 outlier range
# We will truncate them to 500 characters so they don't flood your screen or the prompt
outlier_samples = df[df['text_length'] > 4000]['embeddings_text'].sample(2, random_state=42)

print("\n--- Outlier Samples (Truncated) ---")
for i, text in enumerate(outlier_samples):
    print(f"Sample {i+1} (Len: {len(str(text))}): {str(text)[:512]}... [TRUNCATED]")

# %% [markdown]
# # 3. Embed phase (ran in colab with T4 GPU)

# %% [markdown] id="GvhuBSzE5tTr"
# ## 3.1. Set Up the Resilient Directory Structure

# %% colab={"base_uri": "https://localhost:8080/"} id="9aHCUmVNuePM" outputId="3ec87e0a-e6b9-49f9-f8d6-e78fe4308eff"
# --- CONFIGURATION ---
# Double-check that Colab actually provided a GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device.upper()}")

DRIVE_SAVE_DIR = base_path / 'embeddings_chunks/'
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# %% [markdown] id="FppgtbTqjHOQ"
# ## 3.2. Prepare and sort data

# %% colab={"base_uri": "https://localhost:8080/"} id="mkRWuP-hjF0A" outputId="6daf9915-4398-4b6e-b2e1-4261dd84ed09"
print("Calculating lengths and sorting...")
# We MUST save the original index so we can put the embeddings back in order later
df['text_length'] = df['embeddings_text'].str.len()
df_sorted = df.sort_values('text_length', ascending=False).reset_index(drop=False)

texts_to_embed = df_sorted['embeddings_text'].tolist()
total_texts = len(texts_to_embed)
print(f"Total texts to embed: {total_texts}")

# %% [markdown] id="euq7tK3L55oA"
# ## 3.3. Load the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 800, "referenced_widgets": ["1491a942040b4feab0d9267531720ef2", "fab9e975e2244bd488dbc63afe2e59f8", "64b19e69fc004dfba220e55ad1a38488", "e2cc9eeebd55408cb67ba67cf38addb3", "0b7d02d3eb454f4ea9c07c222e7437ae", "f026946f21b94caf999f94153c28c2da", "810cfc2761bd4a38932bf5bda9a152fa", "d12f6eaaaf3a44f088b1b62477e0f62f", "350d271571cb419d8dbd976f4f9c590c", "c54cd5a11fbc4956b474ec000dd633c7", "1a3fba1f19e045e98bbabf62d6c0794c", "92895f2d2d98421a916411095ce71247", "0f7d6845eb764b96ad0cd340bc004bf9", "31b9a3ccb33c44618a66e2b8eb281e2f", "030b629a83844ddf9878f6e130217805", "f3e4ca4d794842c5a5c68eea2631e89b", "a0526a9d6efc48179c67198c876c74b1", "3f4f2aa245f246acaeaa4279e79ba2b6", "7edb59f139c949e0bd66dd3441595f03", "4e3ebf98f53247739558ecc1a927ab8a", "9cd38b40b868415fa0ba942a0875671a", "c2fffc5ccab0466d9189a9aa8653c6ae", "9e3be69fdcea49f186f3dd4c7760d017", "d805c2ba3af74142be0ac016f7fd1a22", "b9c432d6cba74894aa9f7d1e90aa0c20", "7dae1efde681498c926cf5040489f1b0", "5b7d38dd7d03463f8ca64223568b1204", "93b5c8add49348dbaf22b4cfc8a64a13", "f8d169c970a24901baf2743c4ce93daa", "a0e611116f774ba7b068782d4bdd5c93", "1de1d8860d6844648126e535cdb9c98f", "1b9024b2ef9b4236abe534a07faa7a82", "2f1abe52250145f5961ad66069885f60", "f364ebebca0d43079e6fdaa6b94b4a40", "aae889efc9ec44258cffe8270d56c67d", "bbd41716b78e47368146fc161f6a4287", "cf6ab8faa8bf4c0da8b4e48d145bf61c", "2e26f2782efa4b71982732d6e2b8f969", "f36e5e40367e4b0b90ef99f8a3527c7f", "9f0e858bc36f4adbbf033e4b48b2138f", "f5ec6e3e80754e139d81e71e7b82a28f", "5035628740f7462f9eb9ab8b7d7bf145", "07252bf24d104f6e8501d08a8d675050", "b4f76142860d428082f0d3a80f053634", "88a1f4131a1f4c16b25c3833cfd77a6e", "b86dd172618540a6abbb0f246c9573bd", "6a1337d7f5e54828bf4688bb60a66fa8", "d89518bcdd274470ab4b9f0c3beecf4d", "29257cc0d7c64134be5e6d01de5bda47", "2bdb5c451f134e07b3dbae3fb947bcf1", "60d1eca9ac7a4692bd46fc4c6ce1eb34", "8938a56f3f924d729964b906446ec012", "8ee6859fbf5a453e84f7b8f704d203f8", "f8ebb994670840708d6ab5a2e1368562", "1c90dc8c652945d5ba13f3139a4d7fda", "6f3807799c2140a78e4d545687c2fee0", "1d5fa60d9a994c40ada44ead63187353", "f85d4e2bd48c4bc7b02cf39c0e7481b2", "2eed4adab34e4f459f704fbb1364e7c0", "c3b22aadf8334bb78992df46c40ffdf7", "51ee5dd31e554807bcfbe77028674e16", "e6e9cf4c4d194d328a7a0701f320be8d", "a0093a4ff39e45c78a3b5d47d2d59751", "6aeac1f81d6f4c8d8b70b5de0d1a72f2", "3579b2fe00be4ce7a32370c5f6de5148", "4b5a56d143aa4705a964e87816f63393", "057380ae3cd1469c89277e2753f7c03c", "b279e00c34da4835a2aff4051c45d4fd", "8a616231b2fe4cb48cdd66a239c17b04", "77976ec894bc4604abe2b5538873736a", "8afcf27be5a04f30861234d9cf3c372e", "a467bb6ea5334cb5b0ae1cb805bec587", "197acc140a6d435fb7d5019ea4e72252", "d4ea3507ed664e8fbe0b64e30f599b57", "e63b83c855e44dc694949082d48fb025", "60fbcc8994104bfab2aae388fecb529d", "969eb9a16d0c469fa73dcfc950152156", "4b2ae2d6d5594a4897d49b6d177ab3fa", "854d7d2b19c544dda0043f865a48fa1f", "b5d570c3f6d94d248617f65960840656", "6185f7a42ec84199a20e0df52f7a03a7", "e172421cb244481fa0d4c23d4a0f0bea", "82c118354f8048db9430903a6a3cbbf5", "10ec6ecae851479ca0acaca433e930a5", "0218be5db52449e792e9e15055eafb82", "0a1055fe6ee545c5b0099602e6216127", "2acb64d355174d62be005c4f6a528ef3", "a7f6554d44c1404089fc4a2f861f2b77", "2a918a60ff4c401a8c3d466233404640", "0fae5cffc7094ddbb91c6026046a343c", "5a05710b05d74c66afefac7fa4beaad2", "5b9174fb467148e993d5db7d801033f0", "bb70fa284e8840f483343405170f84c6", "13f8c01b92df41cda58bdbefdf6245c8", "bd4c8c5ed59b4c61bf53fb215d4f7db6", "0d29436743d54dd18245e3dc737587a1", "ff633d3290054c06a3116cd4c4b9255e", "110d4c0326564b679d402c92844ffee0", "f6942b00555545d3add85e82c29da5f4", "6ba14998556d443dbe513fc16d3dee8a", "bff8b69ab66846d09e011175e3af3711", "88c2c6a3ba7e4dc481dff3374069f6e5", "ae339400a7704dcf861e8f1c74ecb15f", "6f2fad1e71534255a8ed63aa55b26ad5", "62cbbb503e20400498704f3866cd3c88", "76215589019f4ea9ad26a83221524ac0", "2edaab789798457981d7a3b1da7acac7", "010ae51eba9846fab4d74c27e007bad4", "f06eb7d10220461d99efbffc8bd6d22b", "510e05a2e0bc4eb394a9cc439ffc71a0", "25db004c05444525b4bbe31566572321", "cf5a3995b0004c768305838631dfe094", "fdebba191d05461cae82bdfae6630433", "54827a824b894d6fb1eb6ef55e56a40c", "8ca3e9e1c8ae4ceeb9e6f8a872a62ee6", "afae127be30640b0b587875e1eaabff8", "0fdca5153681414d8022b1e7b7674bdd", "d1b2b5039d0743f390f5779197e9a661", "457e4cc68e7c4f70886e202e24b31339", "606b4751d3ce4c16827848d0a23eca02", "993fcc949a7c47b58177b43502a8bd4e", "3e6324b79d0c424e9ee6bf6a81f89168", "a8c7abd0f0f24f64b4176f17dfbb432e", "b436441128ea4d66841b8b49fdb1e295", "c171930708114a99a9349ac5382b22f3", "0f03302db36b431486d074630af74b7d", "9bba199e1d614e17a36a4a2857510ee2", "6d5da2a286694bfabe2ce445c1239aab", "9f92a14f9c2d40219d078e3f0ab98270", "ad069aa14bff4dd19ace282f8871a92b", "7b2fedec3f1140abbdbe912943628b05", "0dfca23419c942ada7637c645197c136", "188a7a02bffa414a9fa6773c204cc02e", "6e8f1dea07b743e1bef5a44c29fa18ae", "391f5a1a6ab64946a61dda0ddb883cc0", "8fccb3f0395c4529b391d2fc88f2fa51", "871a2fd89a0441adb8563293a6ad02ca", "02a72b569d09409b9b9c686608752603", "ed32f8fcf50c4e28a1311d2485331f90", "90b97525a6334cfbb0281d330626d4b3", "32195744312a4ec78577fb8e9bed4170", "12ff51b94feb4131a4e8edf68b12d8be", "097ffce18c674afcaa7b42b980372d29"]} id="5NUK9acuu5Cb" outputId="acee2998-d472-4367-8773-46fd74f1b02a"
print("Loading model...")

# Fixes the PyTorch fragmentation issue
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Because we sort by length, we can safely bump these up
BATCH_SIZE = 32
MAX_TOKENS = 1024 # Approx 4000 chars. Covers 98% of the data

model_name = "jinaai/jina-embeddings-v5-text-small"

# trust_remote_code is required for Jina's custom architecture
model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16} # Optimizes for T4 GPU
)

# Set the maximum sequence length on the model to prevent OOM from long texts
model.max_seq_length = MAX_TOKENS

print("Model loaded successfully")

# %% [markdown] id="WC-CPGlN6P9W"
# ## 3.4. The Resilient Encoding Loop

# %%
texts_to_embed = df['embeddings_text'].tolist()
total_texts = len(texts_to_embed)
print(f"Total texts to embed: {total_texts}")

num_batches = int(np.ceil(total_texts / BATCH_SIZE))

for i in tqdm(range(num_batches), desc="Processing Batches"):
    # Define the filename for this specific chunk
    chunk_filename = os.path.join(DRIVE_SAVE_DIR, f"chunk_{i:04d}.npy")

    # If the file already exists, skip encoding
    if os.path.exists(chunk_filename):
        continue

    # Get the specific slice of texts for this batch
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, total_texts)
    batch_texts = texts_to_embed[start_idx:end_idx]

    # Generate embeddings
    with torch.no_grad(): # Extra safety to prevent memory leaks
        embeddings = model.encode(
            batch_texts,
            task="clustering",
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    # Save the numpy array directly to Drive
    np.save(chunk_filename, embeddings)

    # Clear cache aggressively
    torch.cuda.empty_cache()

print("All batches processed!")

# %% [markdown] id="HuL2Cw0C6vfJ"
# ## 3.5. Merge the Chunks

# %%
print("Merging chunks...")
all_embeddings = []
num_batches = int(np.ceil(total_texts / BATCH_SIZE))

for i in tqdm(range(num_batches), desc="Loading Chunks"):
    chunk_filename = os.path.join(DRIVE_SAVE_DIR, f"chunk_{i:04d}.npy")
    chunk_data = np.load(chunk_filename)
    all_embeddings.append(chunk_data)

# Stack vertically into a single array
stacked_embeddings = np.vstack(all_embeddings)
print(f"Stacked shape: {stacked_embeddings.shape}")

# --- THE MAGIC TRICK: UN-SORTING ---
print("Restoring original order...")
original_indices = df_sorted['index'].values
final_ordered_embeddings = np.empty_like(stacked_embeddings)

# This maps the sorted embeddings back to their original row positions
final_ordered_embeddings[original_indices] = stacked_embeddings

# Save the final masterpiece
np.save(base_path / 'processed/final_embeddings_matrix.npy', final_ordered_embeddings)
print("Done. Final embedding matrix ready.")

# %% [markdown]
# # 4. Clustering techniques
#
# First, we load the final embeddings matrix generated in the previous step.

# %%
# Load the embeddings
final_embeddings = np.load(base_path / 'processed/final_embeddings_matrix.npy')
print(f"Loaded embeddings shape: {final_embeddings.shape}")


# %% [markdown]
# ## 4.1. K-Means
# We run KMeans with a predefined K value for fairness comparison with the classic NLP approach.

# %%
print("Reducing dimensions of FULL dataset with UMAP...")
reducer = umap.UMAP(
    n_neighbors=100, 
    min_dist=0.0, 
    n_components=5, 
    metric='cosine', 
    random_state=42,
    n_jobs=-1 # Use all CPU cores
)
full_reduced_embeddings = reducer.fit_transform(final_embeddings)

# Exact K from the classic NLP approach
K_CLASSIC = 14

# 2. Run K-Means on the 5D REDUCED embeddings
print(f"Running KMeans (K={K_CLASSIC}) on UMAP reduced embeddings...")
kmeans_umap = KMeans(n_clusters=K_CLASSIC, random_state=42, n_init='auto')
df['kmeans_cluster'] = kmeans_umap.fit_predict(full_reduced_embeddings)
kmeans_labels = df['kmeans_cluster'].values

# Calculate Silhouette Score 
# (We sample to 20k to prevent RAM exhaustion / slow compute on 100k vectors)
sample_size = min(20000, final_embeddings.shape[0])
kmeans_silhouette = silhouette_score(
    final_embeddings, kmeans_labels, 
    sample_size=sample_size, random_state=42
)

kmeans_calinski = calinski_harabasz_score(final_embeddings, kmeans_labels)
kmeans_davies = davies_bouldin_score(final_embeddings, kmeans_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"K-Means Calinski-Harabasz Score: {kmeans_calinski:.4f}")
print(f"K-Means Davies-Bouldin Score: {kmeans_davies:.4f}")

# %% [markdown]
# ## 4.2. HDBSCAN (with UMAP Dimensionality Reduction)
#
# Density-based clustering struggles in high-dimensional spaces. UMAP is used first, to reduce the embeddings to a dense, lower-dimensional space (e.g., 5 dimensions).

# %% [markdown]
# ### 4.2.1. Hyperparameter Tuning
# To find the best parameters without guessing, we performed a grid search. 
# We subsampled to 20.000 rows for the search to obtain a fair approximation and faster execution.

# %%
print("Starting Grid Search on a 20k sample...")
# 1. Take a 20k sample for fast tuning
np.random.seed(42)
sample_indices = np.random.choice(final_embeddings.shape[0], size=20000, replace=False)
sample_embeddings = final_embeddings[sample_indices]

# 2. Define the grid
param_grid = {
    'umap__n_neighbors': [15, 50, 100],
    'umap__min_dist': [0.0],
    'hdbscan__min_cluster_size': [50, 100, 300],
    'hdbscan__min_samples': [15, 30]
}

results = []
grid = list(ParameterGrid(param_grid))

for params in tqdm(grid, desc="Tuning Parameters"):
    start_time = time.time()
    
    # Run UMAP
    reducer = umap.UMAP(
        n_neighbors=params['umap__n_neighbors'],
        min_dist=params['umap__min_dist'],
        n_components=5,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    reduced_emb = reducer.fit_transform(sample_embeddings)
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params['hdbscan__min_cluster_size'],
        min_samples=params['hdbscan__min_samples'],
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(reduced_emb)
    
    # Calculate Metrics
    n_noise = list(labels).count(-1)
    noise_pct = n_noise / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Silhouette (only on clustered points)
    mask = labels != -1
    sil_score = 0
    if np.sum(mask) > 1 and n_clusters > 1:
        # We calculate silhouette on the ORIGINAL high-dim space to ensure true semantic cohesion
        sil_score = silhouette_score(sample_embeddings[mask], labels[mask], random_state=42)
        
    results.append({
        'n_neighbors': params['umap__n_neighbors'],
        'min_cluster_size': params['hdbscan__min_cluster_size'],
        'min_samples': params['hdbscan__min_samples'],
        'n_clusters': n_clusters,
        'noise_pct': round(noise_pct * 100, 2),
        'silhouette': round(sil_score, 4),
        'time_sec': round(time.time() - start_time, 1)
    })

# Convert to DataFrame and sort by best Silhouette (while keeping an eye on noise)
df_tuning = pd.DataFrame(results)
display(df_tuning.sort_values('silhouette', ascending=False).head(10))


# %% [markdown]
# The set of parameters that give the highest Silhouette score, provided the noise is acceptable (e.g., < 40%) are picked.

# %%
best_n_neighbors = 100
best_min_cluster_size = 50
best_min_samples = 30

# %% [markdown]
# ### 4.2.2. HDBSCAN execution with the best set of parameters

# %%
print("1. Reducing dimensionality with UMAP for HDBSCAN...")
umap_cluster_model = umap.UMAP(
    n_neighbors=best_n_neighbors, 
    n_components=5, 
    min_dist=0.0, 
    metric='cosine', 
    random_state=42
)
embeddings_reduced = umap_cluster_model.fit_transform(final_embeddings)

print("2. Running HDBSCAN...")
# min_cluster_size dictates the minimum size of a grouping to be considered a cluster
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=best_min_cluster_size, 
    min_samples=best_min_samples,
    metric='euclidean',
    cluster_selection_method='eom'
)
hdbscan_labels = hdbscan_model.fit_predict(embeddings_reduced)

df['hdbscan_cluster'] = hdbscan_labels

# Noise and Cluster counts
n_noise = list(hdbscan_labels).count(-1)
noise_percent = (n_noise / len(hdbscan_labels)) * 100
n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

print(f"HDBSCAN found {n_clusters} clusters.")
print(f"Noise points isolated: {n_noise} ({noise_percent:.2f}%)")

# Calculate Silhouette Score EXCLUDING noise points for a fair metric
mask_clustered = hdbscan_labels != -1
if np.sum(mask_clustered) > 0:
    hdbscan_silhouette = silhouette_score(
        final_embeddings[mask_clustered], # Evaluate using the ORIGINAL high-dim space
        hdbscan_labels[mask_clustered], 
        sample_size=min(20000, np.sum(mask_clustered)), 
        random_state=42
    )
    # Calinski and Davies are not strictly designed for density clusters with noise, but we compute them on the clustered points for comparison
    hdbscan_calinski = calinski_harabasz_score(final_embeddings[mask_clustered], hdbscan_labels[mask_clustered])
    hdbscan_davies = davies_bouldin_score(final_embeddings[mask_clustered], hdbscan_labels[mask_clustered])
    
    print(f"HDBSCAN Silhouette Score (excluding noise): {hdbscan_silhouette:.4f}")
    print(f"HDBSCAN Calinski-Harabasz Score (excluding noise): {hdbscan_calinski:.4f}")
    print(f"HDBSCAN Davies-Bouldin Score (excluding noise): {hdbscan_davies:.4f}")
else:
    hdbscan_silhouette, hdbscan_calinski, hdbscan_davies = 0, 0, 0
    print("No clusters found, everything is noise!")

# %% [markdown]
# ### 4.2.3. Interpretation of Internal Metrics
#
# - **Silhouette Score Expected Values:** In high-dimensional text clustering (especially with approximately 100,000 documents and hundreds of granular topics), distances between different abstract concepts are relatively small. A score of ~0.13 indicates that the clusters are valid and cohesive, but they blend softly into each other. This mirrors how academic research functions in reality, where topics are not perfectly isolated spheres.
# - **Calinski-Harabasz and Davies-Bouldin:** The Davies-Bouldin index evaluates the average similarity between clusters, where a lower score represents better separation. The Calinski-Harabasz score measures the ratio of between-cluster dispersion to within-cluster dispersion, where a higher score is better.
# - **Comparison:** The modern approach using HDBSCAN and K-Means on Embeddings outperforms the Classical K-Means and Hierarchical clustering across all three metrics. The Davies-Bouldin score is significantly lower (1.80 for HDBSCAN vs 2.88 for Classic), and the Silhouette score is almost doubled. This empirically demonstrates that embeddings capture tighter, more cohesive semantic groups than sparse word-counting matrices like TF-IDF.

# %% [markdown]
# ### 4.2.4. Noise Analysis
#
# HDBSCAN classified roughly 34% of the dataset as noise. Here is a sample of these documents to verify if they are indeed too vague, short, or unique to form a coherent semantic cluster.

# %%
print("Sampling 5 random noise points...")
noise_samples = df[df['hdbscan_cluster'] == -1].sample(5, random_state=42)
for i, (_, row) in enumerate(noise_samples.iterrows()):
    text_preview = str(row['embeddings_text'])[:300] + ("..." if len(str(row['embeddings_text'])) > 300 else "")
    print(f"\n[Noise Sample {i+1}]:\n{text_preview}")

# %% [markdown]
# In HDBSCAN, "Noise" doesn't mean bad data. It means data that is not surrounded by at least min_cluster_size similar neighbors. Here are the main reasons why a perfectly good abstracts as those sampled become noise:
#
# - **Hyper-Specificity:** As it happens with Sample 2 (conformal GaAs/Si layers) there are incredibly specific records (in this case from quantum physics/materials science). Even if it's a great paper, if there aren't 50 other papers in the dataset specifically about GaAs/Si layers, it won't form a cluster. HDBSCAN intelligently decides to leave it alone rather than forcing it into a generic "Science" blob.
# - **Boundary Points (Interdisciplinary Blur):** If an abstract blends two distinct topics perfectly (e.g., exactly 50% Biology and 50% Law), its vector will sit in the empty "valley" between the dense Biology cluster and the dense Law cluster. Because it's in a sparse valley, it gets marked as noise.
# - **Language Complexity:** Some abstracts (especially in sciences, like Sample 4) use complex, theoretical, or less standardized vocabulary compared to other areas. This scatters them across the semantic space, preventing them from forming tight, dense hubs.
#
# 34% noise means that 34% of the university's production is either highly unique, interdisciplinary, or niche. By filtering them out, HDBSCAN guarantees that the remaining 310 clusters are extremely pure and reliable for recommending thesis advisors.


# %% [markdown]
# ## 4.3. Comparison & Visualization
#
# The original embeddings are projected to 2D specifically to visualize the semantic shapes discovered by both algorithms.

# %%
print("Projecting to 2D for visualization...")
umap_2d_model = umap.UMAP(n_neighbors=best_n_neighbors, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = umap_2d_model.fit_transform(final_embeddings)

df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]

# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- Plot K-Means ---
sns.scatterplot(
    data=df, x='umap_x', y='umap_y', 
    hue='kmeans_cluster', palette='tab20', 
    s=10, alpha=0.6, ax=ax1, legend=False
)
ax1.set_title(f'K-Means Clustering (K={K_CLASSIC})\nSilhouette: {kmeans_silhouette:.4f}')

# --- Plot HDBSCAN ---
# Custom palette so noise points (-1) are light gray
palette = sns.color_palette('tab20', n_clusters)
color_map = {cluster_id: palette[i % len(palette)] for i, cluster_id in enumerate(sorted(list(set(hdbscan_labels) - {-1})))}
color_map[-1] = (0.8, 0.8, 0.8) # Gray for noise

sns.scatterplot(
    data=df, x='umap_x', y='umap_y', 
    hue='hdbscan_cluster', palette=color_map, 
    s=10, alpha=0.6, ax=ax2, legend=False
)
ax2.set_title(f'HDBSCAN Clustering ({n_clusters} clusters)\nNoise: {noise_percent:.1f}% | Silhouette (no noise): {hdbscan_silhouette:.4f}')

plt.tight_layout()
plt.show()

# %% [markdown]
# The grey areas in HDBSCAN clustering are parts of the noise detected. We can interpret the majority of the noise could be happening because of interdisciplinary productions.

# %% [markdown]
# ## 4.4. Quantitative Comparison against Administrative Boundaries (Faculty)
#
# Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI) is used to see how closely the semantic clusters align with the university's administrative faculties.
#
# Note: We injected the faculty name into the embeddings text, so the Modern Pipeline has a structural advantage compared to the Classic NLP baseline for this case

# %%
# Drop any rows where faculty might be null just in case
eval_df = df.dropna(subset=['faculty', 'kmeans_cluster', 'hdbscan_cluster']).copy()

# KMeans Metrics
kmeans_nmi = normalized_mutual_info_score(eval_df['faculty'], eval_df['kmeans_cluster'])
kmeans_ari = adjusted_rand_score(eval_df['faculty'], eval_df['kmeans_cluster'])

# HDBSCAN Metrics (we filter out noise points for a fair evaluation of the formed clusters)
hdbscan_eval_df = eval_df[eval_df['hdbscan_cluster'] != -1]
hdbscan_nmi = normalized_mutual_info_score(hdbscan_eval_df['faculty'], hdbscan_eval_df['hdbscan_cluster'])
hdbscan_ari = adjusted_rand_score(hdbscan_eval_df['faculty'], hdbscan_eval_df['hdbscan_cluster'])

print("--- Alignment with Administrative Faculties ---")
print(f"K-Means -> NMI: {kmeans_nmi:.4f} | ARI: {kmeans_ari:.4f}")
print(f"HDBSCAN -> NMI: {hdbscan_nmi:.4f} | ARI: {hdbscan_ari:.4f}")

# %% [markdown]
# ### 4.4.1. Interdisciplinary Mixing Analysis
#
# The NMI and ARI scores against the faculty column are virtually zero. This proves our core hypothesis: Academic research at UNAL is more flexible than rigid administrative boundaries.
#
# To demonstrate that this is not simply a mistake, here is the composition of one of the largest semantic clusters discovered by HDBSCAN to check whether it covers several faculties.

# %%
# Find the largest non-noise cluster
largest_cluster_id = df[df['hdbscan_cluster'] != -1]['hdbscan_cluster'].value_counts().idxmax()
cluster_df = df[df['hdbscan_cluster'] == largest_cluster_id]

print(f"Composition of Largest Semantic Cluster (Cluster {largest_cluster_id}) by Faculty:")
faculty_counts = cluster_df['faculty'].value_counts()
print(faculty_counts.head(10))

print("\n--- Sampling 1 random record per Faculty from the Largest Cluster ---")
# Sample from the faculties in this cluster to check semantic coherence
for faculty in faculty_counts.index:
    sample_row = cluster_df[cluster_df['faculty'] == faculty].sample(1, random_state=42).iloc[0]
    embeddings_text = str(sample_row.get('embeddings_text', 'No Title'))
    print(f"\n{embeddings_text}")

# %% [markdown]
# ### 4.4.2. Conclusion on Interdisciplinary Mixing
#
# The sampled records demonstrate that while the cluster encompasses a wide array of faculties, there is a recurring underlying macro-theme related to society, public policy, human rights, and urban/rural dynamics (e.g., patient autonomy in Law, urban recovery in Arts, agricultural taxation impacts in Economics). 
#
# The broad nature of this cluster indicates that HDBSCAN has identified a massive interdisciplinary "hub". This confirms the hypothesis that semantic topics do not strictly adhere to administrative boundaries, explaining the near-zero NMI score. To achieve higher granularity and break this macro-topic into more specific sub-disciplines, adjusting the `min_cluster_size` parameter to a lower threshold or applying hierarchical sub-clustering could be explored in future iterations.

# %% [markdown]
# ## 4.5. Latent Expertise Extraction (c-TF-IDF)
#
# To prove that the clusters formed by HDBSCAN and K-Means are semantically coherent, top keywords per cluster are extracted using Class-based TF-IDF (c-TF-IDF).
# Since the corpus is predominantly Spanish (>70%) with a significant English representation, the stopword removal must combine both languages to correctly denoise the topics. We consolidate the logic into a reusable function for both clustering approaches.

# %%
# Download the stopwords dataset if you haven't run this on your machine before
nltk.download('stopwords')

# 0. Sample language distribution to justify custom stopwords
language_dist = df['language'].value_counts(normalize=True) * 100
print("--- Language Distribution in the Corpus ---")
print(language_dist.head(5).round(2).astype(str) + "%")

# Define custom combined stopwords list for both models
custom_stopwords = stopwords.words('spanish') + stopwords.words('english')
# Add domain-specific fluff words to prevent generic topics
custom_stopwords.extend([
    'colombia', 'estudio', 'analisis', 'resultados', 'study', 'analysis', 'results', 'nacional', 'universidad', 'bogota'
])

def extract_cluster_keywords(df, cluster_col, text_col, stop_words, top_n=10, min_df=5, max_df=0.90):
    """
    Groups documents by cluster and applies c-TF-IDF to extract top distinctive keywords per cluster.
    Filters out noise clusters (-1) if they exist.
    """
    print(f"\nGrouping documents by {cluster_col}...")
    if -1 in df[cluster_col].unique():
        docs_per_cluster = df[df[cluster_col] != -1].groupby(cluster_col)[text_col].apply(' '.join).reset_index()
    else:
        docs_per_cluster = df.groupby(cluster_col)[text_col].apply(' '.join).reset_index()

    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    X = vectorizer.fit_transform(docs_per_cluster[text_col])
    words = vectorizer.get_feature_names_out()

    ctfidf_transformer = ClassTfidfTransformer()
    c_tfidf_matrix = ctfidf_transformer.fit_transform(X)

    cluster_keywords = {}
    for i, cluster_id in enumerate(docs_per_cluster[cluster_col]):
        row = c_tfidf_matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-top_n:][::-1]
        keywords = [words[idx] for idx in top_indices]
        cluster_keywords[cluster_id] = keywords

    return cluster_keywords

# --- 1. HDBSCAN Cluster Keywords ---
# We use the clean 'classic_text' here because we want words, not the structural metadata!
print("\nExtracting keywords for HDBSCAN clusters...")
cluster_keywords = extract_cluster_keywords(
    df=df, 
    cluster_col='hdbscan_cluster', 
    text_col='classic_text', 
    stop_words=custom_stopwords, 
    top_n=10, 
    min_df=2, 
    max_df=0.95
)

# Display the keywords for the top 5 largest HDBSCAN clusters
cluster_sizes_hdbscan = df[df['hdbscan_cluster'] != -1]['hdbscan_cluster'].value_counts().head(5)

print("\n--- Top Keywords for the 5 Largest Semantic Clusters (HDBSCAN) ---")
for cluster_id, size in cluster_sizes_hdbscan.items():
    print(f"\nCluster {cluster_id} (Size: {size} documents):")
    print(", ".join(cluster_keywords[cluster_id]))

# %%
# --- 2. K-Means Cluster Keywords ---
print("Extracting keywords for KMeans clusters...")
kmeans_keywords = extract_cluster_keywords(
    df=df, 
    cluster_col='kmeans_cluster', 
    text_col='classic_text', 
    stop_words=custom_stopwords, 
    top_n=10, 
    min_df=5, 
    max_df=0.90
)

# Display results for K-Means
cluster_sizes_kmeans = df['kmeans_cluster'].value_counts()

print("\n--- Top Keywords for Semantic UMAP + K-Means (K=14) ---")
for cluster_id, size in cluster_sizes_kmeans.items():
    print(f"\nCluster {cluster_id} (Size: {size} docs):")
    print(", ".join(kmeans_keywords[cluster_id]))

# %%
print("--- Extracting Top Centroid Documents for LLM Semantic Labeling ---")

# Ensure you have your KMeans model and UMAP embeddings from the previous step
for cluster_id in sorted(df['kmeans_cluster'].unique()):
    
    # 1. Get the centroid of the cluster in the 5D UMAP space
    centroid = kmeans_umap.cluster_centers_[cluster_id].reshape(1, -1)
    
    # 2. Get the indices of documents in this cluster
    cluster_indices = df.index[df['kmeans_cluster'] == cluster_id].tolist()
    cluster_embeddings = full_reduced_embeddings[cluster_indices]
    
    # 3. Calculate distance from centroid
    distances = euclidean_distances(cluster_embeddings, centroid).flatten()
    
    # 4. Get the top 10 closest documents
    closest_relative_indices = distances.argsort()[:10]
    closest_absolute_indices = [cluster_indices[i] for i in closest_relative_indices]
    
    # 5. Print out the prompt block
    size = len(cluster_indices)
    print(f"\nCluster {cluster_id} (Size: {size} documents)")
    print("Titles closest to centroid:")
    
    titles = df.loc[closest_absolute_indices, 'embeddings_text'].dropna().tolist()
    for j, title in enumerate(titles, 1):
        # Limit title length just in case they are massive
        print(f"  {j}. {title[:150]}...")

# %% [markdown]
# # 5. Export Results
# The clustering labels, evaluation metrics, and top terms are exported for comparative analysis by the final evaluator team member.

# %%
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5.1. Export Labels
labels_output_path = OUTPUT_DIR / "05_modern_clustering_labels.parquet"

if "original_row_index" not in df.columns:
    df["original_row_index"] = df.index

export_columns = [
    "original_row_index",
    "kmeans_cluster",
    "hdbscan_cluster"
]

export_df = df[export_columns].rename(columns={
    "kmeans_cluster": "cluster_mod_kmeans",
    "hdbscan_cluster": "cluster_mod_hdbscan"
})

export_df.to_parquet(labels_output_path, index=False)
print(f"Modern clustering labels exported to: {labels_output_path}")

# 5.2. Export Metrics
metrics_output_path = OUTPUT_DIR / "05_modern_clustering_metrics.csv"

metrics_data = {
    "model": ["Modern KMeans", "HDBSCAN"],
    "silhouette": [kmeans_silhouette, hdbscan_silhouette],
    "calinski_harabasz": [kmeans_calinski, hdbscan_calinski],
    "davies_bouldin": [kmeans_davies, hdbscan_davies],
    "nmi": [kmeans_nmi, hdbscan_nmi],
    "ari": [kmeans_ari, hdbscan_ari]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(metrics_output_path, index=False)
print(f"Modern clustering metrics exported to: {metrics_output_path}")

# 5.3. Export Top Terms (HDBSCAN c-TF-IDF)
top_terms_hdbscan_path = OUTPUT_DIR / "05_modern_hdbscan_top_terms.csv"
top_terms_records = []
for cluster_id, words_list in cluster_keywords.items():
    top_terms_records.append({
        "cluster_id": cluster_id,
        "top_terms": ", ".join(words_list)
    })
    
top_terms_df = pd.DataFrame(top_terms_records)
top_terms_df.to_csv(top_terms_hdbscan_path, index=False)
print(f"HDBSCAN top terms exported to: {top_terms_hdbscan_path}")

# 5.4. Export Top Terms (K-Means c-TF-IDF)
top_terms_kmeans_path = OUTPUT_DIR / "05_modern_kmeans_top_terms.csv"
top_terms_records_kmeans = []
for cluster_id, words_list in kmeans_keywords.items():
    top_terms_records_kmeans.append({
        "cluster_id": cluster_id,
        "top_terms": ", ".join(words_list)
    })
    
top_terms_df_kmeans = pd.DataFrame(top_terms_records_kmeans)
top_terms_df_kmeans.to_csv(top_terms_kmeans_path, index=False)
print(f"K-Means top terms exported to: {top_terms_kmeans_path}")

# %% [markdown]
# # 6. Brief interpretation and future work
#
# The modern baseline fundamentally upgraded the text representation from sparse lexical matching (TF-IDF) to dense semantic vectors using `jina-embeddings-v5-text-small`. This effectively mitigated the cross-lingual semantic drift present in the bilingual (Spanish/English) corpus and captured the actual contextual meaning of the academic abstracts. 
#
# To enable a direct comparison with the classical pipeline, **K-Means** was executed using the exact same number of clusters ($k=14$). The modern embeddings outperformed the classical representation across internal metrics, yielding a significantly lower Davies-Bouldin score and a much higher Silhouette score, proving that semantic vectors create tighter, more cohesive groupings than word counts.
#
# Furthermore, **UMAP + HDBSCAN** was introduced to allow the data's natural topology to dictate the number of clusters. This density-based approach successfully identified highly granular research niches. The near-zero NMI and ARI scores against the administrative `faculty` column empirically proved our core hypothesis: academic research at the university is deeply interdisciplinary and consistently transcends rigid administrative boundaries. 
#
# **Pathways for SOTA Improvement (Future Iterations):**
#
# While this pipeline establishes a robust modern baseline, the ~34% noise rate identified by HDBSCAN highlights the limitations of rigid density-based clustering. Many of these "noise" points are actually highly valuable, multi-disciplinary abstracts that sit between distinct clusters. Future iterations to upgrade this pipeline to the 2025-2026 State-of-the-Art could include:
#
# * **Advanced Semantic Representation:** Upgrading the embedding model to **Qwen3-Embedding-8B**. This would allow for instruction-aware encoding (tailoring the latent space specifically for expert profiling) and native 32,000-token context windows to ensure massive historical publication records are never truncated.
# * **Overcoming the Single-Topic Assumption:** Replacing c-TF-IDF and HDBSCAN with **Semantic Component Analysis (SCA)**. By mathematically projecting documents into multiple semantic components, SCA can assign secondary and tertiary topics to a single abstract, virtually eliminating the 34% noise rate caused by interdisciplinary blurring.
# * **Optimal Transport Taxonomy:** Integrating **Hierarchical Contrastive Optimal Transport (HiCOT)** to frame the document-to-topic assignment as a Wasserstein distance problem. This would strictly prevent "topic collapse" and allow the system to map complex, hierarchical intersections of faculty expertise with near-perfect topic diversity.

# %% [markdown]
# # 7. References / Bibliography
#
# * **BERTopic: Neural topic modeling with a class-based TF-IDF procedure**
#   * Paper: [arXiv:2203.05794](https://arxiv.org/abs/2203.05794)
#   * GitHub: [MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)
#   * BibTeX:
#     ```bibtex
#     @article{grootendorst2022bertopic,
#       title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
#       author={Grootendorst, Maarten},
#       journal={arXiv preprint arXiv:2203.05794},
#       year={2022}
#     }
#     ```
