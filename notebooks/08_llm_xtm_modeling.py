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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="9435f127"
# # 08. VAE with LLM-XTM Multi-Phase Refinement
# This notebook implements the LLM-XTM cross-lingual topic model using a decoupled
# workflow to prevent GPU bottlenecking and API rate/cost explosions.

# %% colab={"base_uri": "https://localhost:8080/"} id="272eee18" outputId="a6252f8c-e08a-45dc-8b54-e255ff2217f4"
import os
import json
import asyncio
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm.asyncio import tqdm as async_tqdm
from pathlib import Path
import sys
import importlib.util
import subprocess

try:
    from google.colab import drive, userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def check_and_install(package_name, pip_name=None):
    if pip_name is None:
        pip_name = package_name
    if importlib.util.find_spec(package_name) is None:
        if IN_COLAB:
            print(f"Installing {pip_name} in Colab...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        else:
            print(f"Warning: '{package_name}' is missing. Make sure to install '{pip_name}' in your local .venv")

check_and_install("openai")

from openai import AsyncOpenAI

if IN_COLAB:
    drive.mount('/content/drive')
    BASE_PATH = Path("/content/drive/Shareddrives/Minería/proyecto_horus/")
    # BASE_PATH = Path("/content/drive/MyDrive/proyecto_horus/")
else:
    BASE_PATH = Path("../")

# %% [markdown] id="7bfb470d"
# ## Global Configurations & Constants

# %% id="74db3363"
RANDOM_STATE = 42
MAX_FEATURES = 10000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS_PHASE_0 = 10
EPOCHS_PHASE_1 = 50
EPOCHS_PHASE_3 = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_CANDIDATES = [30, 50, 70]
LLM_REPEATS = 3  # Self-consistency R

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# %% [markdown] id="cd31c2c9"
# ## 1. Data Preparation
# Load the preprocessed dataset and build the Bag-of-Words (BoW) matrix strictly using `tokens_ngrams`.

# %% colab={"base_uri": "https://localhost:8080/"} id="9da2f078" outputId="fc16440f-5be2-4857-e8bf-1b89e2104cfc"
print("Loading data...")
df = pd.read_parquet(BASE_PATH / "data/processed/products_modeling.parquet")

# Build BoW matrix using CountVectorizer with max_features=10000
texts = df['tokens_ngrams'].apply(
    lambda x: " ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x)
).tolist()

print("Vectorizing...")
vectorizer = CountVectorizer(max_features=MAX_FEATURES)
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
id2word = {i: w for i, w in enumerate(vocab)}
word2id = {w: i for i, w in enumerate(vocab)}

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        # Return as a tuple to maintain compatibility with TensorDataset extraction logic
        return (torch.tensor(self.sparse_matrix[idx].toarray()[0], dtype=torch.float32),)

# Create DataLoader dynamically loading dense tensors from the sparse matrix
dataset = SparseDataset(X)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print(f"Data shape: {X.shape}")

# %% [markdown] id="dd36279f"
# ## 2. Base VAE Architecture
# A standard Variational Autoencoder configured for text inputs.

# %% id="b3d35704"
class BaseVAE(nn.Module):
    def __init__(self, vocab_size, hidden_dim, latent_dim):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout(h1)
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3 = self.dropout(h3)
        return torch.log_softmax(self.fc4(h3), dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def compute_elbo_loss(recon_x, x, mu, logvar):
    """Computes the ELBO loss: Reconstruction + KL divergence"""
    # x is BoW counts, recon_x is log_softmax
    recon_loss = -torch.sum(x * recon_x, dim=-1).mean()
    # KL divergence from N(0, 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return recon_loss + kl_loss

def get_top_words(model, id2word_map, top_n=10):
    """Extracts top words per topic from the decoder using the identity matrix."""
    model.eval()
    with torch.no_grad():
        z = torch.eye(model.latent_dim).to(DEVICE)
        out = model.decode(z) # shape [latent_dim, vocab_size]

    topics = []
    for i in range(model.latent_dim):
        top_indices = torch.argsort(out[i], descending=True)[:top_n]
        topic_words = [id2word_map[idx.item()] for idx in top_indices]
        topics.append(topic_words)
    return topics

# %% [markdown] id="5c1d2e70"
# ## 3. Phase 0: Hyperparameter Pre-Search
# Local grid search for optimal K based on NPMI coherence.

# %% colab={"base_uri": "https://localhost:8080/"} id="f8ad2a66" outputId="7a6c1d6b-ff6d-4fce-aeea-2431e24c7f50"
def compute_npmi(topic_words, texts_list):
    """Computes the Normalized Pointwise Mutual Information (NPMI) for topic coherence."""
    from sklearn.feature_extraction.text import CountVectorizer
    import itertools

    cv = CountVectorizer(vocabulary=list(set(itertools.chain(*topic_words))))
    doc_word_matrix = cv.fit_transform(texts_list).toarray()
    vocab_map = cv.vocabulary_

    num_docs = len(texts_list)
    npmi_scores = []

    for words in topic_words:
        word_ids = [vocab_map[w] for w in words if w in vocab_map]
        if len(word_ids) < 2:
            npmi_scores.append(0.0)
            continue

        topic_npmi = []
        for i, j in combinations(word_ids, 2):
            doc_i = doc_word_matrix[:, i] > 0
            doc_j = doc_word_matrix[:, j] > 0

            p_i = doc_i.sum() / num_docs
            p_j = doc_j.sum() / num_docs
            p_ij = (doc_i & doc_j).sum() / num_docs

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

print("--- Starting Phase 0: Hyperparameter Pre-Search ---")
best_k = K_CANDIDATES[0]
best_score = -float('inf')

for k in K_CANDIDATES:
    print(f"Testing K={k}...")
    model = BaseVAE(vocab_size=MAX_FEATURES, hidden_dim=256, latent_dim=k).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS_PHASE_0):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = compute_elbo_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate locally
    topics = get_top_words(model, id2word, top_n=10)
    score = compute_npmi(topics, texts)
    print(f"K={k} | NPMI Score: {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

    # Free GPU memory
    del model
    del optimizer
    torch.cuda.empty_cache()

print(f"Selected Optimal K: {best_k}")

# %% [markdown] id="61369d73"
# ## 4. Phase 1: VAE Backbone Pre-Training
# Fully train the base VAE using the selected optimal K.

# %% colab={"base_uri": "https://localhost:8080/"} id="6e8ab406" outputId="bc8fd76a-fc6b-4c88-ae77-6a699286cfc8"
print(f"--- Starting Phase 1: Pre-training with K={best_k} ---")
final_model = BaseVAE(vocab_size=MAX_FEATURES, hidden_dim=256, latent_dim=best_k).to(DEVICE)
optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

phase_1_ckpt_path = BASE_PATH / "outputs/08_llm_xtm_phase1_checkpoint.pt"
start_epoch_p1 = 0

if os.path.exists(phase_1_ckpt_path):
    print(f"Resuming Phase 1 from checkpoint: {phase_1_ckpt_path}")
    checkpoint = torch.load(phase_1_ckpt_path, map_location=DEVICE)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_p1 = checkpoint['epoch'] + 1
    print(f"Resumed at epoch {start_epoch_p1}")

final_model.train()
for epoch in range(start_epoch_p1, EPOCHS_PHASE_1):
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        optimizer.zero_grad()
        recon_x, mu, logvar = final_model(x)
        loss = compute_elbo_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Epoch checkpointing for Phase 1
    torch.save({
        'epoch': epoch,
        'model_state_dict': final_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, phase_1_ckpt_path)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE_1} | Loss: {total_loss/len(dataloader):.4f}")

# Extract top 15 words for Phase 2 LLM refinement
top_words_k = get_top_words(final_model, id2word, top_n=15)

# %% [markdown] id="542d108e"
# ## 5. Phase 2: Offline Async LLM Refinement
# Send topics to DeepSeek API to align English and Spanish semantics.

# %% colab={"base_uri": "https://localhost:8080/"} id="caf65705" outputId="a4fd327b-6e21-4e71-cd30-3faed93d2c43"
print("--- Starting Phase 2: Async LLM Refinement ---")
if IN_COLAB:
    try:
        deepseek_api_key = userdata.get('DEEPSEEK_API_KEY')
    except Exception as e:
        print("Warning: DEEPSEEK_API_KEY not found in colab secrets. Please set it before running the labeling step.")
        deepseek_api_key = ""
else:
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not deepseek_api_key:
        print("Warning: DEEPSEEK_API_KEY environment variable is missing. API calls will fail.")

checkpoint_path = BASE_PATH / "outputs/08_llm_xtm_word_refinements_checkpoint.json"

def safe_save_checkpoint(data, filepath):
    temp_path = str(filepath) + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, str(filepath))

async def query_llm(client, topic_words, semaphore):
    """Sends a prompt to the DeepSeek API to align semantic words."""
    if not client:
        return []

    prompt = (
        "Align the following words cross-lingually (English and Spanish) and filter out "
        "irrelevant terms to form a cohesive topic. Return ONLY a valid JSON list of "
        f"strings containing the refined words.\nWords: {', '.join(topic_words)}"
    )

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": "You strictly output valid JSON arrays of strings. No markdown, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    extra_body={
                        "thinking": {
                            "type": "disabled"
                        }
                    }
                )
                message = response.choices[0].message
                if getattr(message, 'refusal', None):
                    print(f"LLM API Refusal: {message.refusal}")
                    return []

                content = message.content.strip()

                # Clean potential markdown wrapping
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                refined_words = json.loads(content)
                if isinstance(refined_words, list):
                    return [str(w).lower() for w in refined_words]
                return []
            except Exception as e:
                # SDK automatically handles rate-limits, this loop acts as a fallback for severe/timeout issues
                print(f"LLM API Error: {e}. Retrying in {2 ** attempt}s...")
                await asyncio.sleep(2 ** attempt)
        return []

async def process_topic(client, i, words, semaphore, refined_topics_dict):
    """Queries the LLM R times and applies self-consistency aggregation."""
    # Check if this topic is already checkpointed
    if str(i) in refined_topics_dict:
        return i, refined_topics_dict[str(i)]

    # We can gather the runs concurrently
    tasks = [query_llm(client, words, semaphore) for _ in range(LLM_REPEATS)]
    all_runs = await asyncio.gather(*tasks)

    # Self-consistency aggregation: $f_k(v) = \frac{1}{R}\sum_{r=1}^{R}1\{v \in \tilde{w}_k^{(r)}\}$
    word_counts = {}
    for run in all_runs:
        for w in set(run):  # use set to ensure count is at most 1 per run
            word_counts[w] = word_counts.get(w, 0) + 1

    # Keep words that appeared in at least 2 out of 3 runs
    final_words = [w for w, c in word_counts.items() if c >= 2]

    # Fallback to original top words if the LLM completely failed
    if not final_words:
        final_words = words

    return i, final_words

async def run_phase_2_async():
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            refined_topics_dict = json.load(f)
    else:
        refined_topics_dict = {}

    if deepseek_api_key:
        client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com", max_retries=5)
    else:
        client = None

    semaphore = asyncio.Semaphore(50) # Control concurrency

    tasks = []
    for i, words in enumerate(top_words_k):
        if str(i) not in refined_topics_dict:
            tasks.append(process_topic(client, i, words, semaphore, refined_topics_dict))

    if not tasks:
        print("All topics are already refined.")
        return refined_topics_dict

    print(f"Processing {len(tasks)} missing topics...")

    # Execute async API calls and save incrementally
    for f in async_tqdm.as_completed(tasks):
        i, final_words = await f
        refined_topics_dict[str(i)] = final_words
        safe_save_checkpoint(refined_topics_dict, checkpoint_path)

    return refined_topics_dict

# Execute async Phase 2 (Note: top-level await is supported in Jupyter)
refined_topics_dict = await run_phase_2_async()

print("Phase 2 Complete. Refinements saved to checkpoint.")

# %% [markdown] id="0899ed50"
# ## 6. Phase 3: Multi-Objective Fine-Tuning
# Resume VAE training applying MMD loss and Doc-Align KL Divergence based on LLM outputs.

# %% colab={"base_uri": "https://localhost:8080/"} id="582e854a" outputId="dc02a13f-7c49-4fc3-8d68-34ba63c7bfde"
print("--- Starting Phase 3: Multi-Objective Fine-Tuning ---")

# Prepare Target Distribution Tensors based on LLM refinements
target_distributions = torch.zeros((best_k, MAX_FEATURES), dtype=torch.float32, device=DEVICE)

for i in range(best_k):
    refined_words = refined_topics_dict[str(i)]
    valid_words = [w for w in refined_words if w in word2id]
    if valid_words:
        prob = 1.0 / len(valid_words)
        for w in valid_words:
            target_distributions[i, word2id[w]] = prob
    else:
        # Uniform fallback
        target_distributions[i, :] = 1.0 / MAX_FEATURES

def compute_mmd_loss(z_samples):
    """Computes Maximum Mean Discrepancy (MMD) strictly in float32."""
    z_samples = z_samples.to(torch.float32)
    B, dim = z_samples.shape

    prior_samples = torch.randn(B, dim, dtype=torch.float32, device=DEVICE)

    def rbf_kernel(x, y):
        # x: [B, dim], y: [B, dim]
        xx = torch.matmul(x, x.T)
        yy = torch.matmul(y, y.T)
        zz = torch.matmul(x, y.T)
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dist = rx.T + ry - 2.*zz
        # Inverse multiquadric kernel
        return 2.0 * dim / (2.0 * dim + dist)

    k_xx = rbf_kernel(z_samples, z_samples).mean()
    k_yy = rbf_kernel(prior_samples, prior_samples).mean()
    k_xy = rbf_kernel(z_samples, prior_samples).mean()

    return k_xx + k_yy - 2*k_xy

gamma_1 = 10.0  # MMD weight
gamma_2 = 1.0   # Doc-align weight

phase_3_ckpt_path = BASE_PATH / "outputs/08_llm_xtm_checkpoint.pt"
start_epoch_p3 = 0

if os.path.exists(phase_3_ckpt_path):
    print(f"Resuming Phase 3 from checkpoint: {phase_3_ckpt_path}")
    checkpoint = torch.load(phase_3_ckpt_path, map_location=DEVICE)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_p3 = checkpoint['epoch'] + 1
    print(f"Resumed at epoch {start_epoch_p3}")

final_model.train()
for epoch in range(start_epoch_p3, EPOCHS_PHASE_3):
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        recon_x, mu, logvar = final_model(x)
        z = final_model.reparameterize(mu, logvar)

        # Base ELBO Loss
        vae_loss = compute_elbo_loss(recon_x, x, mu, logvar)

        # MMD Loss
        mmd_loss = compute_mmd_loss(z)

        # Doc-Align Loss (KL Divergence)
        # We align the current decoder's output distribution (log_softmax) with the target distribution
        decoder_log_probs = final_model.decode(torch.eye(best_k, device=DEVICE))
        doc_align_loss = nn.KLDivLoss(reduction='batchmean')(decoder_log_probs, target_distributions)

        # Multi-objective Loss calculation
        loss = vae_loss + gamma_1 * mmd_loss + gamma_2 * doc_align_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Epoch checkpointing for resilience
    torch.save({
        'epoch': epoch,
        'model_state_dict': final_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, phase_3_ckpt_path)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE_3} | Loss: {total_loss/len(dataloader):.4f}")

# %% [markdown] id="f3db8a93"
# ## 7. Evaluation & Outputs
# Extract document-topic probability distributions $\theta$ and calculate metrics.

# %% colab={"base_uri": "https://localhost:8080/"} id="807da1a0" outputId="87754f7c-0975-411a-f009-daa09e423b9b"
print("--- Starting Evaluation ---")
final_model.eval()
all_theta = []

# Create a sequential dataloader to preserve order for evaluation
eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

with torch.no_grad():
    for batch in eval_dataloader:
        x = batch[0].to(DEVICE)
        mu, _ = final_model.encode(x)
        # Convert latent mu to probabilities using softmax
        theta = torch.softmax(mu, dim=1).cpu().numpy()
        all_theta.append(theta)

theta_matrix = np.vstack(all_theta)
labels = np.argmax(theta_matrix, axis=1)

# Ensure valid clustering
if len(set(labels)) > 1:
    # Use sample size for large datasets to avoid memory errors
    # SLICE FIRST, THEN CONVERT TO ARRAY (Prevents OOM)
    X_dense_sample = X[:10000].toarray()

    sil_score = silhouette_score(X_dense_sample, labels[:10000])
    ch_score = calinski_harabasz_score(X_dense_sample, labels[:10000])
    db_score = davies_bouldin_score(X_dense_sample, labels[:10000])
else:
    sil_score, ch_score, db_score = -1, -1, -1

metrics_df = pd.DataFrame([{
    "Model": "LLM-XTM",
    "K": best_k,
    "Silhouette": sil_score,
    "Calinski_Harabasz": ch_score,
    "Davies_Bouldin": db_score
}])

# Export Metrics
metrics_df.to_csv(BASE_PATH / "outputs/08_llm_xtm_metrics.csv", index=False)
print("Saved metrics to outputs/08_llm_xtm_metrics.csv")
print(metrics_df)

# Export Theta distributions
theta_df = pd.DataFrame(theta_matrix, columns=[f"Topic_{i}" for i in range(best_k)])
theta_df.to_parquet(BASE_PATH / "outputs/08_llm_xtm_theta.parquet")
print("Saved theta to outputs/08_llm_xtm_theta.parquet")

print("--- Pipeline Complete ---")

# %% [markdown] id="8b77208b"
# # 8. References / Bibliography
#
# * **LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models**
#   * Paper: [arXiv:2605.03299](https://arxiv.org/abs/2605.03299)
#   * GitHub: [tienphat140205/LLM-XTM](https://github.com/tienphat140205/LLM-XTM)
#   * BibTeX:
#     ```bibtex
#     @misc{xuan2026llmxtmenhancingcrosslingualtopic,
#           title={LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models},
#           author={Minh Chu Xuan and Tien-Phat Nguyen and Linh Ngo Van and Dinh Viet Sang and Nguyen Thi Ngoc Diep and Trung Le},
#           year={2026},
#           eprint={2605.03299},
#           archivePrefix={arXiv},
#           primaryClass={cs.CL},
#           url={https://arxiv.org/abs/2605.03299},
#     }
#     ```

