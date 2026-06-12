# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # 09. PRISM Fine-Tuning for Jina V5
#
# This notebook implements the Precision-Informed Semantic Modeling (PRISM) framework to adapt Jina v5's vector space to the scientific and vocabulary nuances of the Colombian academic dataset.

# %%
import os
import json
import asyncio
from pathlib import Path
import importlib.util
import subprocess
import sys

# Fixes the PyTorch fragmentation issue on T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
check_and_install("transformers")
check_and_install("scipy")

if IN_COLAB:
    print("Upgrading torchao and peft in Colab for compatibility...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0", "peft"])

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from torch.optim import AdamW
from openai import AsyncOpenAI
from scipy.stats import pearsonr, spearmanr

if IN_COLAB:
    drive.mount('/content/drive')
    # BASE_PATH = Path("/content/drive/Shareddrives/Minería/proyecto_horus/")
    BASE_PATH = Path("/content/drive/MyDrive/proyecto_horus/")
else:
    BASE_PATH = Path("../")

# %% [markdown]
# ## 1. Sparse Representative Pair Sampling
# We sample 1000 pairs (500 intra-faculty, 500 inter-faculty) to ensure balanced domain representation.
# This establishes the pairs we will label and use for fine-tuning.

# %%
df = pd.read_parquet(BASE_PATH / 'data/processed/products_modeling.parquet')
print(f"Loaded {len(df)} records.")

# We only need the embeddings_text and faculty, ignoring empty texts
df_subset = df[['embeddings_text', 'faculty']].dropna(subset=['embeddings_text']).reset_index(drop=True)
faculties = df_subset['faculty'].unique()

M = 1000
intra_pairs_count = M // 2
inter_pairs_count = M - intra_pairs_count

pairs = []
np.random.seed(42)

# Intra-faculty sampling (50% of M)
for f in faculties:
    f_data = df_subset[df_subset['faculty'] == f]['embeddings_text'].values
    if len(f_data) < 2:
        continue
    # Evenly distribute pairs across faculties
    n_pairs = intra_pairs_count // len(faculties)
    for _ in range(n_pairs):
        idx1, idx2 = np.random.choice(len(f_data), 2, replace=False)
        pairs.append({"text_A": f_data[idx1], "text_B": f_data[idx2], "type": "intra", "faculty": f})

# Inter-faculty sampling (50% of M)
for _ in range(inter_pairs_count):
    f1, f2 = np.random.choice(faculties, 2, replace=False)
    f1_data = df_subset[df_subset['faculty'] == f1]['embeddings_text'].values
    f2_data = df_subset[df_subset['faculty'] == f2]['embeddings_text'].values
    if len(f1_data) == 0 or len(f2_data) == 0:
        continue
    text_A = np.random.choice(f1_data)
    text_B = np.random.choice(f2_data)
    pairs.append({"text_A": text_A, "text_B": text_B, "type": "inter", "faculty": f"{f1}-{f2}"})

# Slice to exactly M if rounding caused an overflow
if len(pairs) > M:
    pairs = pairs[:M]

print(f"Generated {len(pairs)} document pairs.")

# %% [markdown]
# ## 2. Async Teacher Labeling
# We use `deepseek-v4-pro` to assign continuous semantic similarity scores to each pair.
# We utilize Colab Secrets (`userdata`) and `asyncio` to handle concurrent rate-limited queries.

# %%
if IN_COLAB:
    try:
        DEEPSEEK_API_KEY = userdata.get('DEEPSEEK_API_KEY')
    except Exception as e:
        print("Warning: DEEPSEEK_API_KEY not found in colab secrets. Please set it before running the labeling step.")
        DEEPSEEK_API_KEY = "dummy_key"
else:
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'dummy_key')
    if DEEPSEEK_API_KEY == "dummy_key":
        print("Warning: DEEPSEEK_API_KEY not found in environment variables. Please set it before running the labeling step.")

CHECKPOINT_FILE = BASE_PATH / 'data/processed/prism_labeled_pairs_checkpoint.json'

def safe_save_checkpoint(data, filepath):
    """Safely save checkpoint using a temporary file to avoid corruption on interrupt."""
    temp_path = str(filepath) + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, str(filepath))

async def fetch_similarity(client, pair, semaphore):
    prompt = (
        "You are an expert in academic text analysis. "
        "Evaluate the semantic similarity between the following two academic documents. "
        "Output ONLY a single continuous float value between 0.0 (completely distinct topics/semantics) "
        "and 1.0 (identical or extremely overlapping topics/semantics). Do not output any other text.\n\n"
        f"Document A: {pair['text_A']}\n\n"
        f"Document B: {pair['text_B']}"
    )

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    extra_body={
                        "thinking": {
                            "type": "enabled"
                        },
                        "reasoning_effort": "high"
                    }
                )
                message = response.choices[0].message
                if getattr(message, 'refusal', None):
                    print(f"API Refusal: {message.refusal}")
                    return None
                if not message.content:
                    return None

                score_text = message.content.strip()
                import re
                match = re.search(r'\d+\.\d+', score_text)
                score = float(match.group()) if match else float(score_text)
                return max(0.0, min(1.0, score))
            except Exception as e:
                # The SDK automatically retries standard rate-limits, this loop acts as a fallback for severe/timeout issues
                print(f"API Error: {e}. Retrying in {2 ** attempt}s...")
                await asyncio.sleep(2 ** attempt)
        return None

async def run_teacher_labeling():
    # Resume from checkpoint if it exists
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            labeled_pairs = json.load(f)
    else:
        labeled_pairs = []

    labeled_signatures = {(p['text_A'], p['text_B']) for p in labeled_pairs if 'similarity' in p and p['similarity'] is not None}
    semaphore = asyncio.Semaphore(50) # Control concurrency, 50 is safe for deepseek and Colab network

    # Initialize the AsyncOpenAI client with built-in retries
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", max_retries=5)

    tasks, task_indices = [], []
    for i, pair in enumerate(pairs):
        sig = (pair['text_A'], pair['text_B'])
        if sig not in labeled_signatures:
            tasks.append(fetch_similarity(client, pair, semaphore))
            task_indices.append(i)

    if not tasks:
        print("All pairs are already labeled.")
        return labeled_pairs

    print(f"Fetching labels for {len(tasks)} missing pairs...")
    results = await async_tqdm.gather(*tasks)

    for idx, score in zip(task_indices, results):
        if score is not None:
            pair_copy = pairs[idx].copy()
            pair_copy['similarity'] = score
            labeled_pairs.append(pair_copy)

        # Incremental save
        if len(labeled_pairs) % 50 == 0:
            safe_save_checkpoint(labeled_pairs, CHECKPOINT_FILE)

    # Final save
    safe_save_checkpoint(labeled_pairs, CHECKPOINT_FILE)

    return labeled_pairs

# UNCOMMENT TO RUN THE ASYNC TEACHER LABELING:
labeled_pairs = await run_teacher_labeling()
df_labeled = pd.DataFrame([p for p in labeled_pairs if p.get('similarity') is not None])
print(f"Successfully labeled {len(df_labeled)} pairs.")

# %% [markdown]
# ## 3. Resilient Student LoRA Fine-Tuning
# We load the Jina base model, activate its native `clustering` adapter as a warm-start, and freeze the backbone.
# We then apply the CoSENT loss over the labeled pairs.

# %%
# Ensure the DataFrame is ready for training
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        df_labeled = pd.DataFrame([p for p in json.load(f) if p.get('similarity') is not None])
else:
    # Dummy placeholder so notebook compiles/tests successfully before actual labeling
    print("Warning: Labeling checkpoint not found. Creating dummy labels for testing.")
    df_labeled = pd.DataFrame(pairs)
    df_labeled['similarity'] = np.random.uniform(0, 1, len(pairs))

# Train / Val Split (85% / 15%)
df_labeled = df_labeled.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df_labeled) * 0.85)

class PairDataset(Dataset):
    def __init__(self, df):
        self.texts_A = df['text_A'].tolist()
        self.texts_B = df['text_B'].tolist()
        self.labels = df['similarity'].values.astype(np.float32)

    def __len__(self):
        return len(self.texts_A)

    def __getitem__(self, idx):
        return self.texts_A[idx], self.texts_B[idx], self.labels[idx]

train_loader = DataLoader(PairDataset(df_labeled.iloc[:split_idx]), batch_size=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(PairDataset(df_labeled.iloc[split_idx:]), batch_size=4, shuffle=False, pin_memory=True)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Base model and setup native adapter as warm-start
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True, torch_dtype=torch.float16)
model.set_adapter("clustering") # Activates native clustering adapter weights

# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze ONLY the native clustering LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name and "clustering" in name:
        param.requires_grad = True

unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Unfrozen (trainable) LoRA parameters: {unfrozen_params}")

model.to(device)

def cosent_loss(emb1, emb2, labels, temperature=20.0):
    """
    CoSENT loss for continuous labels.
    Penalizes rank inversions where y_m > y_n but cos(s_m) < cos(s_n).
    """
    cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)
    labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    sim_diff = cosine_sim.unsqueeze(0) - cosine_sim.unsqueeze(1)

    mask = (labels_diff > 0).float()
    exp_term = torch.exp(temperature * sim_diff) * mask
    return torch.log(1 + exp_term.sum())

# %%
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 1. Automatic Mixed Precision (AMP) configuration for T4
# Updated API: torch.cuda.amp.GradScaler is deprecated in newer PyTorch versions
try:
    scaler = torch.amp.GradScaler('cuda')
except AttributeError:
    # Fallback for older PyTorch versions
    scaler = torch.cuda.amp.GradScaler()

EPOCHS = 5
BEST_CHECKPOINT_DIR = BASE_PATH / "models/jina_v5_prism_checkpoint"
os.makedirs(BEST_CHECKPOINT_DIR, exist_ok=True)

# 2. Correction of the metric lower limit
best_metric = -float('inf')

# 3. Move the function outside the training loop
def get_last_token(outputs, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = outputs.last_hidden_state.shape[0]
    return outputs.last_hidden_state[
        torch.arange(batch_size, device=outputs.last_hidden_state.device),
        sequence_lengths
    ]

TRAINING_FLAG = BEST_CHECKPOINT_DIR / "training_completed.flag"

if os.path.exists(TRAINING_FLAG):
    print(f"Training already completed. Best model checkpoint is available at {BEST_CHECKPOINT_DIR}")
else:
    # Training Loop
    accumulation_steps = 8
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (texts_A, texts_B, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            labels = labels.to(device)

            encoded_A = model.tokenizer(list(texts_A), padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            encoded_B = model.tokenizer(list(texts_B), padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

            # Forward pass in mixed precision (FP16)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass automatically routes through the active ('clustering') adapter
                out_A = model(**encoded_A)
                out_B = model(**encoded_B)

                emb_A = get_last_token(out_A, encoded_A['attention_mask'])
                emb_A = F.normalize(emb_A, p=2, dim=1)

                emb_B = get_last_token(out_B, encoded_B['attention_mask'])
                emb_B = F.normalize(emb_B, p=2, dim=1)

                loss = cosent_loss(emb_A, emb_B, labels)
                loss = loss / accumulation_steps

            # Gradient scaling to avoid underflow in FP16
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch+1} Avg Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_sims, val_labels_list = [], []
        with torch.no_grad():
            for texts_A, texts_B, labels in val_loader:
                emb_A = model.encode(list(texts_A), task="clustering")
                emb_B = model.encode(list(texts_B), task="clustering")

                if not isinstance(emb_A, torch.Tensor):
                    emb_A = torch.tensor(emb_A)
                if not isinstance(emb_B, torch.Tensor):
                    emb_B = torch.tensor(emb_B)

                sims = F.cosine_similarity(emb_A, emb_B, dim=1).cpu().numpy()
                val_sims.extend(sims)
                val_labels_list.extend(labels.cpu().numpy())

        pearson_corr, _ = pearsonr(val_labels_list, val_sims)
        spearman_corr, _ = spearmanr(val_labels_list, val_sims)

        print(f"Validation - Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

        # Checkpoint selection based on best combined correlation
        combined_metric = pearson_corr + spearman_corr
        if combined_metric > best_metric:
            best_metric = combined_metric
            # Save ONLY the updated LoRA adapter weights manually because PEFT doesn't support save_pretrained for Jina v5 natively yet
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, BEST_CHECKPOINT_DIR / "adapter_weights.pt")
            model.tokenizer.save_pretrained(BEST_CHECKPOINT_DIR)
            print("--> New best model checkpoint saved!")

    # Mark training as completed
    with open(TRAINING_FLAG, 'w') as f:
        f.write("done")

# %% [markdown]
# ## 4. Re-Extraction
# Apply the best fine-tuned clustering adapter to batch process all rows and extract the final 1024-d embeddings.

# %%
OUTPUT_NPY = BASE_PATH / 'data/processed/prism_embeddings.npy'

if os.path.exists(OUTPUT_NPY):
    print(f"Embeddings already exist at {OUTPUT_NPY}. Skipping extraction.")
else:
    print("Loading best checkpoint for extraction...")
    # Load base model again
    best_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True, torch_dtype=torch.float16)
    best_model.set_adapter("clustering")

    # Inject the fine-tuned LoRA weights
    adapter_weights = torch.load(BEST_CHECKPOINT_DIR / "adapter_weights.pt", map_location=device)
    best_model.load_state_dict(adapter_weights, strict=False)

    best_model.to(device)
    best_model.eval()

    # We MUST save the original index so we can put the embeddings back in order later
    print("Sorting texts by length to optimize extraction speed...")
    df['text_length'] = df['embeddings_text'].str.len()
    df_sorted = df.sort_values('text_length', ascending=False).reset_index(drop=False)

    all_texts = df_sorted['embeddings_text'].tolist()
    total_texts = len(all_texts)
    batch_size = 16

    # Save chunks directly to Google Drive so progress survives Colab disconnects
    CHUNKS_DIR = BASE_PATH / 'data/processed/prism_embeddings_chunks'
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print(f"Extracting embeddings for {total_texts} documents on {device}...")

    with torch.no_grad():
        for i in tqdm(range(0, total_texts, batch_size), desc="Extracting PRISM Embeddings"):
            chunk_file = CHUNKS_DIR / f"chunk_{i}.npy"

            # Skip if this batch is already processed
            if os.path.exists(chunk_file):
                continue

            batch_texts = all_texts[i:i+batch_size]
            batch_embs = best_model.encode(batch_texts, task="clustering", max_length=1024)

            # The AutoModel returns CUDA tensors, we MUST move them to CPU and convert to numpy before saving
            if isinstance(batch_embs, torch.Tensor):
                batch_embs = batch_embs.cpu().numpy()

            np.save(chunk_file, batch_embs)
            torch.cuda.empty_cache()

    print("All chunks processed. Assembling final matrix...")
    all_embeddings = []
    for i in range(0, total_texts, batch_size):
        chunk_file = CHUNKS_DIR / f"chunk_{i}.npy"
        all_embeddings.append(np.load(chunk_file))

    stacked_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Stacked shape: {stacked_embeddings.shape}")

    print("Restoring original order...")
    original_indices = df_sorted['index'].values
    final_matrix = np.empty_like(stacked_embeddings)
    final_matrix[original_indices] = stacked_embeddings

    # Resilient atomic write
    temp_output = OUTPUT_NPY.with_suffix('.tmp.npy')
    np.save(temp_output, final_matrix)
    os.replace(temp_output, OUTPUT_NPY)
    print(f"Saved optimized PRISM embeddings to {OUTPUT_NPY}")

    # Clean up intermediate chunks
    import shutil
    shutil.rmtree(CHUNKS_DIR)

# %% [markdown]
# # 5. References / Bibliography
#
# * **PRISM: LLM-Guided Semantic Clustering for High-Precision Topics**
#   * Paper: [arXiv:2604.03180](https://arxiv.org/abs/2604.03180)
#   * GitHub: [connordouglas10/PRISM](https://github.com/connordouglas10/PRISM)
#   * BibTeX:
#     ```bibtex
#     @inproceedings{douglas2026prism,
#       author    = {Douglas, Connor and Balci, Utkucan and Aylett-Bullock, Joseph},
#       title     = {PRISM: LLM-Guided Semantic Clustering for High-Precision Topics},
#       booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
#       year      = {2026},
#       note      = {arXiv:2604.03180 [cs.LG]}
#     }
#     ```

