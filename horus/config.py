import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Data paths
PRODUCTS_MODELING_PATH = PROCESSED_DATA_DIR / "products_modeling.parquet"
PRISM_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "prism_embeddings.npy"
BASE_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "final_embeddings_matrix.npy"
DOCENTES_INVESTIGADORES_PATH = EXTERNAL_DATA_DIR / "docentes_investigadores.csv"
LEIDEN_COMMUNITIES_PATH = PROCESSED_DATA_DIR / "graph_topics_min_communities.parquet"

# Topic metadata output from eval
TOPIC_EVAL_PATH = PROCESSED_DATA_DIR / "11_llm_top_terms_evaluation.json"

# Models
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v5-text-small"

# Default weights for the hybrid recommender scoring
DEFAULT_WEIGHT_SEMANTIC = 0.5
DEFAULT_WEIGHT_PAGERANK = 0.3
DEFAULT_WEIGHT_DENSITY = 0.2
