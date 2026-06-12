import os
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from horus.config import (
    PRISM_EMBEDDINGS_PATH,
    LEIDEN_COMMUNITIES_PATH,
    EMBEDDING_MODEL_NAME,
    DEFAULT_WEIGHT_SEMANTIC,
    DEFAULT_WEIGHT_PAGERANK,
    DEFAULT_WEIGHT_DENSITY,
    BASE_DIR
)
from horus.graph_utils import (
    load_professors_registry,
    build_or_load_graph,
    parse_coauthors
)

PRISM_CHECKPOINT_DIR = BASE_DIR / "models/jina_v5_prism_checkpoint"

class PrismEmbedder:
    """
    Handles query embedding generation using Hugging Face AutoModel on CPU.
    Injects PRISM LoRA adapter weights for semantic alignment.
    """
    def __init__(self):
        print("Initializing PRISM Embedder on CPU...")
        self.device = torch.device("cpu")
        
        # Load base model and tokenizer
        # trust_remote_code=True is required for Jina v5
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.float32
        )
        self.model.set_adapter("clustering")
        
        # Inject fine-tuned LoRA weights if available
        adapter_weights_path = PRISM_CHECKPOINT_DIR / "adapter_weights.pt"
        if adapter_weights_path.exists():
            print(f"Injecting fine-tuned PRISM weights from {adapter_weights_path}...")
            try:
                state_dict = torch.load(adapter_weights_path, map_location=self.device)
                # Strict=False because we only load trainable LoRA parameters
                self.model.load_state_dict(state_dict, strict=False)
                print("PRISM adapter weights loaded successfully.")
            except Exception as e:
                print(f"Error loading PRISM adapter weights: {e}. Falling back to default clustering adapter.")
        else:
            print(f"PRISM adapter weights not found at {adapter_weights_path}. Using base clustering adapter.")
            
        self.model.eval()

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a 1024D L2-normalized embedding for a query text.
        """
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Retrieve last token representation (matching training scheme in 09_prism_finetuning.py)
            attention_mask = inputs['attention_mask']
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            
            embeddings = outputs.last_hidden_state[
                torch.arange(batch_size, device=self.device),
                sequence_lengths
            ]
            # Convert to float32 first to preserve normalization precision on CPU
            embeddings = embeddings.to(torch.float32)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings.cpu().numpy()[0]


class HorusRecommender:
    """
    Manages the recommendation index and computes hybrid scores for academic advisors.
    """
    def __init__(self):
        self.embedder = None
        self.faiss_index = None
        self.df_meta = None
        self.registry = None
        self.G = None
        self.pagerank = None
        
    def lazy_load(self):
        """
        Performs slow loading operations only when first needed to keep imports instant.
        """
        if self.embedder is not None:
            return
            
        # 1. Load configuration and model registry
        self.registry = load_professors_registry()
        
        # 2. Build or load the co-authorship network and PageRank scores
        self.G, self.pagerank = build_or_load_graph()
        
        # 3. Load metadata containing publications and Leiden communities
        print(f"Loading metadata from {LEIDEN_COMMUNITIES_PATH}...")
        if not LEIDEN_COMMUNITIES_PATH.exists():
            raise FileNotFoundError(f"Metadata file missing: {LEIDEN_COMMUNITIES_PATH}")
        self.df_meta = pd.read_parquet(LEIDEN_COMMUNITIES_PATH)
        
        # 4. Load PRISM embeddings and build FAISS index
        print(f"Loading embeddings from {PRISM_EMBEDDINGS_PATH}...")
        if not PRISM_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Embeddings matrix missing: {PRISM_EMBEDDINGS_PATH}")
        embeddings = np.load(PRISM_EMBEDDINGS_PATH).astype(np.float32)
        
        # Ensure L2 normalization for Cosine Similarity inner product
        faiss.normalize_L2(embeddings)
        
        # Initialize FAISS exact index (Inner Product)
        d = embeddings.shape[1]  # 1024
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings)
        
        # 5. Initialize the CPU embedder
        self.embedder = PrismEmbedder()
        print("HorusRecommender fully initialized.")

    def recommend_advisors(
        self, 
        proposal_text: str, 
        top_k_docs: int = 50,
        weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
        weight_pagerank: float = DEFAULT_WEIGHT_PAGERANK,
        weight_density: float = DEFAULT_WEIGHT_DENSITY,
        active_only: bool = True
    ) -> list[dict]:
        """
        Computes hybrid scores to recommend the best thesis advisors.
        
        Args:
            proposal_text (str): Title and abstract/description of the thesis proposal.
            top_k_docs (int): Number of nearest documents to retrieve for author aggregation.
            weight_semantic (float): Weight for the semantic similarity sub-score (0-1).
            weight_pagerank (float): Weight for the topological PageRank authority sub-score (0-1).
            weight_density (float): Weight for the co-authorship density sub-score (0-1).
            active_only (bool): If True, filters out retired advisors.
            
        Returns:
            list[dict]: Ranked list of recommended advisors and their explanation details.
        """
        self.lazy_load()
        
        # Normalize weights
        total_w = weight_semantic + weight_pagerank + weight_density
        if total_w == 0:
            weight_semantic, weight_pagerank, weight_density = 0.34, 0.33, 0.33
        else:
            weight_semantic /= total_w
            weight_pagerank /= total_w
            weight_density /= total_w
            
        # 1. Compute query embedding
        query_emb = self.embedder.get_embedding(proposal_text)
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # 2. Retrieve top nearest documents in FAISS index
        similarities, indices = self.faiss_index.search(query_emb, top_k_docs)
        similarities = similarities[0]
        doc_indices = indices[0]
        
        # Get dominant Leiden community among the top retrieved documents
        retrieved_docs = self.df_meta.iloc[doc_indices].copy()
        retrieved_docs['similarity'] = similarities
        
        valid_comms = retrieved_docs[retrieved_docs['community_id'] != -1]['community_id']
        dominant_community = int(valid_comms.mode().iloc[0]) if not valid_comms.empty else -1
        
        # 3. Aggregate documents by advisor
        advisor_data = {}
        
        for idx, row in retrieved_docs.iterrows():
            sim = float(row['similarity'])
            coauthors = parse_coauthors(row['active_coauthors'])
            comm_id = int(row['community_id'])
            
            for author in coauthors:
                # We only consider authors registered in the official university registry
                if author not in self.registry:
                    continue
                    
                reg_info = self.registry[author]
                if active_only and reg_info['status'] != "Activo":
                    continue
                    
                if author not in advisor_data:
                    advisor_data[author] = {
                        "name": reg_info["original_name"],
                        "status": reg_info["status"],
                        "total_products": reg_info["product_count"],
                        "matched_papers": [],
                        "max_similarity": 0.0,
                        "similarities": [],
                        "in_dominant_community_count": 0
                    }
                    
                advisor_data[author]["matched_papers"].append({
                    "title": row["original_title"],
                    "similarity": sim,
                    "type": row["type"],
                    "date": str(row["date"]),
                    "faculty": row["faculty"]
                })
                
                advisor_data[author]["similarities"].append(sim)
                if sim > advisor_data[author]["max_similarity"]:
                    advisor_data[author]["max_similarity"] = sim
                    
                if comm_id == dominant_community:
                    advisor_data[author]["in_dominant_community_count"] += 1

        if not advisor_data:
            return []

        # 4. Calculate metrics for normalized scoring
        max_matched_papers = max(len(info["matched_papers"]) for info in advisor_data.values())
        
        # Collect PageRank scores for candidates (normalized by max candidates pagerank)
        candidate_pageranks = {
            author: float(self.pagerank.get(author, 0.0)) for author in advisor_data
        }
        max_pagerank = max(candidate_pageranks.values()) if candidate_pageranks else 1.0
        if max_pagerank == 0:
            max_pagerank = 1.0
            
        # 5. Compute final hybrid scores
        recommendations = []
        for author, info in advisor_data.items():
            # Sub-score 1: Semantic match (average similarity of matched publications)
            sub_semantic = float(np.mean(info["similarities"]))
            
            # Sub-score 2: Topological authority (normalized PageRank)
            sub_pagerank = candidate_pageranks[author] / max_pagerank
            
            # Sub-score 3: Co-occurrence density (matched papers count relative to max candidate)
            sub_density = len(info["matched_papers"]) / max_matched_papers
            
            # Calculate final score
            hybrid_score = (
                weight_semantic * sub_semantic +
                weight_pagerank * sub_pagerank +
                weight_density * sub_density
            )
            
            # Sort matched papers by similarity
            sorted_papers = sorted(info["matched_papers"], key=lambda x: x["similarity"], reverse=True)
            
            recommendations.append({
                "username": author,
                "name": info["name"],
                "status": info["status"],
                "total_products": info["total_products"],
                "hybrid_score": float(hybrid_score),
                "sub_scores": {
                    "semantic": sub_semantic,
                    "pagerank": sub_pagerank,
                    "density": sub_density
                },
                "matched_papers": sorted_papers[:5],  # Keep top 5 papers for display
                "papers_matched_count": len(info["matched_papers"]),
                "in_dominant_community_count": info["in_dominant_community_count"]
            })
            
        # Sort recommendations by final score descending
        recommendations = sorted(recommendations, key=lambda x: x["hybrid_score"], reverse=True)
        return recommendations
