import re
import pickle
import unicodedata
from pathlib import Path
import pandas as pd
import networkx as nx
from collections import Counter
from itertools import combinations
from horus.config import (
    PRODUCTS_MODELING_PATH,
    DOCENTES_INVESTIGADORES_PATH,
    PROCESSED_DATA_DIR
)

CACHE_GRAPH_PATH = PROCESSED_DATA_DIR / "horus_coauthorship_graph.pkl"
CACHE_PAGERANK_PATH = PROCESSED_DATA_DIR / "horus_pagerank.pkl"

def normalize_name(name: str) -> str:
    """
    Normalizes professor names by converting to lowercase, removing accents, 
    removing punctuation (periods and commas), and stripping extra whitespace.
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # Remove accents/diacritics
    name = ''.join(
        char for char in unicodedata.normalize('NFD', name)
        if unicodedata.category(char) != 'Mn'
    )
    # Remove punctuation
    name = re.sub(r'[.,]', '', name)
    return ' '.join(name.split())

def parse_coauthors(coauthors_text: str) -> list[str]:
    """
    Parses a comma-separated co-author string into a list of normalized names.
    """
    if pd.isna(coauthors_text) or not coauthors_text:
        return []
    return [
        normalize_name(name)
        for name in str(coauthors_text).split(',')
        if name.strip()
    ]

def load_professors_registry() -> dict[str, dict]:
    """
    Loads the professors registry from the external CSV, normalizes names,
    and returns a dictionary mapping normalized names to registry records.
    """
    if not DOCENTES_INVESTIGADORES_PATH.exists():
        raise FileNotFoundError(f"Professors registry not found at {DOCENTES_INVESTIGADORES_PATH}")
    
    df_prof = pd.read_csv(DOCENTES_INVESTIGADORES_PATH)
    registry = {}
    for _, row in df_prof.iterrows():
        raw_name = str(row['Nombre'])
        normalized = normalize_name(raw_name)
        registry[normalized] = {
            "original_name": raw_name,
            "status": str(row['Vinculación']).strip(),  # "Activo" or "Retirado"
            "product_count": int(row['Cantidad de productos']) if not pd.isna(row['Cantidad de productos']) else 0
        }
    return registry

def build_or_load_graph(force_rebuild: bool = False) -> tuple[nx.Graph, dict[str, float]]:
    """
    Loads the co-authorship graph and PageRank scores from cache if they exist,
    otherwise builds them from products metadata and caches the results.
    
    Args:
        force_rebuild (bool): If True, rebuilds the graph even if cache exists.
        
    Returns:
        tuple[nx.Graph, dict[str, float]]: The NetworkX co-authorship graph and PageRank dictionary.
    """
    if not force_rebuild and CACHE_GRAPH_PATH.exists() and CACHE_PAGERANK_PATH.exists():
        try:
            with open(CACHE_GRAPH_PATH, 'rb') as f:
                G = pickle.load(f)
            with open(CACHE_PAGERANK_PATH, 'rb') as f:
                pagerank = pickle.load(f)
            return G, pagerank
        except Exception as e:
            print(f"Error loading graph cache: {e}. Rebuilding...")

    print("Building co-authorship network from scratch...")
    
    # 1. Load registry and metadata
    registry = load_professors_registry()
    df_meta = pd.read_parquet(PRODUCTS_MODELING_PATH)
    
    # 2. Extract transactions (co-authors list per paper)
    transactions = df_meta['active_coauthors'].apply(parse_coauthors)
    
    # 3. Filter transactions to exclude single author papers
    transactions = transactions[transactions.apply(len) > 1]
    
    # 4. Count overall frequencies of authors
    all_authors = [author for trans in transactions for author in trans]
    author_counts = Counter(all_authors)
    
    # Keep authors appearing at least 2 times to prevent single-occurrence noise
    MIN_FREQ = 2
    frequent_authors = {auth for auth, count in author_counts.items() if count >= MIN_FREQ}
    
    # 5. Build edge combinations
    cooccurrence = Counter()
    for trans in transactions:
        filtered_trans = [auth for auth in trans if auth in frequent_authors]
        if len(filtered_trans) > 1:
            for pair in combinations(sorted(filtered_trans), 2):
                cooccurrence[pair] += 1
                
    # 6. Construct NetworkX graph
    G = nx.Graph()
    for (a1, a2), weight in cooccurrence.items():
        G.add_edge(a1, a2, weight=weight)
        
    # Mark nodes that belong to official university professors
    for node in G.nodes:
        if node in registry:
            G.nodes[node]['official'] = True
            G.nodes[node]['original_name'] = registry[node]['original_name']
            G.nodes[node]['status'] = registry[node]['status']
        else:
            G.nodes[node]['official'] = False
            G.nodes[node]['original_name'] = node.title()
            G.nodes[node]['status'] = "Coauthor"
            
    # 7. Compute PageRank
    print("Computing PageRank scores...")
    pagerank = nx.pagerank(G, weight='weight')
    
    # 8. Cache results
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_GRAPH_PATH, 'wb') as f:
        pickle.dump(G, f)
    with open(CACHE_PAGERANK_PATH, 'wb') as f:
        pickle.dump(pagerank, f)
        
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, pagerank

def get_professor_ego_network(G: nx.Graph, professor_name: str, depth: int = 1) -> nx.Graph:
    """
    Extracts the ego network (neighborhood sub-graph) centered on a specific professor.
    
    Args:
        G (nx.Graph): The global co-authorship network.
        professor_name (str): Normalized or raw professor name.
        depth (int): Radius of neighborhood. Default is 1 (direct coauthors).
        
    Returns:
        nx.Graph: Sub-graph of neighbors.
    """
    normalized = normalize_name(professor_name)
    if normalized not in G:
        # Return empty graph if professor is not in the network
        return nx.Graph()
        
    subgraph_nodes = nx.single_source_shortest_path_length(G, normalized, cutoff=depth).keys()
    subgraph = G.subgraph(subgraph_nodes).copy()
    return subgraph
