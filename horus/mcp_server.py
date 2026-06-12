import sys
import builtins

# Redirect all stdout prints to stderr to prevent corrupting the MCP stdio JSON-RPC channel
_original_print = builtins.print
def _stderr_print(*args, **kwargs):
    if kwargs.get('file') is None or kwargs.get('file') is sys.stdout:
        kwargs['file'] = sys.stderr
    _original_print(*args, **kwargs)
builtins.print = _stderr_print

from pathlib import Path
# Add project root directory to sys.path to resolve the 'horus' package
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import collections
import networkx as nx
from mcp.server.fastmcp import FastMCP
from horus.recommender import HorusRecommender
from horus.graph_utils import normalize_name

# Initialize the FastMCP server
mcp = FastMCP("Horus Academic Recommender")

# Global recommender instance (lazy loading handles data loading on first request)
recommender = HorusRecommender()

@mcp.tool()
def recommend_advisors(
    proposal_text: str, 
    top_k_docs: int = 50, 
    weight_semantic: float = 0.5, 
    weight_pagerank: float = 0.3,
    weight_density: float = 0.2
) -> str:
    """
    Recommend official university thesis advisors for a given thesis proposal.
    
    Inputs:
    - proposal_text: String containing the title and abstract/description of the thesis.
    - top_k_docs: Number of candidate publications to retrieve from FAISS (10-100).
    - weight_semantic: Importance weight of semantic relevance (0.0 to 1.0).
    - weight_pagerank: Importance weight of professor authority/PageRank in the co-authorship network (0.0 to 1.0).
    - weight_density: Importance weight of matched papers volume (0.0 to 1.0).
    """
    try:
        recs = recommender.recommend_advisors(
            proposal_text=proposal_text,
            top_k_docs=top_k_docs,
            weight_semantic=weight_semantic,
            weight_pagerank=weight_pagerank,
            weight_density=weight_density,
            active_only=True
        )
        if not recs:
            return "No advisors found matching this query."
            
        result = []
        for i, r in enumerate(recs[:10]):  # Return top 10 recommendations
            line = (
                f"{i+1}. {r['name']} (Score: {r['hybrid_score']:.3f}, Total Products: {r['total_products']})\n"
                f"   - Match details: Semantic similarity: {r['sub_scores']['semantic']:.3f}, PageRank: {r['sub_scores']['pagerank']:.3f}, Density: {r['sub_scores']['density']:.3f}\n"
                f"   - Matched papers in dataset: {r['papers_matched_count']}\n"
                f"   - Top Matched Paper: \"{r['matched_papers'][0]['title']}\" (sim: {r['matched_papers'][0]['similarity']:.3f})\n"
            )
            result.append(line)
        return "\n".join(result)
    except Exception as e:
        return f"Error executing advisor recommendation: {str(e)}"

@mcp.tool()
def get_professor_collaborations(professor_name: str) -> str:
    """
    Get the collaboration network (co-authors) of a specific professor.
    
    Inputs:
    - professor_name: The name of the professor to search (partial matches supported by normalization).
    """
    try:
        recommender.lazy_load()
        normalized = normalize_name(professor_name)
        
        # Check direct match or look for closest matches in registry
        matched_norm = None
        if normalized in recommender.G:
            matched_norm = normalized
        else:
            # Simple substring matching as a fallback
            for node in recommender.G.nodes:
                if normalized in node:
                    matched_norm = node
                    break
                    
        if not matched_norm:
            return f"Professor '{professor_name}' not found in the co-authorship network."
            
        neighbors = recommender.G[matched_norm]
        orig_name_center = recommender.G.nodes[matched_norm].get('original_name', matched_norm.title())
        collabs = []
        for neighbor in neighbors:
            edge_data = recommender.G.get_edge_data(matched_norm, neighbor)
            weight = edge_data.get('weight', 1)
            orig_name = recommender.G.nodes[neighbor].get('original_name', neighbor.title())
            status = recommender.G.nodes[neighbor].get('status', 'Unknown')
            collabs.append(f"- {orig_name} ({status}): {weight} shared paper(s)")
            
        return f"Collaborations for {orig_name_center}:\n" + "\n".join(collabs)
    except Exception as e:
        return f"Error loading network: {str(e)}"

@mcp.tool()
def get_collaboration_path(professor_a: str, professor_b: str) -> str:
    """
    Find the shortest path of co-authorship connecting Professor A and Professor B.
    
    Inputs:
    - professor_a: Name of the first professor.
    - professor_b: Name of the second professor.
    """
    try:
        recommender.lazy_load()
        norm_a = normalize_name(professor_a)
        norm_b = normalize_name(professor_b)
        
        # Resolve names
        match_a, match_b = None, None
        for node in recommender.G.nodes:
            if not match_a and norm_a in node:
                match_a = node
            if not match_b and norm_b in node:
                match_b = node
                
        if not match_a:
            return f"Professor '{professor_a}' not found in the network."
        if not match_b:
            return f"Professor '{professor_b}' not found in the network."
            
        try:
            path = nx.shortest_path(recommender.G, source=match_a, target=match_b)
            path_names = [recommender.G.nodes[node].get('original_name', node.title()) for node in path]
            return " Shortest co-authorship path:\n" + " -> ".join(path_names)
        except nx.NetworkXNoPath:
            name_a = recommender.G.nodes[match_a].get('original_name', match_a.title())
            name_b = recommender.G.nodes[match_b].get('original_name', match_b.title())
            return f"No co-authorship path exists between {name_a} and {name_b}."
    except Exception as e:
        return f"Error tracing path: {str(e)}"

@mcp.tool()
def get_leiden_community(community_id: int) -> str:
    """
    Retrieve details about a specific Leiden academic community (publications, faculty distribution, researchers).
    
    Inputs:
    - community_id: Integer identifying the Leiden community.
    """
    try:
        recommender.lazy_load()
        df_meta = recommender.df_meta
        comm_docs = df_meta[df_meta['community_id'] == community_id]
        if comm_docs.empty:
            return f"Leiden Community {community_id} not found or contains no documents."
            
        doc_count = len(comm_docs)
        faculties = comm_docs['faculty'].value_counts()
        fac_dist = ", ".join([f"{fac}: {count}" for fac, count in faculties.items()])
        
        # Sample up to 5 titles
        samples = comm_docs['original_title'].dropna().sample(n=min(5, doc_count), random_state=42).tolist()
        sample_titles = "\n".join([f"- {title}" for title in samples])
        
        # Top co-authors in this community
        all_coauthors = []
        for coauths in comm_docs['active_coauthors'].dropna():
            all_coauthors.extend(coauths.split(','))
            
        counts = collections.Counter([normalize_name(c) for c in all_coauthors if c.strip()])
        top_profs = []
        for u_name, count in counts.most_common(5):
            if u_name in recommender.registry:
                orig_name = recommender.registry[u_name]["original_name"]
                top_profs.append(f"{orig_name} ({count} publications)")
        top_profs_str = ", ".join(top_profs)
        
        return (
            f"Leiden Community ID: {community_id}\n"
            f"Total Publications: {doc_count}\n"
            f"Faculty Distribution: {fac_dist}\n"
            f"Top Researchers: {top_profs_str}\n\n"
            f"Sample Publications:\n{sample_titles}"
        )
    except Exception as e:
        return f"Error retrieving community details: {str(e)}"

if __name__ == "__main__":
    # Running fastmcp default executes the stdio transport
    mcp.run()
