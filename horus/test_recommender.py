import unittest
import numpy as np
import networkx as nx
from horus.graph_utils import load_professors_registry, build_or_load_graph, get_professor_ego_network
from horus.recommender import HorusRecommender

class TestHorusBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = HorusRecommender()
        # Trigger lazy load once for all tests to speed up
        cls.recommender.lazy_load()

    def test_registry_loading(self):
        """Test that the professor registry is loaded correctly as a dictionary."""
        registry = self.recommender.registry
        self.assertIsInstance(registry, dict)
        self.assertGreater(len(registry), 0)
        
        # Check a sample entry
        first_key = list(registry.keys())[0]
        self.assertIn("original_name", registry[first_key])
        self.assertIn("status", registry[first_key])
        self.assertIn("product_count", registry[first_key])

    def test_graph_and_pagerank(self):
        """Test that the co-authorship network is constructed and has valid PageRank values."""
        G = self.recommender.G
        pagerank = self.recommender.pagerank
        self.assertIsInstance(G, nx.Graph)
        self.assertIsInstance(pagerank, dict)
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(len(pagerank), 0)
        
        # PageRank values should sum close to 1
        self.assertAlmostEqual(sum(pagerank.values()), 1.0, places=3)
        
        # Test ego-network extraction
        first_node = list(G.nodes())[0]
        ego_G = get_professor_ego_network(G, first_node, depth=1)
        self.assertIsInstance(ego_G, nx.Graph)
        self.assertGreater(ego_G.number_of_nodes(), 0)

    def test_embedder_and_recommender(self):
        """Test that the PRISM query embedder outputs normalized 1024D vectors and the recommender ranks advisors."""
        self.assertIsNotNone(self.recommender.embedder)
        query = "Modelo de optimización matemática para transporte multimodal urbano en Bogotá"
        emb = self.recommender.embedder.get_embedding(query)
        
        self.assertEqual(emb.shape, (1024,))
        # Normalized L2 norm should be close to 1.0
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)
        
        # Test recommendation algorithm output
        recs = self.recommender.recommend_advisors(
            proposal_text=query,
            top_k_docs=10,
            weight_semantic=0.5,
            weight_pagerank=0.3,
            weight_density=0.2,
            active_only=True
        )
        
        self.assertIsInstance(recs, list)
        if recs:
            first_rec = recs[0]
            self.assertIn("name", first_rec)
            self.assertIn("status", first_rec)
            self.assertIn("hybrid_score", first_rec)
            self.assertIn("sub_scores", first_rec)
            self.assertIn("matched_papers", first_rec)
            self.assertEqual(first_rec["status"], "Activo")
            
            # Scores should be sorted descending
            scores = [r["hybrid_score"] for r in recs]
            self.assertEqual(scores, sorted(scores, reverse=True))

if __name__ == "__main__":
    unittest.main()
