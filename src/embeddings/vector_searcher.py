from .db_queries import get_embeddings_for_entry
from .embedder import Embedder

class VectorSearcher:
    def __init__(self, db_path="spells.db"):
        """Initialize the spell searcher."""
        self.embedder = Embedder(db_path)
    
    def search(self, query, entry_name, top_k=5):
        """
        Search for relevant sentences in entries.
        
        Args:
            query: User query
            entry_name: Optional specific entry name
            top_k: Number of top results to return
        
        Returns:
            List of tuples (sentence_text, sentence_order, similarity_score)
        """
        if not entry_name:
            raise ValueError("An entry name must be provided for search.")
        
        # Create query embedding
        query_embedding = self.embedder.model.encode(query)
                
        # Search within specific entry
        results = get_embeddings_for_entry(self.embedder.conn, query_embedding, entry_name, top_k)

        # TODO Add debug mode
        # print(f"Raw results (distance order) for entry '{entry_name}': {query}")
        # for i, result in enumerate(results):
        #     text, chunk_text, order, distance = result
        #     similarity = 1 - distance
        #     print(f"{i}. [distance: {distance:.3f}, similarity: {similarity:.3f}] {chunk_text}")
        
        # Convert to similarity scores
        similarity_results = [(text, order, 1 - distance) for text, chunk_text, order, distance in results]
        return similarity_results
    
    def close(self):
        """Close database connection."""
        self.embedder.close()