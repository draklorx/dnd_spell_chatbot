from embeddings.data_classes import ChunkResult
from .db_queries import get_embeddings_for_entry
from .embedder import Embedder

class VectorSearcher:
    def __init__(self, db_path):
        """Initialize the spell searcher."""
        self.embedder = Embedder(db_path)
    
    def search(self, query, entry_name, top_k=5):
        """
        Search for relevant context in entries.
        
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

        # Convert to similarity scores
        similarity_results = [ChunkResult(chunk_text=chunk_text, chunk_context=text, position=position, similarity_score=1 - distance) for text, chunk_text, position, distance in results]
        return similarity_results
    
    def close(self):
        """Close database connection."""
        self.embedder.close()