from sentence_transformers import SentenceTransformer
import math

from embeddings.data_classes import ChunkedEntry
from .db_setup import connect, setup
from .db_queries import (
    insert_entry, insert_chunk_context, insert_chunk, insert_embedding
)


class Embedder:
    def __init__(self, db_path, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a database and embedding model.
        
        Args:
            db_path: Path to SQLite database
            model_name: Sentence transformer model name
        """
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.conn = connect(self.db_path)
    
    def process_entries(self, chunked_entries: list[ChunkedEntry]):
        """Process entry data and create embeddings."""

        # Setup database tables
        print("Setting up database...")
        setup(self.conn, self.embedding_dim)

        print ("Processing entries and creating embeddings...")
        for i, entry in enumerate(chunked_entries):
            # Insert spell metadata
            if (i % 10) == 0:
                percent_done = math.floor((i+1) / len(chunked_entries) * 100)
                print(f"Processing entries {percent_done}% complete...")
            entry_id = insert_entry(self.conn, entry.name)
            for chunk_context in entry.chunk_contexts:
                # Insert chunk context
                chunk_context_id = insert_chunk_context(self.conn, entry_id, chunk_context.text, chunk_context.position)
                for chunk in chunk_context.chunks:
                    # Insert chunk
                    chunk_id = insert_chunk(self.conn, chunk_context_id, chunk.text)
                    # Create and insert embedding
                    embedding = self.model.encode(chunk.text)
                    insert_embedding(self.conn, chunk_id, embedding)
        
        print("Processing complete.")
        self.conn.commit()
        self.close()

    def close(self):
        """Close database connection."""
        self.conn.close()