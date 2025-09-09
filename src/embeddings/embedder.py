from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re
import math
from .db_setup import connect, setup
from .db_queries import (
    insert_entry, insert_sentence, insert_chunk, insert_embedding
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    
    def clean_and_split_text(self, text):
        """Clean text and split into sentences."""
        if not text:
            return []
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'#+\s*', '', text)             # Headers
        
        # Split into sentences
        sentences = sent_tokenize(text)

        return [sentence.strip() for sentence in sentences]

    def _chunk_sentence(self, sentence, sentence_idx, target_chunk_size=10):
        """
        Break a sentence into smaller chunks while preserving meaning.
        
        Args:
            sentence: The sentence to chunk
            sentence_idx: Original sentence index
            target_words: Target number of words per chunk
            max_words: Maximum words before forcing a split
        
        Returns:
            List of tuples (chunk_text, original_sentence_idx, chunk_idx)
        """
        if target_chunk_size <= 4:
            raise ValueError("Target chunk size must be greater than 4.")
        
        words = sentence.split()
        min_overlap = math.ceil(target_chunk_size * 0.15)

        # If sentence is short enough, return as single chunk
        if len(words) <= target_chunk_size:
            return [(sentence, sentence_idx, 0)]

        # Determine number of chunks that still meets minimum overlap requirements
        num_chunks = math.ceil((len(words) - min_overlap) / (target_chunk_size - min_overlap))

        chunks = []
        # Evenly distribute chunks over the words
        for chunk_idx in range(num_chunks):
            if chunk_idx == 0:
                chunk_text = ' '.join(words[0:target_chunk_size])
                chunks.append((chunk_text, sentence_idx, chunk_idx))
            elif chunk_idx == num_chunks - 1:
                chunk_text = ' '.join(words[-target_chunk_size:])
                chunks.append((chunk_text, sentence_idx, chunk_idx))
            else:
                chunk_center = (len(words) // (num_chunks -1)) * (chunk_idx)
                chunk_text = ' '.join(words[chunk_center - target_chunk_size//2 : chunk_center + target_chunk_size//2])
                chunks.append((chunk_text, sentence_idx, chunk_idx))

        return chunks
    
    def process_entries(self, entries):
        """Process entry data and create embeddings."""

        # Setup database tables
        print("Setting up database...")
        setup(self.conn, self.embedding_dim)

        print ("Processing entries and creating embeddings...")
        for i, entry in enumerate(entries):
            # Insert spell metadata
            if (i % 10) == 0:
                percent_done = math.floor((i+1) / len(entries) * 100)
                print(f"Processing entries {percent_done}% complete...")
            entry_id = insert_entry(self.conn, entry["name"])
            
            # Process entry text
            if 'text' in entry:
                self._process_text(entry_id, entry['text'])
        
        print("Processing complete.")
        self.conn.commit()

    def _process_text(self, entry_id, text):
        """Process a text for an entry with chunking."""
        sentences = self.clean_and_split_text(text)

        for sentence_order, sentence in enumerate(sentences):
            # Insert sentence
            sentence_id = insert_sentence(self.conn, entry_id, sentence, sentence_order)

            # Create chunks and embeddings
            chunks = self._chunk_sentence(sentence, sentence_id)

            for chunk_order, (chunk_text, original_sentence_id, chunk_order) in enumerate(chunks):
                # Insert chunk
                chunk_id = insert_chunk(self.conn, sentence_id, chunk_text, chunk_order)
                
                # Create embedding for this chunk
                embedding = self.model.encode(chunk_text)
                
                # Insert embedding
                insert_embedding(self.conn, chunk_id, embedding)

    def close(self):
        """Close database connection."""
        self.conn.close()