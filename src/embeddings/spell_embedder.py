import json
import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import re
from rapidfuzz import fuzz, process
from pathlib import Path
import math

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SpellEmbedder:
    def __init__(self, db_path="spells.db", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the spell embedder with a database and embedding model.
        
        Args:
            db_path: Path to SQLite database
            model_name: Sentence transformer model name
        """
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.setup_database()
    
    def setup_database(self):
        """Set up the SQLite database with sqlite-vec extension."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS spells (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                level INTEGER,
                school TEXT,
                classes TEXT,
                full_description TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS spell_sentences (
                id INTEGER PRIMARY KEY,
                spell_id INTEGER,
                sentence_text TEXT NOT NULL,
                sentence_order INTEGER NOT NULL,
                source_field TEXT NOT NULL,
                FOREIGN KEY (spell_id) REFERENCES spells (id)
            )
        ''')
        
        # Add a chunks table to handle multiple embeddings per sentence
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS sentence_chunks (
                id INTEGER PRIMARY KEY,
                sentence_id INTEGER,
                chunk_text TEXT NOT NULL,
                chunk_order INTEGER NOT NULL,
                FOREIGN KEY (sentence_id) REFERENCES spell_sentences (id)
            )
        ''')
        
        # Create vector table for chunk embeddings
        self.conn.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS sentence_embeddings USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]
            )
        ''')
        
        # Create index on spell name for fast lookup
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_spell_name ON spells (name)')
        
        self.conn.commit()
    
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
    
    def process_spells(self, spells_data):
        """Process spells data and create embeddings."""
        spell_names = []
        
        for spell in spells_data['spells']:
            # Insert spell metadata
            cursor = self.conn.execute('''
                INSERT INTO spells (name, level, school, classes, full_description)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                spell['name'],
                spell.get('level', 0),
                spell.get('school', ''),
                json.dumps(spell.get('classes', [])),
                spell.get('description', '')
            ))
            
            spell_id = cursor.lastrowid
            spell_names.append(spell['name'])
            
            # Process description
            if 'description' in spell:
                self._process_text_field(spell_id, spell['description'], 'description')
            
            # Process higherLevelSlot if present
            if 'higherLevelSlot' in spell:
                self._process_text_field(spell_id, spell['higherLevelSlot'], 'higherLevelSlot')
            
            # Process cantripUpgrade if present
            if 'cantripUpgrade' in spell:
                self._process_text_field(spell_id, spell['cantripUpgrade'], 'cantripUpgrade')
        
        self.conn.commit()
        return spell_names
    
    def _process_text_field(self, spell_id, text, field_name):
        """Process a text field for a spell with chunking."""
        sentences = self.clean_and_split_text(text)

        for sentence_idx, sentence in enumerate(sentences):
            # Insert sentence
            cursor = self.conn.execute('''
                INSERT INTO spell_sentences (spell_id, sentence_text, sentence_order, source_field)
                VALUES (?, ?, ?, ?)
            ''', (spell_id, sentence, sentence_idx, field_name))
            
            sentence_id = cursor.lastrowid
            
            # Create chunks and embeddings
            chunks = self._chunk_sentence(sentence, sentence_idx)

            for chunk_idx, (chunk_text, original_sentence_idx, chunk_order) in enumerate(chunks):
                # Insert chunk
                chunk_cursor = self.conn.execute('''
                    INSERT INTO sentence_chunks (sentence_id, chunk_text, chunk_order)
                    VALUES (?, ?, ?)
                ''', (sentence_id, chunk_text, chunk_idx))
                
                chunk_id = chunk_cursor.lastrowid
                
                # Create embedding for this chunk
                embedding = self.model.encode(chunk_text)
                
                # Insert embedding
                self.conn.execute('''
                    INSERT INTO sentence_embeddings (chunk_id, embedding)
                    VALUES (?, ?)
                ''', (chunk_id, embedding.tobytes()))
    
    def search_spells(self, query, spell_name=None, top_k=3):
        """
        Search for relevant sentences in spells.
        
        Args:
            query: User query
            spell_name: Optional specific spell name
            top_k: Number of top results to return
        
        Returns:
            List of tuples (sentence_text, source_field, sentence_order, similarity_score)
        """
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # If no specific spell provided, try to find it using fuzzy matching
        if not spell_name:
            spell_names = self.get_all_spell_names()
            match = process.extractOne(query, spell_names, scorer=fuzz.partial_ratio)
            if match and match[1] > 60:  # Threshold for fuzzy matching
                spell_name = match[0]
                print(f"Found spell match: {spell_name} (confidence: {match[1]}%)")
        
        # Build query
        if spell_name:
            # Search within specific spell
            sql_query = '''
                SELECT ss.sentence_text, sc.chunk_text, ss.source_field, ss.sentence_order,
                    vec_distance_cosine(se.embedding, ?) as distance
                FROM spell_sentences ss
                JOIN sentence_chunks sc ON ss.id = sc.sentence_id
                JOIN sentence_embeddings se ON sc.id = se.chunk_id
                JOIN spells s ON ss.spell_id = s.id
                WHERE s.name = ?
                ORDER BY distance ASC
                LIMIT ?
            '''
            params = (query_embedding.tobytes(), spell_name, top_k)
            print(f"DEBUG: Searching for top {top_k} results in spell '{spell_name}'")
        else:
            # Search across all spells
            sql_query = '''
                SELECT ss.sentence_text, ss.source_field, ss.sentence_order,
                    vec_distance_cosine(se.embedding, ?) as distance,
                    s.name
                FROM spell_sentences ss
                JOIN sentence_chunks sc ON ss.id = sc.sentence_id
                JOIN sentence_embeddings se ON sc.id = se.chunk_id
                JOIN spells s ON ss.spell_id = s.id
                ORDER BY distance ASC
                LIMIT ?
            '''
            params = (query_embedding.tobytes(), top_k)
            print(f"DEBUG: Searching for top {top_k} results across all spells")
        
        results = self.conn.execute(sql_query, params).fetchall()
        print(f"DEBUG: Query returned {len(results)} results")
        
        if spell_name:
            print(f"Raw results (distance order) for spell '{spell_name}': {query}")
            for i, result in enumerate(results):
                text, chunk_text, field, order, distance = result
                similarity = 1 - distance
                print(f"{i}. [distance: {distance:.3f}, similarity: {similarity:.3f}] {chunk_text}")
            
            # Convert to similarity scores
            similarity_results = [(text, field, order, 1 - distance) for text, chunk_text, field, order, distance in results]
            return similarity_results
        else:
            print(f"Raw results (distance order) across all spells: {query}")
            for i, result in enumerate(results):
                text, field, order, distance, name = result
                similarity = 1 - distance
                print(f"{i}. [{name}] [distance: {distance:.3f}, similarity: {similarity:.3f}] {text}")
            
            return [(text, field, order, 1 - distance, name) for text, field, order, distance, name in results]
    
    def get_spell_sentences_ordered(self, spell_name):
        """Get all sentences for a spell in order."""
        return self.conn.execute('''
            SELECT sentence_text, source_field, sentence_order
            FROM spell_sentences ss
            JOIN spells s ON ss.spell_id = s.id
            WHERE s.name = ?
            ORDER BY source_field, sentence_order
        ''', (spell_name,)).fetchall()
    
    def get_all_spell_names(self):
        """Get all spell names for fuzzy matching."""
        return [row[0] for row in self.conn.execute('SELECT name FROM spells').fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def load_and_process_spells():
    """Load spells from JSON and process them."""
    parent_dir = Path(__file__).parent.parent
    with open(parent_dir / 'dnd_spell_chatbot' / 'data' / 'spells.json', 'r', encoding='utf-8') as f:
        spells_data = json.load(f)
    
    embedder = SpellEmbedder()
    spell_names = embedder.process_spells(spells_data)
    
    print(f"Processed {len(spell_names)} spells")
    print("Sample spell names:", spell_names[:5])
    
    return embedder

if __name__ == "__main__":
    # Load and process the spells
    embedder = load_and_process_spells()
    
    # Test the search functionality
    query = "How many darts does magic missile fire?"
    print(f"\nQuery: {query}")
    
    results = embedder.search_spells(query, top_k=3)
    
    print("\nTop results:")
    for i, result in enumerate(results):
        if len(result) == 5:  # Cross-spell search
            text, field, order, score, spell_name = result
            print(f"{i+1}. [{spell_name}] {text} (score: {score:.3f})")
        else:  # Single spell search
            text, field, order, score = result
            print(f"{i+1}. {text} (score: {score:.3f})")
    
    embedder.close()