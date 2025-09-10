import sqlite3
import sqlite_vec

def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def setup(conn, embedding_dim):
    # Remove old tables if they exist
    conn.execute('DROP INDEX IF EXISTS idx_entry_name')
    conn.execute('DROP TABLE IF EXISTS entries')
    conn.execute('DROP TABLE IF EXISTS chunk_context')
    conn.execute('DROP TABLE IF EXISTS chunks')
    conn.execute('DROP TABLE IF EXISTS embeddings')

    # Table creation
    # An entry represents a logical grouping within the full text
    # In a book this could be a chapter or a section
    # In a list of people, places, or things, it could be the name of the person, place, or thing
    conn.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')

    # Chunk context represents more complete context around individual chunks
    # This could be the sentence or paragraph that the chunk is part of
    # Its order is maintained within the entry for further context
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunk_context (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            text TEXT NOT NULL,
            position INTEGER NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries (id)
        )
    ''')

    # A chunk is a small overlapping part of the entry's text
    # It overlaps with other chunks to preserve context
    # This might be broken up by sentence, a fixed number of words, etc.
    # We keep the text in case we want to provide very specific results
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_context_id INTEGER,
            text TEXT NOT NULL,
            FOREIGN KEY (chunk_context_id) REFERENCES chunk_context (id)
        )
    ''')

    # Embeddings table using sqlite_vec
    # It represents a vector applying meaning to a chunk of text
    # We can use this to find similar chunks of text
    # Each chunk has one embedding
    conn.execute(f'''
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        )
    ''')

    conn.execute('CREATE INDEX IF NOT EXISTS idx_entry_name ON entries (name)')
    conn.commit()