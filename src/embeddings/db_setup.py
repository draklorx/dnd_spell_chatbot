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
    conn.execute('DROP TABLE IF EXISTS sentences')
    conn.execute('DROP TABLE IF EXISTS chunks')
    conn.execute('DROP TABLE IF EXISTS embeddings')

    # Table creation
    conn.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            sentence_text TEXT NOT NULL,
            sentence_order INTEGER NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries (id)
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            sentence_id INTEGER,
            chunk_text TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            FOREIGN KEY (sentence_id) REFERENCES sentences (id)
        )
    ''')

    conn.execute(f'''
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        )
    ''')

    conn.execute('CREATE INDEX IF NOT EXISTS idx_entry_name ON entries (name)')
    conn.commit()