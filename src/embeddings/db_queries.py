def insert_entry(conn, name):
    cursor = conn.execute('''
        INSERT INTO entries (name)
        VALUES (?)
    ''', (name,)
    )
    return cursor.lastrowid

def insert_sentence(conn, entry_id, sentence, sentence_id):
    cursor = conn.execute('''
        INSERT INTO sentences (entry_id, sentence_text, sentence_order)
        VALUES (?, ?, ?)
    ''', (entry_id, sentence, sentence_id))
    return cursor.lastrowid

def insert_chunk(conn, sentence_id, chunk_text, chunk_id):
    cursor = conn.execute('''
        INSERT INTO chunks (sentence_id, chunk_text, chunk_order)
        VALUES (?, ?, ?)
    ''', (sentence_id, chunk_text, chunk_id))
    return cursor.lastrowid

def insert_embedding(conn, chunk_id, embedding):
    conn.execute('''
        INSERT INTO embeddings (chunk_id, embedding)
        VALUES (?, ?)
    ''', (chunk_id, embedding.tobytes()))

def get_all_entries(conn):
    return [row[0] for row in conn.execute('SELECT name FROM entries').fetchall()]

def get_sentences_ordered(conn, entry_name):
    return conn.execute('''
        SELECT sentence_text, sentence_order
        FROM sentences s
        JOIN entries e ON s.entry_id = e.id
        WHERE e.name = ?
        ORDER BY sentence_order
    ''', (entry_name,)).fetchall()

def get_embeddings_for_entry(conn, query_embedding, entry_name, top_k):
    # Create query embedding
    return conn.execute('''
        SELECT s.sentence_text, c.chunk_text, s.sentence_order,
            vec_distance_cosine(em.embedding, ?) as distance
        FROM sentences s
        JOIN chunks c ON s.id = c.sentence_id
        JOIN embeddings em ON c.id = em.chunk_id
        JOIN entries e ON s.entry_id = e.id
        WHERE e.name = ?
        ORDER BY distance ASC
        LIMIT ?
    ''', (query_embedding.tobytes(), entry_name, top_k)).fetchall()