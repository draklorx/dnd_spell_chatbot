def insert_entry(conn, name):
    cursor = conn.execute('''
        INSERT INTO entries (name)
        VALUES (?)
    ''', (name,)
    )
    return cursor.lastrowid

def insert_chunk_context(conn, entry_id, text, position):
    cursor = conn.execute('''
        INSERT INTO chunk_context (entry_id, text, position)
        VALUES (?, ?, ?)
    ''', (entry_id, text, position))
    return cursor.lastrowid

def insert_chunk(conn, chunk_context_id, text):
    cursor = conn.execute('''
        INSERT INTO chunks (chunk_context_id, text)
        VALUES (?, ?)
    ''', (chunk_context_id, text))
    return cursor.lastrowid

def insert_embedding(conn, chunk_id, embedding):
    conn.execute('''
        INSERT INTO embeddings (chunk_id, embedding)
        VALUES (?, ?)
    ''', (chunk_id, embedding.tobytes()))

def get_embeddings_for_entry(conn, query_embedding, entry_name, top_k):
    # Create query embedding
    return conn.execute('''
        SELECT cc.text, c.text, cc.position,
            vec_distance_cosine(em.embedding, ?) as distance
        FROM chunk_context cc
        JOIN chunks c ON cc.id = c.chunk_context_id
        JOIN embeddings em ON c.id = em.chunk_id
        JOIN entries e ON cc.entry_id = e.id
        WHERE e.name = ?
        ORDER BY distance ASC
        LIMIT ?
    ''', (query_embedding.tobytes(), entry_name, top_k)).fetchall()