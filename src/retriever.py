# src/retriever.py

"""
Retrieval module: Find relevant chunks using semantic search

This module:
1. Converts a query into an embedding
2. Computes simiarity between query and embeddings
3. Returns the top-k most relevant chunks
"""

import numpy as np
from src.embeddings import create_embedding
from src.vector_store import VectorStore


# ------------------------------------------------------
def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity of two vectors
    Formula: cos(θ) = (A.B) / (|A| x |B|)

    Args:
        vec1 (list or np.array): First vector
        vec2 (list or np.array): Second vector

    Returns:
        float: Similarity score between 0 and 1
        1.0 - identical; 0.0 - completely different

    Example:
    sim = cosinesimilarity([05, 0.3], [0.4, 0.2])
    returns: 0.987 (which means the vectors are very similar)
    """

    # Convert the vectors to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Calculate dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return float(similarity)


# ------------------------------------------------------
def retrieve_relevant_chunks(query, pdf_name, top_k, return_metadata=False):
    """
    Find the most relavant chunks for a query

    Process:
    1. Load chunks and embeddings from storage
    2. Convert query to embedding
    3. Compare similarity between query and all chunks
    4. Return the top_k most similar chunks

    Args:
        query (str): The question asked by user
        pdf_name (str): Which pdf to search (e.g., "chip_huyen_ch_1")
        top_k (int): The number of top similar chunks to return
        return_metadata (bool): If true, return metadata dictionary for logging

    Returns:
        list (if return_metadata is false): List of tuples - (chunk, similarity)
        Sorted by the highest similarity score
        list, dict (if return_metadata is true): List of tuples, dictionary of metadata


    Example:
        results = retrieve_relevant_chunks("What is self-supervision?", "chip_huyen_ch_1", top_k=3)
        Returns:
        [("Self-supervision is a lear...", 0.89),
        ("Language models use self-super....", 0.80),
        ("Training without labels...", 0.65)]
    """

    # Load chunks and embeddings from storage
    store = VectorStore()
    chunks, embeddings = store.load(pdf_name)

    if chunks is None:
        raise ValueError(f"No data found for {pdf_name}. Run process_pdf.py first.")

    # Create embedding for the query
    print(f"🔍 Searching for '{query}'")

    query_embedding = []
    metadata = {}
    if return_metadata:
        query_embedding, metadata = create_embedding(query, return_metadata)
    else:
        query_embedding = create_embedding(query)

    # Calculate cosine similarity between query and embeddings
    similarities = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))

    # Sort by highest similarity (similarity score is second value of each tuple)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Find top_k results
    top_results = similarities[:top_k]

    # Now collect the chunks corresponding to the similarities (we can use the index)
    results = []
    for idx, score in top_results:
        results.append((chunks[idx], score))

    print(f"✓ Found {len(results)} relevant chunks")

    if return_metadata:
        return results, metadata
    else:
        return results
