# src/embeddings.py

"""
Embeddings Module: Convert text to semantic vectors using OpenAI

This module hancles:
1. Loading OpenAI credentials
2. Creating a vector embedding for a text chunk
3. Batch processing above for efficiency
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# load env variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
EMBEDDING_COST_PER_1K_TOKENS = 0.00002  # text-embedding-3-small pricing


# ------------------------------------------------------------------
def create_embedding(text, return_metadata=False):
    """
    Create a vector embedding for a single text

    Args:
        text (str): Text to create embedding for
        return_metadata (bool): If True, return embedding, metadata
                                If False, return just embedding

    Returns:
        list: Vector embedding with 1536 numbers that represent the meaning of the text (if metadata is True)
        tuple: tuple of (embedding, metadata dict)

    Example:
    vector = create_embedding("What is a language model?")
    Returns: [0.234, 0.718, ...., 0.085]
    """

    # Replace all newlines with space because the API works better without newlines
    text = text.replace("\n", " ")

    # Call OpenAI API
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)

    # Extract the embedding vector
    embedding = response.data[0].embedding

    # Extract usage info and cost
    tokens_used = response.usage.total_tokens
    cost = (tokens_used / 1000) * EMBEDDING_COST_PER_1K_TOKENS

    # Create metadata
    metadata = {"tokens_used": tokens_used, "cost_usd": cost, "model": EMBEDDING_MODEL}

    if return_metadata:
        return embedding, metadata
    else:
        return embedding


# ------------------------------------------------------------------
def create_embeddings_batch(texts, show_progress=True, return_metadata=False):
    """
    Create embeddings for multiple chunks of text

    Args:
        texts (list): List of text strings to create embeddings
        show_progress (bool): To visually indicate progress by printing it
        return_metadata (bool): Return metadata for logging or not

    Returns:
        list: A list of vector embeddings (if return_metadata=False)
        list, dict: List of embeddings, dict of metadata (if return_metadata=True)


    Example:
    chunks = ["text1", "text2", "text3"]
    create_embeddings_batch(chunks)
    Returns [[vec1], [vec2], [vec3]]
    Each vec will be in fhe form of [0.321, 0.932, ..., 0.412]
    """

    embeddings = []
    embedding_tokens = 0
    embedding_cost = 0.0

    for i, text in enumerate(texts):
        # print every 10th item and last item
        if show_progress and (i % 10 == 0 or i == len(texts) - 1):
            print(f"Processing {i + 1}/{len(texts)} texts")

        embedding, metadata = create_embedding(text, return_metadata=True)
        embeddings.append(embedding)

        embedding_tokens += metadata["tokens_used"]
        embedding_cost += metadata["cost_usd"]

    total_metadata = {
        "num_embeddings": len(embeddings),
        "embedding_tokens": embedding_tokens,
        "embedding_cost_usd": embedding_cost,
        "avg_tokens_per_embedding": round(embedding_tokens / len(embeddings), 2),
        "embedding_model": EMBEDDING_MODEL,
    }

    if return_metadata:
        return embeddings, total_metadata
    else:
        return embeddings
