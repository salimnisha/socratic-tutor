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


# ------------------------------------------------------------------
def create_embedding(text):
    """
    Create a vector embedding for a single text

    Args:
        text (str): Text to create embedding for

    Returns:
        list: Vector with 1536 numbers that represent the meaning of the text

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

    return embedding


# ------------------------------------------------------------------
def create_embeddings_batch(texts, show_progress=True):
    """
    Create embeddings for multiple chunks of text

    Args:
        texts (list): List of text strings to create embeddings
        show_progress (bool): To visually indicate progress by printing it

    Returns:
        list: A list of vector embeddings

    Example:
    chunks = ["text1", "text2", "text3"]
    create_embeddings_batch(chunks)
    Returns [[vec1], [vec2], [vec3]]
    Each vec will be in fhe form of [0.321, 0.932, ..., 0.412]
    """

    embeddings = []

    for i, text in enumerate(texts):
        # print every 10th item and last item
        if show_progress and (i % 10 == 0 or i == len(texts) - 1):
            print(f"Processing {i + 1}/{len(texts)} texts")

        embedding = create_embedding(text)
        embeddings.append(embedding)

    return embeddings
