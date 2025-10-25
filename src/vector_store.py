# src/vector_store.py

"""
Vector Store: Save and load embeddings with metadata

This module handles:
1. Saving chunks and their embeddings to disk
2. Loading them back
3. Keep track of which pdf they came from
"""

import json
from pathlib import Path
from datetime import datetime


class VectorStore:
    """
    A simple JSON file-based storage for text chunks and vector embeddings
    In Phase 2, we will replace this with ChromaDB
    """

    def __init__(self, storage_dir="vector_store"):
        """
        Initialize the vector store

        Args:
        storage_dir (str): Path of the storage directory
        """

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    def save(self, pdf_name, chunks, embeddings):
        """
        Stores chunk-embedding pairings to disk

        Args:
            pdf_name (str): The pdf that the text is extracted from (e.g. "Ch_1_Chip_Huyen")
            chunks (list): list of text chunks
            embeddings (list): list of embeddings corresponding to the text chunks
        """

        # First create the data structure that will hold all the chunk-embedding pairs
        data = {
            "pdf_name": pdf_name,
            "created_at": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "chunks": [
                {"id": i, "text": chunk, "embedding": embedding}
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        }

        # Save to JSON file
        file = self.storage_dir / f"{pdf_name}.json"
        with open(file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(chunks)} chunks to {file}")

    # --------------------------------------------------
    def load(self, pdf_name):
        """
        Loads chunks and embeddings from disk

        Args:
            pdf_name (str): Name of the source pdf (e.g. Ch_1_Chip_Huyen)

        Returns:
            tuple: (chunks, embeddings) or (None, None) if not found
        """

        # Open the file
        file = self.storage_dir / f"{pdf_name}.json"
        if not file.exists():
            return None, None

        with open(file, "r") as f:
            data = json.load(f)

        # Extract chunks and embeddings
        chunks = [item["text"] for item in data["chunks"]]
        embeddings = [item["embedding"] for item in data["chunks"]]

        print(f"✓ Loaded {len(chunks)} from {pdf_name}")

        return chunks, embeddings
