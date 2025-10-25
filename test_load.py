# test_load.py

from src.vector_store import VectorStore

store = VectorStore()
chunks, embeddings = store.load("chip_huyen_ch_1")

if chunks:
    print(f"✓ Loaded {len(chunks)} chunks")
    print(f"✓ Loaded {len(embeddings)} embeddings")
    print("\nPreview first chunk:")
    print(chunks[0][:200] + "...")
else:
    print("x No data found")
