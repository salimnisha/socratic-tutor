# test_extractor.py

"""Quick test script for PDF extraction"""

from src.pdf_processor import extract_text_from_pdf
from src.pdf_processor import chunk_text
from src.embeddings import create_embedding, create_embeddings_batch

# ----------------------------------------------------------
""" Test extract text from pdf """
pdf_path = "data/Ch 01 Chip Huyen AI Engineering - 025-072.pdf"
print(f"Extracting text from: {pdf_path}")

text = extract_text_from_pdf(pdf_path)

# Show results
print(f"\n✓ Successfully extracted {len(text)} characters")
print("\n✓ First 500 characters")
print(text[:500])
print("\n...")
print("\n✓ Last 500 characters")
print(text[-500:])

# ----------------------------------------------------------
""" Now test chunking """
print("\n" + "=" * 50)
print("TESTING CHUNKING")
print("=" * 50)

chunks = chunk_text(text, chunk_size=1000, overlap=100)

# Show chunk details
print(f"\n✓ Create {len(chunks)} chunks")
print(
    f"\n✓ Avg. chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters"
)

# Show contents of first 3 chunks
print("\n--- First 3 chunks ---")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\nChunk{i}: {len(chunk)} characters")
    print(f"{chunk[:]}...")

# ----------------------------------------------------------
""" Test embeddings creation """
print("\n" + "=" * 50)
print("TESTING EMBEDDINGS")
print("=" * 50)

# Test just one embedding
print("\nTest a single embedding")
test_text = "What exactly is a token?"
embedding = create_embedding(test_text)
print(f"✓ Created embedding of {len(embedding)} dimensions")
print(f"First 10 values: {embedding[:10]}")

# Test batch processing of embeddings with 3 chunks
print("\n\nTesting batch processing embeddings with 3 chunks")
sample_chunks = chunks[:3]
sample_embeddings = create_embeddings_batch(sample_chunks)
print(f"✓ Created {len(sample_embeddings)} embeddings")
print(f"✓ Each embedding has {len(sample_embeddings[0])} dimensions")
