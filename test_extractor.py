# test_extractor.py

"""Quick test script for PDF extraction"""

from src.pdf_processor import extract_text_from_pdf
from src.pdf_processor import chunk_text

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
