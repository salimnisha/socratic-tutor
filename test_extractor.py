# test_extractor.py

"""Quick test script for PDF extraction"""

from src.pdf_processor import extract_text_from_pdf

# Extract text from pdf
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
