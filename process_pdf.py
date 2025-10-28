# process_pdf.py

"""
Complete PDF processing pipeline

This script:
1. Extracts text from PDF
2. Creates chunks from the text
3. Creates embeddings for chunks
4. Saves chunks and embeddings to disk
"""

from src.pdf_processor import extract_text_from_pdf, chunk_text, chunk_text_by_tokens
from src.embeddings import create_embeddings_batch
from src.vector_store import VectorStore


# ---------------------------------------------------------
def process_pdf(pdf_path, pdf_name):
    """
    Process a PDF through the complete pipeline

    Args:
        pdf_path (str): path where the pdf resides (e.g., data/Ch_1_Chip_Huyen.pdf)
        pdf_name (str): name of the pdf (e.g., Ch_1_Chip_Huyen)
    """

    print("=" * 60)
    print(f"Processing {pdf_name}")
    print("=" * 60)

    # Step 1: Extract text from PDF
    print("\n[1/4] Extracting text from {pdf} ...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(text)} chars")

    # Step 2: Chunk the extracted text (by characters)
    # Comment this step out if using chunk_text_by_tokens()
    # print("\n[2/4] Chunking text by characters ...")
    # chunks = chunk_text(text, chunk_size=250, overlap=50)
    # print(f"✓ Created {len(chunks)} chunks")

    # Step 2: Chunk the extracted text (by tokens)
    # Comment this step out if using chunk_text()
    print("\n[2/4] Chunking text by tokens ...")
    chunks = chunk_text_by_tokens(
        text, chunk_size=150, overlap=40, model="text-embedding-3-small"
    )
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Create embeddings for the chunks
    print("\n[3/4] Creating embeddings ...")
    print("⚠️  This will cost approximately ${:.4f}".format(len(chunks) * 0.00002))
    embeddings = create_embeddings_batch(chunks, show_progress=True)
    print(f"✓ Created {len(embeddings)} embeddings")

    # Step 4: Save to disk
    print("\n[4/4] Saving to vector store ...")
    store = VectorStore()
    store.save(pdf_name, chunks, embeddings)

    print("=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)

    return chunks, embeddings


if __name__ == "__main__":
    # Process the pdf
    pdf_path = "data/chip_huyen_ch_1.pdf"
    pdf_name = "chip_huyen_ch_1"

    chunks, embeddings = process_pdf(pdf_path, pdf_name)

    # Verification
    print("\nVerification:")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Embeddings: {len(embeddings)}")
    print(f"  - Embedding dimesnions: {len(embeddings[0])}")
