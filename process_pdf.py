# process_pdf.py

"""
Complete PDF processing pipeline

This script:
1. Extracts text from PDF
2. Creates chunks from the text
3. Creates embeddings for chunks
4. Saves chunks and embeddings to disk
"""

from src.pdf_processor import (
    extract_text_from_pdf,
    chunk_text_by_chars,
    chunk_text_by_tokens,
)
from src.embeddings import create_embeddings_batch
from src.topic_extractor import extract_topics
from src.vector_store import VectorStore

import time
from logger_utils import log_data, create_new_run_id, PROCESS_PDF_COLUMNS


# ---------------------------------------------------------
def process_pdf(pdf_path, pdf_name):
    """
    Process a PDF through the complete pipeline

    Args:
        pdf_path (str): path where the pdf resides (e.g., data/Ch_1_Chip_Huyen.pdf)
        pdf_name (str): name of the pdf (e.g., Ch_1_Chip_Huyen)
    """

    # Start the clock and create run id (for logging)
    start_time = time.time()
    run_id = create_new_run_id()

    print("=" * 60)
    print(f"Processing {pdf_name}")
    print("=" * 60)

    # Step 1: Extract text from PDF
    print("\n[1/6] Extracting text from {pdf} ...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(text)} chars")

    # Step 2: Chunk the extracted text (by characters)
    # Comment this step out if using chunk_text_by_tokens()
    # print("\n[2/6] Chunking text by characters ...")
    # chunks, processing_metadata = chunk_text_by_chars(text, chunk_size=250, overlap=50, return_metadata=True)
    # print(f"✓ Created {len(chunks)} chunks")

    # Step 2: Chunk the extracted text (by tokens)
    # Comment this step out if using chunk_text_by_chars()
    print("\n[2/6] Chunking text by tokens ...")
    chunks, processing_metadata = chunk_text_by_tokens(
        text,
        chunk_size=250,
        overlap=50,
        model="text-embedding-3-small",
        return_metadata=True,
    )
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Create embeddings for the chunks
    print("\n[3/6] Creating embeddings ...")
    print("⚠️  This will cost approximately ${:.4f}".format(len(chunks) * 0.00002))
    embeddings, embeddings_metadata = create_embeddings_batch(
        chunks, show_progress=True, return_metadata=True
    )
    print(f"✓ Created {len(embeddings)} embeddings")

    # Step 4: Save to disk
    print("\n[4/6] Saving chunks and embeddings to vector store ...")
    store = VectorStore()
    store.save(pdf_name, chunks, embeddings)

    # Step 5: Create topic map of the pdf from extracted text
    print(f"\n[5/6] Extracting topics from {pdf_name}...")
    topic_map = extract_topics(text, pdf_name)

    # Step 6: Save topic map to vector store
    print("\n[6/6] Saving topic map to vector store...")
    store.save_topics(pdf_name, topic_map)

    print(f"\n{'=' * 60}")
    print("✓ Processing complete!")
    print("=" * 60)

    # Log data to csv and json log files
    print("\n🗒️ Logging data ...")
    duration = round(time.time() - start_time, 2)
    log_entry = {
        "run_id": run_id,
        "stage": "process_pdf",
        "time_taken_sec": duration,
        "pdf_name": pdf_name,
        "notes": "",
    }
    log_entry.update(processing_metadata)
    log_entry.update(embeddings_metadata)

    log_data(log_entry, PROCESS_PDF_COLUMNS)

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
