# src/pdf_processor.py

"""
PDF Processor: Extracts and chunks text from pdf files

This module handles:
1. Reading pdf files
2. Extracting text from each page
3. Cleaning the extracted text
4. Splitting text into manageable chunks
"""

from PyPDF2 import PdfReader
from pathlib import Path


# --------------------------------------------------
# Function: Extract text from all pages of a pdf file
# --------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a pdf file

    Args:
        pdf_path (str or Path): Path to the pdf file

    Returns:
        str: All text from the pdf concatenated together

    Example:
        text = extract_text_from_pdf("data/textbook.pdf")
    """

    # Convert str path to Path object (if needed)
    pdf_path = Path(pdf_path)

    # Check if file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open and read the pdf
    reader = PdfReader(str(pdf_path))

    # Extract text from each page
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page_text

    return text


# --------------------------------------------------
# Function: Split text into overlapping chunks
# --------------------------------------------------
def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Split text into overlapping chunks

    Strategy:
    1. Split text on newline (not paragraph, because paragraph boundary marker inconsistencies across pdfs)
    2. Combine lines to create chunks that reach chunk_size
    3. Add overlap between chunks for continuity between chunks

    Args:
    text (str): The text to create chunks
    chunk_size: Target size of each chunk (characters)
    overlap: Overlap characters to add to beginning of each chunk to retain context

    Returns:
    list: List of text chunks with overlap
    """

    # First, replace the page number strings (which we added during extraction) with Page Markers
    # Why do this in 2 steps rather than add PAGE_MARKER directly to extract function?
    # For separation of concerns, more modular code
    text = text.replace("--- Page", "PAGE_MARKER").replace("---", "")

    # Split the text into lines (newline)
    lines = text.split("\n")

    # Remove empty lines and whitespaces
    lines = [line.strip() for line in lines if line.strip()]

    chunks = []
    current_chunk = ""
    # Loop through lines to create chunks of target chunk_size and with overlap
    for line in lines:
        # Check if adding line to current_chunk will be within target chunk_size
        if len(current_chunk) + len(line) < chunk_size:
            # Add line to current chunk
            current_chunk += "\n" + line
        else:
            # Append to chunks list, then start new chunk with previous line overlap and line
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] + "\n" + line

    # Also add the final chunk to the list
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
