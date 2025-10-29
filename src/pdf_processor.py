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
import tiktoken
from statistics import mean


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
def chunk_text_by_chars(text, chunk_size=1000, overlap=100, return_metadata=False):
    """
    Split text into overlapping chunks based on character count
    Note: function below (chunk_text_by_tokens) makes chunks based on token count

    Strategy:
    1. Split text on newline (not paragraph, because paragraph boundary marker inconsistencies across pdfs)
    2. Combine lines to create chunks that reach chunk_size
    3. Add overlap between chunks for continuity between chunks

    Args:
    text (str): The text to create chunks
    chunk_size (int): Target size of each chunk (characters)
    overlap (int): Overlap characters to add to beginning of each chunk to retain context
    return_metadata (dict): Data for logfile

    Returns:
    list: List of text chunks with overlap (if return_metadata is False)
    list, dict: List of text chunks and dict for logging (if return_metadata is True)
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

    # Avg chunk size created
    avg_chunk_size_created = mean(len(chunk) for chunk in chunks)

    # Construct metadata for logging
    metadata = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunking_method": "characters",
        "num_chunks": len(chunks),
        "avg_chunk_size_created": round(avg_chunk_size_created, 2),
    }

    if return_metadata:
        return chunks, metadata
    else:
        return chunks


# --------------------------------------------------
# Function: Split text into overlapping tokens
# --------------------------------------------------
def chunk_text_by_tokens(
    text,
    chunk_size=500,
    overlap=50,
    model="text-embedding-3-small",
    return_metadata=False,
):
    """
    Create overlapping chunks based on token count (not character count)
    Note: Function above (create_chunks) creates chunks based on character count

    Args:
        text::(str)
            Text to split into chunks
        chunk_size::(int)
            Target number of tokens per chunk
        overlap::(int)
            Number of tokens to overlap between consecutive chunks
        model::(str)
            Model name to get correct tokenizer
        return_metadata::(bool)
            Return metadata for logfile or not

    Returns:
        list: list of chunks (text strings) with overlap (if return_metadata=False)
        list, dict: list of chunks, dict of metadata for logging (if return_metadata=True)
    """

    # Replace the page number strings with PAGEMARKER
    text = text.replace("--- Page", "PAGEMARKER").replace("---", "")

    # Split the text into lines
    lines = text.split("\n")

    # Remove empty lines and whitespaces
    lines = [line for line in lines if line.strip()]

    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model(model)

    chunks = []
    current_tokens = []

    # loop through lines and chunk based on chunk_size and overlap
    for line in lines:
        # Encode the line into tokens first
        line_tokens = encoding.encode(line)

        # Check if current_tokens + new tokens stay within chunk size limit
        if len(current_tokens) + len(line_tokens) < chunk_size:
            current_tokens.extend(line_tokens)
        else:
            # Decode current_tokens and to the chunks list
            chunk_text = encoding.decode(current_tokens)
            chunks.append(chunk_text)
            # Start a new current_token with an overlap from the previous + current encoded line
            current_tokens = current_tokens[-overlap:] + line_tokens

    # Add the final chunk
    chunk_text = encoding.decode(current_tokens)
    chunks.append(chunk_text)

    # Avg chunk size created
    avg_chunk_size_created = mean(len(encoding.encode(chunk)) for chunk in chunks)

    # Construct metadata for logging
    metadata = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunking_method": "tokens",
        "num_chunks": len(chunks),
        "avg_chunk_size_created": round(avg_chunk_size_created, 2),
    }

    if return_metadata:
        return chunks, metadata
    else:
        return chunks
