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
