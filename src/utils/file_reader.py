import fitz
from typing import List

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text page by page from a PDF.
    Returns: List of strings, one per page.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages_text.append(text.strip())

    return pages_text
