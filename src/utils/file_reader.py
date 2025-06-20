import os
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

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[List[str]]:
    """
    Extracts images from each page and saves to output_dir/images/.
    Returns a list of image filenames per page.
    """
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    all_images = []

    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        page_images = []

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{i+1}_img_{img_index+1}.{image_ext}"

            with open(os.path.join(output_dir, image_filename), "wb") as img_file:
                img_file.write(image_bytes)

            page_images.append(image_filename)

        all_images.append(page_images)

    return all_images