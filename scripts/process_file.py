from utils.file_reader import extract_text_from_pdf,extract_images_from_pdf
from chunking.fixed_chunker import tokenize_text, create_chunks
from src.chunking.semantic_chunker import semantic_chunk
import json
import os

pdf_path = "data/Test.pdf"
output_dir = "outputs/chunks"
image_dir = "outputs/images"

fixed_dir = os.path.join(output_dir, "fixed")
semantic_dir = os.path.join(output_dir, "semantic")
hierarchical_dir = os.path.join(output_dir, "hierarchical")

os.makedirs(fixed_dir, exist_ok=True)
os.makedirs(semantic_dir, exist_ok=True)
os.makedirs(hierarchical_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

pages = extract_text_from_pdf(pdf_path)
page_images = extract_images_from_pdf(pdf_path, image_dir)

for i, (text, images) in enumerate(zip(pages, page_images)):
    page_num = i + 1

    # ---------- Fixed chunking ----------
    tokens = tokenize_text(text)
    fixed_chunks = create_chunks(tokens, page_num, chunk_size=512, overlap=128)

    # Add image placeholders
    for j, image_name in enumerate(images):
        fixed_chunks.append({ "id": f"img-{j+1}", "type": "image", "page": page_num, "image_file": image_name, "text": f"[IMAGE_PRESENT: Page {page_num}, {image_name}]" })

    with open(os.path.join(fixed_dir, f"page_{page_num}.json"), "w") as f:
        json.dump(fixed_chunks, f, indent=2)

    # ---------- Semantic chunking ----------
    semantic_chunks = semantic_chunk(text, page_num, max_tokens=512, overlap=128)

    # Add image placeholders
    for j, image_name in enumerate(images): 
        semantic_chunks.append({ "id": f"img-{j+1}", "type": "image", "page": page_num, "image_file": image_name, "text": f"[IMAGE_PRESENT: Page {page_num}, {image_name}]" })

    with open(os.path.join(semantic_dir, f"page_{page_num}.json"), "w") as f:
        json.dump(semantic_chunks, f, indent=2)

    print(f"Page {page_num}: {len(fixed_chunks)} fixed chunks, {len(semantic_chunks)} semantic chunks")
