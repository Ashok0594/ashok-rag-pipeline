from utils.file_reader import extract_text_from_pdf
from chunking.fixed_chunker import tokenize_text, create_chunks
import json
import os

pdf_path = "data/Test.pdf"
output_dir = "outputs/chunks"
os.makedirs(output_dir, exist_ok=True)

pages = extract_text_from_pdf(pdf_path)

for i, page_text in enumerate(pages):
    tokens = tokenize_text(page_text)
    chunks = create_chunks(tokens, chunk_size=512, overlap=128)

    output_path = os.path.join(output_dir, f"page_{i+1}.json")
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Processed page {i+1}, chunks: {len(chunks)}")
