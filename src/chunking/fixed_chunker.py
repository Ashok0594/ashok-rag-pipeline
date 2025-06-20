from typing import List, Dict
import re

def tokenize_text(text: str) -> List[str]:
    # Simple whitespace tokenizer (replace with tiktoken if needed later)
    return text.split()

def create_chunks(
    tokens: List[str],
    chunk_size: int = 200,
    overlap: int = 50
) -> List[Dict]:
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)

        chunks.append({
            "id": chunk_id,
            "start_token": start,
            "end_token": end,
            "text": chunk_text
        })

        chunk_id += 1
        start += chunk_size - overlap  # move forward by stride

    return chunks
