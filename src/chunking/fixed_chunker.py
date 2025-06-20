from typing import List, Dict
import re

def tokenize_text(text: str) -> List[str]:
    # Simple whitespace tokenizer (replace with tiktoken if needed later)
    return text.split()

def create_chunks(tokens: List[str],  page_num: int, chunk_size: int = 512, overlap: int = 128) -> List[Dict]:
    chunks = []
    total_tokens = len(tokens)
    chunk_id = 1
    
    def build_chunk(cid: int, start: int, end: int) -> Dict:
        chunk_text = " ".join(tokens[start:end])
        return {"chunk_id": cid, "strategy": "Fixed", "page": page_num, "start_token": start, "end_token": end, "text": chunk_text }
        
    # Case 1: Short document — return as a single chunk
    if total_tokens <= chunk_size:
        chunk_text = " ".join(tokens)
        return [build_chunk(chunk_id, 0, total_tokens)]

    # Case 2: Long document — sliding window chunking
    stride = chunk_size - overlap
    for start in range(0, total_tokens, stride):
        end = min(start + chunk_size, total_tokens)
        chunk_length = end - start
        
        # Skip if trailing chunks is smaller than overlap
        if chunk_length < overlap and start != 0:
            break

        chunks.append(build_chunk(chunk_id, start, end))
        chunk_id += 1

    return chunks