import nltk
from typing import List, Dict
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoTokenizer

nltk.download("punkt", quiet=True)

# Initialize tokenizers
sent_tokenizer = PunktSentenceTokenizer()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def semantic_chunk(text: str,  page_num: int, max_tokens: int = 512, overlap: int = 128) -> List[Dict]:
    sentences = sent_tokenizer.tokenize(text)
    chunks = []

    current_chunk = []
    current_tokens = 0
    token_offset = 0
    chunk_id = 1

    def flush_chunk(chunk_data, start, end, cid):
        chunk_text = " ".join(s for s, _ in chunk_data)
        return {"chunk_id": cid, "strategy": "semantic", "page": page_num, "start_token": start, "end_token": end, "text": chunk_text }

    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))

        # If the current sentence fits into current chunk
        if current_tokens + token_count <= max_tokens:
            current_chunk.append((sentence, token_count))
            current_tokens += token_count
            continue

        # If current chunk is empty, handle long sentence as its own chunk
        if not current_chunk:
            chunks.append(flush_chunk([(sentence, token_count)], token_offset, token_offset + token_count, chunk_id))
            chunk_id += 1
            token_offset += token_count
            continue

        # Otherwise, flush current chunk and apply overlap
        chunks.append(flush_chunk(current_chunk, token_offset, token_offset + current_tokens, chunk_id))
        chunk_id += 1
        token_offset += current_tokens - overlap

        # Prepare new chunk with overlap + current sentence
        overlap_chunk = []
        overlap_tokens = 0
        while current_chunk and overlap_tokens < overlap:
            sent, sent_tokens = current_chunk.pop()
            overlap_chunk.insert(0, (sent, sent_tokens))
            overlap_tokens += sent_tokens

        current_chunk = overlap_chunk + [(sentence, token_count)]
        current_tokens = sum(t[1] for t in current_chunk)

    # Final chunk
    if current_chunk:
        chunks.append(flush_chunk(current_chunk, token_offset, token_offset + current_tokens, chunk_id))

    return chunks