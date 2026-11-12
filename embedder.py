from sentence_transformers import SentenceTransformer
from typing import List

# Loaded once and reused for every call (loading the model is the slow part)
# all-MiniLM-L6-v2 -> 384-dimension vectors, free, runs fully on your CPU
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embeds a list of text chunks using a local Sentence-Transformers model."""
    embeddings = _model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_User_query(query: str) -> List[float]:
    """Embeds a single user query using the same local model."""
    embedding = _model.encode(query, convert_to_numpy=True)
    return embedding.tolist()
