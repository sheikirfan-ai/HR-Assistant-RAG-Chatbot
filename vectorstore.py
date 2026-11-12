import chromadb
from typing import List

# Persistent local vector store -> data is saved to ./chroma_db so it
# survives between runs. No account, no API key, no internet needed.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "hr_policy"

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def store_in_chromadb(chunks: List[str], embeddings: List[List[float]], namespace: str = ""):
    """Stores chunks + their embeddings in the local ChromaDB collection."""
    ids = [f"{namespace}chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk, "chunk_index": i} for i, chunk in enumerate(chunks)]

    # Chroma also wants the raw text passed separately as 'documents'
    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")


def search_in_chromadb(query_vector: List[float], top_k: int = 4, namespace: str = ""):
    """Searches the local ChromaDB collection for the top_k most similar chunks."""
    results = _collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    matched_chunks = results.get("documents", [[]])[0]
    print(f"Found {len(matched_chunks)} matches for the query.")
    return matched_chunks
