from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_chromadb
from typing import List

pdf_path = "./resources/HRPolicy.pdf"
def run():
    # Read HR Policy PDF and extract text
    pages = read_pdf(pdf_path)

    # Chunk the extracted text into manageable pieces
    chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)

    embedded_chunks = embed_chunks(chunks)
    
    store_in_chromadb(chunks, embedded_chunks, namespace="")
    print(f"Done! Indexed {len(chunks)} chunks from {pdf_path}.")
   
    
if __name__ == "__main__":
    run()