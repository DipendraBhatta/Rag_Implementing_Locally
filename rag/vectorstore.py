# rag/vectorstore.py
import os
from langchain_chroma import Chroma
from rag.embeddings import get_embedding_model


def create_vector_db(
    chunks,
    db_path="./chroma_db",
    collection_name="rag_chunks",
):
    """Create persistent ChromaDB with cosine similarity"""
    if not chunks:
        print("No chunks provided!")
        return None

    os.makedirs(db_path, exist_ok=True)

    embeddings = get_embedding_model()

    print(f"--- Creating vector database with {len(chunks)} chunks ---")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},   # Cosine Similarity
    )

    print(f" Vector database successfully created at: {db_path}")
    return vector_db


# ====================== TEST BLOCK ======================
if __name__ == "__main__":
    from rag.loader import load_documents
    from rag.splitter import split_documents

    raw_docs = load_documents("data", extension=".txt")
    if raw_docs:
        chunks = split_documents(raw_docs)
        create_vector_db(chunks)