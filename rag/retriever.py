# rag/retriever.py
from langchain_chroma import Chroma
from typing import List, Tuple
from rag.embeddings import get_embedding_model
from langchain_core.documents import Document


def get_retriever(
    db_path: str = "./chroma_db",
    collection_name: str = "rag_chunks",
    k: int = 6,
    score_threshold: float = 0.55   # Lowered for all-MiniLM-L6-v2
):
    """Returns retriever function with cosine similarity"""
    print(f"--- Loading vector store from {db_path} ---")
    
    embeddings = get_embedding_model()
    
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    
    total = vector_db._collection.count()
    print(f"Vector store loaded. Total chunks: {total}")

    def retriever_with_scores(query: str) -> List[Tuple[Document, float]]:
        docs_with_distance = vector_db.similarity_search_with_score(query, k=k*4)  # Fetch more
        
        results = []
        for doc, distance in docs_with_distance:
            similarity = 1 - distance
            if similarity >= score_threshold:
                results.append((doc, similarity))
        
        results = results[:k]
        
        print(f"Query: '{query}'")
        print(f"  → Found {len(results)} relevant chunks (threshold = {score_threshold})")
        
        # Debug: Show top 3 scores
        print("  Top similarities:", [round(1-d, 4) for _, d in docs_with_distance[:3]])
        
        return results

    print(f"Retriever ready (k={k}, threshold={score_threshold})")
    return retriever_with_scores


# ====================== TEST BLOCK ======================
if __name__ == "__main__":
    print("=== Running retriever test ===")
    try:
        retriever = get_retriever()
        print("Retriever created successfully!\n")
        
        test_queries = ["hello", "tell me about google", "What is Tesla working on?"]
        for q in test_queries:
            print("-" * 60)
            results = retriever(q)
            print(f"Final returned chunks: {len(results)}\n")
    except Exception as e:
        print(f"ERROR: {e}")