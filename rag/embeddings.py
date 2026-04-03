# rag/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"--- Loading Hugging Face embeddings: {model_name} ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name

    )
    print(f"Model loaded: {embeddings}")
    return embeddings

if __name__ == "__main__":
    from rag.loader import load_documents
    from rag.splitter import split_documents

    # Load & split documents
    raw_docs = load_documents("data")
    chunks = split_documents(raw_docs)
    print(f"--- Total chunks: {len(chunks)} ---")

    # Load embeddings
    model = get_embedding_model()
    texts = [doc.page_content for doc in chunks]
 
    # Embed documents (no batch_size argument!)
    embeddings_vectors = model.embed_documents(texts)  # just pass chunks/texts
    print(f" Generated {len(embeddings_vectors)} embeddings")