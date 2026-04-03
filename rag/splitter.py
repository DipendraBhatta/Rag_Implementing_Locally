# rag/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.loader import load_documents
from typing import List
from langchain_core.documents import Document


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into chunks with your exact requirement:
    - chunk_size = 1200 characters
    - chunk_overlap ≈ 100 words (≈ 500 characters)
    """
    print("--- Splitting documents into chunks (1200 chars, ~100 words overlap) ---")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,           
        chunk_overlap=200,         # ≈100 words overlap
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    
    print(f" Created {len(chunks)} chunks from {len(docs)} documents.")
    return chunks


# ====================== TEST BLOCK ======================
if __name__ == "__main__":
    raw_docs = load_documents("data", extension=".txt")
    if raw_docs:
        chunks = split_documents(raw_docs)
        
        print(f"\n{'='*70}")
        print("PREVIEW OF FIRST 2 CHUNKS")
        print(f"{'='*70}\n")
        
        for doc in raw_docs:
            doc_chunks = [c for c in chunks if c.metadata['source'] == doc.metadata['source']]
            print(f"\nCompany: {doc.metadata['company']}")
            for i, chunk in enumerate(doc_chunks[:2]):
                print(f"   --- Chunk {i+1} ({len(chunk.page_content)} chars) ---")
                print(chunk.page_content[:400] + "...")
                print("-" * 60)