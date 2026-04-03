# rag/loader.py

import os
from typing import List
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader, 
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


# ====================== LOADER MAPPING ======================
# We map file extensions to their correct LangChain loader
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
}
# ============================================================


def load_documents(directory_path: str = "data", extension: str = ".txt") -> List[Document]:
    """
    Load documents from a folder.
    Supports .txt, .pdf, and .md files.
    """
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found!")

    print(f"--- Loading {extension} files from: {directory_path} ---")

    # Get the correct loader class based on extension
    loader_cls = LOADER_MAPPING.get(extension.lower())
    
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {extension}. Use .txt, .pdf or .md")

    # Create the directory loader
    loader = DirectoryLoader(
        path=directory_path,
        glob=f"**/*{extension}",                    # Search for files with this extension
        loader_cls=loader_cls,
        loader_kwargs={"encoding": "utf-8"} if extension == ".txt" else {},   # Only txt needs encoding
        show_progress=True,
        use_multithreading=True
    )

    # Load all documents
    documents = loader.load()

    # Add useful metadata to each document
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("source", ""))
        doc.metadata["source"] = filename
        doc.metadata["company"] = filename.replace(".txt", "").replace(".pdf", "").replace(".md", "").title()

    print(f" Successfully loaded {len(documents)} document(s).")
    return documents

# ====================== TEST BLOCK ======================
if __name__ == "__main__":
    # You can change extension here to test .pdf or .md later
    docs = load_documents("data", extension=".txt")
    
    print(f"\n{'='*70}")
    print("PREVIEW OF ALL LOADED DOCUMENTS")
    print(f"{'='*70}\n")
    
    print(f"Total Documents Loaded : {len(docs)}\n")
    
    total_chars = 0
    
    for i, doc in enumerate(docs, start=1):
        char_count = len(doc.page_content)
        total_chars += char_count
        
        print(f"[{i}] Company : {doc.metadata['company']}")
        print(f"     Source  : {doc.metadata['source']}")
        print(f"     Length  : {char_count} characters")
        print("-" * 60)
        print(doc.page_content[:500] + "...")   # First 500 characters preview
        print("\n")
    
    print(f"{'='*70}")
    print(f"SUMMARY:")
    print(f"   Total Documents : {len(docs)}")
    print(f"   Total Characters: {total_chars:,}")
    print(f"{'='*70}")