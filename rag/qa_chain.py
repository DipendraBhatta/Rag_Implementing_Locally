# rag/qa_chain.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
from langchain_core.documents import Document

from rag.retriever import get_retriever
from rag.rag_prompts import get_rag_prompt


def format_docs_with_scores(results: List[Tuple[Document, float]]) -> str:
    """Format retrieved documents with similarity scores"""
    if not results:
        return "No sufficiently relevant information found in the provided documents."
    
    formatted = []
    for i, (doc, score) in enumerate(results, 1):
        formatted.append(
            f"--- From {doc.metadata['company']} (Similarity: {score:.4f}) ---\n"
            f"{doc.page_content}\n"
        )
    return "\n\n".join(formatted)


def create_qa_chain(llm):
    """Create full RAG QA chain"""
    # Using same settings as retriever
    retriever_func = get_retriever(k=6, score_threshold=0.55)
    prompt = get_rag_prompt()

    def retrieve_and_format(query: str):
        results = retriever_func(query)
        return format_docs_with_scores(results)

    qa_chain = (
        {
            "context": retrieve_and_format,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG QA Chain created successfully")
    return qa_chain