# rag/main.py
import os
from dotenv import load_dotenv

load_dotenv()

from rag.qa_chain import create_qa_chain
from langchain_ollama import ChatOllama


# ================== LIGHTER LLM SETUP ==================
llm = ChatOllama(
    model="tinyllama",          #  "llama3.2:1b" or "gemma2:2b" if still OOM
    temperature=0.3,
    num_ctx=4096,               
    num_predict=1024,
    base_url="http://localhost:11434"
)
print("Using local Ollama: phi3:mini (num_ctx=4096)")
# ======================================================


def main():
    print("="*90)
    print("Local RAG System - Company Knowledge Assistant")
    print("="*90)

    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print("Error: Vector DB not found!")
        print("Please run: python -m rag.vectorstore")
        return

    qa_chain = create_qa_chain(llm)

    print("\nSystem is ready!")
    print("Ask questions about companies")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        
        if query.lower() in ["exit", "quit", "q", "bye"]:
            print("Goodbye!")
            break
            
        if not query:
            continue

        print("\nThinking...\n")
        answer = qa_chain.invoke(query)
        
        print("Answer:")
        print("-" * 80)
        print(answer)
        print("-" * 80, "\n")


if __name__ == "__main__":
    main()