# pull_models.py
import os
import subprocess

def pull_ollama_model(model_name):
    print(f"📥 Pulling {model_name} ... (this may take a few minutes)")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Successfully pulled {model_name}")
    except Exception as e:
        print(f"❌ Failed to pull {model_name}: {e}")
        print("Make sure Ollama is running!")

print("🚀 Pulling required Ollama models for Local RAG...\n")
pull_ollama_model("phi3:mini")          # LLM
pull_ollama_model("nomic-embed-text")   # Embedding

print("\n🎉 All models downloaded!")
print("Now run: python -m rag.vectorstore  (to create DB)")
print("Then:   python -m rag.main")