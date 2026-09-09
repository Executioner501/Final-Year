import sys

packages = ["langchain", "llama_index", "sentence_transformers", "chromadb", "faiss", "sklearn"]
print(f"Python: {sys.version}")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  {pkg}: AVAILABLE")
    except ImportError as e:
        print(f"  {pkg}: NOT INSTALLED ({e})")
