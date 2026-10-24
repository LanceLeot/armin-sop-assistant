
from pathlib import Path

import chromadb
import requests

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def main():
    question = input("Enter a question about your SOPs: ").strip()

    if not question:
        print("Please enter a question.")
        return

    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": question},
        timeout=120,
    )
    response.raise_for_status()
    embedding = response.json()["embeddings"][0]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("armin_sop_pages")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print("\nTop matching SOP pages:\n")

    for i, (text, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"Result {i}")
        print(f"Document: {metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Distance: {distance:.4f}")
        print(f"Excerpt: {text[:700]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
