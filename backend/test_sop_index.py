
from pathlib import Path

import chromadb
import fitz
import requests

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 20


def get_embeddings(texts):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DOCUMENTS_DIR}")
        return

    pages = []
    seen_page_texts = set()

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()

                    if not text:
                        continue

                    # Normalize whitespace and capitalization
                    # so identical page text is detected consistently.
                    normalized_text = " ".join(text.split()).casefold()

                    if normalized_text in seen_page_texts:
                        print(
                            f"Skipping duplicate page: "
                            f"{pdf_path.name}, page {page_number}"
                        )
                        continue

                    seen_page_texts.add(normalized_text)

                    pages.append(
                        {
                            "id": f"{pdf_path.name}::page::{page_number}",
                            "filename": pdf_path.name,
                            "page": page_number,
                            "text": text,
                        }
                    )

        except Exception as error:
            print(f"Could not read {pdf_path.name}: {error}")

    if not pages:
        print("No readable PDF pages found.")
        return

    print(f"Extracted {len(pages)} unique readable pages.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name="armin_sop_pages",
        metadata={"embedding_model": EMBED_MODEL},
    )

    # Rebuild the collection so removed duplicate pages and
    # outdated document pages do not remain in the index.
    if collection.count() > 0:
        client.delete_collection("armin_sop_pages")
        collection = client.create_collection(
            name="armin_sop_pages",
            metadata={"embedding_model": EMBED_MODEL},
        )

    for start in range(0, len(pages), BATCH_SIZE):
        batch = pages[start : start + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(texts)

        collection.add(
            ids=[item["id"] for item in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "filename": item["filename"],
                    "page": item["page"],
                }
                for item in batch
            ],
        )

        print(
            f"Indexed {min(start + len(batch), len(pages))}"
            f"/{len(pages)} pages."
        )

    print(f"Index complete. Stored {collection.count()} pages.")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
