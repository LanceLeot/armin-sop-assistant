
from pathlib import Path
import re
import logging
import json
import os
import threading
import time
import uuid

import chromadb
import fitz
import httpx
import requests

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
CHROMA_DIR = BASE_DIR / "chroma_db"
ACTIVE_COLLECTION_FILE = CHROMA_DIR / "active_collection.json"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

OLLAMA_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

OLLAMA_TIMEOUT = 180.0
EMBED_TIMEOUT = 120.0

# Retrieve extra candidates first, then filter them.
RETRIEVAL_CANDIDATES = 12
MAX_CONTEXT_PAGES = 5

# Chroma distances are model/index dependent, not probabilities.
# This is a conservative starting point; evaluate it against your PDFs.
MAX_DISTANCE = 0.65

# Avoid allowing one document to dominate every answer.
MAX_PAGES_PER_DOCUMENT = 3

CHROMA_COLLECTION = "armin_sop_pages"
INDEX_BATCH_SIZE = 20
REFRESH_DEBOUNCE_SECONDS = 3.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("armin")

FALLBACK_ANSWER = (
    "I couldn't find enough relevant information in the available "
    "SOP excerpts to answer this question reliably. The requested "
    "procedure or detail could not be verified from the retrieved "
    "pages. Please check the relevant official SOP or department."
)


# --------------------------------------------------
# 2. CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(title="Armin - Organizational SOP Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. REQUEST FORMAT
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str


# --------------------------------------------------
# 4. LOAD PDF DOCUMENTS
# --------------------------------------------------

def load_documents():
    """Extract readable text from every PDF page."""

    loaded_documents = []

    if not DOCUMENTS_DIR.exists():
        logger.error("Documents folder not found: %s", DOCUMENTS_DIR)
        return loaded_documents

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text").strip()

                    if page_text:
                        loaded_documents.append({
                            "filename": pdf_path.name,
                            "page": page_number,
                            "text": page_text,
                        })

            logger.info("Loaded PDF: %s", pdf_path.name)

        except Exception:
            logger.exception("Could not read %s", pdf_path.name)

    logger.info(
        "Total readable PDF pages loaded: %s",
        len(loaded_documents),
    )
    return loaded_documents


documents = load_documents()


# --------------------------------------------------
# 5. CONNECT TO THE PERSISTENT SEMANTIC INDEX
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
index_lock = threading.RLock()
refresh_lock = threading.Lock()
refresh_timer = None
folder_observer = None
index_status = {
    "refreshing": False,
    "last_refresh": None,
    "last_error": None,
    "watched_folder": str(DOCUMENTS_DIR),
}


def read_active_collection_name():
    """Load the last successfully activated collection after a restart."""
    try:
        data = json.loads(ACTIVE_COLLECTION_FILE.read_text(encoding="utf-8"))
        name = data.get("collection")
        if isinstance(name, str) and name:
            return name
    except (OSError, ValueError, AttributeError):
        pass
    return CHROMA_COLLECTION


try:
    active_name = read_active_collection_name()
    sop_collection = chroma_client.get_collection(name=active_name)
    logger.info("Semantic index loaded: %s pages (%s)", sop_collection.count(), active_name)
except Exception:
    try:
        sop_collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
        logger.info("Loaded default semantic index: %s pages", sop_collection.count())
    except Exception:
        sop_collection = None
        logger.exception("Semantic index unavailable")


def get_pdf_snapshot():
    """Return a stable signature of PDF names, sizes, and modification times."""
    if not DOCUMENTS_DIR.exists():
        return ()
    signature = []
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda p: p.name.casefold()):
        try:
            stat = path.stat()
            signature.append((path.name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            # A file may be in the middle of being copied or removed.
            signature.append((path.name, -1, -1))
    return tuple(signature)


last_pdf_snapshot = get_pdf_snapshot()


def build_new_collection():
    """Build a separate collection; never destroy the currently active index."""
    global documents

    fresh_documents = load_documents()
    if not fresh_documents:
        raise RuntimeError("No readable SOP PDF pages were found; keeping the previous index.")

    # Match the indexer's duplicate-page behavior.
    unique_pages = []
    seen_page_texts = set()
    for item in fresh_documents:
        normalized = " ".join(item["text"].split()).casefold()
        if normalized in seen_page_texts:
            continue
        seen_page_texts.add(normalized)
        unique_pages.append(item)

    if not unique_pages:
        raise RuntimeError("No unique readable SOP pages were found; keeping the previous index.")

    new_name = "armin_sop_" + uuid.uuid4().hex[:16]
    new_collection = chroma_client.create_collection(
        name=new_name,
        metadata={"embedding_model": EMBED_MODEL, "created_by": "automatic_refresh"},
    )
    try:
        for start in range(0, len(unique_pages), INDEX_BATCH_SIZE):
            batch = unique_pages[start:start + INDEX_BATCH_SIZE]
            response = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBED_MODEL, "input": [item["text"] for item in batch]},
                timeout=EMBED_TIMEOUT,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            if len(embeddings) != len(batch) or any(not embedding for embedding in embeddings):
                raise RuntimeError("Ollama returned incomplete page embeddings.")
            new_collection.add(
                ids=[f"{item['filename']}::page::{item['page']}" for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=embeddings,
                metadatas=[{"filename": item["filename"], "page": item["page"]} for item in batch],
            )
            logger.info("Automatic index refresh: indexed %s/%s pages", min(start + len(batch), len(unique_pages)), len(unique_pages))

        if new_collection.count() != len(unique_pages):
            raise RuntimeError("The new index page count did not match the extracted page count.")

        # Persist the new active collection name atomically before switching the in-memory pointer.
        temp_marker = ACTIVE_COLLECTION_FILE.with_suffix(".tmp")
        temp_marker.write_text(json.dumps({"collection": new_name}), encoding="utf-8")
        os.replace(temp_marker, ACTIVE_COLLECTION_FILE)

        global sop_collection
        with index_lock:
            sop_collection = new_collection
            documents = fresh_documents
        return new_name, new_collection.count()
    except Exception:
        try:
            chroma_client.delete_collection(name=new_name)
        except Exception:
            logger.exception("Could not clean up failed temporary collection %s", new_name)
        raise


def perform_index_refresh():
    """Run one refresh in a worker thread and preserve the last good index on failure."""
    global last_pdf_snapshot
    if not refresh_lock.acquire(blocking=False):
        logger.info("Index refresh already running; skipping duplicate trigger")
        return
    try:
        current_snapshot = get_pdf_snapshot()
        if current_snapshot == last_pdf_snapshot:
            return
        index_status["refreshing"] = True
        index_status["last_error"] = None
        logger.info("SOP folder change detected; rebuilding semantic index safely")
        new_name, count = build_new_collection()
        last_pdf_snapshot = get_pdf_snapshot()
        index_status["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
        index_status["last_error"] = None
        logger.info("Automatic index refresh complete: %s pages (%s)", count, new_name)
    except Exception as error:
        index_status["last_error"] = str(error)
        logger.exception("Automatic index refresh failed; the previous index remains active")
    finally:
        index_status["refreshing"] = False
        refresh_lock.release()


def schedule_index_refresh():
    """Debounce filesystem events so one file copy triggers only one refresh."""
    global refresh_timer
    if refresh_timer is not None:
        refresh_timer.cancel()
    refresh_timer = threading.Timer(REFRESH_DEBOUNCE_SECONDS, perform_index_refresh)
    refresh_timer.daemon = True
    refresh_timer.start()


class SopFolderEventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        paths = [getattr(event, "src_path", "")]
        destination = getattr(event, "dest_path", "")
        if destination:
            paths.append(destination)
        if any(str(path).lower().endswith(".pdf") for path in paths if path):
            schedule_index_refresh()


def start_folder_monitor():
    """Start monitoring the SOP folder when FastAPI starts."""
    global folder_observer, last_pdf_snapshot
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    last_pdf_snapshot = get_pdf_snapshot()
    folder_observer = Observer()
    folder_observer.schedule(SopFolderEventHandler(), str(DOCUMENTS_DIR), recursive=False)
    folder_observer.daemon = True
    folder_observer.start()
    logger.info("Automatic SOP folder monitoring started: %s", DOCUMENTS_DIR)


def stop_folder_monitor():
    global folder_observer, refresh_timer
    if refresh_timer is not None:
        refresh_timer.cancel()
    if folder_observer is not None:
        folder_observer.stop()
        folder_observer.join(timeout=5)
        folder_observer = None



def embed_question(question: str):
    """Generate a question embedding with Ollama."""

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": question,
        },
        timeout=EMBED_TIMEOUT,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if not embeddings or not embeddings[0]:
        raise RuntimeError(
            "Ollama returned an empty question embedding."
        )

    return embeddings[0]


def normalize_tokens(text: str):
    """Normalize text into simple words for a lexical relevance check."""

    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def lexical_overlap(question: str, page_text: str):
    """
    Calculate a simple overlap score.

    This is only a supporting signal, not a measure of factual accuracy.
    """

    question_tokens = normalize_tokens(question)
    page_tokens = normalize_tokens(page_text)

    # Ignore very short words that often occur in unrelated documents.
    question_tokens = {
        token for token in question_tokens if len(token) > 2
    }

    if not question_tokens or not page_tokens:
        return 0.0

    overlap = question_tokens.intersection(page_tokens)
    return len(overlap) / len(question_tokens)


def find_relevant_pages(question: str):
    """Retrieve, filter, and rank candidate SOP pages."""

    if sop_collection is None:
        raise RuntimeError(
            "The SOP vector collection is unavailable. "
            "Run test_sop_index.py to build the index."
        )

    collection_count = sop_collection.count()

    if collection_count == 0:
        raise RuntimeError(
            "The SOP vector collection is empty. "
            "Run test_sop_index.py to build the index."
        )

    question_embedding = embed_question(question)

    candidate_count = min(
        RETRIEVAL_CANDIDATES,
        collection_count,
    )

    results = sop_collection.query(
        query_embeddings=[question_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    result_documents = results.get("documents") or [[]]
    result_metadatas = results.get("metadatas") or [[]]
    result_distances = results.get("distances") or [[]]

    if not result_documents[0]:
        return []

    candidates = []

    for text, metadata, distance in zip(
        result_documents[0],
        result_metadatas[0],
        result_distances[0],
    ):
        if not text or not metadata:
            continue

        try:
            distance = float(distance)
            page_number = int(metadata["page"])
            filename = str(metadata["filename"])
        except (TypeError, ValueError, KeyError):
            continue

        if distance < 0:
            distance = 0.0

        lexical_score = lexical_overlap(question, text)

        # Discard weak semantic matches unless the page also has
        # meaningful literal overlap with the question.
        if distance > MAX_DISTANCE and lexical_score < 0.20:
            continue

        # Lower distance is better. Lexical overlap is a modest
        # secondary signal to help separate similarly themed SOPs.
        combined_score = (
            (1.0 / (1.0 + distance)) * 0.75
            + lexical_score * 0.25
        )

        candidates.append({
            "filename": filename,
            "page": page_number,
            "text": text,
            "distance": distance,
            "lexical_score": lexical_score,
            "score": combined_score,
        })

    # Rank using the combined score rather than raw distance alone.
    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Limit repeated pages from one document. This prevents one
    # document from crowding every other relevant document out.
    selected = []
    pages_per_document = {}

    for candidate in candidates:
        filename = candidate["filename"]
        used = pages_per_document.get(filename, 0)

        if used >= MAX_PAGES_PER_DOCUMENT:
            continue

        selected.append(candidate)
        pages_per_document[filename] = used + 1

        if len(selected) >= MAX_CONTEXT_PAGES:
            break

    logger.info(
        "Retrieval for %r: %s candidate(s), %s selected page(s)",
        question,
        len(candidates),
        len(selected),
    )

    for item in selected:
        logger.info(
            "Selected %s page %s (distance=%.3f, lexical=%.3f)",
            item["filename"],
            item["page"],
            item["distance"],
            item["lexical_score"],
        )

    return selected


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    index_available = (
        sop_collection is not None
        and sop_collection.count() > 0
    )

    return {
        "status": "ok",
        "sop_loaded": len(documents) > 0,
        "readable_pages": len(documents),
        "documents": sorted({
            document["filename"]
            for document in documents
        }),
        "model": OLLAMA_MODEL,
        "semantic_index_loaded": index_available,
        "indexed_pages": (
            sop_collection.count()
            if sop_collection is not None
            else 0
        ),
        "embedding_model": EMBED_MODEL,
        "retrieval_candidates": RETRIEVAL_CANDIDATES,
        "max_context_pages": MAX_CONTEXT_PAGES,
        "max_distance": MAX_DISTANCE,
        "index_refresh": dict(index_status),
        "auto_refresh_enabled": folder_observer is not None and folder_observer.is_alive(),
    }


# --------------------------------------------------
# 7. ASK ARMIN
# --------------------------------------------------

@app.post("/ask")
async def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    if not documents:
        return {
            "answer": (
                "No readable SOP PDF content was found. "
                "Please check the backend/documents folder."
            ),
            "sources": [],
        }

    try:
        matches = find_relevant_pages(question)

    except requests.RequestException:
        logger.exception("Ollama embedding service error")
        return {
            "answer": (
                "I couldn't search the SOP index. Please check "
                "that Ollama is running and the embedding model "
                "nomic-embed-text is available."
            ),
            "sources": [],
        }

    except RuntimeError:
        logger.exception("Semantic retrieval error")
        return {
            "answer": (
                "The SOP search index is unavailable. "
                "Please check the backend setup."
            ),
            "sources": [],
        }

    except Exception:
        logger.exception("Unexpected semantic search error")
        return {
            "answer": (
                "An unexpected error occurred while searching "
                "the SOP index. Please check the backend terminal."
            ),
            "sources": [],
        }

    if not matches:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    context_parts = []
    sources = []

    for match in matches:
        context_parts.append(
            f"DOCUMENT: {match['filename']}\n"
            f"PAGE: {match['page']}\n"
            f"RETRIEVAL DISTANCE: {match['distance']:.3f}\n"
            f"SOP EXCERPT:\n{match['text']}"
        )

        sources.append({
            "document": match["filename"],
            "page": match["page"],
            "relevance": round(match["score"], 3),
        })

    sop_context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are Armin, an assistant for organizational Standard
Operating Procedures (SOPs).

Answer using only the supplied SOP excerpts.

STRICT EVIDENCE RULES:

1. Use only facts explicitly supported by the excerpts.
   Never invent policy details or use outside knowledge.

2. Each excerpt has a document name and page number.
   Keep every rule attached to its own document.
   Do not transfer a rule from one SOP to another.

3. A similar process is not proof that the same rule applies.
   For example, approval rules for a WFH request do not establish
   approval rules for a promotion.

4. When asked about a named SOP, prioritize evidence from that
   exact SOP. Do not substitute a similar rule from another SOP.
   If the named SOP does not contain the requested detail, say so.
   A different SOP may be mentioned separately only if it explicitly
   provides relevant information, and its document name must be clear.

5. If a requested detail is not documented, name that exact detail.
   Do not claim it is documented merely because another process has
   a similar rule or timeframe.

6. Do not contradict yourself. Before finalizing the answer, check
   whether you have already stated a fact that you later describe as
   missing or unspecified. If the excerpt supports the fact, do not
   say it is unspecified. If the evidence is insufficient, explain
   the limitation precisely.

7. If a procedure is not documented, say so directly. Do not create
   a workflow by combining unrelated SOPs.

8. If only part of the question is answered by the excerpts,
   answer the supported parts and clearly identify the missing parts.

9. Use only source names and page numbers supplied in the excerpts.
   Do not invent citations.

10. Keep the answer focused. Do not list unrelated documents simply
    because they were retrieved.

11. When multiple SOPs have different rules, keep them separate.
    Clearly label each document's rule and scope.

12. Retrieval relevance is not proof. Do not use a page's presence
    in the context as evidence that it answers the question.

ANSWER STYLE:

- Use numbered steps for procedures that are explicitly sequential.
- Use concise bullet points for requirements and responsibilities.
- Cite the document name and page for each key claim in the answer.
- Do not add generic missing-information statements after you have
  already answered the requested detail.
- Do not add irrelevant SLAs, approval rules, or exception procedures.
- Accuracy is more important than completeness.
"""

    user_prompt = (
        f"USER QUESTION:\n{question}\n\n"
        f"RETRIEVED SOP EXCERPTS:\n{sop_context}\n\n"
        "Answer the question using only directly relevant evidence. "
        "Preserve the boundaries between documents. If the user names "
        "a specific SOP, first check whether that SOP supports the answer. "
        "Do not substitute rules from similar procedures. Before replying, "
        "check that the answer contains no unsupported claims and no "
        "contradiction between stated facts and missing information. "
        "Cite only the source document and page that support each claim. "
        "If the requested information is not documented, say exactly "
        "which detail is missing."
    )

    try:
        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT
        ) as client:
            response = await client.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                    },
                },
            )

            response.raise_for_status()
            result = response.json()

        answer = (
            result.get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            return {
                "answer": (
                    "Gemma returned an empty response. "
                    "Please try asking your question again."
                ),
                "sources": sources,
            }

        return {
            "answer": answer,
            "sources": sources,
        }

    except httpx.ConnectError:
        logger.exception("Could not connect to Ollama")
        return {
            "answer": (
                "I couldn't connect to Ollama. Please make sure "
                "Ollama is running and Gemma 3:4B is installed."
            ),
            "sources": [],
        }

    except httpx.TimeoutException:
        logger.exception("Gemma response timed out")
        return {
            "answer": (
                "Gemma took too long to respond. Please try again."
            ),
            "sources": sources,
        }

    except httpx.HTTPStatusError:
        logger.exception("Ollama returned an HTTP error")
        return {
            "answer": (
                "There was a problem communicating with Gemma. "
                "Please check that the Ollama service is working."
            ),
            "sources": sources,
        }

    except Exception:
        logger.exception("Unexpected answer-generation error")
        return {
            "answer": (
                "An unexpected error occurred while generating "
                "the answer. Please check the backend terminal."
            ),
            "sources": sources,
        }


@app.on_event("startup")
def on_startup():
    start_folder_monitor()


@app.on_event("shutdown")
def on_shutdown():
    stop_folder_monitor()


# --------------------------------------------------
# 8. SERVE THE REACT FRONTEND
# --------------------------------------------------

if FRONTEND_DIST.exists():
    assets_folder = FRONTEND_DIST / "assets"

    if assets_folder.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_folder)),
            name="frontend-assets",
        )

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        requested_file = FRONTEND_DIST / full_path

        if full_path and requested_file.is_file():
            return FileResponse(str(requested_file))

        index_file = FRONTEND_DIST / "index.html"

        if index_file.exists():
            return FileResponse(str(index_file))

        return {
            "error": "Frontend has not been built yet.",
            "instruction": (
                "Run npm run build inside the frontend folder."
            ),
        }

else:
    logger.warning(
        "Frontend build not found. Run 'npm run build' "
        "inside the frontend folder."
    )
