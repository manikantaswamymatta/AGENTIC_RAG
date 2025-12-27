# rag/vector_store.py
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2  # for PDF extraction

# ------------------------------------------------------------------
# SentenceTransformer (lazy-loaded to avoid Windows spawn issues)
# ------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """
    INTERNAL helper to load embedding model once.
    Safe to import and call optionally from other files.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[VECTOR] Loading SentenceTransformer model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[VECTOR] SentenceTransformer model loaded.")
    return _embedding_model


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyPDF2.
    """
    print(f"[PDF] Extracting text from: {pdf_path}")
    text_chunks = []

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            print(f"[PDF] Number of pages: {len(reader.pages)}")

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    print(
                        f"[PDF] Extracted page {i + 1}/{len(reader.pages)} "
                        f"(len={len(page_text)})"
                    )
                    text_chunks.append(page_text)
                except Exception as e:
                    print(f"[PDF][ERROR] Failed to read page {i + 1}: {e}")

    except Exception as e:
        print(f"[PDF][ERROR] Could not open PDF {pdf_path}: {e}")
        return ""

    full_text = "\n".join(text_chunks)
    print(f"[PDF] Total extracted text length: {len(full_text)}")
    return full_text


def chunk_text(text: str, max_words: int = 200) -> List[str]:
    """
    Split text into chunks by word count with overlap.
    """
    print(
        f"[CHUNK] Chunking text into word-based chunks "
        f"(max {max_words} words each, with overlap)"
    )

    words = text.split()
    chunks: List[str] = []

    if not words:
        print("[CHUNK] No words found in text.")
        return chunks

    overlap_words = max(max_words // 5, 1)  # 20% overlap
    step = max_words - overlap_words

    print(
        f"[CHUNK] Using overlap of {overlap_words} words. "
        f"Step size: {step} words."
    )

    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    print(f"[CHUNK] Created {len(chunks)} word-based chunks with overlap.")
    return chunks


class VectorStore:
    """
    Vector store using in-memory Chroma DB.
    """

    def __init__(self):
        print("[VECTOR] Setting up in-memory Chroma DB...")
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection = self.client.create_collection(name="bfsi_docs")
        print("[VECTOR] Collection 'bfsi_docs' ready.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using SentenceTransformer.
        """
        if not texts:
            raise ValueError("[VECTOR] embed() received empty text list")

        print(f"[VECTOR] Embedding {len(texts)} text(s)...")

        model = _get_embedding_model()
        vectors = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if len(vectors) != len(texts):
            raise RuntimeError(
                "[VECTOR] Embedding count mismatch "
                f"({len(vectors)} != {len(texts)})"
            )

        print("[VECTOR] Embeddings created successfully.")
        return vectors.tolist()

    def add_documents(self, docs: List[Dict]):
        """
        docs format:
        { "id": str, "text": str, "metadata": dict }
        """
        if not docs:
            print("[VECTOR] No documents to add.")
            return

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]

        embeddings = self.embed(texts)

        print(f"[VECTOR] Adding {len(docs)} docs to the vector DB...")
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print("[VECTOR] Documents added successfully.")

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Perform similarity search against the vector DB.
        """
        print(f"[VECTOR] Similarity search for query: {query!r}")

        query_embedding = self.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        docs = results.get("documents", [[]])[0]
        print(f"[VECTOR] Found {len(docs)} docs.")
        return docs

    def add_pdf(self, pdf_path: str, category: str = "pdf_bfsi"):
        """
        Ingest a single PDF:
        - extract text
        - chunk
        - embed
        - store in Chroma
        """
        print(f"[PDF] Adding PDF to vector DB: {pdf_path}")

        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            print("[PDF] No extractable text, skipping.")
            return None

        chunks = chunk_text(full_text, max_words=200)

        docs: List[Dict] = []
        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"chunk_{idx}",
                    "text": chunk,
                    "metadata": {
                        "category": category,
                        "source": pdf_path,
                        "chunk_index": idx,
                    },
                }
            )

        print(f"[PDF] Inserting {len(docs)} chunk(s) into vector DB...")
        self.add_documents(docs)
        print(f"[PDF] Done adding PDF: {pdf_path}")
