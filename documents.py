"""Document processing: extract text, split into passages, index in Chroma, manage versions."""
import hashlib
import io
import re
import unicodedata

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import storage

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ACTIVE_DOCUMENTS = 5
PASSAGE_TARGET = 1000      # aim for passages of roughly 800-1200 characters
PASSAGE_MIN = 800
PASSAGE_MAX = 1200
PASSAGE_OVERLAP = 150
MIN_CHARS_PER_PDF_PAGE = 20  # below this on every page we assume a scanned (image-only) PDF


class DocumentError(Exception):
    """A problem with the uploaded file that the user can fix (wrong type, scanned PDF...)."""


# ---------- text helpers ----------

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})


def normalize_text(text):
    """Unicode-normalize, unify quotes/dashes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def split_text(text):
    """Split text into ~800-1200 character passages with a small overlap.

    Prefers to break at a paragraph, then a sentence, then a word boundary.
    """
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        if len(text) - start <= PASSAGE_MAX:
            chunks.append(text[start:].strip())
            break
        window = text[start + PASSAGE_MIN:start + PASSAGE_MAX]
        cut = -1
        for sep in ("\n\n", "\n", ". ", " "):
            cut = window.rfind(sep)
            if cut != -1:
                cut += len(sep)
                break
        end = start + PASSAGE_MIN + cut if cut != -1 else start + PASSAGE_TARGET
        chunks.append(text[start:end].strip())
        next_start = end - PASSAGE_OVERLAP
        space = text.find(" ", next_start)  # start the overlap on a word boundary
        start = space + 1 if 0 <= space < end else end
    return [c for c in chunks if c]


# ---------- extraction ----------

def extract_pdf_pages(data):
    """Return [(page_number, text)] for a text-based PDF. Raises DocumentError otherwise."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("This PDF is password-protected. Upload an unencrypted copy.")
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except DocumentError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise DocumentError(f"Could not read this PDF ({type(exc).__name__}). Is the file a valid PDF?") from exc
    if not pages:
        raise DocumentError("This PDF has no pages.")
    if all(len(text.strip()) < MIN_CHARS_PER_PDF_PAGE for _, text in pages):
        raise DocumentError(
            "No selectable text was found in this PDF. It looks like a scanned or image-only PDF. "
            "OCR is not supported in this prototype; upload a text-based PDF or a TXT file."
        )
    return pages


def extract_txt(data):
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError("This TXT file is not valid UTF-8. Save it as UTF-8 and upload again.") from exc
    if not text.strip():
        raise DocumentError("This TXT file is empty.")
    return text


def build_passages(filename, data):
    """Extract text and split it into passages with source locations.

    Returns (file_type, full_text, passages). PDF passages never cross a page boundary,
    so each one has an exact page number. TXT files get passage numbers only.
    """
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("File is larger than the 5 MB limit.")
    lower = filename.lower()
    passages = []
    if lower.endswith(".pdf"):
        file_type = "pdf"
        pages = extract_pdf_pages(data)
        for page_no, page_text in pages:
            chunks = split_text(page_text)
            for i, chunk in enumerate(chunks, start=1):
                location = f"Page {page_no}" if len(chunks) == 1 else f"Page {page_no}, passage {i}"
                passages.append({"page": page_no, "location": location, "text": chunk})
        full_text = "\n\n".join(text for _, text in pages)
    elif lower.endswith(".txt"):
        file_type = "txt"
        full_text = extract_txt(data)
        for i, chunk in enumerate(split_text(full_text), start=1):
            passages.append({"page": None, "location": f"Passage {i}", "text": chunk})
    else:
        raise DocumentError("Only .pdf and .txt files are supported.")
    if not passages:
        raise DocumentError("No text could be extracted from this file.")
    for ordinal, p in enumerate(passages, start=1):
        p["ordinal"] = ordinal
    return file_type, full_text, passages


# ---------- search index ----------

class PassageIndex:
    """Chroma collection holding passage embeddings for every stored version.

    Old versions stay in the index (their passages remain citable), but searches are
    always filtered to the current version IDs taken from SQLite.
    """

    def __init__(self, path, embedding_model):
        self._client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedding_model).strip("-") or "default"
        # One collection per embedding model so vectors from different models never mix.
        self._collection = self._client.get_or_create_collection(
            name=f"passages-{slug}"[:60], embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, passages, embeddings):
        self._collection.upsert(  # upsert = safe if the same passages are indexed twice
            ids=[p["id"] for p in passages],
            embeddings=embeddings,
            documents=[p["text"] for p in passages],
            metadatas=[{"document_id": p["document_id"], "version_id": p["version_id"],
                        "location": p["location"]} for p in passages],
        )

    def has_version(self, version_id):
        return bool(self._collection.get(where={"version_id": version_id}, limit=1)["ids"])

    def delete_version(self, version_id):
        self._collection.delete(where={"version_id": version_id})

    def search(self, query_embeddings, version_ids, k):
        """Return a list (one per query) of passage IDs, best match first."""
        if not version_ids or not query_embeddings:
            return [[] for _ in query_embeddings]
        result = self._collection.query(
            query_embeddings=query_embeddings, n_results=k,
            where={"version_id": {"$in": list(version_ids)}},
        )
        return result["ids"]


def _index_version(conn, index, ai, version_id):
    rows = storage.passages_for_version(conn, version_id)
    passages = [dict(r) for r in rows]
    embeddings = ai.embed([p["text"] for p in passages])
    if len(embeddings) != len(passages):
        raise RuntimeError("Embedding count does not match passage count.")
    index.add(passages, embeddings)


def ensure_index(conn, index, ai):
    """Re-embed any current version missing from Chroma (e.g. index deleted or model changed)."""
    for version_id in storage.current_version_ids(conn):
        if not index.has_version(version_id):
            _index_version(conn, index, ai, version_id)


# ---------- adding and replacing documents ----------

def _process_new_version(conn, index, ai, doc_id, filename, file_type, full_text, passages, progress):
    """Store, embed and activate a version. On any failure the previous version stays current."""
    version_id = storage.create_version(conn, doc_id, filename, file_type, text_hash(full_text), full_text)
    try:
        for p in passages:
            p["id"] = f"v{version_id}-p{p['ordinal']}"
        storage.add_passages(conn, version_id, doc_id, passages)
        progress(f"Creating embeddings for {len(passages)} passages (text is sent to OpenAI)...")
        _index_version(conn, index, ai, version_id)
        storage.activate_version(conn, doc_id, version_id)
    except Exception as exc:
        try:
            index.delete_version(version_id)
        finally:
            storage.fail_version(conn, version_id, str(exc))
        raise
    return version_id


def add_document(conn, index, ai, filename, data, progress=lambda msg: None):
    """Add a brand-new document. Returns the new version ID."""
    existing = storage.get_document_by_name(conn, filename)
    if existing and existing["current_version_id"]:
        raise DocumentError(f"'{filename}' already exists. Use 'Replace a document' to upload a new version.")
    if len(storage.list_active_documents(conn)) >= MAX_ACTIVE_DOCUMENTS:
        raise DocumentError(f"This prototype supports at most {MAX_ACTIVE_DOCUMENTS} active documents.")
    progress("Extracting text...")
    file_type, full_text, passages = build_passages(filename, data)
    new_hash = text_hash(full_text)
    for doc in storage.list_active_documents(conn):
        if storage.get_version(conn, doc["version_id"])["text_hash"] == new_hash:
            raise DocumentError(f"This file has the same text as the existing document '{doc['name']}'.")
    doc_id = existing["id"] if existing else storage.create_document(conn, filename)
    return _process_new_version(conn, index, ai, doc_id, filename, file_type, full_text, passages, progress)


def replace_document(conn, index, ai, doc_id, filename, data, progress=lambda msg: None):
    """Upload a replacement version.

    Returns {"changed": False} if the normalized text is identical (nothing is stored or
    re-indexed), otherwise {"changed": True, "version_id": ...}.
    """
    doc = storage.get_document(conn, doc_id)
    if doc is None or doc["current_version_id"] is None:
        raise DocumentError("Select an existing active document to replace.")
    progress("Extracting text...")
    file_type, full_text, passages = build_passages(filename, data)
    current = storage.get_version(conn, doc["current_version_id"])
    if text_hash(full_text) == current["text_hash"]:
        return {"changed": False}
    version_id = _process_new_version(conn, index, ai, doc_id, filename, file_type, full_text, passages, progress)
    return {"changed": True, "version_id": version_id}


# ---------- retrieval ----------

def retrieve(conn, index, ai, queries, k=4):
    """Search current versions only. Returns passage rows (deduplicated, best first)."""
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return []
    embeddings = ai.embed(queries)
    id_lists = index.search(embeddings, storage.current_version_ids(conn), k)
    ordered = []
    for ids in id_lists:
        for pid in ids:
            if pid not in ordered:
                ordered.append(pid)
    rows = storage.get_passages(conn, ordered)
    current = set(storage.current_version_ids(conn))
    # Double-check against SQLite: never return a passage from a superseded version.
    return [rows[pid] for pid in ordered if pid in rows and rows[pid]["version_id"] in current]
