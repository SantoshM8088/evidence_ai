"""SQLite records: documents, versions, passages, saved questions and answer runs.

SQLite is the source of truth for versions and history. Chroma (see documents.py)
is only a search index and can be rebuilt from the passages stored here.
"""
import json
import sqlite3
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    current_version_id INTEGER,           -- NULL until a first version is processed successfully
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_versions (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    version_no INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,              -- 'pdf' or 'txt'
    text_hash TEXT NOT NULL,              -- sha256 of normalized extracted text
    extracted_text TEXT NOT NULL,
    status TEXT NOT NULL,                 -- processing | active | superseded | failed
    error TEXT,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    UNIQUE (document_id, version_no)
);
CREATE TABLE IF NOT EXISTS passages (
    id TEXT PRIMARY KEY,                  -- e.g. 'v3-p2' (version 3, passage 2)
    version_id INTEGER NOT NULL REFERENCES document_versions(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),
    ordinal INTEGER NOT NULL,
    page INTEGER,                         -- PDF page number; NULL for TXT
    location TEXT NOT NULL,               -- 'Page 2, passage 1' or 'Passage 3'
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,    -- 0 = removed by user (history is kept)
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answer_runs (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES questions(id),
    trigger TEXT NOT NULL,                -- initial | update | manual
    trigger_version_id INTEGER,           -- document version whose upload caused this recheck
    status TEXT NOT NULL,                 -- answered | insufficient | needs_review | failed
    result_json TEXT,                     -- structured findings + evidence
    error TEXT,
    collection_snapshot TEXT NOT NULL,    -- JSON list of document versions searched
    comparison TEXT,                      -- vs previous run: unchanged | changed | review
    created_at TEXT NOT NULL
);
"""

MAX_SAVED_QUESTIONS = 5


def now():
    return datetime.now().isoformat(timespec="seconds")


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# ---------- documents and versions ----------

def list_active_documents(conn):
    return conn.execute(
        """SELECT d.id, d.name, v.id AS version_id, v.version_no, v.filename, v.file_type,
                  v.activated_at, (SELECT COUNT(*) FROM passages p WHERE p.version_id = v.id) AS passages
           FROM documents d JOIN document_versions v ON v.id = d.current_version_id
           ORDER BY d.id"""
    ).fetchall()


def get_document(conn, doc_id):
    return conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()


def get_document_by_name(conn, name):
    return conn.execute("SELECT * FROM documents WHERE name = ?", (name,)).fetchone()


def create_document(conn, name):
    cur = conn.execute("INSERT INTO documents (name, created_at) VALUES (?, ?)", (name, now()))
    conn.commit()
    return cur.lastrowid


def get_version(conn, version_id):
    return conn.execute("SELECT * FROM document_versions WHERE id = ?", (version_id,)).fetchone()


def versions_for_document(conn, doc_id):
    return conn.execute(
        "SELECT * FROM document_versions WHERE document_id = ? ORDER BY version_no", (doc_id,)
    ).fetchall()


def create_version(conn, doc_id, filename, file_type, text_hash, extracted_text):
    row = conn.execute(
        "SELECT COALESCE(MAX(version_no), 0) + 1 FROM document_versions WHERE document_id = ?", (doc_id,)
    ).fetchone()
    cur = conn.execute(
        """INSERT INTO document_versions
           (document_id, version_no, filename, file_type, text_hash, extracted_text, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'processing', ?)""",
        (doc_id, row[0], filename, file_type, text_hash, extracted_text, now()),
    )
    conn.commit()
    return cur.lastrowid


def add_passages(conn, version_id, doc_id, passages):
    """passages: list of dicts with id, ordinal, page, location, text."""
    conn.executemany(
        "INSERT INTO passages (id, version_id, document_id, ordinal, page, location, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(p["id"], version_id, doc_id, p["ordinal"], p["page"], p["location"], p["text"]) for p in passages],
    )
    conn.commit()


def activate_version(conn, doc_id, version_id):
    """Make version_id the current version in one transaction; the old one becomes superseded."""
    with conn:
        conn.execute(
            """UPDATE document_versions SET status = 'superseded'
               WHERE id = (SELECT current_version_id FROM documents WHERE id = ?)""",
            (doc_id,),
        )
        conn.execute(
            "UPDATE document_versions SET status = 'active', activated_at = ? WHERE id = ?", (now(), version_id)
        )
        conn.execute("UPDATE documents SET current_version_id = ? WHERE id = ?", (version_id, doc_id))


def fail_version(conn, version_id, error):
    """Mark a version failed and drop its passages. The document's current version is untouched."""
    with conn:
        conn.execute("DELETE FROM passages WHERE version_id = ?", (version_id,))
        conn.execute(
            "UPDATE document_versions SET status = 'failed', error = ? WHERE id = ?", (error, version_id)
        )


def current_version_ids(conn):
    rows = conn.execute("SELECT current_version_id FROM documents WHERE current_version_id IS NOT NULL")
    return [r[0] for r in rows]


def collection_snapshot(conn):
    """The exact document versions that a question was answered against."""
    return [
        {"document_id": d["id"], "document": d["name"], "version_id": d["version_id"], "version_no": d["version_no"]}
        for d in list_active_documents(conn)
    ]


def replacement_versions(conn):
    """Versions that replaced an earlier version and were activated (newest first)."""
    return conn.execute(
        """SELECT v.*, d.name AS document_name FROM document_versions v
           JOIN documents d ON d.id = v.document_id
           WHERE v.version_no > 1 AND v.status IN ('active', 'superseded')
           ORDER BY v.activated_at DESC, v.id DESC"""
    ).fetchall()


def previous_active_version(conn, doc_id, version_no):
    """The successfully activated version just before version_no (skipping failed uploads)."""
    return conn.execute(
        """SELECT * FROM document_versions WHERE document_id = ? AND version_no < ?
           AND status IN ('active', 'superseded') ORDER BY version_no DESC LIMIT 1""",
        (doc_id, version_no),
    ).fetchone()


# ---------- passages ----------

def get_passages(conn, passage_ids):
    """Return {passage_id: row} including document name and version number."""
    if not passage_ids:
        return {}
    marks = ",".join("?" * len(passage_ids))
    rows = conn.execute(
        f"""SELECT p.*, d.name AS document_name, v.version_no FROM passages p
            JOIN documents d ON d.id = p.document_id
            JOIN document_versions v ON v.id = p.version_id
            WHERE p.id IN ({marks})""",
        list(passage_ids),
    ).fetchall()
    return {r["id"]: r for r in rows}


def passages_for_version(conn, version_id):
    return conn.execute(
        "SELECT * FROM passages WHERE version_id = ? ORDER BY ordinal", (version_id,)
    ).fetchall()


def current_passages(conn):
    return conn.execute(
        """SELECT p.* FROM passages p JOIN documents d ON d.current_version_id = p.version_id"""
    ).fetchall()


# ---------- questions and answer runs ----------

def saved_questions(conn):
    return conn.execute("SELECT * FROM questions WHERE active = 1 ORDER BY id").fetchall()


def add_question(conn, text):
    if len(saved_questions(conn)) >= MAX_SAVED_QUESTIONS:
        raise ValueError(f"You can save at most {MAX_SAVED_QUESTIONS} questions. Remove one first.")
    cur = conn.execute("INSERT INTO questions (text, created_at) VALUES (?, ?)", (text, now()))
    conn.commit()
    return cur.lastrowid


def remove_question(conn, question_id):
    """Hide a saved question. Its answer history stays in the database."""
    conn.execute("UPDATE questions SET active = 0 WHERE id = ?", (question_id,))
    conn.commit()


def add_run(conn, question_id, trigger, status, result, error, snapshot, trigger_version_id=None, comparison=None):
    cur = conn.execute(
        """INSERT INTO answer_runs (question_id, trigger, trigger_version_id, status, result_json, error,
                                    collection_snapshot, comparison, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (question_id, trigger, trigger_version_id, status,
         json.dumps(result) if result is not None else None, error, json.dumps(snapshot), comparison, now()),
    )
    conn.commit()
    return cur.lastrowid


def _decode_run(row):
    if row is None:
        return None
    run = dict(row)
    run["result"] = json.loads(run["result_json"]) if run["result_json"] else None
    run["collection_snapshot"] = json.loads(run["collection_snapshot"])
    return run


def runs_for_question(conn, question_id):
    rows = conn.execute(
        "SELECT * FROM answer_runs WHERE question_id = ? ORDER BY id DESC", (question_id,)
    ).fetchall()
    return [_decode_run(r) for r in rows]


def latest_run(conn, question_id, before_run_id=None, successful_only=False):
    sql = "SELECT * FROM answer_runs WHERE question_id = ?"
    params = [question_id]
    if before_run_id is not None:
        sql += " AND id < ?"
        params.append(before_run_id)
    if successful_only:
        sql += " AND status != 'failed'"
    return _decode_run(conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone())


def run_for_trigger(conn, question_id, trigger_version_id):
    row = conn.execute(
        "SELECT * FROM answer_runs WHERE question_id = ? AND trigger_version_id = ? ORDER BY id DESC LIMIT 1",
        (question_id, trigger_version_id),
    ).fetchone()
    return _decode_run(row)
