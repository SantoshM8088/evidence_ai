"""Offline tests. No network: embeddings and model responses come from FakeAI."""
import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import answering  # noqa: E402
import documents  # noqa: E402
import storage  # noqa: E402
from ai_client import AIServiceError  # noqa: E402
from make_demo_files import simple_pdf_bytes  # noqa: E402

DEMO = ROOT / "demo"


class FakeAI:
    """Deterministic stand-in for OpenAIClient.

    `facts` is a list of (requirement_index, statement, quote). The fake finds the retrieved
    passage containing each quote and cites it, like a well-behaved model would.
    """

    def __init__(self, needs, facts=(), missing=(), dependent=None, conflicts=(), supported=True,
                 extra_citations=()):
        self.needs, self.facts, self.missing = list(needs), list(facts), list(missing)
        self.dependent = dependent          # (established_fact_quote, follow-up query) or None
        self.conflicts, self.supported = list(conflicts), supported
        self.extra_citations = list(extra_citations)  # raw citations to inject (for validation tests)
        self.embed_calls, self.calls = 0, []
        self.fail_embed = False

    def embed(self, texts):
        if self.fail_embed:
            raise AIServiceError("simulated embedding outage")
        self.embed_calls += 1
        vectors = []
        for text in texts:  # hashed bag-of-words, enough for meaningful similarity
            v = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors

    @staticmethod
    def _passages(user_input):
        return re.findall(r'<passage id="([^"]+)"[^>]*>\n(.*?)\n</passage>', user_input, re.S)

    def _find(self, user_input, quote):
        for pid, text in self._passages(user_input):
            if documents.normalize_text(quote) in documents.normalize_text(text):
                return pid
        return None

    def generate_json(self, name, instructions, user_input, schema):
        self.calls.append(name)
        if name == "information_needs":
            return {"information_needs": [{"need": n, "search_query": n} for n in self.needs],
                    "needs_dependent_lookup": self.dependent is not None}
        if name == "dependent_lookups":
            fact, query = self.dependent
            pid = self._find(user_input, fact)
            return {"followups": [{"established_fact": fact, "source_passage_id": pid, "search_query": query}]
                    if pid else []}
        if name == "evidence_answer":
            findings = []
            for idx, statement, quote in self.facts:
                pid = self._find(user_input, quote)
                if pid:
                    findings.append({"requirement_index": idx, "statement": statement,
                                     "citations": [{"passage_id": pid, "quote": quote}]})
            for idx, statement, pid, quote in self.extra_citations:
                findings.append({"requirement_index": idx, "statement": statement,
                                 "citations": [{"passage_id": pid, "quote": quote}]})
            return {"findings": findings, "missing_information": self.missing, "personal_details_needed": [],
                    "conflicts": self.conflicts, "connection_explanation": "fake explanation",
                    "proposed_answer": "fake proposed answer"}
        if name == "support_check":
            n = len(re.findall(r"^Finding \d+:", user_input, re.M))
            return {"checks": [{"finding_index": i, "supported": self.supported, "reason": "fake"} for i in range(n)]}
        raise AssertionError(f"unexpected call {name}")


@pytest.fixture
def env(tmp_path):
    conn = storage.connect(str(tmp_path / "test.db"))
    index = documents.PassageIndex(str(tmp_path / "chroma"), "fake-embedding")
    return conn, index, tmp_path


def load_demo(conn, index, ai):
    for name in ["1_eligibility_rules.pdf", "2_application_guide.txt", "3_deadline_notice_v1.pdf"]:
        documents.add_document(conn, index, ai, name, (DEMO / name).read_bytes())
    return storage.get_document_by_name(conn, "3_deadline_notice_v1.pdf")["id"]


DEADLINE_AI = dict(needs=["application deadline"],
                   facts=[(0, "The deadline is 10 December 2026.", "is 10 December 2026")])
DOCS_AI = dict(needs=["required documents"],
               facts=[(0, "A transcript and student ID are required.", "an official academic transcript")])


# ---------- document processing ----------

def test_extraction_preserves_source_locations():
    _, _, pdf = documents.build_passages("rules.pdf", (DEMO / "1_eligibility_rules.pdf").read_bytes())
    assert [p["page"] for p in pdf] == [1, 2]
    assert "Final-year" in pdf[0]["text"] and "7.5" in pdf[1]["text"]
    assert pdf[1]["location"] == "Page 2"

    _, _, txt = documents.build_passages("guide.txt", (DEMO / "2_application_guide.txt").read_bytes())
    assert all(p["page"] is None and p["location"].startswith("Passage ") for p in txt)


def test_scanned_pdf_is_rejected_with_ocr_message():
    with pytest.raises(documents.DocumentError, match="OCR is not supported"):
        documents.build_passages("scan.pdf", simple_pdf_bytes([[], []]))


def test_split_text_bounded_with_overlap():
    text = " ".join(f"word{i}" for i in range(1500))
    chunks = documents.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= documents.PASSAGE_MAX for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # small overlap between neighbours


# ---------- evidence checking ----------

def test_invalid_passage_ids_and_fabricated_quotes_are_rejected(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    load_demo(conn, index, ai)
    real_pid = storage.passages_for_version(conn, storage.current_version_ids(conn)[0])[0]["id"]
    ai.extra_citations = [(0, "Invented passage", "v999-p1", "anything"),
                          (0, "Fabricated quote", real_pid, "The scholarship is worth 50,000 rupees")]
    result = answering.answer_question(conn, index, ai, "When is the deadline?")
    reasons = " | ".join(r["reason"] for r in result["rejected"])
    assert "does not exist" in reasons and "quote not found" in reasons
    assert all(f["statement"] not in ("Invented passage", "Fabricated quote") for f in result["findings"])
    # labels come from stored metadata
    cite = result["findings"][0]["citations"][0]
    assert cite["document"] == "3_deadline_notice_v1.pdf" and cite["version_no"] == 1 and cite["location"] == "Page 1"


def test_missing_evidence_triggers_abstention(env):
    conn, index, _ = env
    ai = FakeAI(needs=["scholarship amount"], missing=["The documents do not state a scholarship amount."])
    load_demo(conn, index, ai)
    result = answering.answer_question(conn, index, ai, "What is the scholarship amount?")
    assert result["status"] == "insufficient"
    assert result["proposed_answer"] == ""  # no definitive answer when evidence is missing
    assert result["missing_information"]


def test_partial_evidence_is_insufficient_but_keeps_supported_findings(env):
    conn, index, _ = env
    ai = FakeAI(needs=["application deadline", "scholarship amount"], facts=DEADLINE_AI["facts"])
    load_demo(conn, index, ai)
    result = answering.answer_question(conn, index, ai, "Deadline and amount?")
    assert result["status"] == "insufficient"
    assert len(result["findings"]) == 1
    assert any("scholarship amount" in m for m in result["missing_information"])


def test_unsupported_quote_fails_model_support_check(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI, supported=False)
    load_demo(conn, index, ai)
    result = answering.answer_question(conn, index, ai, "When is the deadline?")
    assert result["status"] == "insufficient" and not result["findings"]


def test_conflict_marks_needs_review_but_not_a_conflict_does_not(env):
    conn, index, _ = env
    conflict = {"description": "two deadlines", "passage_ids": [], "assessment": "conflict"}
    ai = FakeAI(**DEADLINE_AI, conflicts=[conflict])
    load_demo(conn, index, ai)
    assert answering.answer_question(conn, index, ai, "Deadline?")["status"] == "needs_review"
    ai.conflicts = [{**conflict, "assessment": "not_a_conflict"}]
    assert answering.answer_question(conn, index, ai, "Deadline?")["status"] == "answered"


def test_multi_hop_question_connects_two_documents(env):
    conn, index, _ = env
    ai = FakeAI(
        needs=["committee named in the application guide", "contact for that committee"],
        dependent=("Merit Scholarship Review Committee", "contact for the Merit Scholarship Review Committee"),
        facts=[(0, "Queries go to the Merit Scholarship Review Committee.",
                "All scholarship queries are handled by the Merit Scholarship Review Committee."),
               (1, "The committee coordinator is Dr. Asha Verma.", "Coordinator: Dr. Asha Verma")])
    load_demo(conn, index, ai)
    result = answering.answer_question(conn, index, ai, "Who handles queries for the committee named in the guide?")
    assert result["status"] == "answered"
    assert result["retrieval"]["round2"], "a second, dependent retrieval round should run"
    docs = {c["document"] for f in result["findings"] for c in f["citations"]}
    assert docs == {"2_application_guide.txt", "3_deadline_notice_v1.pdf"}


def test_api_error_is_not_reported_as_missing_evidence(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    load_demo(conn, index, ai)
    qid = storage.add_question(conn, "When is the deadline?")
    ai.fail_embed = True
    run_id = answering.run_and_save(conn, index, ai, qid, "initial")
    run = storage.latest_run(conn, qid)
    assert run["id"] == run_id and run["status"] == "failed" and "outage" in run["error"]
    with pytest.raises(AIServiceError):
        answering.answer_question(conn, index, ai, "When is the deadline?")


# ---------- versions and updates ----------

def test_retrieval_excludes_superseded_versions_but_keeps_history(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    doc_id = load_demo(conn, index, ai)
    old_version = storage.get_document(conn, doc_id)["current_version_id"]
    out = documents.replace_document(conn, index, ai, doc_id, "3_deadline_notice_v2.pdf",
                                     (DEMO / "3_deadline_notice_v2.pdf").read_bytes())
    assert out["changed"]
    rows = documents.retrieve(conn, index, ai, ["application deadline December"], k=10)
    assert rows and all(r["version_id"] != old_version for r in rows)
    assert any("15 December 2026" in r["text"] for r in rows)
    assert not any("10 December 2026" in r["text"] for r in rows)
    # superseded passages are still stored for old citations
    assert "10 December 2026" in storage.passages_for_version(conn, old_version)[0]["text"]


def test_identical_replacement_skips_indexing_and_rechecks(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    doc_id = load_demo(conn, index, ai)
    storage.add_question(conn, "When is the deadline?")
    embeds_before = ai.embed_calls
    out = documents.replace_document(conn, index, ai, doc_id, "copy.pdf", (DEMO / "3_deadline_notice_v1.pdf").read_bytes())
    assert out == {"changed": False}
    assert ai.embed_calls == embeds_before
    assert len(storage.versions_for_document(conn, doc_id)) == 1
    assert storage.replacement_versions(conn) == []


def test_update_rechecks_all_saved_questions_and_keeps_old_answers(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    doc_id = load_demo(conn, index, ai)
    q_deadline = storage.add_question(conn, "When is the application deadline?")
    answering.run_and_save(conn, index, ai, q_deadline, "initial")
    ai_docs = FakeAI(**DOCS_AI)
    q_docs = storage.add_question(conn, "Which documents must I submit?")
    answering.run_and_save(conn, index, ai_docs, q_docs, "initial")
    old_deadline_run = storage.latest_run(conn, q_deadline)

    out = documents.replace_document(conn, index, ai, doc_id, "3_deadline_notice_v2.pdf",
                                     (DEMO / "3_deadline_notice_v2.pdf").read_bytes())
    assert len(answering.pending_rechecks(conn, out["version_id"])) == 2

    # A model that would now answer with the new deadline; documents question uses its own fake.
    ai.facts = [(0, "The deadline is 15 December 2026.", "is 15 December 2026")]
    calls = {"n": 0}

    class Router:  # route each question to the matching fake model
        def embed(self, texts):
            return ai.embed(texts)

        def generate_json(self, name, instructions, user_input, schema):
            target = ai_docs if "documents must I submit" in user_input else ai
            calls["n"] += 1
            return target.generate_json(name, instructions, user_input, schema)

    run_ids = answering.recheck_all(conn, index, Router(), out["version_id"])
    assert len(run_ids) == 2 and calls["n"] > 0
    assert answering.pending_rechecks(conn, out["version_id"]) == []

    new_deadline = storage.run_for_trigger(conn, q_deadline, out["version_id"])
    new_docs = storage.run_for_trigger(conn, q_docs, out["version_id"])
    assert new_deadline["comparison"] == "changed"
    assert new_docs["comparison"] == "unchanged"
    assert new_deadline["collection_snapshot"] != old_deadline_run["collection_snapshot"]

    # previous answer and its v1 citation remain available
    history = storage.runs_for_question(conn, q_deadline)
    assert len(history) == 2
    old_cite = history[1]["result"]["findings"][0]["citations"][0]
    assert old_cite["version_no"] == 1
    assert "10 December 2026" in storage.get_passages(conn, [old_cite["passage_id"]])[old_cite["passage_id"]]["text"]


def test_failed_replacement_preserves_previous_usable_version(env):
    conn, index, _ = env
    ai = FakeAI(**DEADLINE_AI)
    doc_id = load_demo(conn, index, ai)
    v1 = storage.get_document(conn, doc_id)["current_version_id"]
    ai.fail_embed = True
    with pytest.raises(AIServiceError):
        documents.replace_document(conn, index, ai, doc_id, "3_deadline_notice_v2.pdf",
                                   (DEMO / "3_deadline_notice_v2.pdf").read_bytes())
    ai.fail_embed = False
    assert storage.get_document(conn, doc_id)["current_version_id"] == v1
    statuses = [v["status"] for v in storage.versions_for_document(conn, doc_id)]
    assert statuses == ["active", "failed"]
    rows = documents.retrieve(conn, index, ai, ["application deadline"], k=10)
    assert any("10 December 2026" in r["text"] for r in rows)

    # a later successful replacement still works and skips the failed version number
    out = documents.replace_document(conn, index, ai, doc_id, "3_deadline_notice_v2.pdf",
                                     (DEMO / "3_deadline_notice_v2.pdf").read_bytes())
    assert storage.get_version(conn, out["version_id"])["version_no"] == 3
    assert storage.previous_active_version(conn, doc_id, 3)["id"] == v1


def test_data_survives_restart(env):
    conn, index, tmp_path = env
    ai = FakeAI(**DEADLINE_AI)
    load_demo(conn, index, ai)
    qid = storage.add_question(conn, "When is the deadline?")
    answering.run_and_save(conn, index, ai, qid, "initial")
    conn.close()

    conn2 = storage.connect(str(tmp_path / "test.db"))
    assert len(storage.list_active_documents(conn2)) == 3
    assert storage.latest_run(conn2, qid)["status"] == "answered"


def test_limits_enforced(env):
    conn, index, _ = env
    for i in range(storage.MAX_SAVED_QUESTIONS):
        storage.add_question(conn, f"q{i}")
    with pytest.raises(ValueError):
        storage.add_question(conn, "one too many")
    with pytest.raises(documents.DocumentError, match="5 MB"):
        documents.build_passages("big.txt", b"a" * (documents.MAX_FILE_BYTES + 1))
