# EvidenceGap AI

Prototype for **PS 237 — Multi-Document Evidence Coordination Agent** (Generative AI semester-end hackathon).

EvidenceGap AI helps college students check scholarship eligibility and application requirements
across several documents. It answers by connecting evidence from different documents, shows the
exact supporting quotes with document name, version and page/passage, and **abstains** when the
evidence is incomplete.

**Our extra feature:** save up to five important questions. When you upload a revised document, the
app automatically rechecks **all** saved questions and shows the previous and updated answers side by
side with their evidence. (It reruns every saved question. It does not track which questions depend
on which document.)

> Privacy: document text and questions are sent to the OpenAI API for embeddings and answers.

---

## Setup (Windows, PowerShell)

```powershell
cd evidencegap
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then edit .env
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env`

### Environment variables (in `.env`)

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key. Kept only in `.env`, which git ignores. Never shown in the UI. |
| `OPENAI_CHAT_MODEL` | A chat model that supports Structured Outputs through the Responses API. The configured value is `gpt-6.1-sol`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, used for both documents and queries. Example: `text-embedding-3-small`. |
| `EVIDENCEGAP_DATA_DIR` | Optional. Where runtime data is stored. Defaults to `data/`. |

If any of the three OpenAI variables is missing, the app shows setup instructions and does not
produce answers. It never shows fabricated results.

## Run

```powershell
streamlit run app.py
```

Runtime data is kept in `data/`: the SQLite database `evidencegap.db` and the Chroma index
`chroma/`. It survives restarts. To start fresh, stop the app and delete `data/`.

## Run the offline tests

```powershell
python -m pytest -q tests
```

The tests use a fake model and fake embeddings. They make no network calls and cost nothing.

---

## How it works

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface with three tabs: Documents, Questions, Changes |
| `documents.py` | PDF/TXT extraction, passage splitting, hashing, Chroma index, adding and replacing documents, retrieval of current versions only |
| `answering.py` | The bounded question-answering steps, citation validation, answer status, rechecks and comparison |
| `storage.py` | SQLite tables: documents, document_versions, passages, questions, answer_runs |
| `ai_client.py` | Thin OpenAI SDK wrapper with a finite timeout (60 s), retries set to 0 (`MAX_RETRIES`) and clear error messages |

**SQLite** is the record of versions and history. **Chroma** is only the search index. Every
version's passages stay in both, so old answers can still show their citations. Every search is
filtered to the *current* version IDs taken from SQLite.

### Document processing
- **PDF:** text is extracted page by page with pypdf. A passage never crosses a page boundary, so
  every passage has an exact page number. If no page has selectable text, the PDF is rejected as
  scanned ("OCR is not supported").
- **TXT:** must be UTF-8. Locations are "Passage N". Page numbers are never invented for TXT files.
- Text is split into roughly 800–1200 character passages with about 150 characters of overlap.
- A SHA-256 hash of the normalized text (Unicode NFKC, unified quotes, collapsed whitespace)
  detects identical content. An identical replacement reports **"No text changes"**, and nothing
  is stored, re-indexed or rechecked.
- A new version is activated only after its passages are stored and embedded. If any step fails,
  the version is marked `failed` and the previous version stays current.
- Limits: 5 active documents, 5 MB per file, 5 saved questions.
- Indexing happens only when the **Add** or **Upload replacement** button is clicked. Streamlit
  reruns do not trigger it, and a hash check plus Chroma `upsert` prevent duplicate entries.

### Question answering: a fixed sequence of steps, at most two retrieval rounds
1. **Information needs.** The model lists the facts needed, each with a search query.
2. **Retrieval round 1.** The question and every need are searched in the current documents.
3. **Dependent lookup (round 2, only if needed).** The model names a fact established in a round-1
   passage, such as a committee name, and a follow-up query that uses it, such as that committee's
   contact. The follow-up is accepted only if it points to a passage that was actually retrieved.
4. **Combine.** Structured output (strict JSON schema) contains: findings per required item with
   passage IDs and exact quotes, missing information, personal details needed, potential conflicts
   (each assessed as `conflict`, `unclear` or `not_a_conflict`), an explanation of how the evidence
   connects, and a proposed answer.
5. **Validate in Python:**
   - the output has the expected structure;
   - each cited passage ID exists;
   - the passage belongs to a *current* document version;
   - the quote appears in that passage after normalizing whitespace, quotes and case.

   Document names, versions and page/passage labels are filled in from SQLite, not from model text.
   A **model support check** then asks whether each verified quote actually supports its finding.
6. **Status:**
   - **Answered:** every required item has a supported finding and no conflict is unresolved.
   - **Insufficient evidence:** at least one required item is unsupported, or personal details are
     needed. No definitive answer is shown. Only the supported partial findings and the missing
     items are listed.
   - **Needs review:** sources appear to conflict, or their meaning is unclear.

   No confidence percentages are shown.

Uploaded documents are treated as **untrusted data**. Passages are wrapped in `<passage>` tags, and
every prompt states that instructions inside passages must be ignored and that general knowledge
must not fill gaps.

### Document updates and rechecks
After a changed replacement is activated, **every saved question** is rechecked against the current
collection. Each recheck is saved as a new answer run with a snapshot of the document versions it
used. No answer history is ever deleted. The **Changes** tab shows, for each update:

- a line diff of the document text (Python `difflib`);
- the answer before and after, each with its quotes and versions;
- a comparison label computed in Python, which ignores rephrasing:
  - **Changed:** the status changed, or a quote from the old answer no longer appears in the
    current documents.
  - **Unchanged:** same status and the same set of quotes.
  - **Rechecked — review comparison:** anything else, such as different quotes chosen from
    unchanged text.
- **Recheck failed:** an API error during a recheck. The old answer is kept and marked *not
  verified against the new documents*.
- **Pending:** a saved question with no recheck for this update yet, for example if the app
  stopped midway. A **Run pending rechecks** button is provided.

API failures are always shown as failures. They are never reported as "insufficient evidence".

---

## Demo

The `demo/` folder holds **clearly labelled fictional** documents about a fictional "Northbridge
Institute of Technology". These are not real university rules. To regenerate them, run
`python demo/make_demo_files.py`.

| File | Content |
|---|---|
| `1_eligibility_rules.pdf` (2 pages) | Page 1: final-year B.Tech CSE students may apply. Page 2: minimum CGPA 7.5. |
| `2_application_guide.txt` | Required documents are a transcript and student ID. Queries are handled by the *Merit Scholarship Review Committee*. |
| `3_deadline_notice_v1.pdf` | Deadline 10 December 2026, plus the contact for the Merit Scholarship Review Committee (Dr. Asha Verma, msrc@northbridge.example, Room B-214) |
| `3_deadline_notice_v2.pdf` | Replacement: deadline 15 December 2026. Everything else is unchanged. |

None of the documents states a scholarship amount, so the app should abstain on that question.

### Demo script
1. **Documents tab:** add the three files `1_…`, `2_…` and `3_deadline_notice_v1.pdf`.
2. **Questions tab:** ask and save each question below.
3. **Documents tab → Replace a document:** select `3_deadline_notice_v1.pdf` and upload
   `3_deadline_notice_v2.pdf`. All saved questions are rechecked.
   (Uploading `3_deadline_notice_v1.pdf` again as a replacement shows "No text changes".)
4. **Changes tab:** compare the answers before and after.
5. Restart `streamlit run app.py`. Documents, questions and history are still there.

| # | Question | Intended result |
|---|---|---|
| 1 | Can a final-year CSE student with CGPA 7.95 apply, what documents are required, and when is the deadline? | Answered, combining all three documents: eligible (7.95 ≥ 7.5, final-year CSE); transcript and student ID; 10 December 2026 |
| 2 | When is the application deadline? | 10 December 2026 before the update and 15 December 2026 after. Expected label: Changed. |
| 3 | Who handles queries for the committee named in the application guide? | Merit Scholarship Review Committee (guide) → Dr. Asha Verma / msrc@northbridge.example (deadline notice), using a round-2 lookup |
| 4 | What is the scholarship amount? | Insufficient evidence, with no amount given |
| 5 | Which documents must I submit? | Transcript and student ID. Expected label after the update: Unchanged (or "review comparison" if the model picks different quotes). |

These are the **intended** outcomes. The full scenario has been checked offline with a mocked model
(see `tests/test_core.py`), but real model behaviour can vary, so verify it with a live run.

---

## Known limitations
- Live answer quality depends on the chosen OpenAI model. Retrieval quality depends on the
  embedding model.
- Matching an exact quote proves the text exists. The **support check is done by a model and can be
  wrong**, and conflict detection is also model-based.
- Quote matching ignores case and whitespace differences.
- Rechecks run every saved question, one after another, while the page waits. There is no
  selective dependency tracking and no background queue.
- Comparison labels are a heuristic based on status and quotes. "Rechecked — review comparison"
  means a person should look.
- There is no OCR, no login and a single local user. Scanned PDFs are rejected. Encrypted PDFs are
  rejected.
- Complex PDF layouts (tables, multiple columns) may extract in an awkward order.
- Prompt-injection defences are instructions to the model, so they reduce risk but cannot fully
  prevent it.
- Removing a saved question hides it. Its history stays in the database.
