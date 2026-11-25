"""EvidenceGap AI - Streamlit interface.  Run:  streamlit run app.py"""
import difflib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import answering
import documents
import storage
from ai_client import AIServiceError, OpenAIClient, load_config

load_dotenv()
DATA_DIR = Path(os.environ.get("EVIDENCEGAP_DATA_DIR") or Path(__file__).parent / "data")
STATUS_ICONS = {"answered": "✅", "insufficient": "⚠️", "needs_review": "🔍", "failed": "❌"}
STATUS_COLORS = {"answered": "green", "insufficient": "orange", "needs_review": "violet", "failed": "red"}
COMPARISON_LABELS = {
    "unchanged": "Unchanged — same status and same evidence",
    "changed": "Changed — status or supporting evidence changed",
    "review": "Rechecked — review comparison",
}
RECHECK_NOTE = "All saved questions are rechecked after each document update."

# Minimal styling: navy sidebar, blue primary buttons, larger answer text, tighter page padding.
CSS = """
<style>
.block-container {padding-top: 4rem; padding-bottom: 2rem;}  /* clears Streamlit's fixed 3.75rem header */
[data-testid="stTab"][data-selected], [data-testid="stTab"][data-hovered],
[data-testid="stTab"][data-focus-visible] {color: #1f5eff;}
[data-testid="stTabs"] .react-aria-SelectionIndicator {background-color: #1f5eff;}
[data-testid="stSidebar"] {background: #0f1f3d;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stFileUploader"] *,
[data-testid="stSidebarHeader"] * {color: #e6ecf7;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileChip"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileChip"] * {color: #1b2433;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {opacity: 0.75;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {border-color: rgba(255, 255, 255, 0.25);}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: transparent; border: 1.5px dashed rgba(255, 255, 255, 0.45);}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent; color: #e6ecf7; border-color: rgba(255, 255, 255, 0.4);}
[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color: inherit;}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: #1f5eff; border-color: #1f5eff; color: #ffffff;}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: #174bd1; border-color: #174bd1; color: #ffffff;}
[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] p {font-size: 1.3rem; font-weight: 700; line-height: 1.4;}
</style>
"""

st.set_page_config(page_title="EvidenceGap AI", page_icon="📑", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_conn(data_dir):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return storage.connect(str(Path(data_dir) / "evidencegap.db"))


@st.cache_resource
def get_ai_and_index(data_dir, api_key, chat_model, embedding_model):
    return OpenAIClient(api_key, chat_model, embedding_model), \
        documents.PassageIndex(str(Path(data_dir) / "chroma"), embedding_model)


conn = get_conn(str(DATA_DIR))
config, missing_config = load_config()

if missing_config:
    st.error("OpenAI is not configured, so documents cannot be processed and questions cannot be answered.")
    st.markdown(
        "**Setup:** copy `.env.example` to `.env` in the project folder and fill in:\n\n"
        + "\n".join(f"- `{name}`" for name in missing_config)
        + "\n\nThen restart the app with `streamlit run app.py`."
    )
    ai, index = None, None
else:
    ai, index = get_ai_and_index(str(DATA_DIR), config["OPENAI_API_KEY"], config["OPENAI_CHAT_MODEL"],
                                 config["OPENAI_EMBEDDING_MODEL"])


def require_ai():
    if ai is None:
        st.warning("Configure OpenAI first (see setup instructions).")
        return False
    return True


# ---------- rendering helpers ----------

def citation_label(c):
    return f"{c['document']} · v{c['version_no']} · {c['location']}"


def short(text, limit=110):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_result(result, key):
    status = result["status"]
    st.badge(answering.STATUS_LABELS[status], icon=STATUS_ICONS[status], color=STATUS_COLORS[status])
    if status == "answered":
        with st.container(key=f"answer-{key}"):
            st.markdown(result["proposed_answer"])
    elif status == "insufficient":
        st.warning("No complete answer: the documents do not support every part of this question.")
    else:
        st.warning("No definitive answer: the sources may conflict or are unclear. Review the evidence below.")

    if result["personal_details_needed"]:
        st.markdown("**Please tell me these details so eligibility can be checked:**")
        for d in result["personal_details_needed"]:
            st.markdown(f"- {d}")
    if result["missing_information"]:
        st.markdown("**Missing information:**")
        for m in result["missing_information"]:
            st.markdown(f"- {m}")
    real_conflicts = [c for c in result["conflicts"] if c["assessment"] != "not_a_conflict"]
    if real_conflicts:
        st.markdown("**Potential conflicts:**")
        for c in real_conflicts:
            st.markdown(f"- ({c['assessment']}) {c['description']} — passages: {', '.join(c['passage_ids']) or 'n/a'}")

    if result["findings"]:
        st.markdown("**Supporting sources**" if status == "answered" else "**Supported partial findings**")
        cited = {c["passage_id"] for f in result["findings"] for c in f["citations"]}
        passages = storage.get_passages(conn, cited)
        for f in result["findings"]:
            st.markdown(f"✔ {f['statement']}")
            for c in f["citations"]:
                with st.expander(f"📄 **{citation_label(c)}** — “{short(c['quote'])}”"):
                    st.markdown(f"> {c['quote']}")
                    p = passages.get(c["passage_id"])
                    if p:
                        st.caption(f"Full passage `{c['passage_id']}`")
                        st.text(p["text"])
    if result["connection_explanation"]:
        st.markdown(f"**Evidence connection:** {result['connection_explanation']}")

    with st.expander("Retrieval details"):
        r = result["retrieval"]
        st.markdown("**Information required:** " + "; ".join(
            f"{'✔' if n.get('supported') else '✘'} {n['need']}" for n in result["information_required"]))
        st.markdown("**Round 1 searches:** " + "; ".join(r["round1_queries"]))
        if r["round2"]:
            for fu in r["round2"]:
                st.markdown(f"**Round 2 search:** `{fu['search_query']}` — using established fact "
                            f"“{fu['established_fact']}” from `{fu['source_passage_id']}`")
        else:
            st.markdown("**Round 2:** not needed")
        if result["rejected"]:
            st.markdown("**Rejected by validation:**")
            for rej in result["rejected"]:
                st.markdown(f"- {rej['statement']} — *{rej['reason']}*")
        st.caption("Each quote is checked to appear in its cited current passage, ignoring differences in "
                   "whitespace, letter case and quote/dash style. A separate model check judges whether each "
                   "quote supports its finding; that check can still be wrong.")


def render_run(run, key):
    snapshot = ", ".join(f"{s['document']} v{s['version_no']}" for s in run["collection_snapshot"])
    st.caption(f"Run #{run['id']} · {run['created_at']} · trigger: {run['trigger']} · documents: {snapshot}")
    if run["status"] == "failed":
        what = "Answer generation failed" if run["trigger"] == "initial" else "Recheck failed"
        st.error(f"❌ {what}: {run['error']}")
    else:
        render_result(run["result"], key)


# ---------- sidebar: documents ----------

active = storage.list_active_documents(conn)

with st.sidebar:
    st.markdown("## 📑 EvidenceGap AI")
    st.caption("Document evidence assistant")
    st.divider()

    # Outcome of the last add/replace, kept across the refresh that follows it.
    notice = st.session_state.pop("doc_notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])

    st.markdown(f"#### Documents {len(active)}/{documents.MAX_ACTIVE_DOCUMENTS}")
    new_file = st.file_uploader("Upload PDF or TXT (maximum 5 MB per file)", type=["pdf", "txt"], key="new_file")
    if st.button("Add document", icon=":material/add:", type="primary", width="stretch",
                 disabled=new_file is None) and require_ai():
        added = False
        with st.status(f"Processing {new_file.name}...", expanded=True) as status:
            try:
                documents.add_document(conn, index, ai, new_file.name, new_file.getvalue(), progress=st.write)
                status.update(label=f"Added {new_file.name}", state="complete")
                added = True
            except documents.DocumentError as exc:
                status.update(label="Upload rejected", state="error")
                st.error(str(exc))
            except AIServiceError as exc:
                status.update(label="OpenAI error — document not added", state="error")
                st.error(str(exc))
        if added:
            # The collection changed: drop an unsaved answer (it is never rechecked) and refresh the
            # sidebar. A rerun does not re-click the button, so no API call is repeated.
            if st.session_state.get("last_answer", ("",))[0] == "result":
                del st.session_state["last_answer"]
            st.session_state["doc_notice"] = ("success", f"Added {new_file.name}")
            st.rerun()

    with st.expander("🔒 Privacy: text is sent to OpenAI"):
        st.markdown("Document text and your questions are sent to the OpenAI API to create embeddings "
                    "and answers. Do not upload confidential documents. AI answers can be wrong — always "
                    "check the quoted passages.")

    if active:
        for d in active:
            st.markdown(f"📄 **{d['name']}** `v{d['version_no']}`")
            details = f"{d['file_type'].upper()} · {d['passages']} passages"
            if d["filename"] != d["name"]:
                details += f" · current file: {d['filename']}"
            st.caption(details)
    else:
        st.caption("No documents yet. Upload up to five PDF or TXT files.")

    with st.expander("🔄 Replace a document"):
        if active:
            st.caption(RECHECK_NOTE)
            by_id = {d["id"]: d for d in active}
            choice = by_id[st.selectbox("Document to replace", list(by_id),
                                        format_func=lambda i: f"{by_id[i]['name']} (v{by_id[i]['version_no']})")]
            repl_file = st.file_uploader("Revised version", type=["pdf", "txt"], key="replacement")
            if st.button("Upload replacement", width="stretch", disabled=repl_file is None) and require_ai():
                with st.status(f"Processing replacement for {choice['name']}...", expanded=True) as status:
                    try:
                        outcome = documents.replace_document(conn, index, ai, choice["id"], repl_file.name,
                                                             repl_file.getvalue(), progress=st.write)
                    except documents.DocumentError as exc:
                        status.update(label="Replacement rejected — previous version kept", state="error")
                        st.error(str(exc))
                        outcome = None
                    except AIServiceError as exc:
                        status.update(label="OpenAI error — previous version kept", state="error")
                        st.error(str(exc))
                        outcome = None
                    if outcome and not outcome["changed"]:
                        status.update(label="No text changes — nothing reprocessed or rechecked", state="complete")
                    elif outcome:
                        questions = storage.saved_questions(conn)
                        st.write(f"New version active. Rechecking all {len(questions)} saved question(s)...")
                        bar = st.progress(0.0)
                        run_ids = answering.recheck_all(
                            conn, index, ai, outcome["version_id"],
                            progress=lambda done, total, text: bar.progress(done / total if total else 1.0,
                                                                            text=text[:80]))
                        # Report the recorded outcome of each recheck, not an assumed success.
                        failed = sum(conn.execute("SELECT status FROM answer_runs WHERE id = ?", (rid,))
                                     .fetchone()[0] == "failed" for rid in run_ids)
                        message = (f"{choice['name']}: new version active. Rechecked all {len(run_ids)} saved "
                                   f"question(s): {len(run_ids) - failed} succeeded, {failed} failed. "
                                   "See the Changes tab.")
                        status.update(label="Replacement processed", state="complete")
                if outcome and outcome["changed"]:
                    if st.session_state.get("last_answer", ("",))[0] == "result":
                        del st.session_state["last_answer"]
                    st.session_state["doc_notice"] = ("warning" if failed else "success", message)
                    st.rerun()
        else:
            st.caption("Add a document first.")

    with st.expander("🕘 Version history"):
        if not active:
            st.caption("No versions yet.")
        for d in active:
            for v in storage.versions_for_document(conn, d["id"]):
                extra = f" — error: {v['error']}" if v["error"] else ""
                st.markdown(f"- **{d['name']}** v{v['version_no']} ({v['filename']}): {v['status']}{extra}")

    st.divider()
    st.caption("Local prototype · one user · up to 5 documents and 5 saved questions")


# ---------- main area ----------

tab_questions, tab_changes = st.tabs(["Questions", "Changes"])

with tab_questions:
    st.title("Ask about your documents")
    st.caption("Answers with traceable supporting evidence. If eligibility depends on your details "
               "(for example year, programme or CGPA), include them in the question.")
    saved = storage.saved_questions(conn)
    with st.form("ask", border=False):
        col_q, col_ask = st.columns([6, 1], vertical_alignment="bottom")
        question = col_q.text_input("Question", label_visibility="collapsed",
                                    placeholder="Type your question about the uploaded documents")
        asked = col_ask.form_submit_button("Ask", type="primary", width="stretch")
        save = st.checkbox(f"Save for automatic rechecks · {len(saved)} of {storage.MAX_SAVED_QUESTIONS} saved",
                           value=len(saved) < storage.MAX_SAVED_QUESTIONS,
                           disabled=len(saved) >= storage.MAX_SAVED_QUESTIONS)
    if asked and question.strip() and require_ai():
        with st.status("Answering...", expanded=True) as status:
            try:
                if save:
                    qid = storage.add_question(conn, question.strip())
                    run_id = answering.run_and_save(conn, index, ai, qid, "initial", progress=st.write)
                    run = storage.latest_run(conn, qid)
                    st.session_state["last_answer"] = ("run", run_id)
                    if run["status"] == "failed":
                        status.update(label="Failed — see error below", state="error")
                    else:
                        status.update(label="Done — question saved", state="complete")
                else:
                    result = answering.answer_question(conn, index, ai, question.strip(), progress=st.write)
                    st.session_state["last_answer"] = ("result", result)
                    status.update(label="Done (not saved)", state="complete")
            except (AIServiceError, ValueError) as exc:
                status.update(label="Could not answer", state="error")
                st.error(f"{exc} (This is a service/setup problem, not a finding about your documents.)"
                         if isinstance(exc, AIServiceError) else str(exc))
        saved = storage.saved_questions(conn)

    last = st.session_state.get("last_answer")
    if last:
        if last[0] == "run":
            run = conn.execute("SELECT question_id FROM answer_runs WHERE id = ?", (last[1],)).fetchone()
            run = storage.latest_run(conn, run["question_id"]) if run else None
            if run:
                q_text = conn.execute("SELECT text FROM questions WHERE id=?", (run["question_id"],)).fetchone()[0]
                st.markdown(f"**Q:** {q_text}")
                render_run(run, "latest")
        else:
            st.markdown(f"**Q:** {last[1]['question']}")
            render_result(last[1], "latest")

    st.divider()
    with st.expander(f"🔖 Saved questions ({len(saved)})"):
        if not saved:
            st.caption("No saved questions yet. Tick “Save for automatic rechecks” when asking.")
        for q in saved:
            runs = storage.runs_for_question(conn, q["id"])
            latest = runs[0] if runs else None
            icon = STATUS_ICONS.get(latest["status"], "") if latest else "⏳"
            with st.container(border=True):
                st.markdown(f"{icon} **{q['text']}**")
                if latest:
                    render_run(latest, f"saved-{q['id']}")
                    if latest["status"] == "failed":
                        good = storage.latest_run(conn, q["id"], successful_only=True)
                        if good:
                            st.warning("Showing the last successful answer below. It has NOT been verified "
                                       "against the current documents.")
                            render_run(good, f"saved-good-{q['id']}")
                c1, c2 = st.columns(2)
                if c1.button("Recheck now", key=f"recheck-{q['id']}", width="stretch") and require_ai():
                    with st.spinner("Rechecking..."):
                        answering.run_and_save(conn, index, ai, q["id"], "manual")
                    st.rerun()
                if c2.button("Remove (history is kept)", key=f"remove-{q['id']}", width="stretch"):
                    storage.remove_question(conn, q["id"])
                    st.rerun()
                if len(runs) > 1:
                    st.markdown("**History**")
                    for r in runs[1:]:
                        label = answering.STATUS_LABELS[r["status"]]
                        st.caption(f"Run #{r['id']} · {r['created_at']} · {r['trigger']} · {label}")

with tab_changes:
    st.title("Document updates")
    st.caption(f"{RECHECK_NOTE} Previous and updated answers are shown side by side.")
    updates = storage.replacement_versions(conn)
    if not updates:
        st.info("No document has been replaced yet. Use “Replace a document” in the sidebar to see "
                "previous and updated answers here.")
    for upd in updates:
        prev_version = storage.previous_active_version(conn, upd["document_id"], upd["version_no"])
        with st.container(border=True):
            st.subheader(f"{upd['document_name']}: v{prev_version['version_no'] if prev_version else '?'} "
                         f"→ v{upd['version_no']}")
            st.caption(f"Activated {upd['activated_at']} · new file: {upd['filename']}")
            if prev_version:
                with st.expander("Text changes in the document"):
                    diff = difflib.unified_diff(prev_version["extracted_text"].splitlines(),
                                                upd["extracted_text"].splitlines(),
                                                f"v{prev_version['version_no']}", f"v{upd['version_no']}",
                                                lineterm="", n=0)
                    st.code("\n".join(diff) or "(no line-level differences)", language="diff")

            pending = answering.pending_rechecks(conn, upd["id"])
            if pending:
                st.warning(f"⏳ {len(pending)} saved question(s) have not been rechecked against this update. "
                           "Their answers are NOT verified against the new documents.")
                if st.button("Run pending rechecks", key=f"pending-{upd['id']}") and require_ai():
                    with st.spinner("Rechecking..."):
                        for q in pending:
                            answering.run_and_save(conn, index, ai, q["id"], "update", upd["id"])
                    st.rerun()

            for q in storage.saved_questions(conn):
                new_run = storage.run_for_trigger(conn, q["id"], upd["id"])
                if new_run is None:
                    if q["id"] in {p["id"] for p in pending}:
                        st.markdown(f"**⏳ Pending:** {q['text']}")
                    continue
                old_run = storage.latest_run(conn, q["id"], before_run_id=new_run["id"], successful_only=True)
                if new_run["status"] == "failed":
                    header, color = "Recheck failed — previous answer NOT verified against new documents", "red"
                else:
                    header = COMPARISON_LABELS.get(new_run["comparison"], "Rechecked — review comparison")
                    color = {"unchanged": "green", "changed": "orange"}.get(new_run["comparison"], "blue")
                st.divider()
                st.markdown(f"**{q['text']}**")
                st.badge(header, color=color)
                left, right = st.columns(2)
                with left:
                    st.markdown("##### Before")
                    if old_run:
                        render_run(old_run, f"old-{new_run['id']}")
                    else:
                        st.write("No earlier answer.")
                with right:
                    st.markdown("##### After")
                    render_run(new_run, f"new-{new_run['id']}")
                    if new_run["status"] == "failed":
                        st.info("The previous answer is kept as history on the left. Use “Recheck now” on "
                                "the Questions tab to try again.")
