import os
import logging
from flask import Flask, request, jsonify

from atexit import register
from apscheduler.schedulers.background import BackgroundScheduler

from memory_manager import (
    SESSION_MEMORY,
    get_or_create_conversation,
    clear_session,
    append_user_message,
    append_assistant_message,
)
from text_detection import (
    is_thanks,
    is_small_talk,
    get_small_talk_response_llm,
    is_question_vague_in_context,
    is_typo_correction_request,
    correct_typos_llm,
)
from vectorstore_utils import answer_with_persistent_chroma
from summarization import summarize_data_approach1
from requery import requery_for_regulations
from stale_session_cleaner import clear_stale_sessions
# ^ Make sure to create the 'scheduler' inside that module or here.

os.environ["OPENAI_API_KEY"] = "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"

##############################################################################
# Create Flask app
##############################################################################
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

##############################################################################
# SCHEDULER: Clear stale sessions
##############################################################################
scheduler = BackgroundScheduler()
scheduler.add_job(func=clear_stale_sessions, trigger="interval", minutes=30)
scheduler.start()
register(lambda: scheduler.shutdown())

##############################################################################
# CLEAR MEMORY ENDPOINT
##############################################################################
@app.route("/clear_memory", methods=["POST"])
def clear_memory_endpoint():
    data = request.json or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"status": "error", "error": "Missing session_id"}), 400

    if session_id in SESSION_MEMORY:
        clear_session(session_id)
        return jsonify({
            "status": "ok",
            "message": f"Cleared session memory for session ID: {session_id}"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "error": f"No session found with ID '{session_id}'"
        }), 404

##############################################################################
# Approach 1 Endpoint: short or detailed via wants_details
##############################################################################
@app.route("/ask", methods=["POST"])
def ask_approach1():
    """
    Approach 1:
      - If wants_details=True => produce a longer answer, prefixed with "Details:: "
      - Otherwise => short summary
    """
    data = request.json or {}
    session_id = data.get("session_id", "global")
    question = data.get("question", "").strip()
    wants_details = data.get("wants_details", False)

    if not question:
        return jsonify({"status": "error", "error": "No question provided."}), 400

    # Check for small talk, thanks, vague, etc.
    if is_thanks(question):
        clear_session(session_id)
        return jsonify({
            "status": "ok",
            "answer": "You're welcome! The conversation has ended.",
            "references": [],
            "details": None
        })

    if is_small_talk(question):
        st_response = get_small_talk_response_llm(session_id, question)
        return jsonify({
            "status": "ok",
            "answer": st_response,
            "references": [],
            "details": None
        })

    if is_question_vague_in_context(session_id, question):
        vague_answer = "You might need to add more context."
        return jsonify({
            "status": "ok",
            "answer": vague_answer,
            "references": [],
            "details": None
        })

    if is_typo_correction_request(question):
        append_user_message(session_id, question)
        corrected_text = correct_typos_llm(question)
        answer_msg = f"Here is your corrected text:\n\n{corrected_text}"
        append_assistant_message(session_id, answer_msg)
        return jsonify({
            "status": "ok",
            "answer": answer_msg,
            "references": [],
            "details": None
        })

    # Normal Q&A
    append_user_message(session_id, question)

    direct_answer, relevant_data, retrieved_docs = answer_with_persistent_chroma(
        session_id, question, top_k=3
    )

    if not relevant_data.strip() or "I don’t have enough information" in direct_answer:
        fallback = "I don’t have enough information to answer that."
        append_assistant_message(session_id, fallback)
        return jsonify({
            "status": "ok",
            "answer": fallback,
            "references": [],
            "details": None
        })

    detail_level = "long" if wants_details else "short"
    summarized_answer = summarize_data_approach1(relevant_data, question, detail_level=detail_level)

    headings_set = set()
    for doc in retrieved_docs:
        if "heading_key" in doc.metadata:
            headings_set.add(doc.metadata["heading_key"])

    # Only display references if detail_level == "long"
    if detail_level == "long" and headings_set:
        summarized_answer += "\n\nReferences:\n"
        for heading_text in headings_set:
            summarized_answer += f"- {heading_text}\n"

    # You might still want to store headings in the session regardless of detail_level
    if headings_set:
        SESSION_MEMORY[session_id]["preferred_heading_keys"] = headings_set

    append_assistant_message(session_id, summarized_answer)


    # Attempt to detect CFR references in the summarized_answer
    import re
    cited_references = re.findall(
        r"\(\d+\s?CFR\s\d+\.\d+\(\w+\)\)|\u00a7\s?\d+\.\d+\(\w+\)|\u00a7\s?\d+\.\d+|Section\s\d+\.\d+|"
        r"\d+\s?CFR\s(?:Part\s\d+|\d+\.\d+\(\w+\)|\d+\.\d+)|"
        r"\d+\s?CFR\sAppendix-[A-Za-z0-9-]+(?:\s\d+\.\d+)?",
        summarized_answer
    )
    details = None
    if wants_details and cited_references:
        details = requery_for_regulations(cited_references, question)

    return jsonify({
        "status": "ok",
        "answer": summarized_answer,
        #"Citations": list(headings_set),
        "details": details
    })

##############################################################################
# Home route
##############################################################################
@app.route("/")
def home():
    return (
        "LLM-based QA with 2 Approaches:\n"
        "1) /ask -> short vs long response via 'wants_details'\n"
        "POST to /clear_memory to clear a session."
    )

##############################################################################
# Run Flask
##############################################################################
if __name__ == "__main__":
    # Make sure the OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "YOUR-OPENAI-KEY":
        logging.error("OpenAI API key not set. Please set the 'OPENAI_API_KEY' environment variable.")
        exit(1)
    
    app.run(host="0.0.0.0", port=6000, debug=True)

