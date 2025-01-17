import logging
from datetime import datetime

##############################################################################
# In-memory conversation data
##############################################################################
SESSION_MEMORY = {}

def get_or_create_conversation(session_id: str):
    """
    Retrieve an existing session or create a new one.
    Each session stores:
      - 'messages': list of conversation messages
      - 'last_updated': timestamp for stale session cleanup
      - 'last_question': used only in Approach 2 (not shown here)
      - 'preferred_heading_keys': set of user-preferred heading metadata
    """
    from datetime import datetime
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {
            "messages": [],
            "last_updated": datetime.utcnow(),
            "last_question": None,
            "preferred_heading_keys": set()
        }
        logging.debug(f"Created new session with ID: {session_id}")
    else:
        SESSION_MEMORY[session_id]["last_updated"] = datetime.utcnow()
    return SESSION_MEMORY[session_id]["messages"]

def clear_session(session_id: str):
    """Remove a session from SESSION_MEMORY."""
    if session_id in SESSION_MEMORY:
        del SESSION_MEMORY[session_id]
        logging.debug(f"Cleared session with ID: {session_id}")

def append_user_message(session_id: str, content: str):
    conversation = get_or_create_conversation(session_id)
    conversation.append({"role": "user", "content": content})
    logging.debug(f"Appended user message to session {session_id}: {content}")

def append_assistant_message(session_id: str, content: str):
    conversation = get_or_create_conversation(session_id)
    conversation.append({"role": "assistant", "content": content})
    logging.debug(f"Appended assistant message to session {session_id}: {content}")

