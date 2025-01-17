import logging
from datetime import datetime, timedelta
from memory_manager import SESSION_MEMORY

def clear_stale_sessions():
    now = datetime.utcnow()
    to_remove = []
    for sid, data in SESSION_MEMORY.items():
        last_updated = data["last_updated"]
        if (now - last_updated) > timedelta(hours=1):
            to_remove.append(sid)
    for sid in to_remove:
        logging.debug(f"Session {sid} has been stale for over 1 hour. Removing.")
        del SESSION_MEMORY[sid]

