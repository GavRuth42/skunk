import re
import string
import logging
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from memory_manager import get_or_create_conversation

def is_thanks(user_input: str) -> bool:
    pattern = r"(?i)^\s*thanks[!?.,]{0,4}\s*$"
    return re.match(pattern, user_input.strip()) is not None

def is_small_talk(user_input: str) -> bool:
    small_talk_phrases = [
        "hello", "hi", "hey", "greetings", "good morning",
        "good evening", "good afternoon", "how are you",
        "what's up", "nice to meet you", "thank you", "thanks",
        "how is it going", "how's it going", "how are you doing", "yo yo", "awesome",
        "whats up", "what's new", "yo", "hey there"
    ]
    lower_input = user_input.lower().strip()
    lower_input = lower_input.strip(string.punctuation)
    return lower_input in small_talk_phrases

def get_small_talk_response_llm(session_id: str, user_input: str, temperature=0.8) -> str:
    conversation = get_or_create_conversation(session_id)
    
    from message_utils import dict_to_chat_message  # If using a separate module

    # Convert the conversation
    chat_messages = [dict_to_chat_message(m) for m in conversation]

    # Add a system message that instructs the LLM on formatting
    system_msg = SystemMessage(content=(
        "You are a friendly chatbot.  Appropriately respond to greeting in no more than 3 words."
    ))
    chat_messages.insert(0, system_msg)

    chat_messages.append(HumanMessage(content=user_input))

    # Additional instruction to keep it short and friendly
    chat_messages.append(SystemMessage(content=(
        "Respond in a short, friendly, and natural way. "
        "Use a conversational tone and format your answer with short paragraphs or bullets."
    )))

    llm_small_talk = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    response = llm_small_talk(messages=chat_messages)
    return response.content

def can_trigger_vague_detection(session_id: str) -> bool:
    conversation = get_or_create_conversation(session_id)
    if not conversation:
        return True  # conversation is empty
    for msg in conversation:
        if len(msg["content"].split()) > 10:
            return False
    return True

def is_question_vague_in_context(session_id: str, new_question: str) -> bool:
    if not can_trigger_vague_detection(session_id):
        return False

    from langchain.schema import SystemMessage, HumanMessage

    conversation = get_or_create_conversation(session_id)
    system_instruction = (
        "You are ChatGPT, a large language model trained by OpenAI. Your task is to decide if the user's question is too vague. "
        "A question is 'vague' if it lacks necessary context or clarity to be answered. Otherwise, it is 'specific'."
    )
    messages = [SystemMessage(content=system_instruction)]

    conversation_str = "".join(f"{msg['role'].upper()}:\n{msg['content']}\n\n" for msg in conversation)
    user_msg_content = (
        f"Conversation so far:\n{conversation_str}\n"
        f"New question: '{new_question}'\n\n"
        "Respond with 'vague' or 'specific'."
    )
    messages.append(HumanMessage(content=user_msg_content))

    llm_classifier = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=10)
    response = llm_classifier(messages=messages)
    classification = response.content.strip().lower()
    return classification == "vague"

def is_typo_correction_request(user_input: str) -> bool:
    triggers = [
        "correct my text",
        "typo correction",
        "spell check",
        "please correct any errors",
        "fix my text",
        "correct my sentence",
        "fix my grammar",
        "proofread",
        "proof read"
    ]
    lower_input = user_input.lower()
    return any(t in lower_input for t in triggers)

def correct_typos_llm(question: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "You are ChatGPT, a large language model with excellent grammar/spelling skills. "
                "Correct any typos or minor grammatical errors in the user's question without changing its meaning. "
                "Return only the corrected text."
            )
        ),
        HumanMessage(content=question)
    ]
    llm_corrector = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=100)
    response = llm_corrector(messages=messages)
    return response.content.strip()

