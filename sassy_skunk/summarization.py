import logging
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def llm_is_yes_no_question(question: str) -> bool:
    try:
        messages = [
            SystemMessage(content=(
                "You are ChatGPT, a large language model. Determine if the question can be answered yes/no."
            )),
            HumanMessage(content=(
                f"The user asked: '{question}'\n\n"
                "If it's a yes/no question, respond 'yesno'. Otherwise respond 'notyesno'."
            ))
        ]
        llm_classifier = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, max_tokens=10)
        response = llm_classifier(messages=messages)
        result = response.content.strip().lower()
        if result not in {"yesno", "notyesno"}:
            logging.warning(f"Unexpected classification result: {result}")
            return False
        return result == "yesno"
    except Exception as e:
        logging.error(f"Failed to classify question as yes/no: {e}")
        return False

def summarize_data_approach1(data: str, question: str, detail_level="short") -> str:
    #from .summarization import llm_is_yes_no_question  # or just define above, as shown.

    yesno = llm_is_yes_no_question(question)

    system_prompt = (
        "You are ChatGPT, an expert on US regulations and CFR documents. "
        "Always provide the **best possible answer** based on the data.\n"
        "Use **only** the data provided to summarize or elaborate on the question.\n"
        "Cite CFR references as needed.\n"
        "If the data doesn't fully address the question, say so.\n"
    )

    if detail_level == "long":
        style_prompt = (
            "Use only the relevant data provided. Be direct and factual. "
            "give a **detailed** explanation. "
            "Use bullet points or paragraphs for clarity where appropriate."
            "If the data does not conclusively support either, respond with 'Unable to determine conclusively.' "
        )
    else:  # short
        if yesno:
            style_prompt = (
                "If the information provided conclusively supports a 'Yes' or 'No' answer, "
                "respond with 'Yes' or 'No' (on its own line) followed by a brief explanation. "
                "Provide a concise (short) summary with 2-3 bullets or a short paragraph. "
                "If the data does not conclusively support either, respond with 'Unable to determine conclusively.' "
                "Use bullet points or paragraphs for clarity."
            )
        else:
            style_prompt = (
                "Provide a concise (short) summary with 2-3 bullets or a short paragraph. "
                "If the data does not conclusively support either, respond with 'Unable to determine conclusively.' "
                "Use bullet points or paragraphs for clarity."
            )

    user_msg = HumanMessage(content=(
        f"{style_prompt}\n\n"
        f"**Question**: {question}\n"
        f"**Relevant Data**:\n{data}\n\n"
        "Begin your final answer now:"
    ))

    system_msg = SystemMessage(content=system_prompt)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    response = llm(messages=[system_msg, user_msg])
    answer = response.content.strip()

    if detail_level == "long":
        answer = f"{answer}"

    return answer

