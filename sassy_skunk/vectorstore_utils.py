from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from memory_manager import get_or_create_conversation, SESSION_MEMORY

def answer_with_persistent_chroma(session_id: str, question: str, persist_dir="chroma_db", top_k=2):
    """
    Retrieve relevant docs from Chroma store and build an LLM answer.
    Prioritizes any headings that have been flagged in the session memory.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 1) Retrieve docs from Chroma
    result = retriever.get_relevant_documents(question)

    relevant_data = "\n".join(doc.page_content for doc in result)
    if not relevant_data.strip():
        return "I don’t have enough information to answer that.", relevant_data, result

    # 2) Attempt to reorder docs if we have "preferred_heading_keys"
    session_data = SESSION_MEMORY.get(session_id, {})
    preferred_headings = session_data.get("preferred_heading_keys", set())

    if preferred_headings:
        preferred_docs = []
        other_docs = []
        for doc in result:
            doc_head = doc.metadata.get("heading_key")
            if doc_head in preferred_headings:
                preferred_docs.append(doc)
            else:
                other_docs.append(doc)
        if preferred_docs:
            result = preferred_docs + other_docs
            relevant_data = "\n".join(doc.page_content for doc in result)

    # 3) Build the conversation prompt
    conversation = get_or_create_conversation(session_id)
    conversation_str = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation
    )

    system_content = (
        "You are ChatGPT, a large language model trained by OpenAI. "
        "Use chain-of-thought reasoning internally, but present a clear final answer. "
        "When conversation conflicts with retrieved data, favor conversation context or disclaim. "
        "If doc headings do not match the scenario, disclaim them.\n"
    )
    system_msg = SystemMessage(content=system_content)

    user_prompt = (
        f"Conversation History:\n{conversation_str}\n\n"
        f"Relevant Data:\n{relevant_data}\n\n"
        f"Question: {question}\n"
        "Please provide the best possible answer. If the doc headings do not match the conversation context, disclaim or ignore them."
    )
    user_msg = HumanMessage(content=user_prompt)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    response = llm([system_msg, user_msg])
    direct_answer = response.content.strip()

    return direct_answer, relevant_data, result

