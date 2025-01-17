from langchain.schema import SystemMessage, HumanMessage, AIMessage

def dict_to_chat_message(msg_dict: dict):
    role = msg_dict["role"]
    content = msg_dict["content"]
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    else:
        raise ValueError(f"Unknown role {role}")

