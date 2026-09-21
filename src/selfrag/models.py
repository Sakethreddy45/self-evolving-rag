from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from selfrag.settings import Role, settings


def chat(role: Role, **overrides) -> ChatOpenAI:
    s = settings()
    spec = s.models[role]
    kw = {
        "model": spec.name,
        "api_key": s.openai_api_key,
        "timeout": spec.timeout_s,
        "max_retries": spec.max_retries,
    }
    # reasoning models reject temperature, so it's opt-out per spec
    if spec.temperature is not None:
        kw["temperature"] = spec.temperature
    return ChatOpenAI(**kw, **overrides)


def embeddings() -> OpenAIEmbeddings:
    s = settings()
    return OpenAIEmbeddings(model=s.embed_model, api_key=s.openai_api_key)