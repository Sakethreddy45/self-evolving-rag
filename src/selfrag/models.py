from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from selfrag.settings import Role, settings


def chat(role: Role, **overrides) -> ChatOpenAI:
    s = settings()
    spec = s.models[role]
    return ChatOpenAI(
        model=spec.name,
        api_key=s.openai_api_key,
        temperature=spec.temperature,
        timeout=spec.timeout_s,
        max_retries=spec.max_retries,
        **overrides,
    )


def embeddings() -> OpenAIEmbeddings:
    s = settings()
    return OpenAIEmbeddings(model=s.embed_model, api_key=s.openai_api_key)