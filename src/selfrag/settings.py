from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Role(StrEnum):
    ROUTE = "route"
    GRADE = "grade"
    REWRITE = "rewrite"
    GENERATE = "generate"


class ModelSpec(BaseModel):
    name: str
    temperature: float = 0.0
    timeout_s: float = 30.0
    max_retries: int = 3
    max_concurrency: int = 16


class RetrievalPolicy(BaseModel):
    k: int = 8
    min_surviving_docs: int = 2
    max_rewrites: int = 2
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_regens: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SELFRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # validation_alias escapes env_prefix so the conventional name works,
    # which matters because the openai sdk looks for it too
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    embed_model: str = "text-embedding-3-large"

    models: dict[Role, ModelSpec] = Field(
        default_factory=lambda: {
            Role.ROUTE: ModelSpec(name="gpt-4o"),
            Role.GRADE: ModelSpec(name="gpt-4o", max_concurrency=24),
            Role.REWRITE: ModelSpec(name="gpt-4o", temperature=0.3),
            Role.GENERATE: ModelSpec(name="gpt-4o", timeout_s=60.0),
        }
    )

    retrieval: RetrievalPolicy = RetrievalPolicy()
    data_dir: Path = Path("data")
    collection: str = "docs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def settings() -> Settings:
    return Settings()