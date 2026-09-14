from pydantic import BaseModel, Field


class Relevance(BaseModel):
    on_topic: bool = Field(description="helps answer the query")
    satisfies_constraints: bool = Field(
        description="meets explicit constraints such as dates or named sources; "
        "true if the query states none"
    )
    reason: str = Field(description="one short clause")


class Plan(BaseModel):
    queries: list[str] = Field(
        description="one self-contained search query per distinct thing the "
        "question asks about; each names exactly one topic",
        min_length=1,
        max_length=3,
    )