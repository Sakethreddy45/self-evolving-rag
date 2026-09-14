import asyncio

from pydantic import BaseModel, Field

from selfrag.evals.dataset import Case
from selfrag.models import chat
from selfrag.settings import Role

CHECK = """Is this fact present in the answer? Wording may differ; judge meaning.

Fact: {fact}

Answer:
{answer}"""


class Present(BaseModel):
    present: bool = Field(description="the fact appears in the answer")


class Score(BaseModel):
    case_id: str
    found: list[str]
    missing: list[str]

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missing)
        return len(self.found) / total if total else 1.0


async def score(case: Case, answer: str) -> Score:
    llm = chat(Role.GRADE).with_structured_output(Present)
    verdicts = await asyncio.gather(
        *(llm.ainvoke(CHECK.format(fact=f, answer=answer)) for f in case.must_contain)
    )
    found = [f for f, v in zip(case.must_contain, verdicts) if v.present]
    return Score(
        case_id=case.id,
        found=found,
        missing=[f for f in case.must_contain if f not in found],
    )