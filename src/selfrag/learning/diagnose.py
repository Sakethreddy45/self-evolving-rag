from pydantic import BaseModel, Field

from selfrag.evals.dataset import Case
from selfrag.models import chat
from selfrag.settings import Role

DIAGNOSE = """These questions share a failure pattern. Work out why.

Each shows the question, what the system planned to search for, and which
expected facts never appeared in the answer.

{evidence}

Name the single root cause, then say whether better instructions to the
planner could fix it, or whether the retriever lacks a capability no wording
can supply.

If it's fixable by instruction, write that instruction — a short paragraph to
append to the planner's prompt, concrete about what to do differently."""


class Diagnosis(BaseModel):
    cause: str = Field(description="root cause in one or two sentences")
    fixable_by_prompt: bool = Field(
        description="false when the retriever lacks a capability wording can't supply"
    )
    instruction: str = Field(
        default="", description="the prompt fragment, empty when not fixable"
    )


def evidence(cases: list[Case], traces: list[dict]) -> str:
    out = []
    for case, t in zip(cases, traces):
        plan = next((s for s in t["steps"] if s["node"] == "plan"), {})
        grade = next((s for s in t["steps"] if s["node"] == "grade"), {})
        out.append(
            f"question: {case.question}\n"
            f"  planned: {plan.get('queries')}\n"
            f"  retrieved {grade.get('of')} chunks, {grade.get('kept')} judged relevant\n"
            f"  documents carry metadata: source, published date\n"
            f"  missing from answer: {t['missing']}"
        )
    return "\n\n".join(out)


async def diagnose(cases: list[Case], traces: list[dict]) -> Diagnosis:
    llm = chat(Role.GENERATE).with_structured_output(Diagnosis)
    return await llm.ainvoke(DIAGNOSE.format(evidence=evidence(cases, traces)))