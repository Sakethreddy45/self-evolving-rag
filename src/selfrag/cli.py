import asyncio
import sys

from selfrag.retrieval.vector import VectorRetriever, open_store
from selfrag.runner import Runner


async def main(question: str) -> None:
    out = await Runner(VectorRetriever(store=open_store())).ask(question)

    for s in out["steps"]:
        print(" ", s)
    print()
    print(out["answer"])


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))