import asyncio
import sys

from selfrag.graph.build import build
from selfrag.retrieval.vector import VectorRetriever, open_store


async def main(question: str) -> None:
    graph = build(VectorRetriever(store=open_store()))
    out = await graph.ainvoke({"question": question})

    for s in out["steps"]:
        print(" ", s)
    print()
    print(out["answer"])


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))