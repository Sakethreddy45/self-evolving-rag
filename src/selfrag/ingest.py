import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from selfrag.corpus import SOURCES, SourceDoc
from selfrag.loader import fetch
from selfrag.retrieval.vector import open_store
from selfrag.settings import settings

log = logging.getLogger(__name__)


def _chunk_id(source: str, offset: int, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"{source}:{offset}:{digest}"


def load(sources: list[str] | None = None) -> list[Document]:
    docs = []
    for url in sources or SOURCES:
        meta = SourceDoc.parse(url)
        doc = fetch(url)
        doc.metadata |= {
            "source": meta.slug,
            "url": meta.url,
            "published": meta.published.isoformat(),
            # int form because chroma's $gte/$lte need numbers, not iso strings
            "published_ts": int(meta.published.strftime("%Y%m%d")),
            "year": meta.published.year,
        }
        docs.append(doc)
        log.info("fetched %s (%d chars)", meta.slug, len(doc.page_content))
    return docs


def split(docs: list[Document]) -> tuple[list[str], list[Document]]:
    p = settings().retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=p.chunk_size,
        chunk_overlap=p.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    ids = [_chunk_id(c.metadata["source"], c.metadata["start_index"], c.page_content) for c in chunks]
    return ids, chunks


def ingest(sources: list[str] | None = None) -> int:
    docs = load(sources)
    ids, chunks = split(docs)
    # content-hashed ids make this an upsert, so re-running never duplicates
    open_store().add_documents(chunks, ids=ids)
    log.info("ingested %d chunks from %d documents", len(chunks), len(docs))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest()