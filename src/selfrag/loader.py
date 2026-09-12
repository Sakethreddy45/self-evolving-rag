import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document

UA = "selfrag/0.1"

DROP = ["script", "style", "nav", "footer", "header", "aside"]


def fetch(url: str, *, selector: str = "article") -> Document:
    r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(DROP):
        tag.decompose()

    main = soup.select_one(selector) or soup.body
    lines = (ln.strip() for ln in main.get_text("\n").splitlines())
    return Document(page_content="\n".join(ln for ln in lines if ln), metadata={"url": url})