import re
from dataclasses import dataclass
from datetime import date

SOURCES = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
]

_SLUG = re.compile(r"/posts/(\d{4})-(\d{2})-(\d{2})-([^/]+)/?$")


@dataclass(frozen=True, slots=True)
class SourceDoc:
    url: str
    slug: str
    published: date

    @classmethod
    def parse(cls, url: str) -> "SourceDoc":
        m = _SLUG.search(url)
        if not m:
            raise ValueError(f"cannot derive date from {url}")
        y, mo, d, slug = m.groups()
        return cls(url=url, slug=slug, published=date(int(y), int(mo), int(d)))