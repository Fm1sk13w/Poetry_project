"""
Fetch and parse individual poem pages into Poem objects.
"""

from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from poetry_project.config import HEADERS
from poetry_project.models import Poem, Author


def extract_lines(stanza_div: Tag) -> List[str]:
    """
    Extract text lines from a stanza <div> element.

    Args:
        stanza_div: BeautifulSoup Tag for the stanza container.

    Returns:
        A list of cleaned lines.
    """
    lines: List[str] = []
    for span in stanza_div.find_all("span"):
        if not span.contents:
            continue
        text = " ".join(
            part.text.strip() if hasattr(part, "text") else str(part).strip()
            for part in span.contents
        ).strip()
        if text:
            lines.append(text)
    return lines


def scrape_poem_content(url: str, author: Optional[Author] = None) -> Poem:
    """
    Download poem page and build a Poem object.

    Args:
        url: URL of the poem page.
        author: Optional Author instance; if not provided, name is inferred.

    Returns:
        A Poem instance with content and minimal metadata.
    """
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    parts = url.rstrip("/").split("/")
    inferred_author = " ".join(parts[-2].split("-")).title()
    inferred_title = " ".join(parts[-1].split("-")).title()
    lines = []
    stanza = soup.find("div", class_="stanza")
    if stanza:
        lines = extract_lines(stanza)
    content = "\n".join(lines) or "Poem content not found."

    author_obj = author or Author(name=inferred_author)
    poem = Poem(
        title=inferred_title,
        author=author_obj,
        content=content,
        source_url=url
    )
    author_obj.poems.append(poem)
    return poem
