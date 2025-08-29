"""
Extract lists of poets and poem URLs from poemanalysis.com.
"""

import re
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

from poetry_project.config import HEADERS


def get_poets_by_poem_range(
    min_poems: int = 10,
    max_poems: float = float("inf")
) -> List[Tuple[str, int]]:
    """
    Retrieve all poets whose poem-counts fall within [min_poems, max_poems].

    Returns:
        A list of (author_name, poem_count), sorted descending by poem_count.
    """
    url = "https://poemanalysis.com/poets-a-z-list/full/"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results: List[Tuple[str, int]] = []
    for anchor in soup.select("a.generatepress.smooth-scroll"):
        name_el = anchor.select_one("span.category-name")
        count_el = anchor.select_one("span.poems-label")
        if not name_el or not count_el:
            continue

        match = re.search(r"\d+", count_el.text)
        if not match:
            continue

        count = int(match.group())
        if min_poems <= count <= max_poems:
            results.append((name_el.text.strip(), count))

    return sorted(results, key=lambda x: x[1], reverse=True)


def get_poem_urls_by_author(author_name: str) -> List[str]:
    """
    Retrieve all poem URLs for a given author by paging through their archive.

    Args:
        author_name: Name of the poet, e.g. "Emily Dickinson".

    Returns:
        A sorted list of poem URLs.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', author_name.lower()).strip('-')
    base = f"https://poemanalysis.com/{slug}/poems/"
    all_urls: set[str] = set()
    page = 1

    while True:
        page_url = base if page == 1 else f"{base}?_paged={page}"
        resp = requests.get(page_url, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        urls_on_page = {
            a["href"].rstrip("/")
            for h2 in soup.find_all("h2", class_="entry-title")
            if (a := h2.find("a", href=True))
        }

        new = urls_on_page - all_urls
        if not new:
            break

        all_urls |= new
        page += 1

    return sorted(all_urls)
