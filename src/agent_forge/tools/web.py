from __future__ import annotations

"""Web tools: search and URL fetching."""

from agent_forge.tools import register


@register("web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return the top results.
    Returns titles, URLs, and snippets. Use for current events or general research.
    """
    try:
        from ddgs import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        )
    except Exception as exc:
        return f"Web search failed for '{query}': {exc}"


@register("fetch_url")
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetch the text content of a URL.
    Returns the raw page text (truncated to max_chars). Use to read articles or docs.
    """
    try:
        import httpx

        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; agent-forge/1.0)"},
        )
        response.raise_for_status()

        # Strip HTML tags with a simple regex if it looks like HTML
        text = response.text
        if "<html" in text.lower():
            import re
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars]
    except Exception as exc:
        return f"Failed to fetch '{url}': {exc}"
