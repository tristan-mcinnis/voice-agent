"""Web and external-API tools — search, weather, and other network calls."""

from __future__ import annotations

import os

import httpx

from tools.registry import REGISTRY, BaseTool


# ---------------------------------------------------------------------------
# Web search via Serper
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Google search via Serper. Returns formatted top results."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Web search unavailable: SERPER_API_KEY is not set."
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Web search failed: {exc}"

    results = data.get("organic", [])[:max_results]
    if not results:
        return f"No results found for: {query}"
    lines = [f"Top results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')} — {r.get('snippet', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: weather (dummy data)
# ---------------------------------------------------------------------------

def get_current_weather(location: str = "", format: str = "fahrenheit") -> dict:
    """Dummy weather to demonstrate function calling end-to-end."""
    return {"conditions": "nice", "temperature": "75"}


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

@REGISTRY.register
class WebSearchTool(BaseTool):
    name = "web_search"
    category = "web"
    speak_text = "Searching the web."
    description = (
        "Search the web with Google (via Serper) for current information. "
        "Use for facts, news, or anything you don't already know."
    )
    parameters = {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {
            "type": "integer",
            "description": "How many results (1-10). Default 5.",
        },
    }
    required = ["query"]

    def execute(self, query: str, max_results: int = 5) -> str:
        return web_search(query, max_results)


@REGISTRY.register
class GetCurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    category = "demo"
    speak_text = "Let me check on that."
    description = "Get the current weather (demo — returns dummy data)."
    parameters = {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location.",
        },
    }
    required = ["location", "format"]

    def execute(self, location: str = "", format: str = "fahrenheit") -> dict:
        return get_current_weather(location, format)
