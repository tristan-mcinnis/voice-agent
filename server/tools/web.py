"""Web and external-API tools — search, weather, and other network calls.

All implementations live in BaseTool.execute() methods — the canonical interface.
Thin backward-compat callables are re-exported from __init__.py.
"""

from __future__ import annotations

import os

import httpx

from tools.registry import REGISTRY, BaseTool


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
        return {"conditions": "nice", "temperature": "75"}
