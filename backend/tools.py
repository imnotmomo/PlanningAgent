"""
Tool functions and OpenAI-compatible JSON schemas.

Tools available:
- tavily_search: live web search via Tavily API. Used by Research and Budget
  to ground their outputs in real data (real places, real prices).
- python_exec: short Python snippets for arithmetic / date math / list ops.
  Used by Budget (cost arithmetic) and Critic (time-feasibility math).

Security note for python_exec: the snippet runs in a subprocess with a 5s
timeout. The model is the caller, but tool outputs (e.g. Tavily content) can
contain prompt-injection. For production, swap to a real sandbox (e.g. e2b,
Modal, Docker). For a class project this is acceptable.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any

import urllib.request
import urllib.error

TAVILY_API = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 20
PYTHON_TIMEOUT = 5


def _tavily_search_sync(
    query: str,
    max_results: int = 5,
    include_images: bool = False,
    search_depth: str = "basic",
) -> dict:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return {"error": "TAVILY_API_KEY not set"}
    body = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": max(1, min(max_results, 10)),
        "include_answer": True,
        "include_images": include_images,
        "search_depth": search_depth,
    }).encode()
    req = urllib.request.Request(
        TAVILY_API, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TAVILY_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    out: dict = {
        "answer": data.get("answer"),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("content") or "")[:500],
            }
            for r in (data.get("results") or [])[:max_results]
        ],
    }
    if include_images:
        # Tavily returns either a list of URL strings or [{url, description}] objects
        imgs: list[str] = []
        for img in (data.get("images") or [])[:6]:
            if isinstance(img, str):
                imgs.append(img)
            elif isinstance(img, dict) and img.get("url"):
                imgs.append(img["url"])
        out["images"] = imgs
    return out


def _python_exec_sync(code: str) -> dict:
    try:
        proc = subprocess.run(
            ["python3", "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {PYTHON_TIMEOUT}s"}
    return {
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-500:],
        "returncode": proc.returncode,
    }


async def tavily_search(query: str, max_results: int = 5) -> dict:
    return await asyncio.to_thread(_tavily_search_sync, query, max_results)


async def tavily_search_detailed(query: str) -> dict:
    """Deep search with images, used by /candidate-detail."""
    return await asyncio.to_thread(
        _tavily_search_sync, query, 5, True, "advanced"
    )


async def python_exec(code: str) -> dict:
    return await asyncio.to_thread(_python_exec_sync, code)


# OpenAI / vLLM tool-calling schemas
TAVILY_TOOL = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": (
            "Search the live web for current information about real places, prices, "
            "opening hours, transit, events, weather, and travel logistics. Use this "
            "whenever you need facts that may not be in your training data, or when "
            "you need to verify that something exists. Returns a synthesized 'answer' "
            "plus the top result snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural-language web query"},
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
}

PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute a short Python snippet for arithmetic, date math, or simple list "
            "manipulation. The snippet runs in an isolated subprocess (no network, no "
            "filesystem persistence). MUST print results to stdout for them to come back. "
            "No imports beyond the standard library; no long-running code (5s timeout)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code. Must print() its results.",
                },
            },
            "required": ["code"],
        },
    },
}


DISPATCH = {
    "tavily_search": tavily_search,
    "python_exec": python_exec,
}


async def run_tool(name: str, raw_args: str) -> str:
    """Invoke a tool by name with stringified JSON args. Returns JSON string."""
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return json.dumps({"error": f"bad JSON in tool args: {raw_args[:200]}"})
    try:
        result = await fn(**args)
    except TypeError as e:
        return json.dumps({"error": f"bad args: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(result)
