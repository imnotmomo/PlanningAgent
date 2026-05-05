"""
FastAPI server for TripWise.

Endpoints:
- GET  /health                — liveness
- POST /plan    (SSE stream)  — runs the multi-agent pipeline, streams events
- POST /revise                — apply a user-requested change to an itinerary

Run:
  cd /Users/zhelichen/Downloads/travel-agent
  uvicorn backend.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import agents, orchestrator, tools

app = FastAPI(title="TripWise", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    request: str


class ResearchRequest(BaseModel):
    preferences: dict
    destinations: list[dict] | None = None  # if None, fallback to preferences.destinations


class BuildRequest(BaseModel):
    preferences: dict
    research: dict
    arrival: dict | None = None
    selections: dict | None = None  # {"places":[], "restaurants":[], "hotels":[]} or None=all


class ReviseRequest(BaseModel):
    # `result` is the full prior PlanResult (preferences, places, hotels,
    # destinations, arrival, budget, itinerary, …). It gives the smart-router
    # enough context to decide whether to do a text-only edit or re-run
    # route/itinerary/critic. For backwards compatibility we also accept the
    # legacy {itinerary, change} shape and synthesize a minimal result.
    result: dict | None = None
    itinerary: dict | None = None
    change: str


class EnrichItem(BaseModel):
    name: str
    city: str | None = None
    category: str  # "place" | "restaurant" | "hotel"


class EnrichRequest(BaseModel):
    items: list[EnrichItem]
    destination: str | None = None  # fallback when item.city is missing


class DetailRequest(BaseModel):
    name: str
    city: str | None = None
    category: str | None = None  # "place" | "restaurant" | "hotel"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(stream_factory):
    """Wrap an async generator of dict events into an SSE response, with
    error reporting as a final `error` event."""
    async def gen():
        try:
            async for event in stream_factory():
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            err = {
                "event": "error",
                "payload": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "trace": traceback.format_exc(limit=4),
                },
            }
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/plan")
async def plan(req: PlanRequest):
    """Full pipeline (no picker). Streams every step end-to-end."""
    return _sse(lambda: orchestrator.run_plan(req.request))


@app.post("/destinations")
async def destinations(req: PlanRequest):
    """Phase 0: extract preferences. If user gave only a country/region,
    also runs the destination suggester so the UI can show candidate cities
    for the user to pick from. Ends with a `destinations_complete` event."""
    return _sse(lambda: orchestrator.run_destinations(req.request))


@app.post("/research")
async def research(req: ResearchRequest):
    """Phase 1: arrival (if origin) + research per leg. Caller passes the
    resolved destinations from phase 0 (either user-confirmed or auto)."""
    return _sse(lambda: orchestrator.run_research(req.preferences, destinations=req.destinations))


@app.post("/build")
async def build(req: BuildRequest):
    """Phase 2: route + budget + itinerary + critic, given the user's
    selections from the picker."""
    return _sse(lambda: orchestrator.run_build(
        prefs=req.preferences,
        research=req.research,
        arrival=req.arrival,
        selections=req.selections,
    ))


@app.post("/revise")
async def revise(req: ReviseRequest):
    """SSE streaming revision. Routes the change through one of three paths
    (text / structural / budget) and emits the same step events as the
    planning pipeline so the UI can show which agent is running."""
    result = req.result or {"itinerary": req.itinerary or {}}
    return _sse(lambda: orchestrator.run_revise(result, req.change))


@app.post("/enrich-candidates")
async def enrich_candidates(req: EnrichRequest) -> dict:
    """Look up each user-supplied custom candidate via Tavily and return a
    short description so it can be merged into the research candidate set
    before route/itinerary run.
    Tavily calls are issued in parallel; one failure doesn't block the rest.
    """
    import asyncio as _asyncio

    async def _enrich(item: EnrichItem) -> dict:
        loc = item.city or req.destination or ""
        if item.category == "hotel":
            qbits = [item.name, loc, "hotel"]
        elif item.category == "restaurant":
            qbits = [item.name, loc, "restaurant"]
        else:
            qbits = [item.name, loc, "attraction"]
        query = " ".join(b for b in qbits if b).strip()
        try:
            data = await tools.tavily_search(query, max_results=3)
        except Exception:
            data = {}
        # Prefer Tavily's `answer` if present; else stitch first snippet.
        desc = (data.get("answer") or "").strip()
        if not desc:
            for r in data.get("results") or []:
                snip = (r.get("snippet") or "").strip()
                if snip:
                    desc = snip[:240]
                    break
        if not desc:
            desc = f"User-added {item.category} in {loc or 'the trip'}."
        # Keep descriptions concise so they look like the rest of research.
        words = desc.split()
        if len(words) > 40:
            desc = " ".join(words[:40]) + "…"
        return {"name": item.name, "city": item.city or req.destination, "description": desc}

    enriched = await _asyncio.gather(*(_enrich(it) for it in req.items))
    return {"items": list(enriched)}


@app.post("/candidate-detail")
async def candidate_detail(req: DetailRequest) -> dict:
    """Fetch deeper info + images about a single candidate via Tavily.
    Used by the picker's 'Research more' button."""
    parts = [req.name]
    if req.city:
        parts.append(req.city)
    if req.category == "hotel":
        parts.append("hotel review price")
    elif req.category == "restaurant":
        parts.append("restaurant review menu")
    elif req.category == "place":
        parts.append("travel attraction")
    query = " ".join(parts)
    data = await tools.tavily_search_detailed(query)
    return {
        "name": req.name,
        "city": req.city,
        "summary": data.get("answer") or "",
        "images": data.get("images") or [],
        "sources": [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
            for r in (data.get("results") or [])
        ],
    }
