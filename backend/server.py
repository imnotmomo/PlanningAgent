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
    itinerary: dict
    change: str


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
async def revise(req: ReviseRequest) -> dict:
    return await agents.revision_agent(req.itinerary, req.change)


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
