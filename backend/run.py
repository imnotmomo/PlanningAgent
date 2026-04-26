"""End-to-end pipeline runner. Useful for ad-hoc testing before wiring FastAPI.

Usage:
  cd /Users/zhelichen/Downloads/travel-agent
  python -m backend.run "Plan a 3-day trip to Kyoto for 2 adults..."
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from . import agents


async def run_pipeline(user_request: str, change_requests: list[str] | None = None) -> dict:
    prefs = await agents.preference_agent(user_request)
    missing = agents.missing_info_agent(prefs)
    if missing:
        return {"status": "incomplete", "missing_fields": missing, "preferences": prefs}

    places = await agents.research_agent(prefs)
    route_groups = await agents.route_agent(
        places, prefs["trip_length_days"], prefs["destination"]
    )
    budget = await agents.budget_agent(prefs, places)
    itinerary = await agents.itinerary_agent(prefs, places, route_groups)
    critique = await agents.critic_agent(itinerary, prefs)

    for change in (change_requests or []):
        itinerary = await agents.revision_agent(itinerary, change)

    return {
        "status": "ok",
        "preferences": prefs,
        "places": places,
        "route_groups": route_groups,
        "budget": budget,
        "itinerary": itinerary,
        "critique": critique,
    }


def main():
    req = sys.argv[1] if len(sys.argv) > 1 else (
        "Plan a 3-day trip to Tokyo for 2 friends. We like anime, food, "
        "and city views. Medium budget, prefer public transit."
    )
    result = asyncio.run(run_pipeline(req))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
