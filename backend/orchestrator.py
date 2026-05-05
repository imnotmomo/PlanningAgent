"""
Two-phase agent pipeline so the user can review research candidates before
the rest of the planning runs. Supports multi-destination trips: research,
route, and itinerary run once per leg, then results are stitched together
with day numbers offset across the full trip.

  run_research()  → preference, missing_info, arrival (if origin),
                    research × N legs (parallel)
                    Ends with a `research_complete` event carrying all data.

  run_build()     → route × N legs, budget (whole trip), itinerary × N legs,
                    critic (whole trip), retry-replan if critic score < 7
                    Takes the user's selections from the picker UI.

  run_plan()      → composes both phases with no picker (selections = all).
"""
from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from . import agents


def _parse_price(s) -> int:
    """Pull the first integer dollar value out of a price string."""
    if not isinstance(s, str):
        return 0
    cleaned = s.replace(",", "").replace("$", "")
    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not nums:
        return 0
    return int(float(nums[0]))


def _normalize_destinations(prefs: dict) -> list[dict]:
    """Coerce destinations into a clean list of {city, country, days}."""
    dests = prefs.get("destinations") or []
    out: list[dict] = []
    for d in dests:
        if not isinstance(d, dict):
            continue
        city = (d.get("city") or "").strip()
        if not city:
            continue
        out.append({
            "city": city,
            "country": (d.get("country") or "").strip(),
            "days": int(d.get("days") or 0),
        })
    return out


def _city_prefs(prefs: dict, leg: dict) -> dict:
    """Build a per-city prefs dict the agents can consume."""
    return {
        **prefs,
        "destination": leg["city"],
        "country": leg.get("country"),
        "trip_length_days": leg["days"],
    }


async def _research_one_leg(prefs: dict, leg: dict) -> tuple[dict, dict]:
    """Returns (leg, research_dict). Tags each candidate with the leg's city."""
    res = await agents.research_agent(_city_prefs(prefs, leg))
    for cat in ("places", "restaurants", "hotels"):
        for item in res.get(cat, []):
            item["city"] = leg["city"]
    return leg, res


async def run_destinations(user_request: str) -> AsyncIterator[dict]:
    """Phase 0: extract preferences. If the user gave a country/region only
    (no specific cities), also run the destination suggester so the UI can
    show candidate cities to pick from."""
    yield {"event": "started", "payload": {"request": user_request}}

    yield {"event": "step", "payload": {"name": "preference", "status": "running"}}
    prefs = await agents.preference_agent(user_request)
    yield {"event": "step", "payload": {"name": "preference", "status": "done", "output": prefs}}

    yield {"event": "step", "payload": {"name": "missing_info", "status": "running"}}
    missing = agents.missing_info_agent(prefs)
    yield {"event": "step", "payload": {"name": "missing_info", "status": "done", "output": missing}}
    if missing:
        yield {"event": "incomplete", "payload": {"missing_fields": missing, "preferences": prefs}}
        return

    legs = _normalize_destinations(prefs)
    suggester: dict | None = None
    needs_resolution = (len(legs) == 0) and bool((prefs.get("country_or_region") or "").strip())

    if needs_resolution:
        yield {"event": "step", "payload": {"name": "destination_suggester", "status": "running"}}
        suggester = await agents.destination_suggester_agent(prefs)
        yield {"event": "step", "payload": {"name": "destination_suggester", "status": "done", "output": suggester}}

    yield {
        "event": "destinations_complete",
        "payload": {
            "preferences": prefs,
            "destinations": legs,           # may be empty if needs_resolution
            "needs_resolution": needs_resolution,
            "suggester": suggester,         # null if not needed
        },
    }


async def run_research(
    prefs: dict,
    destinations: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """Phase 1: arrival + per-leg research. Caller passes resolved destinations
    (either user-confirmed from the destination picker, or extracted directly
    from the user's prompt for explicit single/multi-city requests)."""
    yield {"event": "started", "payload": {"phase": "research"}}

    # Normalize whatever destinations we got — use prefs.destinations as
    # fallback for legacy single-shot callers.
    legs: list[dict] = []
    src = destinations if destinations is not None else (prefs.get("destinations") or [])
    for d in src:
        if isinstance(d, dict) and (d.get("city") or "").strip():
            legs.append({
                "city": d["city"].strip(),
                "country": (d.get("country") or "").strip(),
                "days": int(d.get("days") or 0),
            })
    if not legs:
        yield {"event": "incomplete", "payload": {"missing_fields": ["destinations"], "preferences": prefs}}
        return

    # Bake the resolved destinations into prefs so downstream agents see them.
    prefs = {**prefs, "destinations": legs, "trip_length_days": sum(d["days"] for d in legs) or prefs.get("trip_length_days")}

    first_city = legs[0]["city"]
    last_city = legs[-1]["city"]
    arrival_coro = agents.arrival_agent({
        **prefs,
        "destination": first_city,
        "first_city": first_city,
        "last_city": last_city,
    }) if prefs.get("origin") else None
    research_coros = [_research_one_leg(prefs, leg) for leg in legs]

    if arrival_coro is not None:
        yield {"event": "step", "payload": {"name": "arrival", "status": "running"}}
    yield {"event": "step", "payload": {"name": "research", "status": "running"}}

    if arrival_coro is not None:
        arrival, *research_results = await asyncio.gather(arrival_coro, *research_coros)
    else:
        arrival = None
        research_results = await asyncio.gather(*research_coros)

    if arrival is not None:
        yield {"event": "step", "payload": {"name": "arrival", "status": "done", "output": arrival}}

    aggregated: dict = {"places": [], "restaurants": [], "hotels": [], "by_city": {}}
    for leg, res in research_results:
        aggregated["by_city"][leg["city"]] = res
        for cat in ("places", "restaurants", "hotels"):
            aggregated[cat].extend(res.get(cat, []))
    yield {"event": "step", "payload": {"name": "research", "status": "done", "output": aggregated}}

    yield {
        "event": "research_complete",
        "payload": {
            "preferences": prefs,
            "destinations": legs,
            "arrival": arrival,
            "research": aggregated,
        },
    }


def _filter_by_names(items: list[dict], allowed: list[str] | None) -> list[dict]:
    if not allowed:
        return items
    allow_set = {a.strip().lower() for a in allowed}
    return [it for it in items if it.get("name", "").strip().lower() in allow_set]


def _airfare_from_choices(arrival_choices: dict | None) -> dict | None:
    """Convert user's outbound/return picks into an arrival blob. Computes the
    actual round-trip total per person in code (sum of one-way prices) so the
    LLM cannot mis-sum it."""
    if not isinstance(arrival_choices, dict):
        return None
    out = arrival_choices.get("outbound")
    ret = arrival_choices.get("return")
    if not out and not ret:
        return None
    out_price = _parse_price(out.get("price")) if isinstance(out, dict) else 0
    ret_price = _parse_price(ret.get("price")) if isinstance(ret, dict) else 0
    total = out_price + ret_price

    parts: list[str] = []
    for label, opt in (("Outbound", out), ("Return", ret)):
        if isinstance(opt, dict):
            line = f"{label}: {opt.get('mode','')} {opt.get('carrier','')} {opt.get('class','')} — {opt.get('price','')}"
            parts.append(line.strip())

    blob: dict = {
        "options": [{
            "mode": "round_trip",
            "carrier": " + ".join([o.get("carrier", "") for o in (out, ret) if isinstance(o, dict)]),
            "duration": ", ".join([o.get("duration", "") for o in (out, ret) if isinstance(o, dict)]),
            "price_range": " ; ".join(parts) + (f" (round-trip total: ${total:,})" if total else ""),
            "notes": "User-selected outbound and return.",
        }],
        "user_chosen": True,
    }
    if total > 0:
        blob["computed_airfare"] = {"low": total, "high": total}
    return blob


async def run_build(
    prefs: dict,
    research: dict,
    arrival: dict | None = None,
    selections: dict | None = None,
) -> AsyncIterator[dict]:
    """selections = {"places":[str],"restaurants":[str],"hotels":[str],
                     "arrival_choices": {"outbound":{...},"return":{...}}|None } or None."""
    sel = selections or {}
    by_city: dict = research.get("by_city") or {}

    # If the user picked specific outbound/return options, honor them in budget.
    user_arrival = _airfare_from_choices(sel.get("arrival_choices"))
    arrival_for_budget = user_arrival if user_arrival is not None else arrival

    # Apply user selections to the per-city candidates.
    legs = _normalize_destinations(prefs)
    if not legs:
        # fallback: synthesise one leg from a single-city prefs shape (legacy)
        legs = [{"city": prefs.get("destination") or "", "country": "", "days": prefs.get("trip_length_days", 0)}]

    # Selections become priority hints rather than hard filters: if the user
    # picked items, the pipeline prefers them; if they picked nothing, all
    # candidates are equally optional.
    place_picks = set((sel.get("places") or []))
    rest_picks = set((sel.get("restaurants") or []))

    # ----- route per leg -----
    yield {"event": "step", "payload": {"name": "route", "status": "running"}}
    leg_routes: dict[str, dict] = {}
    aggregated_route_groups: dict[str, list[str]] = {}
    aggregated_meal_plan: dict[str, dict] = {}
    aggregated_transit_notes: dict[str, str] = {}
    aggregated_day_schedule: dict[str, list] = {}
    day_offset = 0
    for leg in legs:
        city = leg["city"]
        leg_research = by_city.get(city) or {"places": [], "restaurants": [], "hotels": []}
        all_places = [p["name"] for p in leg_research.get("places", [])]
        all_rests = [r["name"] for r in leg_research.get("restaurants", [])]
        # Split into priority (user-picked) and optional within this city.
        attr_priority = [n for n in all_places if n in place_picks]
        attr_optional = [n for n in all_places if n not in place_picks]
        rest_priority = [n for n in all_rests if n in rest_picks]
        rest_optional = [n for n in all_rests if n not in rest_picks]
        # Pick the hotel for this city: the user-selected one if any (within
        # this city's candidates), else the first candidate as a fallback so
        # transit_notes still has a hotel to reference.
        hotel_picks = set((sel.get("hotels") or []))
        leg_hotels = leg_research.get("hotels", [])
        leg_hotel_name: str | None = None
        for h in leg_hotels:
            if h.get("name") in hotel_picks:
                leg_hotel_name = h["name"]
                break
        if leg_hotel_name is None and leg_hotels:
            leg_hotel_name = leg_hotels[0].get("name")

        leg_route = await agents.route_agent(
            {"priority": attr_priority, "optional": attr_optional},
            {"priority": rest_priority, "optional": rest_optional},
            leg["days"],
            city,
            pace=prefs.get("pace", "medium"),
            interests=prefs.get("interests", []),
            budget_level=prefs.get("budget_level", "medium"),
            hotel_name=leg_hotel_name,
        )
        leg_routes[city] = leg_route
        for i in range(1, leg["days"] + 1):
            inner_key = f"Day {i}"
            outer_key = f"Day {day_offset + i} · {city}"
            aggregated_route_groups[outer_key] = leg_route["route_groups"].get(inner_key, [])
            if leg_route["meal_plan"].get(inner_key):
                aggregated_meal_plan[outer_key] = leg_route["meal_plan"][inner_key]
            if leg_route["transit_notes"].get(inner_key):
                aggregated_transit_notes[outer_key] = leg_route["transit_notes"][inner_key]
            sched = leg_route.get("day_schedule", {}).get(inner_key)
            if isinstance(sched, list) and sched:
                aggregated_day_schedule[outer_key] = sched
        day_offset += leg["days"]

    aggregated_route = {
        "route_groups": aggregated_route_groups,
        "meal_plan": aggregated_meal_plan,
        "transit_notes": aggregated_transit_notes,
        "day_schedule": aggregated_day_schedule,
        "by_city": leg_routes,
    }
    yield {"event": "step", "payload": {"name": "route", "status": "done", "output": aggregated_route}}

    # Build the flat selected_for_lora list across all legs (for budget context).
    selected_for_lora: list[str] = []
    for city, leg_route in leg_routes.items():
        for day_list in (leg_route.get("route_groups") or {}).values():
            if isinstance(day_list, list):
                selected_for_lora.extend(day_list)
        for meals in (leg_route.get("meal_plan") or {}).values():
            if isinstance(meals, dict):
                for k in ("lunch", "dinner"):
                    v = meals.get(k)
                    if isinstance(v, str) and v and v not in selected_for_lora:
                        selected_for_lora.append(v)

    # ----- budget (whole trip) -----
    yield {"event": "step", "payload": {"name": "budget", "status": "running"}}
    budget = await agents.budget_agent(prefs, selected_for_lora, arrival=arrival_for_budget)
    yield {"event": "step", "payload": {"name": "budget", "status": "done", "output": budget}}

    # ----- itinerary per leg, concatenated with day-offset -----
    yield {"event": "step", "payload": {"name": "itinerary", "status": "running"}}
    combined_days: list[dict] = []
    leg_summaries: list[str] = []
    backup_options: list[str] = []
    travel_tips: list[str] = []
    day_offset = 0
    for leg in legs:
        city = leg["city"]
        leg_route = leg_routes.get(city, {})
        leg_selected = []
        for day_list in (leg_route.get("route_groups") or {}).values():
            if isinstance(day_list, list):
                leg_selected.extend(day_list)
        for meals in (leg_route.get("meal_plan") or {}).values():
            if isinstance(meals, dict):
                for k in ("lunch", "dinner"):
                    v = meals.get(k)
                    if isinstance(v, str) and v and v not in leg_selected:
                        leg_selected.append(v)
        leg_itin = await agents.itinerary_agent(
            _city_prefs(prefs, leg),
            leg_selected,
            leg_route.get("route_groups", {}),
        )
        if leg_itin.get("trip_summary"):
            leg_summaries.append(leg_itin["trip_summary"])
        # Truncate to leg's day count and renumber sequentially with offset.
        # The LoRA occasionally outputs duplicate or extra day objects; this
        # is the single safety net.
        leg_days = leg_itin.get("daily_itinerary") or []
        if not isinstance(leg_days, list):
            leg_days = []
        leg_days = leg_days[: int(leg.get("days") or 0) or len(leg_days)]
        for i, day_obj in enumerate(leg_days):
            if isinstance(day_obj, dict):
                day_obj["day"] = day_offset + i + 1
                day_obj["city"] = city
                combined_days.append(day_obj)
        backup_options.extend(leg_itin.get("backup_options") or [])
        travel_tips.extend(leg_itin.get("travel_tips") or [])
        day_offset += int(leg.get("days") or 0)
    # Final safety net: cap to trip_length_days regardless of leg overflows.
    expected_total = int(prefs.get("trip_length_days") or 0)
    if expected_total and len(combined_days) > expected_total:
        combined_days = combined_days[:expected_total]

    cities_label = ", ".join(leg["city"] for leg in legs)
    combined_summary = f"A {prefs.get('trip_length_days')}-day trip across {cities_label}."
    if leg_summaries:
        combined_summary += " " + " ".join(leg_summaries)
    itinerary = {
        "trip_summary": combined_summary,
        "daily_itinerary": combined_days,
        "budget_summary": "",
        "backup_options": backup_options[:8],
        "travel_tips": list(dict.fromkeys(travel_tips))[:8],
    }
    # Replace the LoRA's templated tips with ones grounded in the actual
    # day plan (specific days, places, constraints).
    specific_tips = await agents.tips_agent(itinerary, prefs)
    if specific_tips:
        itinerary["travel_tips"] = specific_tips
    yield {"event": "step", "payload": {"name": "itinerary", "status": "done", "output": itinerary}}

    # ----- critic (whole trip) with retry-replan if score < 7 -----
    yield {"event": "step", "payload": {"name": "critic", "status": "running"}}
    critique = await agents.critic_agent(itinerary, prefs, budget=budget)
    yield {"event": "step", "payload": {"name": "critic", "status": "done", "output": critique}}

    # When critic fails, re-run route + itinerary across all legs with the
    # critic's feedback in the route prompt. This is more effective than
    # text-only revision because route can re-group days structurally.
    MAX_RETRIES = 2
    retries = 0
    while not critique.get("passed", True) and retries < MAX_RETRIES:
        retries += 1
        feedback_lines = critique.get("issues", []) + critique.get("suggested_revisions", [])
        feedback = (
            f"Previous plan scored {critique.get('score','?')}/10 (must be ≥7). "
            f"Apply these fixes while preserving every priority item:\n- "
            + "\n- ".join(feedback_lines)
        )

        # ---- Replan: route × N legs with feedback ----
        yield {"event": "step", "payload": {"name": "route", "status": "running", "is_retry": True, "retry_round": retries}}
        leg_routes_v2: dict[str, dict] = {}
        new_route_groups: dict[str, list[str]] = {}
        new_meal_plan: dict[str, dict] = {}
        new_transit_notes: dict[str, str] = {}
        day_offset = 0
        for leg in legs:
            city = leg["city"]
            leg_research = by_city.get(city) or {"places": [], "restaurants": [], "hotels": []}
            all_places = [p["name"] for p in leg_research.get("places", [])]
            all_rests = [r["name"] for r in leg_research.get("restaurants", [])]
            attr_priority = [n for n in all_places if n in place_picks]
            attr_optional = [n for n in all_places if n not in place_picks]
            rest_priority = [n for n in all_rests if n in rest_picks]
            rest_optional = [n for n in all_rests if n not in rest_picks]
            hotel_picks = set((sel.get("hotels") or []))
            leg_hotels = leg_research.get("hotels", [])
            leg_hotel_name = next((h["name"] for h in leg_hotels if h.get("name") in hotel_picks), None) \
                or (leg_hotels[0].get("name") if leg_hotels else None)
            leg_route_v2 = await agents.route_agent(
                {"priority": attr_priority, "optional": attr_optional},
                {"priority": rest_priority, "optional": rest_optional},
                leg["days"], city,
                pace=prefs.get("pace", "medium"),
                interests=prefs.get("interests", []),
                budget_level=prefs.get("budget_level", "medium"),
                hotel_name=leg_hotel_name,
                feedback=feedback,
            )
            leg_routes_v2[city] = leg_route_v2
            lr_route = leg_route_v2.get("route_groups") if isinstance(leg_route_v2.get("route_groups"), dict) else {}
            lr_meal = leg_route_v2.get("meal_plan") if isinstance(leg_route_v2.get("meal_plan"), dict) else {}
            lr_transit = leg_route_v2.get("transit_notes") if isinstance(leg_route_v2.get("transit_notes"), dict) else {}
            for i in range(1, leg["days"] + 1):
                inner_key = f"Day {i}"
                outer_key = f"Day {day_offset + i} · {city}"
                new_route_groups[outer_key] = lr_route.get(inner_key, []) if isinstance(lr_route, dict) else []
                if lr_meal.get(inner_key):
                    new_meal_plan[outer_key] = lr_meal[inner_key]
                if lr_transit.get(inner_key):
                    new_transit_notes[outer_key] = lr_transit[inner_key]
            day_offset += leg["days"]
        leg_routes = leg_routes_v2
        aggregated_route_groups = new_route_groups
        aggregated_meal_plan = new_meal_plan
        aggregated_transit_notes = new_transit_notes
        yield {"event": "step", "payload": {"name": "route", "status": "done", "output": {
            "route_groups": aggregated_route_groups,
            "meal_plan": aggregated_meal_plan,
            "transit_notes": aggregated_transit_notes,
        }, "is_retry": True, "retry_round": retries}}

        # ---- Replan: itinerary per leg ----
        yield {"event": "step", "payload": {"name": "itinerary", "status": "running", "is_retry": True, "retry_round": retries}}
        combined_days_v2: list[dict] = []
        leg_summaries_v2: list[str] = []
        backup_options_v2: list[str] = []
        travel_tips_v2: list[str] = []
        day_offset = 0
        for leg in legs:
            city = leg["city"]
            lr = leg_routes.get(city, {})
            lr_route = lr.get("route_groups") if isinstance(lr.get("route_groups"), dict) else {}
            lr_meal = lr.get("meal_plan") if isinstance(lr.get("meal_plan"), dict) else {}
            leg_selected: list[str] = []
            for dl in lr_route.values():
                if isinstance(dl, list):
                    leg_selected.extend(dl)
            for meals in lr_meal.values():
                if isinstance(meals, dict):
                    for k in ("lunch", "dinner"):
                        v = meals.get(k)
                        if isinstance(v, str) and v and v not in leg_selected:
                            leg_selected.append(v)
            leg_itin = await agents.itinerary_agent(_city_prefs(prefs, leg), leg_selected, lr.get("route_groups", {}))
            if leg_itin.get("trip_summary"):
                leg_summaries_v2.append(leg_itin["trip_summary"])
            leg_days = leg_itin.get("daily_itinerary") or []
            if not isinstance(leg_days, list):
                leg_days = []
            leg_days = leg_days[: int(leg.get("days") or 0) or len(leg_days)]
            for i, day_obj in enumerate(leg_days):
                if isinstance(day_obj, dict):
                    day_obj["day"] = day_offset + i + 1
                    day_obj["city"] = city
                    combined_days_v2.append(day_obj)
            backup_options_v2.extend(leg_itin.get("backup_options") or [])
            travel_tips_v2.extend(leg_itin.get("travel_tips") or [])
            day_offset += int(leg.get("days") or 0)
        expected_total_v2 = int(prefs.get("trip_length_days") or 0)
        if expected_total_v2 and len(combined_days_v2) > expected_total_v2:
            combined_days_v2 = combined_days_v2[:expected_total_v2]
        cities_label = ", ".join(leg["city"] for leg in legs)
        itinerary = {
            "trip_summary": f"A {prefs.get('trip_length_days')}-day trip across {cities_label}." +
                            ((" " + " ".join(leg_summaries_v2)) if leg_summaries_v2 else ""),
            "daily_itinerary": combined_days_v2,
            "budget_summary": "",
            "backup_options": backup_options_v2[:8],
            "travel_tips": list(dict.fromkeys(travel_tips_v2))[:8],
        }
        specific_tips_retry = await agents.tips_agent(itinerary, prefs)
        if specific_tips_retry:
            itinerary["travel_tips"] = specific_tips_retry
        yield {"event": "step", "payload": {"name": "itinerary", "status": "done", "output": itinerary, "is_retry": True, "retry_round": retries}}

        # ---- Re-critic ----
        yield {"event": "step", "payload": {"name": "critic", "status": "running", "is_retry": True, "retry_round": retries}}
        critique = await agents.critic_agent(itinerary, prefs, budget=budget)
        yield {"event": "step", "payload": {"name": "critic", "status": "done", "output": critique, "is_retry": True, "retry_round": retries}}

    yield {
        "event": "complete",
        "payload": {
            "preferences": prefs,
            "destinations": legs,
            "arrival": arrival,
            "arrival_choices": sel.get("arrival_choices") or None,
            "places": research.get("places", []),
            "restaurants": research.get("restaurants", []),
            "hotels": research.get("hotels", []),
            "route_groups": aggregated_route_groups,
            "meal_plan": aggregated_meal_plan,
            "transit_notes": aggregated_transit_notes,
            "day_schedule": aggregated_day_schedule,
            "budget": budget,
            "itinerary": itinerary,
            "critique": critique,
        },
    }


async def run_plan(user_request: str) -> AsyncIterator[dict]:
    """Full pipeline (no pickers): destinations → research → build."""
    prefs: dict | None = None
    legs: list[dict] = []
    suggester: dict | None = None

    async for ev in run_destinations(user_request):
        yield ev
        if ev["event"] == "destinations_complete":
            prefs = ev["payload"]["preferences"]
            legs = ev["payload"]["destinations"]
            suggester = ev["payload"].get("suggester")
        elif ev["event"] == "incomplete":
            return
    if prefs is None:
        return

    if not legs and suggester:
        legs = suggester.get("default_split") or []

    research_payload: dict | None = None
    async for ev in run_research(prefs, destinations=legs):
        yield ev
        if ev["event"] == "research_complete":
            research_payload = ev["payload"]
        elif ev["event"] == "incomplete":
            return
    if research_payload is None:
        return

    async for ev in run_build(
        prefs=research_payload["preferences"],
        research=research_payload["research"],
        arrival=research_payload["arrival"],
        selections=None,
    ):
        yield ev


# --------------------------- streaming revise ---------------------------

def _names(items: list) -> list[str]:
    out = []
    for it in items or []:
        if isinstance(it, dict):
            n = it.get("name")
            if n:
                out.append(n)
        elif isinstance(it, str):
            out.append(it)
    return out


async def run_revise(result: dict, change: str) -> AsyncIterator[dict]:
    """Smart-routed, streaming revision. Picks one of:
      - text:        revision_agent only (cheap, surface tweaks)
      - structural:  route + itinerary + critic (re-shapes the day plan)
      - budget:      budget recompute + itinerary text revision
    Emits agent step events the same way the planning pipeline does, then
    a `complete` event with the partial PlanResult fields that changed.
    """
    itinerary = result.get("itinerary") or {}
    prefs = result.get("preferences") or {}
    arrival = result.get("arrival")

    # Router
    yield {"event": "step", "payload": {"name": "revision_router", "status": "running"}}
    decision = await agents.revision_router(itinerary, change)
    yield {"event": "step", "payload": {"name": "revision_router", "status": "done", "output": decision}}
    category = decision["category"]

    place_names = _names(result.get("places", []))
    restaurant_names = _names(result.get("restaurants", []))
    hotel_names = _names(result.get("hotels", []))
    hotel_name = hotel_names[0] if hotel_names else None

    if category == "text":
        yield {"event": "step", "payload": {"name": "revision", "status": "running"}}
        new_itin = await agents.revision_agent(itinerary, change)
        yield {"event": "step", "payload": {"name": "revision", "status": "done"}}
        yield {"event": "complete", "payload": {
            "category": category,
            "reason": decision["reason"],
            "itinerary": new_itin,
        }}
        return

    if category == "structural":
        destinations = result.get("destinations") or []
        trip_length_days = sum(int(d.get("days", 0) or 0) for d in destinations) \
            or len(itinerary.get("daily_itinerary") or []) \
            or int(prefs.get("trip_length_days") or 1)
        destination_label = ", ".join(d.get("city", "") for d in destinations if d.get("city")) \
            or prefs.get("destination") or "trip"

        # First pass uses the user's change as feedback. If the critic fails
        # we run another round with the critic's issues appended to the
        # feedback, mirroring the planning path's replan loop.
        feedback = change
        new_itin: dict = {}
        route_data: dict = {}
        critique: dict = {}
        MAX_REVISE_RETRIES = 2
        for attempt in range(MAX_REVISE_RETRIES + 1):
            is_retry = attempt > 0
            retry_evt = {"is_retry": True, "retry_round": attempt} if is_retry else {}

            yield {"event": "step", "payload": {"name": "route", "status": "running", **retry_evt}}
            route_data = await agents.route_agent(
                attractions=place_names,
                restaurants=restaurant_names,
                trip_length_days=trip_length_days,
                destination=destination_label,
                pace=prefs.get("pace", "medium"),
                interests=prefs.get("interests", []),
                budget_level=prefs.get("budget_level", "medium"),
                hotel_name=hotel_name,
                feedback=feedback,
            )
            yield {"event": "step", "payload": {"name": "route", "status": "done", "output": route_data, **retry_evt}}

            yield {"event": "step", "payload": {"name": "itinerary", "status": "running", **retry_evt}}
            new_itin = await agents.itinerary_agent(
                prefs, place_names + restaurant_names, route_data.get("route_groups", {})
            )
            rev_tips = await agents.tips_agent(new_itin, prefs)
            if rev_tips:
                new_itin["travel_tips"] = rev_tips
            yield {"event": "step", "payload": {"name": "itinerary", "status": "done", **retry_evt}}

            yield {"event": "step", "payload": {"name": "critic", "status": "running", **retry_evt}}
            critique = await agents.critic_agent(new_itin, prefs, budget=result.get("budget") or {})
            yield {"event": "step", "payload": {"name": "critic", "status": "done", "output": critique, **retry_evt}}

            if critique.get("passed", True) or attempt == MAX_REVISE_RETRIES:
                break
            # Append critic feedback for next round, keeping original change first.
            issues = "; ".join(critique.get("issues") or [])
            sugg = "; ".join(critique.get("suggested_revisions") or [])
            feedback = (
                f"User change: {change}\n"
                f"Previous attempt failed critic ({critique.get('score')}/10).\n"
                f"Issues: {issues}\n"
                f"Suggested fixes: {sugg}"
            )

        yield {"event": "complete", "payload": {
            "category": category,
            "reason": decision["reason"],
            "itinerary": new_itin,
            "route_groups": route_data.get("route_groups", {}),
            "meal_plan": route_data.get("meal_plan", {}),
            "transit_notes": route_data.get("transit_notes", {}),
            "day_schedule": route_data.get("day_schedule", {}),
            "critique": critique,
        }}
        return

    # category == "budget"
    # A budget tier shift means the actual hotels/restaurants/places at the
    # new tier are different — recomputing totals over the existing
    # candidates would be wrong. Re-research at the new tier, then re-run
    # route + itinerary + critic + budget like a fresh build.
    new_tier = decision.get("new_budget_level") or prefs.get("budget_level") or "medium"
    new_prefs = dict(prefs)
    new_prefs["budget_level"] = new_tier
    new_prefs["constraints"] = list(prefs.get("constraints") or []) + [f"BUDGET CHANGE: {change}"]

    destinations = result.get("destinations") or []
    if not destinations:
        destinations = [{
            "city": prefs.get("destination") or "trip",
            "days": int(prefs.get("trip_length_days") or len(itinerary.get("daily_itinerary") or []) or 1),
        }]

    # Re-research per leg at the new tier (parallel)
    yield {"event": "step", "payload": {"name": "research", "status": "running"}}
    leg_research = []
    research_coros = []
    for leg in destinations:
        leg_prefs = dict(new_prefs)
        leg_prefs["destination"] = leg.get("city") or new_prefs.get("destination")
        leg_prefs["trip_length_days"] = int(leg.get("days") or 0) or new_prefs.get("trip_length_days")
        research_coros.append(agents.research_agent(leg_prefs))
    leg_research = await asyncio.gather(*research_coros)
    aggregated_places: list[dict] = []
    aggregated_restaurants: list[dict] = []
    aggregated_hotels: list[dict] = []
    for leg, r in zip(destinations, leg_research):
        for p in r.get("places") or []:
            if isinstance(p, dict):
                p.setdefault("city", leg.get("city"))
                aggregated_places.append(p)
        for r_ in r.get("restaurants") or []:
            if isinstance(r_, dict):
                r_.setdefault("city", leg.get("city"))
                aggregated_restaurants.append(r_)
        for h in r.get("hotels") or []:
            if isinstance(h, dict):
                h.setdefault("city", leg.get("city"))
                aggregated_hotels.append(h)
    yield {"event": "step", "payload": {"name": "research", "status": "done", "output": {
        "places": aggregated_places, "restaurants": aggregated_restaurants, "hotels": aggregated_hotels,
    }}}

    new_place_names = _names(aggregated_places)
    new_restaurant_names = _names(aggregated_restaurants)
    new_hotel_names = _names(aggregated_hotels)
    new_hotel = new_hotel_names[0] if new_hotel_names else None
    trip_length_days = sum(int(d.get("days", 0) or 0) for d in destinations) or int(new_prefs.get("trip_length_days") or 1)
    destination_label = ", ".join(d.get("city", "") for d in destinations if d.get("city")) or new_prefs.get("destination") or "trip"

    yield {"event": "step", "payload": {"name": "route", "status": "running"}}
    route_data = await agents.route_agent(
        attractions=new_place_names,
        restaurants=new_restaurant_names,
        trip_length_days=trip_length_days,
        destination=destination_label,
        pace=new_prefs.get("pace", "medium"),
        interests=new_prefs.get("interests", []),
        budget_level=new_tier,
        hotel_name=new_hotel,
        feedback=change,
    )
    yield {"event": "step", "payload": {"name": "route", "status": "done", "output": route_data}}

    yield {"event": "step", "payload": {"name": "itinerary", "status": "running"}}
    new_itin = await agents.itinerary_agent(
        new_prefs, new_place_names + new_restaurant_names, route_data.get("route_groups", {})
    )
    rev_tips = await agents.tips_agent(new_itin, new_prefs)
    if rev_tips:
        new_itin["travel_tips"] = rev_tips
    yield {"event": "step", "payload": {"name": "itinerary", "status": "done"}}

    yield {"event": "step", "payload": {"name": "budget", "status": "running"}}
    new_budget = await agents.budget_agent(
        new_prefs, new_place_names + new_restaurant_names, arrival=arrival
    )
    yield {"event": "step", "payload": {"name": "budget", "status": "done", "output": new_budget}}

    yield {"event": "step", "payload": {"name": "critic", "status": "running"}}
    critique = await agents.critic_agent(new_itin, new_prefs, budget=new_budget)
    yield {"event": "step", "payload": {"name": "critic", "status": "done", "output": critique}}

    yield {"event": "complete", "payload": {
        "category": category,
        "reason": decision["reason"],
        "new_budget_level": new_tier,
        "itinerary": new_itin,
        "budget": new_budget,
        "places": aggregated_places,
        "restaurants": aggregated_restaurants,
        "hotels": aggregated_hotels,
        "route_groups": route_data.get("route_groups", {}),
        "meal_plan": route_data.get("meal_plan", {}),
        "transit_notes": route_data.get("transit_notes", {}),
        "day_schedule": route_data.get("day_schedule", {}),
        "critique": critique,
        "preferences": new_prefs,
    }}
