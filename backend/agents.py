"""
Agent function skeletons for the TripWise multi-agent pipeline.

Each non-itinerary agent calls `orch_complete` (or `orch_complete_with_tools`
when web search / Python execution is useful). The Itinerary agent calls
`itin_complete` (the fine-tuned model).

Tool-using agents:
- Research: tavily_search (find real places at the destination)
- Budget:   tavily_search (real prices) + python_exec (arithmetic)
- Critic:   python_exec (time/feasibility math)

System prompts here are starting points. Tune as you build evaluation.
"""
from __future__ import annotations

import json

from .llm import orch_complete, orch_complete_with_tools, itin_complete
from .tools import TAVILY_TOOL, PYTHON_EXEC_TOOL, run_tool


def _extract_json(text: str) -> dict | list:
    """Robust JSON extraction. Tries: direct parse, fenced ```json blocks,
    then ALL balanced `{...}` / `[...]` substrings — not just the first.
    This is critical for Qwen3 thinking mode, which often shows malformed
    example JSON inside its thinking preamble before emitting the real
    answer below."""
    s = text.strip()

    # Strip code fences if the entire content is a single ```json...``` block
    if s.startswith("```"):
        rest = s[3:]
        if rest.lower().startswith("json"):
            rest = rest[4:]
        end = rest.rfind("```")
        if end >= 0:
            try:
                return json.loads(rest[:end].strip())
            except json.JSONDecodeError:
                pass

    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Walk every balanced block; return the first that parses cleanly.
    # The "real" answer is usually the LAST block (after thinking), so we
    # collect candidates and prefer the last successful parse.
    candidates: list = []
    for opener, closer in (("{", "}"), ("[", "]")):
        pos = 0
        while True:
            start = s.find(opener, pos)
            if start < 0:
                break
            depth = 0
            in_str = False
            esc = False
            end_idx = -1
            for i in range(start, len(s)):
                c = s[i]
                if esc:
                    esc = False
                    continue
                if c == "\\" and in_str:
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx < 0:
                break  # unbalanced, move on
            try:
                candidates.append(json.loads(s[start:end_idx + 1]))
            except json.JSONDecodeError:
                pass
            pos = start + 1
    if candidates:
        # Prefer the last (i.e. the answer after thinking) — and prefer dicts
        # over lists when both are present at equal trailing positions.
        return candidates[-1]
    raise ValueError(f"could not extract JSON from: {text[:500]}")


# --------------------------- preference ---------------------------

PREFERENCE_SYSTEM = """You extract structured travel preferences from a free-form user request.

Output a single JSON object with exactly these fields:
{
  "destinations": [                    // list of legs — empty if not specified
    {"city": str, "country": str, "days": int}
  ],
  "country_or_region": str | null,     // set if user gave a country/region but no specific city
  "trip_length_days": int,             // total trip length
  "origin": str | null,                // city/airport user is traveling FROM, null if not stated
  "travelers": str,
  "budget_level": "low" | "medium" | "high" | "luxury",
  "interests": [str],
  "pace": "relaxed" | "medium" | "packed",
  "constraints": [str]
}

Rules for `destinations` and `country_or_region`:

1. SINGLE-CITY EXPLICIT — "3 days in Tokyo", "weekend in Paris" →
   destinations=[{city,country,days}], country_or_region=null.

2. MULTI-CITY EXPLICIT — user lists cities OR uses tour-like wording with
   specific cities:
   - "3 days Paris, 2 days Rome"     → destinations with the exact split
   - "Japan trip: Tokyo and Kyoto"   → both cities, you allocate days
   destinations populated, country_or_region=null.

3. COUNTRY-ONLY ("5 days in Japan", "trip to Italy") →
   destinations=[], country_or_region="Japan" (or "Italy", etc.)
   The downstream destination-suggester will propose cities for the user
   to pick from. DO NOT pre-pick cities here.

4. REGION ONLY ("a week in Europe", "Southeast Asia trip", "Balkans tour") →
   destinations=[], country_or_region="Europe" (or "Southeast Asia", etc.)
   Same: leave empty, downstream agent suggests cities.

5. TOUR-LIKE COUNTRY ("tour of Italy", "across Spain", "Japan tour") →
   treat as country-only — leave destinations empty so user can pick cities.

6. If destinations IS populated, the sum of `days` must equal trip_length_days.

When the user is ambiguous about cities, leave destinations EMPTY and set
country_or_region. Never silently default to a single city for a country.

Rules for other fields:
- origin: only set if the user explicitly says where they're traveling FROM
  (e.g. "from San Francisco"). If not stated, null.
- "extra high"/"unlimited"/"VIP"/"Aman-tier" → budget_level: "luxury".
- "expensive"/"premium" → "high". "budget"/"backpacker"/"cheap" → "low".

If a field is not stated, use null for strings/ints, [] for lists, "medium" for
budget_level/pace. Output JSON only, no preamble or trailing text."""


async def preference_agent(user_request: str) -> dict:
    raw = await orch_complete(PREFERENCE_SYSTEM, user_request, response_format_json=True)
    return _extract_json(raw)


# --------------------------- missing-info ---------------------------

REQUIRED_FIELDS = ("destinations_or_region", "trip_length_days")


def missing_info_agent(prefs: dict) -> list[str]:
    """Pure Python — no LLM call. We require trip_length_days, plus EITHER
    a populated destinations list OR a country_or_region the suggester can
    use. Empty-destinations + empty-country_or_region is the only fail."""
    missing: list[str] = []
    if not prefs.get("trip_length_days"):
        missing.append("trip_length_days")
    dests = prefs.get("destinations") or []
    region = (prefs.get("country_or_region") or "").strip()
    if (not isinstance(dests, list) or len(dests) == 0) and not region:
        missing.append("destination")
    return missing


# --------------------------- destination suggester (country/region) -----------

DESTINATION_SUGGESTER_SYSTEM = """You suggest candidate cities for a trip when
the user only specified a country or region.

Output JSON with EXACTLY:
{
  "candidates": [
    {
      "city": str,
      "country": str,
      "description": str,    // 1-2 sentences: vibe + signature interests
      "suggested_days": int  // a sensible standalone duration for this city
    },
    ...
  ],
  "default_split": [
    {"city": str, "country": str, "days": int}    // sums to trip_length_days
  ]
}

NUMBER OF CANDIDATES — the input includes `REQUIRED_candidate_count`. You
MUST output EXACTLY that many UNIQUE cities. Examples:
   3 days → 2 candidates    4 days → 3    5 days → 4
   7 days → 6                10+ days → 7
Each city name must be unique — do not repeat the same city twice.
If the country/region doesn't have enough distinct iconic cities, broaden
into well-known smaller cities or named regions, but still produce N unique
entries.

DEFAULT SPLIT (conservative — favor fewer cities):
  trip ≤ 5 days   → 1 city only (the single best fit for interests)
  6-9 days        → 2 cities
  10-14 days      → 3 cities
  15+ days        → 4 cities
  When using >1 city, distribute days roughly proportional to each city's
  typical depth (e.g. Tokyo 4 / Kyoto 3 for 7 days).

Other rules:
- For a country: include the obvious capital/largest city plus alternates
  aligned to the user's interests (Japan+temples → Tokyo, Kyoto, Nara, Osaka,
  Kanazawa; Italy+food → Rome, Florence, Bologna, Naples).
- For a region: pick well-known cities distributed across the region.
- The default_split's days MUST sum to trip_length_days.
- Output JSON only.
"""


def _target_candidate_count(days: int) -> int:
    return min(7, max(2, days - 1))


def _target_default_city_count(days: int) -> int:
    if days <= 5:
        return 1
    if days <= 9:
        return 2
    if days <= 14:
        return 3
    return 4


def _normalize_default_split(split: list, days: int) -> list:
    """Trim/pad and rescale days so the default_split has the right number of
    cities and sums to trip_length_days."""
    target_n = _target_default_city_count(days)
    cities = [d for d in (split or []) if isinstance(d, dict) and d.get("city")]
    if not cities or days <= 0:
        return cities
    # Trim to target count
    cities = cities[:target_n]
    # Rescale days proportional to original allocation, then snap to int sum=days
    total = sum(int(d.get("days", 0) or 0) for d in cities) or 1
    if total <= 0:
        # If model gave no day counts, distribute evenly
        even = days // len(cities)
        rem = days - even * len(cities)
        for i, d in enumerate(cities):
            d["days"] = even + (1 if i < rem else 0)
        return cities
    ratio = days / total
    for d in cities:
        d["days"] = max(1, round(int(d.get("days", 0) or 0) * ratio))
    diff = days - sum(d["days"] for d in cities)
    if diff != 0 and cities:
        cities[0]["days"] = max(1, cities[0]["days"] + diff)
    return cities


async def destination_suggester_agent(prefs: dict) -> dict:
    days = int(prefs.get("trip_length_days") or 0)
    target_n = _target_candidate_count(days)
    target_default = _target_default_city_count(days)
    payload = json.dumps({
        "country_or_region": prefs.get("country_or_region"),
        "trip_length_days": days,
        "interests": prefs.get("interests", []),
        "budget_level": prefs.get("budget_level", "medium"),
        "pace": prefs.get("pace", "medium"),
        "travelers": prefs.get("travelers"),
        "REQUIRED_candidate_count": target_n,
        "REQUIRED_default_split_city_count": target_default,
    })
    raw = await orch_complete(DESTINATION_SUGGESTER_SYSTEM, payload, response_format_json=True)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        obj = {}
    raw_candidates = obj.get("candidates") or []
    # Dedupe by city name (case-insensitive) — the LLM occasionally repeats
    # the same city. Then trim to target.
    seen: set[str] = set()
    candidates: list[dict] = []
    for c in raw_candidates:
        if not isinstance(c, dict):
            continue
        key = (c.get("city") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(c)
        if len(candidates) >= target_n:
            break
    default_split = _normalize_default_split(obj.get("default_split") or [], days)
    return {"candidates": candidates, "default_split": default_split}


# --------------------------- research ---------------------------

RESEARCH_SYSTEM = """You research candidates for the user's trip — grounded in
live web search. The downstream itinerary agent will SELECT from these
candidates; you do not need to be minimal. Provide a healthy menu.

Process:
1. Call `tavily_search` 1-3 times. Suggested queries:
   - "<destination> top attractions for <interests>"
   - "<budget_level> restaurants <destination>"
   - For hotels, query within the user's price tier (see HOTEL TIERS below).
     For medium: "boutique hotels <destination> $150 per night" or
                 "4 star hotels <destination>".
     For luxury: "Aman <destination>" / "<destination> Ritz Carlton".
2. From results, pick specific real names with a short description grounded
   in the search.

Output JSON with EXACTLY these three top-level keys, each a list of objects
with `name` and `description` (1-2 short sentences):
{
  "places":      [{"name": str, "description": str}, ...],   // 6-12 items
  "restaurants": [{"name": str, "description": str}, ...],   // 4-8 items
  "hotels":      [{"name": str, "description": str}, ...]    // 3-6 items
}

HOTEL TIERS — match the user's budget_level by CATEGORY, not absolute price.
Prices vary by destination and season; pick by what kind of place this is.
  low:    hostels, capsule hotels, budget chains, simple guesthouses
            (Toyoko Inn, APA, K's House, hostelworld picks)
  medium: 3-4 star hotels, mid-range boutique
            (Hyatt Place, Marriott Courtyard, Hotel Sunroute, Mitsui Garden,
             Hotel Granvia, Hyatt Centric, Holiday Inn Express, Citadines)
  high:   4-5 star, premium boutique
            (Park Hyatt, Conrad, Andaz, ANA InterContinental, JW Marriott,
             Westin, Sheraton Premier, Edition)
  luxury: ultra-luxury hotels and exclusive ryokan
            (Aman, Four Seasons, Ritz-Carlton, Mandarin Oriental, Rosewood,
             Hoshinoya, Bulgari, St. Regis, Capella)

CRITICAL: Pick hotels in the user's category. Do NOT downgrade — for medium
do not include hostels; for high do not include 3-star chains. When searching
Tavily, query by category, e.g. "3 star hotels in <destination>" or
"4 star boutique <destination>" or "Aman <destination>". Mention rough rate
in description ONLY when the search told you a number, otherwise just describe
the hotel's category and what makes it distinctive.

RESTAURANT TIERS — same principle: pick by category, not by absolute price:
  low:    casual eateries, street food, chains, neighborhood izakaya
  medium: nice sit-down, well-reviewed neighborhood favorites
  high:   fine dining, chef-driven concepts, premium tasting menus
  luxury: Michelin-starred (1-3 stars), kaiseki masters, top-tier omakase

Other rules:
- Every entry must be a REAL specific name (e.g. "Nakamura Tokichi", not
  "a tea shop"). Description should mention what makes it distinctive plus
  price/tier when relevant.
- Output JSON only, no preamble.
"""


async def research_agent(prefs: dict) -> dict:
    raw = await orch_complete_with_tools(
        RESEARCH_SYSTEM,
        json.dumps(prefs),
        tools=[TAVILY_TOOL],
        run_tool=run_tool,
        response_format_json=True,
        max_steps=8,
    )
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        obj = {}

    def _normalize(items: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if isinstance(it, str):
                out.append({"name": it, "description": ""})
            elif isinstance(it, dict) and "name" in it:
                out.append({
                    "name": str(it["name"]),
                    "description": str(it.get("description", "") or ""),
                })
        return out

    return {
        "places": _normalize(obj.get("places")),
        "restaurants": _normalize(obj.get("restaurants")),
        "hotels": _normalize(obj.get("hotels")),
    }


# --------------------------- route ---------------------------

ROUTE_SYSTEM = """You build a feasible day plan: pick attraction stops, assign
lunch + dinner restaurants, and write a per-day transit note matching the
budget tier.

INPUT SHAPE: `attractions` and `restaurants` may each contain a `priority` list
(items the user explicitly asked for) and an `optional` list (the rest of the
candidates).

RETRY FEEDBACK (NON-NEGOTIABLE WHEN PRESENT): if the input includes a
non-null `feedback` field, a critic just rejected a previous plan. Re-group
the days to address every issue mentioned. Common fixes the critic asks for:
  - Split crowded days across two days
  - Move geographically distant stops to different days
  - Add missing-interest activities (e.g. matcha, food) to under-utilized days
  - Re-order so transit between same-day stops is short
You still must respect the priority rules below.

PRIORITY RULES (NON-NEGOTIABLE):
- Every name in `priority` MUST appear in your output `route_groups` (or the
  `meal_plan` for restaurants). No exceptions.
- If `priority` count is larger than the per-day cap × `trip_length_days`,
  include them all anyway — go over the cap. The user explicitly asked for
  them; that overrides pace.
- Only AFTER all `priority` items are placed should you fill remaining slots
  from `optional`.
- For restaurants: `priority` ones become lunch and dinner first; `optional`
  fills the rest of the meals.
- For places: `priority` ones become route_groups stops first; `optional` fills.

ATTRACTION COUNT BY PACE (this is the count of ATTRACTIONS only — restaurants
do NOT count toward this):
  relaxed:  1-2 attractions per day
  medium:   2-3 attractions per day
  packed:   3-4 attractions per day

TRANSIT NOTES — for each day, write a COMPLETE chain that covers EVERY leg:
  hotel → stop 1 → stop 2 → ... → last stop → hotel.

Format: leg-by-leg, semicolons or "→" arrows between segments. Include mode
plus rough fare OR duration per leg. Total length under 35 words. Use
"hotel" generically if a specific hotel name isn't in the input.

Examples by budget tier (note: every leg covered, including return):
  low:
    "Hotel → Kinkaku-ji (bus #205, 25min); Kinkaku-ji → Fushimi Inari
     (subway 35min, walk); subway back to hotel"
  medium:
    "Hotel → Kinkaku-ji (Uber ~$15, 20min); subway Kinkaku-ji → Fushimi
     (~30min); Uber back to hotel (~$18)"
  high:
    "Hotel → Kinkaku-ji → Fushimi → hotel by Uber (~$15-25 per leg, 4 rides)"
  luxury:
    "Hotel chauffeur for all stops; private car waits between visits;
     direct return to hotel"

Use plain "Uber" not "Uber Black" — Uber Black is reserved for luxury tier
where you should prefer "private chauffeur" / "hotel car" wording instead.

If the input includes `hotel_name`, refer to it explicitly:
  "Aman Kyoto → Kinkaku-ji (chauffeur 20min); ..."

PROCESS:
1. Pick attractions per day matching the pace count above. Drop the rest.
2. Pick a lunch and a dinner restaurant per day. Prefer ones geographically
   near that day's attractions; vary cuisine across days.
3. Write a 1-2 sentence transit note per day in the style required by the
   budget tier.

DAY SCHEDULE (timeline) — for each day also output a chronological list of
clock-timed entries covering hotel-out, every attraction, lunch (~12:00-14:00),
dinner (~18:30-21:00), and the transit between EVERY consecutive pair.

Each schedule entry is one of:
  {"type": "stop",    "name": str, "start": "HH:MM", "end": "HH:MM",
                      "kind": "attraction"|"lunch"|"dinner"}
  {"type": "transit", "from": str, "to": str,
                      "minutes": int, "mode": str}

Time anchors (REQUIRED — do not violate):
  - Hotel-out / first transit: ~09:00
  - Lunch: between **12:00 and 14:00**, lasting 60-90 min
  - Afternoon attraction(s): between lunch end and ~17:30
  - Dinner: between **18:30 and 21:00**, lasting 90-120 min
                 (longer for kaiseki/Michelin/luxury tier)
  - Last transit (return to hotel): right after dinner

Duration heuristics:
  - typical attraction visit: 60-120 min (temples 60-90, big museums 90-150,
    quick photo stops 30, hikes/parks 90-180)
  - walking: 5-20 min in compact areas
  - subway/bus: 15-45 min between districts of the same city
  - Uber/taxi: 10-30 min between most pairs

If a relaxed pace doesn't fill the afternoon, leave a free gap (just no entry
between, say, 16:00 and 18:00) — DO NOT compress dinner earlier than 18:30.
Use the `hotel_name` from input as the hotel reference if provided.

Output JSON with EXACTLY these keys:
{
  "route_groups":  {"Day 1": [attraction_names...], ...},   // ATTRACTIONS only
  "meal_plan":     {"Day 1": {"lunch": str, "dinner": str}, ...},
  "transit_notes": {"Day 1": str, ...},
  "day_schedule":  {"Day 1": [<timeline entries>], ...}
}

Use exactly `trip_length_days` keys in each section. Names in route_groups
must come from the input `attractions` list. Names in meal_plan must come
from the input `restaurants` list. Day_schedule entries reference the same
names plus the hotel. Do NOT include hotels in route_groups."""


async def route_agent(
    attractions: list[str] | dict,
    restaurants: list[str] | dict,
    trip_length_days: int,
    destination: str,
    pace: str = "medium",
    interests: list[str] | None = None,
    budget_level: str = "medium",
    hotel_name: str | None = None,
    feedback: str | None = None,
) -> dict:
    """attractions / restaurants may be either:
       - list[str]  (legacy, no priority)
       - {"priority": list[str], "optional": list[str]}  (priority semantics)
    `hotel_name` (if known) lets the transit_notes reference the hotel by name.
    `feedback` (if provided) is critic feedback from a previous failed plan —
    apply it while still respecting all other rules and priority items.
    """
    payload = json.dumps({
        "attractions": attractions,
        "restaurants": restaurants,
        "trip_length_days": trip_length_days,
        "destination": destination,
        "pace": pace,
        "interests": interests or [],
        "budget_level": budget_level,
        "hotel_name": hotel_name,
        "feedback": feedback,
    })
    raw = await orch_complete(ROUTE_SYSTEM, payload, response_format_json=True, max_tokens=3072)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return {"route_groups": {}, "meal_plan": {}, "transit_notes": {}}
    # Defensive: accept either wrapped or top-level
    if "route_groups" not in obj and "Day 1" in obj:
        obj = {"route_groups": obj, "meal_plan": {}, "transit_notes": {}}

    def _ensure_dict(v):
        return v if isinstance(v, dict) else {}

    return {
        "route_groups": _ensure_dict(obj.get("route_groups")),
        "meal_plan": _ensure_dict(obj.get("meal_plan")),
        "transit_notes": _ensure_dict(obj.get("transit_notes")),
        "day_schedule": _ensure_dict(obj.get("day_schedule")),
    }


# --------------------------- budget ---------------------------

BUDGET_SYSTEM = """You estimate trip costs in PER-PERSON, PER-DAY buckets.
The backend will sum them — DO NOT compute totals yourself.

Process:
1. Call `tavily_search` ONCE.
   - If the input includes `selected_hotel.name`, search for THAT specific
     hotel's nightly rate (e.g. "Park Hyatt Kyoto price per night USD").
     Override the budget_level anchor with what you find: a Park Hyatt
     priced at $700/night should produce hotel.high_per_day ≈ 350-400 even
     if budget_level is "medium".
   - Otherwise search a tier-typical hotel for the destination (e.g.
     "Aman <destination> price per night" for luxury, "boutique hotel
     <destination> $150 per night" for medium).
2. Fill in bucket numbers grounded in what you actually saw.
   - hotel: derive from the search above (per person — divide a double room
     by 2 if the trip has 2 travelers).
   - local_transit: if the route_agent leans heavily on Uber/taxi, bias
     toward the higher end of the anchor (5-7 rides/day × ~$15-25 = $75-175).
   - meals/attractions: standard tier anchor.
3. If the input includes `arrival`, populate the airfare bucket using ONE of
   its options matching budget_level. Else airfare = 0.

Output JSON with this exact shape (numbers are integers in USD per person):
{
  "buckets": {
    "hotel":         {"low_per_day": <int>, "high_per_day": <int>, "rationale": str},
    "local_transit": {"low_per_day": <int>, "high_per_day": <int>, "rationale": str},
    "meals":         {"low_per_day": <int>, "high_per_day": <int>, "rationale": str},
    "attractions":   {"low_per_day": <int>, "high_per_day": <int>, "rationale": str}
  },
  "airfare": {"low": <int>, "high": <int>, "note": str},   // round-trip per person TOTAL (not per day); 0/0 if no origin
  "notes":   [str]                                          // 2-4 short caveats
}

Anchor ranges (each is per person per day, except airfare which is total round-trip):

Hotel (per PERSON per night — divide a double-room rate by 2 if travelers share.
Anchor to typical nightly room rates in the destination):
  low:    $25-$80     (hostel/capsule/budget chain — full room often $30-$100)
  medium: $60-$180    (3-4 star, mid-range — full room often $120-$300)
  high:   $180-$500   (4-5 star — full room often $350-$900)
  luxury: $800-$2500+ (Aman / Four Seasons / Ritz suite — full room $1500-$4000+)
Pick numbers reflecting the LOW end of "what an actual hotel at this tier
costs", not below it. Medium should NEVER produce $50/night per person.

Local transit (regular UberX averages ~$15-30 per ride in major cities; budget
~6-10 rides/day for high tier; chauffeur day rate ~$300-600 for luxury):
  low: $5-$25    medium: $15-$60    high: $60-$180   luxury: $200-$500

Meals:
  low: $20-$60   medium: $50-$150   high: $150-$400  luxury: $300-$1000

Attractions:
  low: $0-$30    medium: $20-$80    high: $50-$200   luxury: $150-$500

Airfare (round trip total, per person):
  low/medium: $300-$1500 economy
  high:       $2000-$8000 business
  luxury:     $5000-$15000 first / $30000-$60000 private jet share

Pick numbers WITHIN these ranges based on the user's exact budget_level.
Do NOT output any totals or daily/airfare strings — backend computes those.
Output JSON only.
"""


def _compute_budget_totals(obj: dict, prefs: dict, arrival: dict | None) -> dict:
    """Sum the LLM's structured buckets into daily/total/airfare strings.
    Defensive: when the LLM returns a list where a dict was asked for, fall
    back to empty so we don't crash the pipeline."""
    if not isinstance(obj, dict):
        obj = {}
    buckets = obj.get("buckets")
    if not isinstance(buckets, dict):
        buckets = {}
    airfare = obj.get("airfare")
    if not isinstance(airfare, dict):
        airfare = {}
    days = int(prefs.get("trip_length_days") or 0)

    daily_low = 0
    daily_high = 0
    for v in buckets.values():
        if isinstance(v, dict):
            daily_low += int(v.get("low_per_day", 0) or 0)
            daily_high += int(v.get("high_per_day", 0) or 0)

    has_origin = bool((prefs.get("origin") or "").strip())
    # If the user picked specific flights, the orchestrator pre-computed the
    # round-trip total in `arrival.computed_airfare` — that overrides whatever
    # the LLM put in the airfare bucket. Otherwise, only count the LLM's value
    # when an origin was actually provided.
    computed = arrival.get("computed_airfare") if isinstance(arrival, dict) else None
    if isinstance(computed, dict) and (computed.get("low") or computed.get("high")):
        af_low = int(computed.get("low") or 0)
        af_high = int(computed.get("high") or 0)
    elif has_origin:
        af_low = int(airfare.get("low", 0) or 0)
        af_high = int(airfare.get("high", 0) or 0)
    else:
        af_low = af_high = 0
    total_low = daily_low * days + af_low
    total_high = daily_high * days + af_high

    daily_str = f"${daily_low:,}-${daily_high:,} per person" if (daily_low or daily_high) else ""
    airfare_str = (
        f"${af_low:,}-${af_high:,} round trip per person"
        if has_origin and (af_low or af_high) else ""
    )
    total_str = (
        f"${total_low:,}-${total_high:,} per person all-in for {days} days"
        if (total_low or total_high) else ""
    )

    return {
        "daily_estimate": daily_str,
        "airfare_estimate": airfare_str,
        "total_estimate": total_str,
        "buckets": buckets,
        "airfare": airfare,
        "notes": obj.get("notes") or [],
    }


async def budget_agent(
    prefs: dict,
    places: list[str],
    arrival: dict | None = None,
    selected_hotel: dict | None = None,
) -> dict:
    """`selected_hotel = {name, city}` lets the agent price the actual hotel
    the user picked rather than the budget_level anchor. Crucial when the
    user picks a luxury hotel under a medium budget tier."""
    payload = json.dumps({
        "prefs": prefs,
        "places": places,
        "arrival": arrival or None,
        "selected_hotel": selected_hotel or None,
    })
    raw = await orch_complete_with_tools(
        BUDGET_SYSTEM,
        payload,
        tools=[TAVILY_TOOL],
        run_tool=run_tool,
        response_format_json=True,
        max_steps=4,
    )
    obj = _extract_json(raw)
    return _compute_budget_totals(obj if isinstance(obj, dict) else {}, prefs, arrival)


# --------------------------- arrival (origin → destination) ---------------------------

ARRIVAL_SYSTEM = """You suggest round-trip transportation. Note: the outbound
leg may go to a DIFFERENT city than the return leg comes from — the user can
fly into one city and out of another (multi-city trip).

  - Outbound:  origin → first_city
  - Return:    last_city → origin

If `first_city` == `last_city`, it's a normal round trip into the same city.

Process:
1. Call `tavily_search` 1-2 times — for example:
   - "<origin> to <first_city> flight options price economy business first"
   - "<last_city> to <origin> flight return"
   (If first_city == last_city, one query is enough.)
2. Build TWO lists: `outbound_options` (origin → first_city) and
   `return_options` (last_city → origin). Each list should have 5-8 options
   spanning multiple classes for the same carriers, so the user can compare
   classes/prices side-by-side.

Output JSON:
{
  "outbound_options": [
    {
      "mode": "flight" | "train" | "bus" | "drive" | "private_jet",
      "carrier": str,                   // e.g. "ANA" or "Shinkansen" or "Gulfstream G650 charter"
      "duration": str,                  // "11h direct" / "13h 1 stop"
      "stops": int,                     // 0 = direct
      "class": "economy" | "premium_economy" | "business" | "first" | "private",
      "price": str,                     // ONE-WAY price per person, e.g. "$1,200"
      "notes": str                      // 1 short line — anything notable
    },
    ...
  ],
  "return_options": [ ... same shape ... ]
}

Rules:
- ONE-WAY price per person (not round trip — backend doubles it when both
  legs are chosen).
- Cover at least 3 classes when commercial flight is the mode.
- mode="private_jet" only for actual private aviation. Never "drive" for a jet.
- Match user's budget_level: low → emphasise economy; high → mostly business;
  luxury → first + private_jet at the top.
- Output JSON only, no preamble."""


async def arrival_agent(prefs: dict) -> dict:
    first_city = (prefs.get("first_city") or prefs.get("destination") or "").strip()
    last_city = (prefs.get("last_city") or first_city).strip()
    payload = json.dumps({
        "origin": prefs.get("origin"),
        "first_city": first_city,
        "last_city": last_city,
        "travelers": prefs.get("travelers"),
        "budget_level": prefs.get("budget_level", "medium"),
        "trip_length_days": prefs.get("trip_length_days"),
    })
    raw = await orch_complete_with_tools(
        ARRIVAL_SYSTEM,
        payload,
        tools=[TAVILY_TOOL],
        run_tool=run_tool,
        response_format_json=True,
        max_steps=4,
    )
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return {"outbound_options": [], "return_options": []}
    return {
        "outbound_options": obj.get("outbound_options") or [],
        "return_options": obj.get("return_options") or [],
    }


# --------------------------- itinerary (fine-tuned) ---------------------------

async def itinerary_agent(prefs: dict, places: list[str], route_groups: dict) -> dict:
    """Calls the FINE-TUNED model. Input shape must match the training schema
    exactly — do not add or rename fields without retraining."""
    user_input = {
        "destination": prefs.get("destination"),
        "trip_length_days": prefs.get("trip_length_days"),
        "travelers": prefs.get("travelers"),
        "budget_level": prefs.get("budget_level"),
        "interests": prefs.get("interests", []),
        "pace": prefs.get("pace", "medium"),
        "constraints": prefs.get("constraints", []),
        "selected_places": places,
        "route_groups": route_groups,
    }
    raw = await itin_complete(user_input)
    return _extract_json(raw)


# --------------------------- critic ---------------------------

CRITIC_SYSTEM = """You critique a travel itinerary and SCORE it 0-10.

You receive `itinerary`, `prefs`, AND `budget` (a computed object with
`daily_estimate`, `airfare_estimate`, `total_estimate`, `notes`). Use the
`budget` numbers — not the itinerary's prose — for any total-cost check.

Check for:
- Rushed days (count attractions per day vs. pace: relaxed=1-2, medium=2-3,
  packed=3-4 — restaurants don't count toward this cap)
- Places too far apart (geographic incoherence on the same day)
- Budget alignment: compare `budget.total_estimate` (and `daily_estimate`)
  against `prefs.budget_level`. Per-day spend excluding airfare:
    low     ≈ <$150/day
    medium  ≈ $300-$500/day
    high    ≈ $500-$800/day
    luxury  ≈ $800+/day
  Don't flag the itinerary's `budget_summary` for being short — totals come
  from `budget`, not the itinerary text.
- Unaddressed user interests
- Missed constraints (transit preference, mobility, dietary, etc.)
- Day-to-day flow problems (lodging changes, long transfers between cities)

SCORING (be strict, but fair — most well-formed plans should score 7-9):
  10  perfect, no issues
  8-9 minor cosmetic concerns only
  7   small issues but trip is still viable as-is — PASSES
  4-6 multiple real problems that would degrade the trip
  0-3 fundamental issues; trip needs rework

Plans scoring < 7 will be sent BACK for replanning.

Output JSON:
{
  "score": <int 0-10>,
  "passed": <bool>,                       // true iff score >= 7
  "issues": [str],                        // empty if perfect; each ref a concrete day or place
  "suggested_revisions": [str]            // 2-5 short, actionable fixes
}
Be terse. Output JSON only.
"""


# --------------------------- itinerary-specific travel tips ---------------

TIPS_SYSTEM = """You generate 3-5 travel tips that are SPECIFIC to the
provided day-by-day itinerary. Reference the actual cities, places, and
days the traveler is visiting — not generic advice.

Bad (generic, do NOT output):
  "Check current opening hours."
  "Bring comfortable shoes."
  "Carry cash."

Good (specific, DO output):
  "Visit Senso-ji on Day 1 before 9am — it gets crowded fast and the
   morning light on the temple is the best photo window."
  "Day 2's Fushimi Inari is at the south end of Kyoto; if you're staying
   near Gion, take the JR Nara line — about 20 min vs. 40+ on bus."
  "TeamLab Planets requires advance tickets — book before your trip; the
   walk-up line on Day 3 will eat your morning."

Output JSON: {"travel_tips": [str, str, str, ...]}  (3-5 items)
Each tip must reference at least one concrete day, place, or constraint
from the itinerary. Be terse (one or two sentences each)."""


async def tips_agent(itinerary: dict, prefs: dict) -> list[str]:
    """Generate itinerary-specific travel tips. Replaces the LoRA's templated
    tips with ones that actually reference the user's plan."""
    payload = json.dumps({
        "trip_summary": itinerary.get("trip_summary"),
        "daily_itinerary": itinerary.get("daily_itinerary") or [],
        "destination": prefs.get("destination"),
        "interests": prefs.get("interests") or [],
        "constraints": prefs.get("constraints") or [],
    })
    raw = await orch_complete(TIPS_SYSTEM, payload, response_format_json=True, max_tokens=1024)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return []
    tips = obj.get("travel_tips") or []
    if not isinstance(tips, list):
        return []
    out: list[str] = []
    for t in tips:
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out[:5]


async def critic_agent(itinerary: dict, prefs: dict, budget: dict | None = None) -> dict:
    payload = json.dumps({"itinerary": itinerary, "prefs": prefs, "budget": budget or {}})
    raw = await orch_complete(CRITIC_SYSTEM, payload, response_format_json=True)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return {"score": 0, "passed": False, "issues": ["Critic output unparseable"], "suggested_revisions": []}
    score = int(obj.get("score") if isinstance(obj.get("score"), (int, float)) else 0)
    obj["score"] = max(0, min(10, score))
    obj["passed"] = bool(obj.get("passed", obj["score"] >= 7))
    obj.setdefault("issues", [])
    obj.setdefault("suggested_revisions", [])
    return obj


# --------------------------- revision ---------------------------

REVISION_SYSTEM = """You apply a user-requested change to an existing itinerary.

The user message contains the current itinerary and a change_request. Apply
the change. Only modify what's needed; preserve everything else.

Output JSON with EXACTLY these top-level keys (the same schema as the input):
  trip_summary, daily_itinerary, budget_summary, backup_options, travel_tips

Do NOT wrap your output in an extra "itinerary" key or any other wrapper.
Do NOT include the change_request in the output. Output the revised itinerary
directly as the top-level JSON object."""


async def revision_agent(itinerary: dict, change_request: str) -> dict:
    payload = json.dumps({"current_itinerary": itinerary, "change_request": change_request})
    raw = await orch_complete(REVISION_SYSTEM, payload, response_format_json=True)
    obj = _extract_json(raw)
    # Defensive: if the model still wrapped the result, unwrap.
    if isinstance(obj, dict) and "daily_itinerary" not in obj:
        for k in ("itinerary", "current_itinerary", "revised_itinerary", "result"):
            inner = obj.get(k)
            if isinstance(inner, dict) and "daily_itinerary" in inner:
                return inner
    return obj


# --------------------------- revision router ---------------------------

REVISION_ROUTER_SYSTEM = """You classify a user's change request against the
current itinerary. Pick exactly one category:

- "text"        — surface tweaks: rewording, adding tips, swapping a single
                  attraction for a similar one in the same area, adjusting
                  evening plans without time/transit implications.
- "structural"  — anything that changes the day plan shape: adding/removing
                  attractions in a way that reshuffles the day, changing
                  pace or order of cities/days, swapping a place that's
                  geographically distant from current ones, time-window
                  changes (e.g., must end by 6pm), changing the trip length.
- "budget"      — the user shifts the price tier of the trip: "make it
                  cheaper", "stay under $X/day", "upgrade to luxury hotels",
                  "drop the high-end dinner", "private jet". This category
                  triggers a FULL re-research at the new tier (different
                  hotels, restaurants, places) — not just a math recompute.
                  Always also output `new_budget_level` for this category.

For "budget" infer the new tier from the change:
  low     ≈ <$150/day       (hostels, budget chains, casual dining, public transit)
  medium  ≈ $300-$500/day   (3-4 star, mid-range restaurants, mix of metro + Uber)
  high    ≈ $500-$800/day   (4-5 star, fine dining, mostly Uber/private cars)
  luxury  ≈ $800+/day       (5-star, Michelin, chauffeur, business/first flights)

Output JSON:
  {"category": "text"|"structural"|"budget",
   "reason": "<one short sentence>",
   "new_budget_level": "low"|"medium"|"high"|"luxury"  (only for budget)}
"""


async def revision_router(itinerary: dict, change_request: str) -> dict:
    payload = json.dumps({"itinerary_summary": {
        "trip_summary": itinerary.get("trip_summary"),
        "days": [d.get("theme") for d in (itinerary.get("daily_itinerary") or [])],
    }, "change_request": change_request})
    raw = await orch_complete(REVISION_ROUTER_SYSTEM, payload, response_format_json=True, max_tokens=512)
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        return {"category": "text", "reason": "router output unparseable; defaulting to text"}
    cat = obj.get("category")
    if cat not in ("text", "structural", "budget"):
        cat = "text"
    out = {"category": cat, "reason": str(obj.get("reason") or "")[:200]}
    if cat == "budget":
        tier = obj.get("new_budget_level")
        if tier in ("low", "medium", "high", "luxury"):
            out["new_budget_level"] = tier
    return out
