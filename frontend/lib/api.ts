// Where to send API calls. In dev set NEXT_PUBLIC_API_URL=http://localhost:8000
// in frontend/.env.local — this bypasses Next.js's rewrite proxy, which
// buffers streaming responses and breaks SSE. Without the env var we fall
// back to the rewrite (works for production with a real reverse proxy).
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const apiUrl = (path: string): string =>
  API_BASE ? `${API_BASE}${path}` : `/api${path}`;

export type AgentName =
  | "preference"
  | "missing_info"
  | "destination_suggester"
  | "arrival"
  | "research"
  | "route"
  | "budget"
  | "itinerary"
  | "critic"
  | "revision_router"
  | "revision";

export interface ResearchPayload {
  places: Candidate[];
  restaurants: Candidate[];
  hotels: Candidate[];
  by_city?: Record<string, { places: Candidate[]; restaurants: Candidate[]; hotels: Candidate[] }>;
}

export interface ResearchComplete {
  preferences: Record<string, unknown>;
  destinations: Destination[];
  arrival: ArrivalData | null;
  research: ResearchPayload;
}

export interface CityCandidate {
  city: string;
  country: string;
  description: string;
  suggested_days: number;
}

export interface DestinationSuggester {
  candidates: CityCandidate[];
  default_split: Destination[];
}

export interface DestinationsComplete {
  preferences: Record<string, unknown>;
  destinations: Destination[];
  needs_resolution: boolean;
  suggester: DestinationSuggester | null;
}

export type ReviseCompletePayload = {
  category: "text" | "structural" | "budget";
  reason: string;
  itinerary: Itinerary;
  budget?: PlanResult["budget"];
  route_groups?: Record<string, string[]>;
  meal_plan?: Record<string, DayMeals>;
  transit_notes?: Record<string, string>;
  day_schedule?: Record<string, ScheduleEntry[]>;
  critique?: PlanResult["critique"];
};

export type StreamEvent =
  | { event: "started"; payload: { request?: string; phase?: string } }
  | { event: "step"; payload: { name: AgentName; status: "running" | "done"; output?: unknown; is_retry?: boolean; retry_round?: number } }
  | { event: "destinations_complete"; payload: DestinationsComplete }
  | { event: "research_complete"; payload: ResearchComplete }
  | { event: "incomplete"; payload: { missing_fields: string[]; preferences: Record<string, unknown> } }
  | { event: "complete"; payload: PlanResult | ReviseCompletePayload }
  | { event: "error"; payload: { type: string; message: string; trace?: string } };

export interface Selections {
  places: string[];
  restaurants: string[];
  hotels: string[];
  arrival_choices?: {
    outbound: ArrivalOption | null;
    return: ArrivalOption | null;
  } | null;
}

export interface DailyEntry {
  day: number;
  theme: string;
  morning: string;
  afternoon: string;
  evening: string;
  estimated_cost: string;
  transportation_note: string;
  feasibility_note: string;
  city?: string; // present on multi-destination trips
}

export interface Itinerary {
  trip_summary: string;
  daily_itinerary: DailyEntry[];
  budget_summary?: string;
  backup_options?: string[];
  travel_tips?: string[];
}

export interface Candidate {
  name: string;
  description: string;
  city?: string; // tagged when the trip spans multiple cities
}

export interface Destination {
  city: string;
  country?: string;
  days: number;
}

export interface ArrivalOption {
  mode: string;
  carrier?: string;
  duration?: string;
  stops?: number;
  class?: string;          // "economy" | "premium_economy" | "business" | "first" | "private"
  price?: string;          // one-way per person
  price_range?: string;    // legacy round-trip range (kept for back-compat)
  notes?: string;
}

export interface ArrivalData {
  outbound_options?: ArrivalOption[];
  return_options?: ArrivalOption[];
  options?: ArrivalOption[]; // legacy round-trip combined list
}

export interface DayMeals {
  lunch?: string;
  dinner?: string;
}

export type ScheduleEntry =
  | { type: "stop"; name: string; start?: string; end?: string; kind?: "attraction" | "lunch" | "dinner" | string }
  | { type: "transit"; from: string; to: string; minutes?: number; mode?: string };

export interface PlanResult {
  preferences: Record<string, unknown>;
  destinations?: Destination[];
  arrival: ArrivalData | null;
  places: Candidate[];
  restaurants: Candidate[];
  hotels: Candidate[];
  route_groups: Record<string, string[]>;
  meal_plan: Record<string, DayMeals>;
  transit_notes: Record<string, string>;
  day_schedule?: Record<string, ScheduleEntry[]>;
  budget: {
    daily_estimate?: string;
    airfare_estimate?: string;
    total_estimate?: string;
    notes?: string[];
  };
  itinerary: Itinerary;
  critique: {
    score?: number;
    passed?: boolean;
    issues: string[];
    should_revise?: boolean;
    suggested_revisions?: string[];
  };
}

async function* parseSSE(res: Response): AsyncGenerator<StreamEvent> {
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      try {
        yield JSON.parse(json) as StreamEvent;
      } catch {
        // skip malformed line
      }
    }
  }
}

export async function* streamPlan(request: string): AsyncGenerator<StreamEvent> {
  const res = await fetch(apiUrl("/plan"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ request }),
    cache: "no-store",
  });
  yield* parseSSE(res);
}

export async function* streamDestinations(request: string): AsyncGenerator<StreamEvent> {
  const res = await fetch(apiUrl("/destinations"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ request }),
    cache: "no-store",
  });
  yield* parseSSE(res);
}

export async function* streamResearch(
  preferences: Record<string, unknown>,
  destinations: Destination[],
): AsyncGenerator<StreamEvent> {
  const res = await fetch(apiUrl("/research"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ preferences, destinations }),
    cache: "no-store",
  });
  yield* parseSSE(res);
}

export async function* streamBuild(
  preferences: Record<string, unknown>,
  research: ResearchPayload,
  arrival: ArrivalData | null,
  selections: Selections | null,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(apiUrl("/build"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ preferences, research, arrival, selections }),
    cache: "no-store",
  });
  yield* parseSSE(res);
}

export interface CandidateDetail {
  name: string;
  city?: string;
  summary: string;
  images: string[];
  sources: { title?: string; url?: string; snippet?: string }[];
}

export interface CustomCandidateInput {
  name: string;
  city?: string;
  category: "place" | "restaurant" | "hotel";
}

export async function enrichCustomCandidates(
  items: CustomCandidateInput[],
  destination?: string,
): Promise<Candidate[]> {
  const res = await fetch(apiUrl("/enrich-candidates"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, destination: destination ?? null }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { items: Candidate[] };
  return data.items;
}

export async function fetchCandidateDetail(
  name: string,
  city?: string,
  category?: "place" | "restaurant" | "hotel",
): Promise<CandidateDetail> {
  const res = await fetch(apiUrl("/candidate-detail"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, city, category }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function* streamRevise(
  result: PlanResult,
  change: string,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(apiUrl("/revise"), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ result, change }),
    cache: "no-store",
  });
  yield* parseSSE(res);
}
