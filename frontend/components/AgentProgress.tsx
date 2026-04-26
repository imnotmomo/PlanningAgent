"use client";
import { AgentName } from "@/lib/api";

export type StepStatus = "pending" | "running" | "done";

export interface StepState {
  name: AgentName;
  status: StepStatus;
  output?: unknown;
  retryRound?: number; // > 0 if this step is part of a critic-driven replan
}

const ORDER: AgentName[] = [
  "preference",
  "missing_info",
  "destination_suggester",
  "arrival",
  "research",
  "route",
  "budget",
  "itinerary",
  "critic",
];

const LABELS: Record<AgentName, string> = {
  preference: "Preferences",
  missing_info: "Required fields check",
  destination_suggester: "Destination suggestions",
  arrival: "Getting there",
  research: "Place research",
  route: "Route + meals + transit",
  budget: "Budget estimate",
  itinerary: "Itinerary draft",
  critic: "Feasibility critique",
};

const SUBLABELS: Record<AgentName, string> = {
  preference: "Extracting your trip parameters",
  missing_info: "Confirming we have what we need",
  destination_suggester: "Suggesting cities for your country/region trip",
  arrival: "Flight/train options from your origin",
  research: "Candidate places, restaurants, hotels",
  route: "Picking attractions, lunch+dinner, daily transit",
  budget: "Estimating per-day costs (incl. lodging)",
  itinerary: "Generating the day-by-day plan (fine-tuned model)",
  critic: "Checking for rushed or unrealistic days",
};

interface AgentProgressProps {
  steps: Map<AgentName, StepState>;
}

export function AgentProgress({ steps }: AgentProgressProps) {
  // Hide conditional steps that didn't run (arrival = no origin given,
  // destination_suggester = user gave specific cities).
  const visible = ORDER.filter((n) => {
    if (n === "arrival" || n === "destination_suggester") return steps.has(n);
    return true;
  });
  return (
    <div className="card p-6 md:p-8">
      <div className="text-uppercase-cta mb-5" style={{ color: "#777169" }}>
        Pipeline
      </div>
      <ol className="flex flex-col gap-1">
        {visible.map((name) => {
          const state = steps.get(name);
          const status: StepStatus = state?.status ?? "pending";
          const isItin = name === "itinerary";
          return (
            <li
              key={name}
              className={`step-${status} flex items-start gap-4 py-3 border-b border-border/60 last:border-b-0`}
            >
              <StatusDot status={status} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[15px] font-medium" style={{ letterSpacing: "0.15px" }}>
                    {LABELS[name]}
                  </span>
                  {isItin && (
                    <span
                      className="text-[10px] px-2 py-0.5 rounded-full"
                      style={{
                        background: "rgba(245,242,239,0.9)",
                        color: "#4e4e4e",
                        letterSpacing: "0.7px",
                        textTransform: "uppercase",
                        fontWeight: 700,
                      }}
                    >
                      Fine-tuned
                    </span>
                  )}
                  {state?.retryRound && state.retryRound > 0 && (
                    <span
                      className="text-[10px] px-2 py-0.5 rounded-full"
                      style={{
                        background: "#7c2d12",
                        color: "#fff",
                        letterSpacing: "0.7px",
                        textTransform: "uppercase",
                        fontWeight: 700,
                      }}
                    >
                      Replan {state.retryRound}
                    </span>
                  )}
                </div>
                <div className="text-caption mt-0.5">{SUBLABELS[name]}</div>
                {status === "done" && state?.output != null && (
                  <StepOutput output={state.output} />
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function StatusDot({ status }: { status: StepStatus }) {
  const base = "step-dot mt-1 h-2 w-2 rounded-full shrink-0";
  if (status === "done") return <span className={`${base} bg-ink`} aria-label="done" />;
  if (status === "running")
    return <span className={`${base} bg-ink`} aria-label="running" />;
  return <span className={`${base} bg-border`} aria-label="pending" />;
}

const RESEARCH_LABELS: Record<string, string> = {
  places: "Places",
  restaurants: "Restaurants",
  hotels: "Stay",
};

function isStringList(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

function ChipList({ items }: { items: string[] }) {
  if (items.length === 0) return <span className="text-caption">— none —</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.slice(0, 16).map((v, i) => (
        <span
          key={i}
          className="text-[13px] px-2.5 py-1 rounded-full bg-surface"
          style={{ color: "#4e4e4e", letterSpacing: "0.14px" }}
        >
          {v}
        </span>
      ))}
    </div>
  );
}

function StepOutput({ output }: { output: unknown }) {
  if (output == null) return null;
  if (Array.isArray(output) && output.length === 0) {
    return <div className="text-caption mt-1">— none —</div>;
  }
  if (typeof output === "string" || typeof output === "number") {
    return (
      <div className="text-caption mt-1" style={{ color: "#4e4e4e" }}>
        {String(output)}
      </div>
    );
  }
  if (isStringList(output)) {
    return (
      <div className="mt-2">
        <ChipList items={output} />
      </div>
    );
  }
  // Research output: {places, restaurants, hotels} where each is now a list
  // of {name, description} candidates. Render each category with named items
  // and dim descriptions.
  if (
    output &&
    typeof output === "object" &&
    !Array.isArray(output) &&
    Object.keys(output as Record<string, unknown>).every((k) => k in RESEARCH_LABELS)
  ) {
    const obj = output as Record<string, unknown>;
    return (
      <div className="mt-3 flex flex-col gap-4">
        {Object.keys(RESEARCH_LABELS).map((k) => {
          const v = obj[k];
          if (!Array.isArray(v) || v.length === 0) return null;
          return (
            <div key={k}>
              <div
                className="text-[11px] mb-2"
                style={{
                  color: "#777169",
                  letterSpacing: "0.7px",
                  textTransform: "uppercase",
                  fontWeight: 700,
                }}
              >
                {RESEARCH_LABELS[k]} · candidates
              </div>
              <ul className="flex flex-col gap-1.5">
                {v.slice(0, 12).map((it, i) => {
                  if (typeof it === "string") {
                    return (
                      <li key={i} className="text-[14px]" style={{ color: "#4e4e4e", letterSpacing: "0.14px" }}>
                        · {it}
                      </li>
                    );
                  }
                  if (it && typeof it === "object" && "name" in it) {
                    const item = it as { name?: string; description?: string };
                    return (
                      <li key={i} className="leading-snug">
                        <span className="text-[14px] font-medium" style={{ color: "#000" }}>
                          {item.name}
                        </span>
                        {item.description && (
                          <span className="text-[13px]" style={{ color: "#777169" }}> — {item.description}</span>
                        )}
                      </li>
                    );
                  }
                  return null;
                })}
              </ul>
            </div>
          );
        })}
      </div>
    );
  }
  return (
    <pre className="text-[12.5px] mt-2 p-3 rounded-card bg-surface overflow-auto max-h-48 font-mono" style={{ color: "#4e4e4e", lineHeight: 1.6 }}>
      {JSON.stringify(output, null, 2)}
    </pre>
  );
}
