"use client";
import { useEffect, useRef, useState } from "react";
import { TripForm } from "@/components/TripForm";
import { AgentProgress, StepState, AGENT_LABELS } from "@/components/AgentProgress";
import { ItineraryView } from "@/components/ItineraryView";
import { RevisionForm } from "@/components/RevisionForm";
import { CandidatePicker } from "@/components/CandidatePicker";
import { ResumePrompt } from "@/components/ResumePrompt";
import {
  AgentName,
  Destination,
  DestinationsComplete,
  PlanResult,
  ResearchComplete,
  Selections,
  streamDestinations,
  streamResearch,
  streamBuild,
  streamRevise,
  Itinerary,
} from "@/lib/api";
import { DestinationPicker } from "@/components/DestinationPicker";
import {
  clearSession,
  loadSession,
  newSessionId,
  saveSession,
  type SavedSession,
} from "@/lib/session";

type Phase =
  | "idle"
  | "resolving"
  | "choosing_destinations"
  | "researching"
  | "picking"
  | "planning"
  | "done"
  | "revising"
  | "error"
  | "incomplete";

export default function Page() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [steps, setSteps] = useState<Map<AgentName, StepState>>(new Map());
  const [result, setResult] = useState<PlanResult | null>(null);
  const [research, setResearch] = useState<ResearchComplete | null>(null);
  const [destResolution, setDestResolution] = useState<DestinationsComplete | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [showPipeline, setShowPipeline] = useState(false);
  const [lastSelections, setLastSelections] = useState<Selections | null>(null);
  const [resumePrompt, setResumePrompt] = useState<SavedSession | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const requestRef = useRef<string>("");

  // On first mount, surface a saved unfinished session if there is one.
  useEffect(() => {
    const s = loadSession();
    if (s) setResumePrompt(s);
  }, []);

  const persist = (
    nextPhase: SavedSession["phase"],
    overrides: Partial<SavedSession> = {},
  ) => {
    if (!sessionIdRef.current) sessionIdRef.current = newSessionId();
    saveSession({
      id: sessionIdRef.current,
      request: requestRef.current,
      phase: nextPhase,
      destResolution: overrides.destResolution ?? destResolution,
      research: overrides.research ?? research,
      result: overrides.result ?? result,
      lastSelections: overrides.lastSelections ?? lastSelections,
    });
  };

  const handleContinueResume = () => {
    if (!resumePrompt) return;
    sessionIdRef.current = resumePrompt.id;
    requestRef.current = resumePrompt.request;
    if (resumePrompt.destResolution) setDestResolution(resumePrompt.destResolution);
    if (resumePrompt.research) setResearch(resumePrompt.research);
    if (resumePrompt.result) setResult(resumePrompt.result);
    if (resumePrompt.lastSelections !== undefined) setLastSelections(resumePrompt.lastSelections ?? null);
    setPhase(resumePrompt.phase);
    setResumePrompt(null);
  };

  const handleDismissResume = () => {
    clearSession();
    sessionIdRef.current = null;
    setResumePrompt(null);
  };

  const handleStepEvent = (ev: { event: "step"; payload: { name: AgentName; status: "running" | "done"; output?: unknown; retry_round?: number } }) => {
    const { name, status, output } = ev.payload;
    const retryRound = ev.payload.retry_round;
    setSteps((prev) => {
      const next = new Map(prev);
      next.set(name, { name, status, output, retryRound });
      return next;
    });
  };

  const startPlan = async (request: string) => {
    setPhase("resolving");
    setSteps(new Map());
    setResult(null);
    setResearch(null);
    setDestResolution(null);
    setMissing([]);
    setErrMsg(null);
    // Fresh planning session — drop any prior saved checkpoint
    clearSession();
    sessionIdRef.current = newSessionId();
    requestRef.current = request;

    try {
      for await (const ev of streamDestinations(request)) {
        if (ev.event === "step") handleStepEvent(ev as never);
        else if (ev.event === "destinations_complete") {
          setDestResolution(ev.payload);
          if (ev.payload.needs_resolution && ev.payload.suggester) {
            setPhase("choosing_destinations");
            persist("choosing_destinations", { destResolution: ev.payload });
          } else {
            // Destinations already resolved by user prompt — auto-proceed.
            await runResearchPhase(ev.payload.preferences, ev.payload.destinations);
            return;
          }
        } else if (ev.event === "incomplete") {
          setMissing(ev.payload.missing_fields);
          setPhase("incomplete");
        } else if (ev.event === "error") {
          setErrMsg(`${ev.payload.type}: ${ev.payload.message}`);
          setPhase("error");
        }
      }
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  const runResearchPhase = async (
    prefs: Record<string, unknown>,
    destinations: Destination[],
  ) => {
    setPhase("researching");
    try {
      for await (const ev of streamResearch(prefs, destinations)) {
        if (ev.event === "step") handleStepEvent(ev as never);
        else if (ev.event === "research_complete") {
          setResearch(ev.payload);
          setPhase("picking");
          persist("picking", { research: ev.payload });
        } else if (ev.event === "incomplete") {
          setMissing(ev.payload.missing_fields);
          setPhase("incomplete");
        } else if (ev.event === "error") {
          setErrMsg(`${ev.payload.type}: ${ev.payload.message}`);
          setPhase("error");
        }
      }
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  const continueWithDestinations = async (destinations: Destination[]) => {
    if (!destResolution) return;
    await runResearchPhase(destResolution.preferences, destinations);
  };

  const continueWithSelections = async (selections: Selections | null) => {
    if (!research) return;
    setLastSelections(selections);
    setPhase("planning");
    try {
      for await (const ev of streamBuild(
        research.preferences,
        research.research,
        research.arrival,
        selections,
      )) {
        if (ev.event === "step") {
          const { name, status, output } = ev.payload;
          const retryRound = ev.payload.retry_round;
          setSteps((prev) => {
            const next = new Map(prev);
            next.set(name, { name, status, output, retryRound });
            return next;
          });
        } else if (ev.event === "complete") {
          // /build always emits a full PlanResult; streamRevise narrows in its own handler
          const planResult = ev.payload as PlanResult;
          setResult(planResult);
          setPhase("done");
          persist("done", { result: planResult, lastSelections: selections });
        } else if (ev.event === "error") {
          setErrMsg(`${ev.payload.type}: ${ev.payload.message}`);
          setPhase("error");
        }
      }
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  const applyRevision = async (change: string) => {
    if (!result) return;
    setPhase("revising");
    // Clear only the revision-specific agent slots so the inline indicator
    // shows fresh state. Don't blow away showPipeline or earlier planning
    // steps — the user is actively viewing the itinerary above.
    setSteps((prev) => {
      const next = new Map(prev);
      next.delete("revision_router");
      next.delete("revision");
      next.delete("route");
      next.delete("itinerary");
      next.delete("critic");
      next.delete("budget");
      return next;
    });
    try {
      let nextResult = result;
      for await (const ev of streamRevise(result, change)) {
        if (ev.event === "step") handleStepEvent(ev as never);
        else if (ev.event === "complete") {
          const p = ev.payload as {
            category?: string;
            itinerary?: Itinerary;
            budget?: typeof result.budget;
            route_groups?: typeof result.route_groups;
            meal_plan?: typeof result.meal_plan;
            transit_notes?: typeof result.transit_notes;
            day_schedule?: typeof result.day_schedule;
            critique?: typeof result.critique;
          };
          nextResult = {
            ...result,
            ...(p.itinerary ? { itinerary: p.itinerary } : {}),
            ...(p.budget ? { budget: p.budget } : {}),
            ...(p.route_groups ? { route_groups: p.route_groups } : {}),
            ...(p.meal_plan ? { meal_plan: p.meal_plan } : {}),
            ...(p.transit_notes ? { transit_notes: p.transit_notes } : {}),
            ...(p.day_schedule ? { day_schedule: p.day_schedule } : {}),
            ...(p.critique ? { critique: p.critique } : {}),
          };
          setResult(nextResult);
          setPhase("done");
          persist("done", { result: nextResult });
          break;
        } else if (ev.event === "error") {
          setErrMsg(`${ev.payload.type}: ${ev.payload.message}`);
          setPhase("done");
          break;
        }
      }
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : String(e));
      setPhase("done");
    }
  };

  const reset = () => {
    setPhase("idle");
    setSteps(new Map());
    setResult(null);
    setResearch(null);
    setDestResolution(null);
    setMissing([]);
    setErrMsg(null);
    setShowPipeline(false);
    setLastSelections(null);
    clearSession();
    sessionIdRef.current = null;
    requestRef.current = "";
  };

  return (
    <main className="min-h-screen pb-24">
      {resumePrompt && (
        <ResumePrompt
          session={resumePrompt}
          onContinue={handleContinueResume}
          onDismiss={handleDismissResume}
        />
      )}
      <Header onReset={reset} />

      <section className="px-6 pt-20 md:pt-28 pb-12">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="h-display text-[44px] md:text-[64px]" style={{ letterSpacing: "-0.96px" }}>
            Plan a trip the way<br className="hidden md:inline" /> a careful friend would.
          </h1>
          <p className="text-body mt-6 max-w-xl mx-auto">
            TripWise is a multi-agent system. Each agent does one thing — extracts your preferences,
            researches places, plans routes, estimates the budget, drafts the itinerary, and checks the result.
            The itinerary draft uses a fine-tuned model that always returns clean, schema-conformant JSON.
          </p>
        </div>
      </section>

      <section className="px-6 pb-16">
        <TripForm
          onSubmit={startPlan}
          disabled={
            phase === "resolving" || phase === "choosing_destinations" ||
            phase === "researching" || phase === "picking" ||
            phase === "planning" || phase === "revising"
          }
        />
      </section>

      {phase !== "idle" && (
        <section className="px-6 pb-12 max-w-5xl mx-auto">
          <div className="grid grid-cols-1 gap-6">
            <PipelineStatus
              phase={phase}
              steps={steps}
              show={showPipeline}
              onToggle={() => setShowPipeline((v) => !v)}
            />
            {showPipeline && <AgentProgress steps={steps} />}

            {phase === "choosing_destinations" && destResolution?.suggester && (
              <DestinationPicker
                countryOrRegion={String(destResolution.preferences.country_or_region || "")}
                tripLengthDays={Number(destResolution.preferences.trip_length_days || 0)}
                suggester={destResolution.suggester}
                onContinue={continueWithDestinations}
                onCancel={reset}
                disabled={false}
              />
            )}

            {phase === "picking" && research && (
              <CandidatePicker
                research={research.research}
                arrival={research.arrival}
                preferences={research.preferences}
                onContinue={continueWithSelections}
                onCancel={reset}
              />
            )}

            {phase === "incomplete" && (
              <div className="card p-6" style={{ background: "rgba(245,242,239,0.55)" }}>
                <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>
                  More info needed
                </div>
                <p className="text-body" style={{ fontSize: 16 }}>
                  Please include: <strong>{missing.join(", ")}</strong>. Add to your request and try again.
                </p>
              </div>
            )}

            {phase === "error" && errMsg && (
              <div className="card p-6 border" style={{ borderColor: "#e5e5e5" }}>
                <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>
                  Error
                </div>
                <p className="text-body" style={{ fontSize: 16 }}>{errMsg}</p>
                <p className="text-caption mt-2">
                  Check that the FastAPI server is running on <code>localhost:8000</code> and that{" "}
                  <code>backend/.env</code> has a valid <code>ORCH_API_KEY</code>.
                </p>
              </div>
            )}

            {result && (
              <>
                <ItineraryView
                  itinerary={result.itinerary}
                  budget={result.budget}
                  critique={result.critique}
                  restaurants={result.restaurants}
                  hotels={result.hotels}
                  arrival={result.arrival}
                  mealPlan={result.meal_plan}
                  transitNotes={result.transit_notes}
                  daySchedule={result.day_schedule}
                  selectedHotels={lastSelections?.hotels}
                  arrivalChoices={lastSelections?.arrival_choices ?? null}
                />
                {phase === "revising" && (() => {
                  const running = Array.from(steps.values()).find((s) => s.status === "running");
                  const lastDone = Array.from(steps.values()).filter((s) => s.status === "done").slice(-1)[0];
                  const showName = running?.name ?? lastDone?.name;
                  const verb = running ? "Running" : "Finished";
                  return showName ? (
                    <div
                      className="card-section px-4 py-3 flex items-center gap-3 text-[14px]"
                      style={{ color: "#4e4e4e" }}
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-full"
                        style={{
                          backgroundColor: running ? "#bf6e3a" : "#7aa274",
                          animation: running ? "pulse 1.4s ease-in-out infinite" : undefined,
                        }}
                      />
                      <span className="font-medium">{verb}: {AGENT_LABELS[showName]}</span>
                    </div>
                  ) : null;
                })()}
                <RevisionForm onSubmit={applyRevision} disabled={phase === "revising"} />
              </>
            )}
          </div>
        </section>
      )}

      <Footer />
    </main>
  );
}

function PipelineStatus({
  phase,
  steps,
  show,
  onToggle,
}: {
  phase: Phase;
  steps: Map<AgentName, StepState>;
  show: boolean;
  onToggle: () => void;
}) {
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
  // Treat conditional steps as visible only if they actually ran.
  const visible = ORDER.filter((n) => {
    if (n === "arrival" || n === "destination_suggester") return steps.has(n);
    return true;
  });
  const doneCount = visible.filter((n) => steps.get(n)?.status === "done").length;
  const running = visible.find((n) => steps.get(n)?.status === "running");

  let label = "Pipeline idle";
  // Detect critic-driven retry rounds
  const maxRetry = Math.max(0, ...visible.map((n) => steps.get(n)?.retryRound ?? 0));
  const retrySuffix = maxRetry > 0 ? ` · replan ${maxRetry}` : "";

  if (phase === "resolving") {
    label = running
      ? `Resolving destinations: ${running.replace("_", " ")}…`
      : "Resolving destinations…";
  } else if (phase === "choosing_destinations") {
    label = "Pick cities to visit — pipeline paused";
  } else if (phase === "researching") {
    label = running
      ? `Researching: ${running.replace("_", " ")}…`
      : "Researching…";
  } else if (phase === "picking") {
    label = "Review candidates and pick favorites — pipeline paused";
  } else if (phase === "planning") {
    label = running
      ? `Working on ${running.replace("_", " ")}…${retrySuffix} (${doneCount}/${visible.length})`
      : `Starting…`;
  } else if (phase === "revising") {
    label = "Applying your revision…";
  } else if (phase === "done") {
    label = `Pipeline complete · ${visible.length}/${visible.length} steps${retrySuffix}`;
  } else if (phase === "incomplete") {
    label = "Pipeline paused — needs more info";
  } else if (phase === "error") {
    label = "Pipeline error";
  }

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <Indicator phase={phase} />
        <span className="text-[14px] truncate" style={{ color: "#4e4e4e", letterSpacing: "0.14px" }}>
          {label}
        </span>
      </div>
      <button onClick={onToggle} className="btn-secondary text-[13px] py-2 px-3">
        {show ? "Hide pipeline" : "Show pipeline"}
      </button>
    </div>
  );
}

function Indicator({ phase }: { phase: Phase }) {
  if (phase === "resolving" || phase === "researching" || phase === "planning" || phase === "revising") {
    return (
      <span
        className="h-2 w-2 rounded-full bg-ink"
        style={{ animation: "step-pulse 1.4s ease-in-out infinite" }}
      />
    );
  }
  if (phase === "choosing_destinations" || phase === "picking") {
    return <span className="h-2 w-2 rounded-full" style={{ background: "#777169" }} />;
  }
  if (phase === "error") {
    return <span className="h-2 w-2 rounded-full" style={{ background: "#7c2d12" }} />;
  }
  if (phase === "incomplete") {
    return <span className="h-2 w-2 rounded-full" style={{ background: "#777169" }} />;
  }
  return <span className="h-2 w-2 rounded-full bg-ink" />;
}

function Header({ onReset }: { onReset: () => void }) {
  return (
    <header
      className="sticky top-0 z-10 backdrop-blur-md bg-white/80"
      style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}
    >
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <button onClick={onReset} className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-ink" />
          <span className="text-[15px] font-medium tracking-nav">TripWise</span>
        </button>
        <nav className="flex items-center gap-4">
          <a
            href="https://github.com/imnotmomo/PlanningAgent"
            className="text-[15px] font-medium hidden md:inline"
            style={{ color: "#4e4e4e", letterSpacing: "0.15px" }}
          >
            GitHub
          </a>
          <button onClick={onReset} className="btn-primary">
            New trip
          </button>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="px-6 pt-10 pb-12 mt-12 border-t border-border/60">
      <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-3">
        <span className="text-caption">
          Itinerary agent fine-tuned on Qwen2.5-7B-Instruct · LoRA r=16
        </span>
        <span className="text-caption">© TripWise</span>
      </div>
    </footer>
  );
}
