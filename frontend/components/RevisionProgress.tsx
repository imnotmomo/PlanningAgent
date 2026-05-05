"use client";

import { useEffect, useState } from "react";
import { AgentName } from "@/lib/api";
import { AGENT_LABELS, StepState } from "./AgentProgress";

const HINT: Partial<Record<AgentName, string>> = {
  itinerary: "the fine-tuned LoRA · 30–60 s",
  route: "Cerebras · few seconds",
  critic: "Cerebras · few seconds",
  budget: "Cerebras + Tavily · few seconds",
  revision: "Cerebras · ~1 s",
  revision_router: "Cerebras · instant",
};

type Category = "text" | "structural" | "budget";

const CATEGORY_AGENTS: Record<Category, AgentName[]> = {
  text:       ["revision_router", "revision"],
  structural: ["revision_router", "route", "itinerary", "critic"],
  budget:     ["revision_router", "budget", "revision"],
};

const CATEGORY_DESC: Record<Category, string> = {
  text:       "rephrase only — Cerebras edit, no LoRA",
  structural: "re-shape day plan — runs route + LoRA itinerary + critic",
  budget:     "recompute totals — budget agent + Cerebras edit",
};

function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
}

export function RevisionProgress({ steps }: { steps: Map<AgentName, StepState> }) {
  const [now, setNow] = useState(Date.now());
  const [runningSince, setRunningSince] = useState<{ name: AgentName; t0: number } | null>(null);

  const router = steps.get("revision_router");
  const routerOut = router?.output as { category?: Category; reason?: string } | undefined;
  const category = routerOut?.category;
  const reason = routerOut?.reason;
  const agents = category ? CATEGORY_AGENTS[category] : ["revision_router" as AgentName];

  const running = Array.from(steps.values()).find((s) => s.status === "running");

  useEffect(() => {
    if (running) {
      if (!runningSince || runningSince.name !== running.name) {
        setRunningSince({ name: running.name, t0: Date.now() });
      }
    } else {
      setRunningSince(null);
    }
  }, [running?.name]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [running]);

  const elapsedMs = running && runningSince ? now - runningSince.t0 : 0;
  const allDone = !running && agents.every((a) => steps.get(a)?.status === "done");

  return (
    <div className="card p-4 space-y-3">
      {/* header line — path + reason */}
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-uppercase-cta" style={{ color: "#777169" }}>
          Revision pipeline
        </span>
        {category ? (
          <>
            <span className="text-[14px] font-medium text-ink">path: {category}</span>
            <span className="text-[13px] text-warm">— {CATEGORY_DESC[category]}</span>
          </>
        ) : (
          <span className="text-[13px] text-warm">deciding which agents to run…</span>
        )}
      </div>
      {reason && (
        <div className="text-[13px]" style={{ color: "#7a7a7a", lineHeight: 1.5 }}>
          {reason}
        </div>
      )}

      {/* per-agent status row */}
      <div className="flex flex-wrap gap-2">
        {agents.map((a) => {
          const s = steps.get(a);
          const status = s?.status ?? "pending";
          const isRunning = status === "running";
          const isDone = status === "done";
          const dot = isRunning ? "#bf6e3a" : isDone ? "#7aa274" : "#cfcfcf";
          return (
            <div
              key={a}
              className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[13px]"
              style={{ background: "rgba(245,242,239,0.8)" }}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{
                  backgroundColor: dot,
                  animation: isRunning ? "pulse 1.4s ease-in-out infinite" : undefined,
                }}
              />
              <span style={{ color: isDone ? "#4e4e4e" : isRunning ? "#bf6e3a" : "#999" }}>
                {AGENT_LABELS[a]}
              </span>
              {isRunning && runningSince?.name === a && (
                <span className="opacity-60">{fmt(elapsedMs)}</span>
              )}
              {HINT[a] && (isRunning || isDone) && (
                <span className="opacity-50">· {HINT[a]}</span>
              )}
            </div>
          );
        })}
      </div>

      {allDone && (
        <div className="text-[13px]" style={{ color: "#7aa274" }}>
          ✓ Revision applied
        </div>
      )}
    </div>
  );
}
