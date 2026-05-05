"use client";

import { useEffect, useState } from "react";
import { AgentName } from "@/lib/api";
import { AGENT_LABELS, StepState } from "./AgentProgress";

const HINT: Partial<Record<AgentName, string>> = {
  itinerary: "the fine-tuned model can take 30–60 sec",
  route: "few seconds",
  critic: "few seconds",
  budget: "few seconds",
  revision: "a few seconds",
  revision_router: "instant",
};

function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
}

export function RevisionProgress({ steps }: { steps: Map<AgentName, StepState> }) {
  const [now, setNow] = useState(Date.now());
  const [runningSince, setRunningSince] = useState<{ name: AgentName; t0: number } | null>(null);

  // Pick the currently-running step. When it changes, reset the timer.
  const running = Array.from(steps.values()).find((s) => s.status === "running");
  const lastDone = Array.from(steps.values()).filter((s) => s.status === "done").slice(-1)[0];

  useEffect(() => {
    if (running) {
      if (!runningSince || runningSince.name !== running.name) {
        setRunningSince({ name: running.name, t0: Date.now() });
      }
    } else {
      setRunningSince(null);
    }
  }, [running?.name]); // eslint-disable-line react-hooks/exhaustive-deps

  // Tick once per second so the elapsed counter updates
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [running]);

  const showStep = running ?? lastDone;
  if (!showStep) return null;
  const isRunning = !!running;
  const elapsedMs = isRunning && runningSince ? now - runningSince.t0 : 0;

  return (
    <div
      className="card-section px-4 py-3 flex items-center gap-3 text-[14px] flex-wrap"
      style={{ color: "#4e4e4e" }}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{
          backgroundColor: isRunning ? "#bf6e3a" : "#7aa274",
          animation: isRunning ? "pulse 1.4s ease-in-out infinite" : undefined,
        }}
      />
      <span className="font-medium">
        {isRunning ? "Running" : "Finished"}: {AGENT_LABELS[showStep.name]}
      </span>
      {isRunning && (
        <>
          <span className="opacity-60 text-[13px]">· {fmt(elapsedMs)}</span>
          {HINT[showStep.name] && (
            <span className="opacity-60 text-[13px]">· {HINT[showStep.name]}</span>
          )}
        </>
      )}
    </div>
  );
}
