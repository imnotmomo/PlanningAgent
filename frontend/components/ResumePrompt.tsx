"use client";

import { describeAge, type SavedSession } from "../lib/session";

interface ResumePromptProps {
  session: SavedSession;
  onContinue: () => void;
  onDismiss: () => void;
}

const phaseLabel: Record<SavedSession["phase"], string> = {
  choosing_destinations: "choosing destinations",
  picking: "picking places",
  done: "viewing the finished itinerary",
};

export function ResumePrompt({ session, onContinue, onDismiss }: ResumePromptProps) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center px-6"
      style={{ backgroundColor: "rgba(28, 26, 23, 0.45)" }}
    >
      <div className="card max-w-md w-full p-6 shadow-2xl">
        <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>
          Resume previous trip
        </div>
        <h2 className="text-2xl font-medium leading-tight mb-3 text-ink">
          You have an unfinished plan
        </h2>
        <p className="text-[15px] mb-2 text-ink" style={{ lineHeight: 1.55 }}>
          <span className="opacity-70">Last activity: </span>
          {describeAge(session.updated_at)} · {phaseLabel[session.phase]}
        </p>
        <p className="text-[14px] mb-6 text-warm" style={{ lineHeight: 1.55 }}>
          &ldquo;{session.request.length > 140
            ? session.request.slice(0, 140) + "…"
            : session.request}&rdquo;
        </p>
        <div className="flex gap-3 justify-end">
          <button onClick={onDismiss} className="btn-secondary">
            Start fresh
          </button>
          <button onClick={onContinue} className="btn-primary">
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}
