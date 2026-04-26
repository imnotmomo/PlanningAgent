"use client";
import { useState } from "react";

interface TripFormProps {
  onSubmit: (request: string) => void;
  disabled?: boolean;
}

const EXAMPLES = [
  "3 days in Kyoto for 2 adults, medium budget, temples and matcha, relaxed pace, prefer public transit.",
  "5 days in Japan for 2 friends, anime + food + city views, medium budget, public transit only.",
  "Weekend in Lisbon for a couple, viewpoints and food, no rush, medium budget.",
];

export function TripForm({ onSubmit, disabled }: TripFormProps) {
  const [text, setText] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    onSubmit(text.trim());
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div className="card-section p-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe your trip — destination, dates, travelers, interests, budget, pace, anything specific…"
          rows={5}
          disabled={disabled}
          className="w-full resize-none bg-transparent px-4 py-3 text-[16px] text-ink placeholder:text-warm focus:outline-none disabled:opacity-50"
          style={{ letterSpacing: "0.16px", lineHeight: 1.55 }}
        />
        <div className="flex items-center justify-between gap-3 px-2 pb-2 pt-1">
          <span className="text-caption">
            ⌘+Enter to send · Cmd+K to clear
          </span>
          <button
            type="submit"
            className="btn-warm"
            disabled={disabled || !text.trim()}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handleSubmit(e);
            }}
          >
            Plan trip
            <span aria-hidden>→</span>
          </button>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-2 justify-center">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setText(ex)}
            disabled={disabled}
            className="text-caption px-3 py-1.5 rounded-full bg-stone/70 hover:bg-stone transition-colors disabled:opacity-40"
            style={{ color: "#4e4e4e" }}
          >
            {ex.slice(0, 55)}…
          </button>
        ))}
      </div>
    </form>
  );
}
