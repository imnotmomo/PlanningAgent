"use client";
import { useState } from "react";

interface RevisionFormProps {
  onSubmit: (change: string) => Promise<void> | void;
  disabled?: boolean;
}

const SUGGESTIONS = [
  "Make it more relaxed",
  "Remove museums",
  "Add more food places",
  "Keep each day under $100",
  "Avoid long walking",
];

export function RevisionForm({ onSubmit, disabled }: RevisionFormProps) {
  const [text, setText] = useState("");

  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault();
        const change = text.trim();
        if (!change) return;
        await onSubmit(change);
        setText("");
      }}
      className="card-section p-2"
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="What would you like to change about this itinerary?"
        rows={3}
        disabled={disabled}
        className="w-full resize-none bg-transparent px-4 py-3 text-[15px] text-ink placeholder:text-warm focus:outline-none disabled:opacity-50"
        style={{ letterSpacing: "0.16px", lineHeight: 1.55 }}
      />
      <div className="flex items-center justify-between gap-3 px-2 pb-2 pt-1 flex-wrap">
        <div className="flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setText(s)}
              disabled={disabled}
              className="text-[13px] px-2.5 py-1 rounded-full bg-stone/70 hover:bg-stone transition-colors disabled:opacity-40"
              style={{ color: "#4e4e4e" }}
            >
              {s}
            </button>
          ))}
        </div>
        <button type="submit" className="btn-secondary" disabled={disabled || !text.trim()}>
          Apply revision
        </button>
      </div>
    </form>
  );
}
