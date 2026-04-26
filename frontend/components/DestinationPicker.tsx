"use client";
import { useState } from "react";
import { CityCandidate, Destination, DestinationSuggester } from "@/lib/api";

interface DestinationPickerProps {
  countryOrRegion: string;
  tripLengthDays: number;
  suggester: DestinationSuggester;
  onContinue: (destinations: Destination[]) => void;
  onCancel: () => void;
  disabled?: boolean;
}

export function DestinationPicker({
  countryOrRegion,
  tripLengthDays,
  suggester,
  onContinue,
  onCancel,
  disabled,
}: DestinationPickerProps) {
  // Each city can be selected with a custom day count. Selection order is
  // tracked so the first picked city = arrival city for the outbound flight,
  // last picked = departure city for the return flight.
  const [picks, setPicks] = useState<Record<string, number>>({});
  const [order, setOrder] = useState<string[]>([]);

  const total = Object.values(picks).reduce((a, b) => a + b, 0);
  const remaining = tripLengthDays - total;

  const togglePick = (c: CityCandidate) => {
    setPicks((prev) => {
      const next = { ...prev };
      if (next[c.city] != null) {
        delete next[c.city];
        setOrder((prevOrder) => prevOrder.filter((x) => x !== c.city));
      } else {
        next[c.city] = Math.min(c.suggested_days || 2, Math.max(1, tripLengthDays - total));
        setOrder((prevOrder) => [...prevOrder, c.city]);
      }
      return next;
    });
  };

  const setDays = (city: string, days: number) => {
    setPicks((prev) => ({ ...prev, [city]: Math.max(1, days) }));
  };

  const useSuggested = () => {
    const next: Record<string, number> = {};
    const ord: string[] = [];
    for (const d of suggester.default_split) {
      next[d.city] = d.days;
      ord.push(d.city);
    }
    setPicks(next);
    setOrder(ord);
  };

  const proceed = (useDefault: boolean) => {
    let dests: Destination[];
    if (useDefault) {
      dests = suggester.default_split;
    } else {
      // Preserve user's selection order so first = arrival city, last = return city
      dests = order
        .filter((city) => picks[city] != null)
        .map((city) => {
          const c = suggester.candidates.find((x) => x.city === city);
          return { city, country: c?.country || "", days: picks[city] };
        });
    }
    if (dests.length === 0) {
      dests = suggester.default_split;
    }
    onContinue(dests);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="card p-6">
        <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>
          {countryOrRegion} — choose where to go
        </div>
        <p className="text-body" style={{ fontSize: 15 }}>
          You said <strong>{countryOrRegion}</strong> for <strong>{tripLengthDays} days</strong>.
          Pick the cities you want to visit and how many days in each, or skip
          and we&apos;ll use the suggested split below.
        </p>
        <div className="mt-3 text-caption">
          Suggested split:{" "}
          {suggester.default_split.map((d, i) => (
            <span key={i}>
              {i > 0 && " → "}
              <strong style={{ color: "#000" }}>{d.city} ({d.days}d)</strong>
            </span>
          ))}
        </div>
      </div>

      <section className="card p-6">
        <div className="flex items-baseline justify-between gap-3 mb-4 flex-wrap">
          <div>
            <div className="text-uppercase-cta" style={{ color: "#777169" }}>
              Candidate cities
            </div>
            <div className="text-caption mt-0.5">
              Order matters — your first pick is where the outbound flight lands; your last pick is where the return flight departs.
            </div>
          </div>
          <button
            type="button"
            onClick={useSuggested}
            disabled={disabled}
            className="text-[12px] px-2.5 py-1 rounded-full bg-stone/70 hover:bg-stone disabled:opacity-40"
            style={{ color: "#4e4e4e" }}
          >
            Use suggested split
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {suggester.candidates.map((c, i) => {
            const isPicked = picks[c.city] != null;
            const pickIndex = order.indexOf(c.city);
            return (
              <div
                key={i}
                className="rounded-card p-4"
                style={{
                  background: isPicked ? "rgba(245,242,239,0.95)" : "#fff",
                  boxShadow: isPicked
                    ? "rgba(78,50,23,0.08) 0px 6px 16px, rgba(0,0,0,0.075) 0px 0px 0px 1px inset"
                    : "rgba(0,0,0,0.06) 0px 0px 0px 1px",
                  opacity: disabled ? 0.5 : 1,
                }}
              >
                <button
                  type="button"
                  onClick={() => togglePick(c)}
                  disabled={disabled}
                  className="text-left w-full"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="text-[15px] font-medium" style={{ color: "#000" }}>
                      {c.city}
                    </span>
                    <span className="text-[12px]" style={{ color: "#777169" }}>
                      {c.country}
                    </span>
                    {isPicked && (
                      <span
                        className="ml-auto h-5 w-5 rounded-full flex items-center justify-center text-white"
                        style={{ background: "#000", fontSize: 11 }}
                        title={`Pick #${pickIndex + 1}`}
                      >
                        {pickIndex + 1}
                      </span>
                    )}
                  </div>
                  {c.description && (
                    <div
                      className="text-[13px] mt-1.5"
                      style={{ color: "#4e4e4e", letterSpacing: "0.14px", lineHeight: 1.45 }}
                    >
                      {c.description}
                    </div>
                  )}
                  <div className="text-caption mt-2">
                    suggested: {c.suggested_days} day{c.suggested_days === 1 ? "" : "s"}
                  </div>
                </button>
                {isPicked && (
                  <div className="mt-3 flex items-center gap-2">
                    <span className="text-caption">days:</span>
                    <button
                      type="button"
                      onClick={() => setDays(c.city, picks[c.city] - 1)}
                      disabled={disabled || picks[c.city] <= 1}
                      className="h-6 w-6 rounded-full text-[12px]"
                      style={{ background: "rgba(245,242,239,0.9)", color: "#4e4e4e" }}
                    >
                      −
                    </button>
                    <span className="text-[14px] font-medium" style={{ minWidth: 16, textAlign: "center" }}>
                      {picks[c.city]}
                    </span>
                    <button
                      type="button"
                      onClick={() => setDays(c.city, picks[c.city] + 1)}
                      disabled={disabled}
                      className="h-6 w-6 rounded-full text-[12px]"
                      style={{ background: "rgba(245,242,239,0.9)", color: "#4e4e4e" }}
                    >
                      +
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <div
        className="sticky bottom-4 flex flex-wrap items-center justify-between gap-3 card p-4"
        style={{ background: "rgba(255,255,255,0.95)", backdropFilter: "blur(8px)" }}
      >
        <span className="text-caption">
          {Object.keys(picks).length === 0
            ? `No selection — clicking continue uses the suggested split (${tripLengthDays} days)`
            : (() => {
                const n = Object.keys(picks).length;
                const arrival = order[0];
                const ret = order[order.length - 1];
                const flightLabel = arrival && arrival === ret
                  ? `flights to/from ${arrival}`
                  : arrival && ret
                    ? `arrive ${arrival}, depart ${ret}`
                    : "";
                return `Selected ${n} ${n === 1 ? "city" : "cities"} · ${total}/${tripLengthDays} days${remaining !== 0 ? ` (${remaining > 0 ? remaining + " unallocated" : Math.abs(remaining) + " over"})` : ""}${flightLabel ? " · " + flightLabel : ""}`;
              })()}
        </span>
        <div className="flex gap-2 flex-wrap">
          <button onClick={onCancel} disabled={disabled} className="btn-secondary text-[14px] py-2 px-3">
            Back
          </button>
          <button
            onClick={() => proceed(true)}
            disabled={disabled}
            className="btn-secondary text-[14px] py-2 px-3"
          >
            Use suggested
          </button>
          <button
            onClick={() => proceed(false)}
            disabled={disabled || (Object.keys(picks).length > 0 && total !== tripLengthDays)}
            className="btn-warm"
          >
            {Object.keys(picks).length === 0 ? "Continue with suggested" : "Continue with selection"}
            <span aria-hidden>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
