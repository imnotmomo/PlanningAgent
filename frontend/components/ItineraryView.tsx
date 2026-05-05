"use client";
import { Itinerary, Candidate, ArrivalData, ArrivalOption, DayMeals, ScheduleEntry } from "@/lib/api";

interface ItineraryViewProps {
  itinerary: Itinerary;
  budget?: { daily_estimate?: string; total_estimate?: string; notes?: string[] };
  critique?: {
    score?: number;
    passed?: boolean;
    issues: string[];
    should_revise?: boolean;
    suggested_revisions?: string[];
  };
  hotels?: Candidate[];
  arrival?: ArrivalData | null;
  arrivalChoices?: { outbound: ArrivalOption | null; return: ArrivalOption | null } | null;
  mealPlan?: Record<string, DayMeals>;
  transitNotes?: Record<string, string>;
  daySchedule?: Record<string, ScheduleEntry[]>;
  selectedHotels?: string[]; // names user picked (subset of hotels[])
}

export function ItineraryView({
  itinerary,
  budget,
  critique,
  hotels,
  arrival,
  mealPlan,
  transitNotes,
  selectedHotels,
  arrivalChoices,
  daySchedule,
}: ItineraryViewProps) {
  // Build a per-city -> chosen hotel map. If user picked one (or more), use
  // those; else pick the first candidate per city as a default.
  const hotelByCity: Record<string, Candidate> = {};
  if (hotels && hotels.length > 0) {
    const picks = new Set(selectedHotels || []);
    for (const h of hotels) {
      const city = h.city || "_";
      if (hotelByCity[city]) continue;
      if (picks.size === 0 || picks.has(h.name)) {
        hotelByCity[city] = h;
      }
    }
    // Fallbacks if no pick covered a city
    for (const h of hotels) {
      const city = h.city || "_";
      if (!hotelByCity[city]) hotelByCity[city] = h;
    }
  }
  return (
    <div className="flex flex-col gap-6">
      {itinerary.trip_summary && (
        <p className="text-body" style={{ fontSize: 20 }}>
          {itinerary.trip_summary}
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {itinerary.daily_itinerary?.map((d) => {
          const dayKeyCity = d.city ? `Day ${d.day} · ${d.city}` : `Day ${d.day}`;
          const meals = mealPlan?.[dayKeyCity] || mealPlan?.[`Day ${d.day}`];
          const transitNote =
            transitNotes?.[dayKeyCity] ||
            transitNotes?.[`Day ${d.day}`] ||
            d.transportation_note;
          const schedule = daySchedule?.[dayKeyCity] || daySchedule?.[`Day ${d.day}`];
          return (
            <article key={d.day} className="card p-6">
              <div className="flex items-baseline justify-between gap-3 mb-2">
                <span className="text-uppercase-cta" style={{ color: "#777169" }}>
                  Day {d.day}{d.city ? ` · ${d.city}` : ""}
                </span>
                <span className="text-caption">
                  {budget?.daily_estimate || d.estimated_cost}
                </span>
              </div>
              <h3 className="h-card text-[24px] mb-4">{d.theme}</h3>
              {d.city && hotelByCity[d.city] && (
                <div
                  className="mb-3 -mx-1 px-3 py-2 rounded-card flex items-baseline gap-2 flex-wrap"
                  style={{ background: "rgba(245,242,239,0.7)" }}
                >
                  <span
                    className="text-[10px]"
                    style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
                  >
                    Stay
                  </span>
                  <span className="text-[14px] font-medium" style={{ color: "#000" }}>
                    {hotelByCity[d.city].name}
                  </span>
                  {hotelByCity[d.city].description && (
                    <span className="text-caption">
                      {hotelByCity[d.city].description.split(/\s+/).slice(0, 8).join(" ") + "…"}
                    </span>
                  )}
                </div>
              )}
              {schedule && schedule.length > 0 && (
                <div className="mb-4">
                  <div className="text-[11px] mb-2" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>
                    Timeline
                  </div>
                  <ScheduleTimeline entries={schedule} />
                </div>
              )}
              <div className="flex flex-col gap-3">
                <Slot label="Morning" body={d.morning} />
                <Slot label="Afternoon" body={d.afternoon} />
                <Slot label="Evening" body={d.evening} />
              </div>
              {(meals?.lunch || meals?.dinner) && (
                <div className="mt-4 pt-3 border-t border-border/60 grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {meals?.lunch && (
                    <div>
                      <div className="text-[11px] mb-0.5" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>Lunch</div>
                      <div className="text-[14px]" style={{ color: "#000" }}>{meals.lunch}</div>
                    </div>
                  )}
                  {meals?.dinner && (
                    <div>
                      <div className="text-[11px] mb-0.5" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>Dinner</div>
                      <div className="text-[14px]" style={{ color: "#000" }}>{meals.dinner}</div>
                    </div>
                  )}
                </div>
              )}
              <div className="mt-5 pt-4 border-t border-border/60 flex flex-col gap-1.5">
                <div className="text-caption">
                  <span className="font-medium" style={{ color: "#000" }}>Transit ·</span> {transitNote}
                </div>
                <div className="text-caption">
                  <span className="font-medium" style={{ color: "#000" }}>Feasibility ·</span> {d.feasibility_note}
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {arrivalChoices && (arrivalChoices.outbound || arrivalChoices.return) && (
        <section className="card p-6">
          <div className="text-uppercase-cta mb-3" style={{ color: "#777169" }}>
            Your flights
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <FlightLine label="Outbound · go" opt={arrivalChoices.outbound} />
            <FlightLine label="Return · back" opt={arrivalChoices.return} />
          </div>
        </section>
      )}

      {(budget?.daily_estimate || itinerary.budget_summary) && (
        <section className="card p-6">
          <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>
            Budget
          </div>
          {budget?.daily_estimate && (
            <div className="text-body">
              <span style={{ color: "#000" }}>{budget.daily_estimate}</span>
              <span className="text-caption ml-1"> per person, on the ground</span>
            </div>
          )}
          {(budget as { airfare_estimate?: string } | undefined)?.airfare_estimate && (
            <div className="text-body mt-1.5">
              <span style={{ color: "#000" }}>
                {(budget as { airfare_estimate?: string }).airfare_estimate}
              </span>
              <span className="text-caption ml-1"> round trip</span>
            </div>
          )}
          {budget?.total_estimate && (
            <div
              className="text-body mt-1.5 font-medium"
              style={{ color: "#000" }}
            >
              Total: {budget.total_estimate}
            </div>
          )}
          {itinerary.budget_summary && (
            <div className="text-caption mt-2">{itinerary.budget_summary}</div>
          )}
          {budget?.notes && budget.notes.length > 0 && (
            <ul className="mt-3 flex flex-col gap-1">
              {budget.notes.map((n, i) => (
                <li key={i} className="text-caption">
                  · {n}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {(itinerary.travel_tips?.length || itinerary.backup_options?.length) ? (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {itinerary.travel_tips && itinerary.travel_tips.length > 0 && (
            <div className="card p-6">
              <div className="text-uppercase-cta mb-3" style={{ color: "#777169" }}>
                Travel tips
              </div>
              <ul className="flex flex-col gap-2">
                {itinerary.travel_tips.map((tip, i) => (
                  <li key={i} className="text-body" style={{ fontSize: 16 }}>
                    · {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {itinerary.backup_options && itinerary.backup_options.length > 0 && (
            <div className="card p-6">
              <div className="text-uppercase-cta mb-3" style={{ color: "#777169" }}>
                Backup options
              </div>
              <ul className="flex flex-col gap-2">
                {itinerary.backup_options.map((b, i) => (
                  <li key={i} className="text-body" style={{ fontSize: 16 }}>
                    · {b}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      ) : null}

      {/* Critic feedback is intentionally not rendered here. It still drives
          the replan loop in the orchestrator (and the retry pill in the
          pipeline panel), but the user sees the final passing plan. */}
    </div>
  );
}

function ScheduleTimeline({ entries }: { entries: ScheduleEntry[] }) {
  return (
    <ol className="flex flex-col gap-0">
      {entries.map((e, i) => {
        if (e.type === "transit") {
          return (
            <li key={i} className="pl-[68px] py-1 relative">
              <span
                className="absolute left-[26px] top-0 bottom-0 w-px"
                style={{ background: "#e5e5e5" }}
                aria-hidden
              />
              <span className="text-caption" style={{ color: "#777169" }}>
                ↓ {e.mode || "transit"}
                {typeof e.minutes === "number" && ` · ${e.minutes} min`}
                <span className="opacity-60"> · {e.from} → {e.to}</span>
              </span>
            </li>
          );
        }
        const start = e.start || "";
        const end = e.end || "";
        const kindStyle =
          e.kind === "lunch" || e.kind === "dinner"
            ? { color: "#000", fontWeight: 600 }
            : { color: "#000" };
        return (
          <li key={i} className="flex items-baseline gap-3 py-1">
            <span
              className="text-caption shrink-0"
              style={{ color: "#000", fontVariantNumeric: "tabular-nums", minWidth: 56 }}
            >
              {start}
              {end && <span style={{ color: "#777169" }}>{" – " + end}</span>}
            </span>
            <span className="text-[14px]" style={kindStyle}>
              {e.name}
              {e.kind && e.kind !== "attraction" && (
                <span
                  className="ml-2 text-[10px] px-1.5 py-0.5 rounded-full"
                  style={{
                    background: "rgba(245,242,239,0.9)",
                    color: "#4e4e4e",
                    letterSpacing: "0.7px",
                    textTransform: "uppercase",
                    fontWeight: 700,
                  }}
                >
                  {e.kind}
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function FlightLine({ label, opt }: { label: string; opt: ArrivalOption | null }) {
  if (!opt) {
    return (
      <div>
        <div className="text-[11px] mb-1" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
        <div className="text-caption">Not chosen</div>
      </div>
    );
  }
  return (
    <div>
      <div className="text-[11px] mb-1" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div className="text-[15px] font-medium" style={{ color: "#000" }}>
        {opt.carrier || opt.mode}
        {opt.class && <span className="text-caption ml-2">({opt.class.replace("_", " ")})</span>}
      </div>
      <div className="text-caption">
        {opt.duration}
        {typeof opt.stops === "number" && <> · {opt.stops === 0 ? "direct" : `${opt.stops} stop${opt.stops > 1 ? "s" : ""}`}</>}
        {opt.price && <> · {opt.price} one-way</>}
      </div>
    </div>
  );
}

function Slot({ label, body }: { label: string; body: string }) {
  return (
    <div>
      <div
        className="text-[12px] mb-1"
        style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
      >
        {label}
      </div>
      <div className="text-[15px]" style={{ color: "#000", letterSpacing: "0.16px", lineHeight: 1.5 }}>
        {body}
      </div>
    </div>
  );
}
