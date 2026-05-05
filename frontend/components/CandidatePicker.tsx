"use client";
import { useState } from "react";
import {
  Candidate,
  ResearchPayload,
  ArrivalOption,
  ArrivalData,
  Selections,
  CandidateDetail,
  fetchCandidateDetail,
  enrichCustomCandidates,
} from "@/lib/api";

interface CandidatePickerProps {
  research: ResearchPayload;
  arrival: ArrivalData | null;
  preferences: Record<string, unknown>;
  /** Called with the final selections; second arg is enriched custom
   *  candidates (with Tavily-fetched descriptions) the caller should
   *  merge into research.places/restaurants/hotels before /build. */
  onContinue: (
    selections: Selections | null,
    customs: { places: Candidate[]; restaurants: Candidate[]; hotels: Candidate[] } | null,
  ) => void;
  onCancel: () => void;
  disabled?: boolean;
}

type Cat = "places" | "restaurants" | "hotels";
const CAT_TO_CATEGORY: Record<Cat, "place" | "restaurant" | "hotel"> = {
  places: "place",
  restaurants: "restaurant",
  hotels: "hotel",
};

const CATEGORY_META: Record<keyof ResearchPayload | "by_city", { title: string; sublabel: string; cat: "place" | "restaurant" | "hotel" }> = {
  places: { title: "Places", sublabel: "Attractions, viewpoints, neighborhoods", cat: "place" },
  restaurants: { title: "Restaurants", sublabel: "Lunch + dinner candidates", cat: "restaurant" },
  hotels: { title: "Hotels", sublabel: "Where to stay", cat: "hotel" },
  by_city: { title: "", sublabel: "", cat: "place" },
};

function shortDescription(desc: string | undefined, maxWords: number = 5): string {
  if (!desc) return "";
  const words = desc.split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) return desc;
  return words.slice(0, maxWords).join(" ") + "…";
}

export function CandidatePicker({
  research,
  arrival,
  preferences,
  onContinue,
  onCancel,
  disabled,
}: CandidatePickerProps) {
  // Default: NOTHING selected. User opts in to priority items.
  const [picks, setPicks] = useState<Selections>({
    places: [],
    restaurants: [],
    hotels: [],
    arrival_choices: null,
  });
  const [outbound, setOutbound] = useState<ArrivalOption | null>(null);
  const [returnFlight, setReturnFlight] = useState<ArrivalOption | null>(null);
  const [decideLater, setDecideLater] = useState(false);
  const [detailOpen, setDetailOpen] = useState<{ item: Candidate; cat: "place" | "restaurant" | "hotel" } | null>(null);

  // User-added custom candidates per category. Populated by the "+ Add custom"
  // card; descriptions are filled in via Tavily on Continue.
  const [customs, setCustoms] = useState<{ places: Candidate[]; restaurants: Candidate[]; hotels: Candidate[] }>({
    places: [],
    restaurants: [],
    hotels: [],
  });
  // The category currently showing the inline add form (one at a time)
  const [addingFor, setAddingFor] = useState<Cat | null>(null);
  const [enriching, setEnriching] = useState(false);
  const [enrichErr, setEnrichErr] = useState<string | null>(null);

  // Distinct cities present in research, for the city picker on multi-leg trips.
  const cityOptions: string[] = (() => {
    const set = new Set<string>();
    for (const arr of [research.places, research.restaurants, research.hotels]) {
      for (const it of arr || []) if (it.city) set.add(it.city);
    }
    return Array.from(set);
  })();

  const addCustom = (cat: Cat, name: string, city: string | undefined) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setCustoms((prev) => ({
      ...prev,
      [cat]: [...prev[cat], { name: trimmed, city, description: "" }],
    }));
    // Auto-pick the new custom so it makes it into selections
    setPicks((prev) => {
      const set = new Set(prev[cat]);
      set.add(trimmed);
      return { ...prev, [cat]: Array.from(set) };
    });
    setAddingFor(null);
  };

  const removeCustom = (cat: Cat, name: string) => {
    setCustoms((prev) => ({
      ...prev,
      [cat]: prev[cat].filter((c) => c.name !== name),
    }));
    setPicks((prev) => ({
      ...prev,
      [cat]: prev[cat].filter((n) => n !== name),
    }));
  };

  const toggle = (cat: Cat, name: string) => {
    setPicks((prev) => {
      const set = new Set(prev[cat]);
      if (set.has(name)) set.delete(name);
      else set.add(name);
      return { ...prev, [cat]: Array.from(set) };
    });
  };

  const totalPicked = picks.places.length + picks.restaurants.length + picks.hotels.length;
  const totalAvail =
    research.places.length + research.restaurants.length + research.hotels.length;

  const outboundOpts = arrival?.outbound_options || [];
  const returnOpts = arrival?.return_options || [];
  const hasRoundTrip = outboundOpts.length > 0 || returnOpts.length > 0;

  const totalCustom = customs.places.length + customs.restaurants.length + customs.hotels.length;

  const handleContinue = async () => {
    const arrivalChoices =
      hasRoundTrip && !decideLater && (outbound || returnFlight)
        ? { outbound, return: returnFlight }
        : null;
    const finalSelections: Selections = {
      ...picks,
      arrival_choices: arrivalChoices,
    };
    const anyPick = totalPicked > 0 || arrivalChoices != null;

    // If the user added customs, enrich them via Tavily so route + itinerary
    // see real descriptions, not blanks. Block the Continue button while this
    // runs (typically <5 s for a few items).
    let enrichedCustoms: { places: Candidate[]; restaurants: Candidate[]; hotels: Candidate[] } | null = null;
    if (totalCustom > 0) {
      setEnriching(true);
      setEnrichErr(null);
      try {
        const inputs = (["places", "restaurants", "hotels"] as const).flatMap((cat) =>
          customs[cat].map((c) => ({ name: c.name, city: c.city, category: CAT_TO_CATEGORY[cat] })),
        );
        const dest = (preferences as { destination?: string }).destination;
        const got = await enrichCustomCandidates(inputs, dest);
        // Map results back into per-category lists, preserving order.
        enrichedCustoms = { places: [], restaurants: [], hotels: [] };
        let idx = 0;
        for (const cat of ["places", "restaurants", "hotels"] as const) {
          for (let i = 0; i < customs[cat].length; i++) {
            const enr = got[idx++];
            if (enr) enrichedCustoms[cat].push(enr);
          }
        }
      } catch (e) {
        setEnrichErr(e instanceof Error ? e.message : String(e));
        setEnriching(false);
        return;
      }
      setEnriching(false);
    }
    onContinue(anyPick ? finalSelections : null, enrichedCustoms);
  };

  return (
    <div className="flex flex-col gap-6">
      <PreferenceSummary preferences={preferences} arrival={arrival} />

      {hasRoundTrip && (
        <RoundTripPicker
          outboundOpts={outboundOpts}
          returnOpts={returnOpts}
          outbound={outbound}
          returnFlight={returnFlight}
          decideLater={decideLater}
          onPickOutbound={(o) => { setOutbound(o); setDecideLater(false); }}
          onPickReturn={(o) => { setReturnFlight(o); setDecideLater(false); }}
          onDecideLater={() => { setOutbound(null); setReturnFlight(null); setDecideLater(true); }}
          disabled={disabled}
        />
      )}

      {(["places", "restaurants", "hotels"] as const).map((cat) => {
        const items = research[cat] || [];
        const customItems = customs[cat];
        const totalCount = items.length + customItems.length;
        // Always render the section so the "+ Add custom" card is reachable
        // even when research returned zero candidates for this category.
        const meta = CATEGORY_META[cat];
        const picked = new Set(picks[cat]);
        return (
          <section key={cat} className="card p-6">
            <div className="flex items-baseline justify-between gap-3 mb-4 flex-wrap">
              <div>
                <div className="text-uppercase-cta" style={{ color: "#777169" }}>
                  {meta.title} <span style={{ fontSize: 11 }}>
                    · {totalCount} candidate{totalCount === 1 ? "" : "s"}
                    {customItems.length > 0 && ` (${customItems.length} custom)`}
                  </span>
                </div>
                <div className="text-caption mt-0.5">{meta.sublabel}</div>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {items.map((it: Candidate, i) => {
                const isPicked = picked.has(it.name);
                return (
                  <button
                    key={i}
                    type="button"
                    onClick={() => toggle(cat, it.name)}
                    disabled={disabled}
                    aria-pressed={isPicked}
                    className={`text-left rounded-card p-4 transition-all relative ${isPicked ? "" : "hover:shadow-outline"}`}
                    style={{
                      background: isPicked ? "rgba(245,242,239,0.95)" : "#fff",
                      boxShadow: isPicked
                        ? "rgba(78,50,23,0.08) 0px 6px 16px, rgba(0,0,0,0.075) 0px 0px 0px 1px inset"
                        : "rgba(0,0,0,0.06) 0px 0px 0px 1px, rgba(0,0,0,0.04) 0px 1px 2px",
                      minHeight: 100,
                      opacity: disabled ? 0.5 : 1,
                    }}
                  >
                    {isPicked && (
                      <span
                        className="absolute top-2 right-2 h-5 w-5 rounded-full flex items-center justify-center text-white"
                        style={{ background: "#000", fontSize: 11 }}
                        aria-hidden
                      >
                        ✓
                      </span>
                    )}
                    <div className="flex items-baseline gap-2 flex-wrap pr-6">
                      <span className="text-[15px] font-medium" style={{ color: "#000", letterSpacing: "0.15px", lineHeight: 1.25 }}>
                        {it.name}
                      </span>
                      {it.city && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded-full"
                          style={{ background: isPicked ? "#fff" : "rgba(245,242,239,0.9)", color: "#4e4e4e", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
                        >
                          {it.city}
                        </span>
                      )}
                    </div>
                    {it.description && (
                      <div
                        className="text-[13px] mt-1.5"
                        style={{ color: "#4e4e4e", letterSpacing: "0.14px", lineHeight: 1.45 }}
                      >
                        {shortDescription(it.description, 6)}
                      </div>
                    )}
                    <div className="mt-3">
                      <span
                        role="link"
                        tabIndex={disabled ? -1 : 0}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!disabled) setDetailOpen({ item: it, cat: meta.cat });
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            e.stopPropagation();
                            if (!disabled) setDetailOpen({ item: it, cat: meta.cat });
                          }
                        }}
                        className="text-[12px] underline cursor-pointer"
                        style={{ color: "#777169", letterSpacing: "0.14px" }}
                      >
                        Show more
                      </span>
                    </div>
                  </button>
                );
              })}

              {/* User-added customs render alongside research items */}
              {customItems.map((it, i) => {
                const isPicked = picked.has(it.name);
                return (
                  <div
                    key={`custom-${i}-${it.name}`}
                    role="button"
                    tabIndex={disabled ? -1 : 0}
                    aria-pressed={isPicked}
                    onClick={() => toggle(cat, it.name)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggle(cat, it.name);
                      }
                    }}
                    className="text-left rounded-card p-4 transition-all relative cursor-pointer"
                    style={{
                      background: isPicked ? "rgba(245,242,239,0.95)" : "#fff",
                      boxShadow: isPicked
                        ? "rgba(78,50,23,0.08) 0px 6px 16px, rgba(0,0,0,0.075) 0px 0px 0px 1px inset"
                        : "rgba(0,0,0,0.06) 0px 0px 0px 1px, rgba(0,0,0,0.04) 0px 1px 2px",
                      minHeight: 100,
                      opacity: disabled ? 0.5 : 1,
                    }}
                  >
                    {isPicked && (
                      <span
                        className="absolute top-2 right-2 h-5 w-5 rounded-full flex items-center justify-center text-white"
                        style={{ background: "#000", fontSize: 11 }}
                        aria-hidden
                      >
                        ✓
                      </span>
                    )}
                    <div className="flex items-baseline gap-2 flex-wrap pr-6">
                      <span className="text-[15px] font-medium" style={{ color: "#000", letterSpacing: "0.15px", lineHeight: 1.25 }}>
                        {it.name}
                      </span>
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded-full"
                        style={{ background: "#bf6e3a", color: "#fff", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
                      >
                        custom
                      </span>
                      {it.city && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded-full"
                          style={{ background: isPicked ? "#fff" : "rgba(245,242,239,0.9)", color: "#4e4e4e", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
                        >
                          {it.city}
                        </span>
                      )}
                    </div>
                    <div className="text-[13px] mt-1.5" style={{ color: "#4e4e4e", letterSpacing: "0.14px", lineHeight: 1.45 }}>
                      Description fetched on Continue.
                    </div>
                    <div className="mt-3">
                      <span
                        role="link"
                        tabIndex={disabled ? -1 : 0}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!disabled) removeCustom(cat, it.name);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            e.stopPropagation();
                            if (!disabled) removeCustom(cat, it.name);
                          }
                        }}
                        className="text-[12px] underline cursor-pointer"
                        style={{ color: "#777169", letterSpacing: "0.14px" }}
                      >
                        Remove
                      </span>
                    </div>
                  </div>
                );
              })}

              {/* "+ Add custom" card */}
              {addingFor === cat ? (
                <CustomAddForm
                  cityOptions={cityOptions}
                  defaultCity={cityOptions.length === 1 ? cityOptions[0] : undefined}
                  onCancel={() => setAddingFor(null)}
                  onSave={(name, city) => addCustom(cat, name, city)}
                  disabled={disabled}
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setAddingFor(cat)}
                  disabled={disabled}
                  className="rounded-card p-4 flex items-center justify-center gap-2 transition-colors"
                  style={{
                    minHeight: 100,
                    border: "1.5px dashed rgba(78,50,23,0.25)",
                    color: "#777169",
                    background: "transparent",
                    opacity: disabled ? 0.4 : 1,
                  }}
                  aria-label={`Add a custom ${meta.cat}`}
                >
                  <span style={{ fontSize: 18, lineHeight: 1 }}>+</span>
                  <span className="text-[14px]" style={{ letterSpacing: "0.14px" }}>
                    Add your own {meta.cat}
                  </span>
                </button>
              )}
            </div>
          </section>
        );
      })}

      <div
        className="sticky bottom-4 flex flex-wrap items-center justify-between gap-3 card p-4"
        style={{ background: "rgba(255,255,255,0.95)", backdropFilter: "blur(8px)" }}
      >
        <span className="text-caption">
          {totalPicked === 0
            ? `No candidate preference — pipeline considers all ${totalAvail} equally`
            : `${totalPicked} prioritized · ${totalAvail - totalPicked} optional`}
          {totalCustom > 0 && ` · ${totalCustom} custom`}
          {hasRoundTrip && (
            outbound || returnFlight
              ? ` · flights: ${outbound ? "go ✓" : "go —"}, ${returnFlight ? "back ✓" : "back —"}`
              : decideLater
                ? " · flights: decide later"
                : " · flights: not chosen"
          )}
          {enriching && " · researching custom candidates…"}
          {enrichErr && (
            <span style={{ color: "#7c2d12" }}> · enrich failed: {enrichErr}</span>
          )}
        </span>
        <div className="flex gap-2 flex-wrap">
          <button onClick={onCancel} disabled={disabled || enriching} className="btn-secondary text-[14px] py-2 px-3">
            Back
          </button>
          {totalPicked > 0 && (
            <button
              onClick={() => {
                setPicks({ places: [], restaurants: [], hotels: [], arrival_choices: null });
                setCustoms({ places: [], restaurants: [], hotels: [] });
              }}
              disabled={disabled || enriching}
              className="btn-secondary text-[14px] py-2 px-3"
            >
              Clear selection
            </button>
          )}
          <button
            onClick={handleContinue}
            disabled={disabled || enriching}
            className="btn-warm"
          >
            {enriching ? "Looking up customs…" : "Continue planning"}
            <span aria-hidden>→</span>
          </button>
        </div>
      </div>

      {detailOpen && (
        <DetailModal
          item={detailOpen.item}
          category={detailOpen.cat}
          onClose={() => setDetailOpen(null)}
        />
      )}
    </div>
  );
}

function CustomAddForm({
  cityOptions,
  defaultCity,
  onCancel,
  onSave,
  disabled,
}: {
  cityOptions: string[];
  defaultCity?: string;
  onCancel: () => void;
  onSave: (name: string, city: string | undefined) => void;
  disabled?: boolean;
}) {
  const [name, setName] = useState("");
  const [city, setCity] = useState(defaultCity ?? "");
  const showCity = cityOptions.length > 1;
  return (
    <div
      className="rounded-card p-3 flex flex-col gap-2"
      style={{
        minHeight: 100,
        border: "1.5px solid rgba(78,50,23,0.35)",
        background: "rgba(245,242,239,0.6)",
      }}
    >
      <input
        type="text"
        value={name}
        autoFocus
        placeholder="Name (e.g., Pontocho Alley)"
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && name.trim()) {
            e.preventDefault();
            onSave(name, city || undefined);
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
        disabled={disabled}
        className="w-full bg-transparent border-0 outline-none text-[14px] font-medium"
        style={{ color: "#000", letterSpacing: "0.15px" }}
      />
      {showCity && (
        <select
          value={city}
          onChange={(e) => setCity(e.target.value)}
          disabled={disabled}
          className="w-full bg-transparent border-0 outline-none text-[12px]"
          style={{ color: "#4e4e4e", padding: "2px 0" }}
        >
          <option value="">Pick a city…</option>
          {cityOptions.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      )}
      <div className="flex justify-end gap-2 mt-auto">
        <button
          type="button"
          onClick={onCancel}
          disabled={disabled}
          className="text-[12px] px-2 py-1 rounded-full"
          style={{ background: "transparent", color: "#777169" }}
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={() => onSave(name, city || undefined)}
          disabled={disabled || !name.trim()}
          className="text-[12px] px-2.5 py-1 rounded-full"
          style={{ background: "#000", color: "#fff", opacity: !name.trim() ? 0.4 : 1 }}
        >
          Add
        </button>
      </div>
    </div>
  );
}


function RoundTripPicker({
  outboundOpts,
  returnOpts,
  outbound,
  returnFlight,
  decideLater,
  onPickOutbound,
  onPickReturn,
  onDecideLater,
  disabled,
}: {
  outboundOpts: ArrivalOption[];
  returnOpts: ArrivalOption[];
  outbound: ArrivalOption | null;
  returnFlight: ArrivalOption | null;
  decideLater: boolean;
  onPickOutbound: (o: ArrivalOption | null) => void;
  onPickReturn: (o: ArrivalOption | null) => void;
  onDecideLater: () => void;
  disabled?: boolean;
}) {
  return (
    <section className="card p-6">
      <div className="flex items-baseline justify-between gap-3 mb-4 flex-wrap">
        <div>
          <div className="text-uppercase-cta" style={{ color: "#777169" }}>
            Round-trip · choose go and back
          </div>
          <div className="text-caption mt-0.5">
            {decideLater
              ? "You chose to decide later — budget will use a generic estimate"
              : "Or skip to decide later. Your selection is included in the total budget."}
          </div>
        </div>
        <button
          type="button"
          onClick={onDecideLater}
          disabled={disabled}
          className={`text-[12px] px-2.5 py-1 rounded-full ${decideLater ? "" : ""}`}
          style={{
            background: decideLater ? "#000" : "rgba(245,242,239,0.9)",
            color: decideLater ? "#fff" : "#4e4e4e",
          }}
        >
          Decide later
        </button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <FlightDirectionList
          label="Outbound · go"
          opts={outboundOpts}
          selected={outbound}
          onPick={onPickOutbound}
          disabled={disabled}
        />
        <FlightDirectionList
          label="Return · back"
          opts={returnOpts}
          selected={returnFlight}
          onPick={onPickReturn}
          disabled={disabled}
        />
      </div>
    </section>
  );
}

function FlightDirectionList({
  label,
  opts,
  selected,
  onPick,
  disabled,
}: {
  label: string;
  opts: ArrivalOption[];
  selected: ArrivalOption | null;
  onPick: (o: ArrivalOption | null) => void;
  disabled?: boolean;
}) {
  if (opts.length === 0) {
    return (
      <div>
        <div className="text-[11px] mb-2" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>
          {label}
        </div>
        <div className="text-caption">No options.</div>
      </div>
    );
  }
  return (
    <div>
      <div className="text-[11px] mb-2" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>
        {label}
      </div>
      <ul className="flex flex-col gap-2">
        {opts.map((o, i) => {
          const active = selected === o;
          return (
            <li key={i}>
              <button
                type="button"
                onClick={() => onPick(active ? null : o)}
                disabled={disabled}
                className="w-full text-left rounded-card p-3 transition-all"
                style={{
                  background: active ? "rgba(245,242,239,0.95)" : "#fff",
                  boxShadow: active
                    ? "rgba(78,50,23,0.08) 0px 6px 16px, rgba(0,0,0,0.075) 0px 0px 0px 1px inset"
                    : "rgba(0,0,0,0.06) 0px 0px 0px 1px",
                  opacity: disabled ? 0.5 : 1,
                }}
              >
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-[14px] font-medium" style={{ color: "#000" }}>
                    {o.carrier || o.mode}
                  </span>
                  {o.class && (
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded-full"
                      style={{ background: "rgba(245,242,239,0.9)", color: "#4e4e4e", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}
                    >
                      {o.class.replace("_", " ")}
                    </span>
                  )}
                  {active && (
                    <span className="ml-auto h-5 w-5 rounded-full flex items-center justify-center text-white" style={{ background: "#000", fontSize: 11 }}>✓</span>
                  )}
                </div>
                <div className="text-caption mt-1">
                  {o.duration && <>{o.duration}</>}
                  {typeof o.stops === "number" && (
                    <> · {o.stops === 0 ? "direct" : `${o.stops} stop${o.stops > 1 ? "s" : ""}`}</>
                  )}
                  {o.price && <> · <span style={{ color: "#000" }}>{o.price}</span> one-way</>}
                </div>
                {o.notes && <div className="text-caption">{o.notes}</div>}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function DetailModal({
  item,
  category,
  onClose,
}: {
  item: Candidate;
  category: "place" | "restaurant" | "hotel";
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<CandidateDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onResearch = async () => {
    setLoading(true);
    setErr(null);
    try {
      const d = await fetchCandidateDetail(item.name, item.city, category);
      setDetail(d);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(2px)" }}
      onClick={onClose}
    >
      <div
        className="card-section w-full max-w-2xl max-h-[85vh] overflow-y-auto p-6 relative"
        style={{ background: "#fff" }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 h-8 w-8 rounded-full text-[14px]"
          style={{ background: "rgba(245,242,239,0.9)", color: "#4e4e4e" }}
          aria-label="Close"
        >
          ×
        </button>
        <div className="flex items-baseline gap-2 flex-wrap pr-8 mb-2">
          <h2 className="h-card text-[24px]">{item.name}</h2>
          {item.city && (
            <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(245,242,239,0.9)", color: "#4e4e4e", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>
              {item.city}
            </span>
          )}
          <span className="text-caption">· {category}</span>
        </div>
        {item.description && (
          <p className="text-body mb-4" style={{ fontSize: 15 }}>
            {item.description}
          </p>
        )}

        {!detail && !loading && (
          <button onClick={onResearch} className="btn-warm">
            Research more (with images)
            <span aria-hidden>→</span>
          </button>
        )}
        {loading && <div className="text-caption">Searching the web…</div>}
        {err && <div className="text-caption" style={{ color: "#7c2d12" }}>Error: {err}</div>}

        {detail && (
          <div className="flex flex-col gap-4">
            {detail.images.length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {detail.images.slice(0, 6).map((src, i) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={i}
                    src={src}
                    alt={`${item.name} ${i + 1}`}
                    className="rounded-card w-full h-32 object-cover"
                    style={{ background: "#f5f5f5" }}
                    referrerPolicy="no-referrer"
                  />
                ))}
              </div>
            )}
            {detail.summary && (
              <div className="text-body" style={{ fontSize: 15 }}>{detail.summary}</div>
            )}
            {detail.sources.length > 0 && (
              <div>
                <div className="text-uppercase-cta mb-2" style={{ color: "#777169" }}>Sources</div>
                <ul className="flex flex-col gap-1.5">
                  {detail.sources.slice(0, 5).map((s, i) => (
                    <li key={i} className="text-[14px]" style={{ color: "#4e4e4e" }}>
                      {s.url ? (
                        <a
                          href={s.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline"
                          style={{ color: "#000" }}
                        >
                          {s.title || s.url}
                        </a>
                      ) : (
                        s.title
                      )}
                      {s.snippet && (
                        <div className="text-caption">{s.snippet.slice(0, 200)}…</div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <button onClick={onResearch} className="btn-secondary text-[13px] py-1.5 px-3 self-start">
              Search again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function PreferenceSummary({
  preferences,
  arrival,
}: {
  preferences: Record<string, unknown>;
  arrival: ArrivalData | null;
}) {
  const p = preferences as {
    destination?: string;
    destinations?: { city: string; country?: string; days: number }[];
    origin?: string | null;
    trip_length_days?: number;
    travelers?: string;
    budget_level?: string;
    pace?: string;
    interests?: string[];
  };
  const dests = p.destinations || (p.destination ? [{ city: p.destination, days: p.trip_length_days || 0 }] : []);
  const tripLabel =
    dests.length > 1
      ? dests.map((d) => `${d.city} (${d.days}d)`).join(" → ")
      : dests.map((d) => d.city).join("");
  return (
    <div className="card p-6">
      <div className="text-uppercase-cta mb-3" style={{ color: "#777169" }}>
        Trip summary
      </div>
      <div className="text-[16px]" style={{ color: "#000", letterSpacing: "0.16px", lineHeight: 1.55 }}>
        <strong>{tripLabel}</strong>
        {p.origin && <> from {p.origin}</>}
        {p.trip_length_days && <> · {p.trip_length_days} days total</>}
        {p.travelers && <> · {p.travelers}</>}
        {p.pace && <> · {p.pace} pace</>}
        {p.budget_level && <> · {p.budget_level} budget</>}
      </div>
      {p.interests && p.interests.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {p.interests.map((i, idx) => (
            <span
              key={idx}
              className="text-[13px] px-2.5 py-1 rounded-full bg-surface"
              style={{ color: "#4e4e4e", letterSpacing: "0.14px" }}
            >
              {i}
            </span>
          ))}
        </div>
      )}
      {arrival && (arrival.outbound_options || arrival.options) && (
        <div className="mt-4 pt-4 border-t border-border/60">
          <div className="text-[11px] mb-2" style={{ color: "#777169", letterSpacing: "0.7px", textTransform: "uppercase", fontWeight: 700 }}>
            Round-trip options below
          </div>
          <div className="text-caption">
            {arrival.outbound_options?.length || 0} outbound and {arrival.return_options?.length || 0} return options found.
            Pick one of each (or skip) below.
          </div>
        </div>
      )}
    </div>
  );
}
