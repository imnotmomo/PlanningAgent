// Session persistence for in-progress trip planning. Saves the latest
// successful checkpoint (after destinations, after research, or final result)
// to localStorage so the user can resume after a refresh or browser close.
//
// We store a single slot keyed by `tripwise.session`. The contained `id` is
// useful if we later cloud-sync; for now it's just an identifier.

import type {
  DestinationsComplete,
  PlanResult,
  ResearchComplete,
  Selections,
} from "./api";

export type ResumablePhase = "choosing_destinations" | "picking" | "done";

export interface SavedSession {
  id: string;
  created_at: number;
  updated_at: number;
  request: string;
  phase: ResumablePhase;
  destResolution?: DestinationsComplete | null;
  research?: ResearchComplete | null;
  result?: PlanResult | null;
  lastSelections?: Selections | null;
}

const STORAGE_KEY = "tripwise.session";
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

export function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  // Fallback for older runtimes
  return "ses-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function loadSession(): SavedSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw) as SavedSession;
    if (!obj.id || !obj.phase) return null;
    if (Date.now() - obj.updated_at > MAX_AGE_MS) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return obj;
  } catch {
    return null;
  }
}

export function saveSession(s: Omit<SavedSession, "created_at" | "updated_at">): void {
  if (typeof window === "undefined") return;
  const existing = loadSession();
  const now = Date.now();
  const next: SavedSession = {
    ...s,
    created_at: existing?.created_at ?? now,
    updated_at: now,
  };
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Quota exceeded or storage disabled — silently no-op
  }
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function describeAge(updated_at: number): string {
  const diff = Math.max(0, Date.now() - updated_at);
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days > 1 ? "s" : ""} ago`;
}
