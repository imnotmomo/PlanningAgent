# PlanningAgent — TripWise

A multi-agent travel planning assistant. Free-form trip request in (e.g. *"5 days in Japan from San Francisco, 2 friends, anime + food, medium budget"*); structured day-by-day itinerary, real candidate places, round-trip flights, and an all-in budget out.

- **Live**: https://trip.23333.info — gated by Cloudflare Access; sign in with a `@columbia.edu` email (one-time PIN).
- **Merged itinerary model on HF Hub**: [`aquaqua/tripwise-7b-merged-bf16`](https://huggingface.co/aquaqua/tripwise-7b-merged-bf16)

## Architecture

| Role | Model | Where it runs |
|---|---|---|
| Orchestrator | `gpt-oss-120b` | Cerebras (`api.cerebras.ai/v1`), `reasoning_effort=low` |
| Itinerary | `tripwise-mlx-bf16` (Qwen2.5-7B + tripwise LoRA, merged) | oMLX on the Mac (`localhost:5620`), exposed to prod via `cloudflared` |

The orchestrator runs every agent except `itinerary` (research, route, budget, critic, tips, revision-router, etc.) and uses Tavily for web search and a Python subprocess tool for arithmetic. The fine-tune does one job: produce schema-conformant JSON for the day-by-day plan.

## Repo layout

```
backend/                 FastAPI multi-agent backend
  server.py              SSE endpoints + /enrich-candidates + /candidate-detail
  orchestrator.py        run_destinations / run_research / run_build / run_revise
  agents.py              11 agents + system prompts
  llm.py                 OpenAI clients, tool-loop, 90 s itin retry
  tools.py               tavily_search + python_exec
frontend/                Next.js 16 + Tailwind 3 (App Router)
  app/page.tsx           state machine + SSE consumers
  components/            TripForm, AgentProgress, CandidatePicker (with custom
                          "+" cards), DestinationPicker, ItineraryView,
                          RevisionForm, RevisionProgress, ResumePrompt
  lib/api.ts             SSE clients
  lib/session.ts         localStorage-backed plan resume (UUID, 7-day TTL)
train.py / eval.py       LoRA fine-tuning + held-out evaluation
serve_lora.py            Historical CUDA serving (vLLM/peft); now superseded
                          by the published merged MLX model on HF
travel_finetune_examples_1_200.jsonl   199 chat-format training examples
tripwise-itinerary-lora/ Trained LoRA adapter (~40 MB)
eval_output/             Per-example fine-tuned vs base outputs + metrics
report.md                Training + evaluation report
```

## Live deployment

> Requires a `@columbia.edu` email — Cloudflare Access sends a one-time PIN.

```
trip.23333.info  →  Cloudflare (proxied + Access)  →  Ubuntu 159.223.182.62
                                                       nginx :80/:443
                                                         /     → Next.js :3000
                                                         /api/ → FastAPI :8000
                                                       (systemd × 2)
                                                                │
                                                                ▼  itinerary calls
                                  cloudflared tunnel  ←  taxagent.23333.info
                                          │
                                  oMLX :5620 on the Mac
                                  → tripwise-mlx-bf16
```

- TLS via Let's Encrypt + `certbot.timer`.
- nginx `/api/` location has `proxy_buffering off` for SSE.
- `itin_complete` has a 90 s per-attempt timeout + 2 retries on connection drops.

## Setup

Local dev needs four pieces:

```
   frontend (Next.js)   localhost:3000
   backend  (FastAPI)   localhost:8000
   orch LLM             api.cerebras.ai      gpt-oss-120b
   itin LLM             localhost:5620       tripwise-mlx-bf16 on oMLX
```

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env       # then edit keys
uvicorn backend.server:app --port 8000     # from repo root
```

`backend/.env`:

| Var | Example | Notes |
|---|---|---|
| `ORCH_BASE_URL` | `https://api.cerebras.ai/v1` | any OpenAI-compat endpoint with tool calling |
| `ORCH_API_KEY` | `csk-…` | Cerebras key |
| `ORCH_MODEL` | `gpt-oss-120b` | `llm.py` auto-detects Qwen vs reasoning models for `extra_body` |
| `ITIN_BASE_URL` | `http://localhost:5620/v1` | oMLX, vLLM, or a cloudflared tunnel |
| `ITIN_API_KEY` | `omlx-…` |  |
| `ITIN_MODEL` | `tripwise-mlx-bf16` |  |
| `TAVILY_API_KEY` | `tvly-…` | https://tavily.com (free 1k/month) |

### Itinerary model

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download aquaqua/tripwise-7b-merged-bf16 --local-dir ~/Models/tripwise-mlx-bf16

# add to ~/.omlx/settings.json:  model.model_dirs += ["~/Models/tripwise-mlx-bf16"]
# restart oMLX, then:
curl localhost:5620/v1/models | jq '.data[].id'   # should include tripwise-mlx-bf16
```

### Frontend

```bash
cd frontend
cp .env.example .env.local                # leave NEXT_PUBLIC_API_URL unset for /api/* rewrite
npm install && npm run dev                # http://localhost:3000
```

## Pipeline

Three SSE-streamed phases plus a streaming revision endpoint.

| Phase | Agent | LLM | Tools |
|---|---|---|---|
| `/destinations` | preference, missing_info, destination_suggester | Cerebras | — |
| `/research` | arrival, research × N legs (parallel) | Cerebras | tavily |
| `/build` | route, budget, **itinerary**, tips, critic | Cerebras + LoRA | tavily |
|  | replan loop: re-runs route + itinerary + critic on critic.score < 7 (max 2 retries) | | |
| `/revise` | revision_router, then text / structural / budget path | varies | tavily on budget |

**Budget path** isn't a math recompute — when the user says *"keep each day under $100"* or *"upgrade to luxury"*, the router infers a new tier and runs research × N legs at the new tier (different hotels, restaurants, places) before re-routing and regenerating.

**Critic-driven replan** runs on both initial planning and structural revisions: failed score (< 7) → re-run with the critic's issues passed back as feedback to `route_agent`, up to 2 retries.

**Custom candidates** — the picker has a dashed `+ Add your own X` card in each section. New entries are sent to `/enrich-candidates`, which fires a focused Tavily search per item to fetch a description, then merged into research before `/build` runs.

**Resume** — the latest checkpoint (after destinations / research / final) is persisted to `localStorage` with a UUID and 7-day TTL. On page mount, a modal offers to continue or start fresh.

## API

| Method + Path | Body | Streams |
|---|---|---|
| `GET /health` | — | — |
| `POST /destinations` | `{request}` | SSE |
| `POST /research` | `{preferences, destinations}` | SSE |
| `POST /build` | `{preferences, research, arrival, selections}` | SSE |
| `POST /plan` | `{request}` | SSE (full pipeline, no pickers) |
| `POST /revise` | `{result, change}` | SSE |
| `POST /enrich-candidates` | `{items: [{name, city?, category}]}` | — |
| `POST /candidate-detail` | `{name, city?, category?}` | — |

SSE events: `started · step{name,status,output?,is_retry?,retry_round?} · destinations_complete · research_complete · complete · incomplete · error`.

## The LoRA

### What it does

Given preferences + selected places + per-day route groups, emit JSON in this exact schema:

```json
{
  "trip_summary": "…",
  "daily_itinerary": [{
    "day": 1, "theme": "…", "morning": "…", "afternoon": "…", "evening": "…",
    "estimated_cost": "$X-$Y per person",
    "transportation_note": "…", "feasibility_note": "…"
  }],
  "budget_summary": "…", "backup_options": ["…"], "travel_tips": ["…"]
}
```

Hotels, real budget numbers, transit, multi-leg day numbering, and itinerary-specific tips are produced by orchestrator agents and merged on top — the LoRA's only job is the day-by-day skeleton in the right structure.

### Training

| | |
|---|---|
| Hardware | 2× RTX 5090 (Vast.ai) |
| Method | bf16 LoRA, `r=16`, `alpha=32`, `target = q,k,v,o_proj` |
| Data | 199 chat-format examples (189 train / 10 eval) |
| Schedule | 3 epochs, lr `2e-4` cosine, eff. batch 16, `max_length=2048` |
| Wall clock | ~63 s |
| Loss | train 1.96 → 0.26, eval 1.07 → 0.26 (no overfitting) |

```bash
ulimit -n 65536
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/.hf_home
torchrun --nproc_per_node=2 train.py     # → tripwise-itinerary-lora/
python eval.py                            # → eval_output/eval_results.json
python report.py                          # → report.md
```

### Evaluation (10 held-out, greedy decoding)

| Metric | Fine-tuned | Base (Qwen2.5-7B-Instruct) |
|---|---:|---:|
| `json_valid` | 1.000 | 0.900 |
| `schema_complete` | **1.000** | **0.000** |
| `day_count_correct` | **1.000** | **0.000** |
| `per_day_complete` | 1.000 | 0.900 |
| `no_extra_places_in_themes` | 0.900 | 0.900 |
| `avg_latency_s` | ~11 | ~7 |

Schema and day-count compliance was the goal; base Qwen makes a fine plan in *its own* schema (`{itinerary: {Day 1: {morning: {activity, …}}}}`), the LoRA produces *our* schema reliably so downstream code doesn't need parser hacks.

Full per-example outputs in [`eval_output/eval_results.json`](eval_output/eval_results.json), narrative writeup in [`report.md`](report.md).

### Using the LoRA elsewhere

The adapter is a generic Qwen2.5-7B → JSON itinerary adapter for the trained input shape.

```bash
# Apple Silicon — pre-merged MLX from HF
hf download aquaqua/tripwise-7b-merged-bf16 --local-dir ./tripwise-mlx-bf16

# CUDA — adapter on top of the base
python infer.py        # transformers + peft, see source

# CUDA — vLLM with multi-LoRA
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules tripwise=/path/to/tripwise-itinerary-lora
```

Use `backend/llm.py:ITINERARY_SYSTEM_PROMPT` verbatim, pass the user message as JSON with the trained keys (`destination, trip_length_days, travelers, budget_level, interests, pace, constraints, selected_places, route_groups`), and decode greedily (`temperature=0`).

## Privacy / secrets

- `backend/.env` and `frontend/.env.local` are gitignored.
- LoRA adapter (~40 MB) and training set are committed; the merged HF Hub repo is private.
- SSH keys, training checkpoints, `.next`, `node_modules`, `.venv` are excluded.
