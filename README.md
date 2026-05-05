# PlanningAgent — TripWise

A multi-agent travel planning assistant. Free-form trip request goes in (e.g. *"5 days in Japan from San Francisco, 2 friends, medium budget, anime + food + city views"*); a structured day-by-day itinerary, real candidate places/restaurants/hotels, round-trip flight options, and an all-in budget come out.

**Live deployment**: https://trip.23333.info
**Fine-tuned merged model on Hugging Face**: https://huggingface.co/aquaqua/tripwise-7b-merged-bf16

The current architecture uses **Cerebras `gpt-oss-120b` for orchestration + a fine-tuned, LoRA-merged Qwen2.5-7B (MLX bf16) for itinerary generation**:

| Role | Model | Where it runs | Used by |
|---|---|---|---|
| **Orchestrator** | `gpt-oss-120b` | Cerebras Inference (`https://api.cerebras.ai/v1`) | Preference, Missing-info, Destination-suggester, Arrival, Research, Route, Budget, Critic, Tips, Revision-router, Revision agents (everything except Itinerary). `reasoning_effort=low` to stay within budget. |
| **Itinerary** | `tripwise-mlx-bf16` (Qwen2.5-7B-Instruct + tripwise LoRA, merged to MLX bf16) | [oMLX](https://github.com/aquaqua/oMLX) on the Mac at `http://localhost:5620/v1` | Itinerary agent only |

The orchestrator handles everything that needs world knowledge or tool use (Tavily web search, Python arithmetic). The fine-tuned itinerary model is scoped narrowly: produce schema-conformant JSON for the day-by-day plan. The split lets a small, fast LoRA enforce structure consistency while a much larger reasoning model drives the agentic side — at very low cost.

The merged model is **published on Hugging Face** as [`aquaqua/tripwise-7b-merged-bf16`](https://huggingface.co/aquaqua/tripwise-7b-merged-bf16) (private repo) so any Apple Silicon machine running oMLX (or a CUDA host running vLLM/transformers) can serve it directly without re-merging the adapter.

## What's in this repo

```
.
├── backend/                          FastAPI multi-agent backend
│   ├── server.py                     /destinations, /research, /build, /plan,
│   │                                 /revise (SSE), /candidate-detail, /health
│   ├── orchestrator.py               three-phase async pipeline + run_revise
│   │                                 (text / structural / budget paths)
│   ├── agents.py                     11 agent functions + system prompts
│   │                                 (preference, missing_info, destination_suggester,
│   │                                  arrival, research, route, budget, itinerary,
│   │                                  critic, tips, revision_router, revision)
│   ├── llm.py                        AsyncOpenAI clients (orch + itinerary),
│   │                                 tool-calling loop, 90 s itin retry+timeout
│   ├── tools.py                      tavily_search, tavily_search_detailed, python_exec
│   ├── run.py                        end-to-end CLI runner
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                         Next.js 16 + Tailwind 3 (App Router)
│   ├── app/page.tsx                  state machine: idle → resolving →
│   │                                 choosing_destinations → researching →
│   │                                 picking → planning → done → revising
│   ├── components/
│   │   ├── TripForm.tsx              free-form request box
│   │   ├── AgentProgress.tsx         live SSE pipeline progress with replan badges
│   │   ├── DestinationPicker.tsx     when user gives only a country/region
│   │   ├── CandidatePicker.tsx       blocks with descriptions, modal, round-trip flights
│   │   ├── ItineraryView.tsx         day cards + hotel + budget (no candidates panels)
│   │   ├── RevisionForm.tsx          freeform "make it more relaxed" / chip suggestions
│   │   ├── RevisionProgress.tsx      live revision pipeline panel (path + per-agent
│   │   │                             pills + elapsed timer)
│   │   └── ResumePrompt.tsx          modal to continue a saved unfinished plan
│   ├── lib/api.ts                    SSE clients (streamDestinations / streamResearch /
│   │                                 streamBuild / streamRevise)
│   ├── lib/session.ts                localStorage-backed plan resume (UUID, 7-day TTL)
│   └── …config (Next, Tailwind, TS, PostCSS)
│
├── train.py                          LoRA fine-tuning (DDP-aware, runs on Vast.ai)
├── eval.py                           held-out eval: fine-tuned vs base
├── report.py                         renders training+eval results into report.md
├── infer.py                          standalone inference script (transformers + peft)
├── serve_lora.py                     historical OpenAI-compatible HF wrapper
│                                     (used during the vLLM-on-Vast.ai phase; now
│                                      superseded by the merged MLX deployment)
├── travel_finetune_examples_1_200.jsonl   199 training examples (chat format)
├── tripwise-itinerary-lora/          ⚙️  TRAINED LoRA ADAPTER (~40 MB)
│   ├── adapter_model.safetensors     LoRA weights (r=16, alpha=32)
│   ├── adapter_config.json
│   ├── tokenizer.json + chat_template.jinja
│   ├── eval_examples.jsonl           the 10 held-out eval examples
│   └── metadata.json                 hyperparams + train/eval sizes + elapsed
├── eval_output/eval_results.json     full per-example fine-tuned vs base outputs
├── report.md                         human-readable training + evaluation report
└── MID_PROJECT_SLIDES.md / .pdf      mid-project status slides
```

## Live deployment

> **Access restricted.** The live site is gated by **Cloudflare Access**: a valid `@columbia.edu` email is required to log in. Cloudflare sends a one-time PIN to the address; once verified, the session lasts 24 hours. Visitors without a Columbia email will see Cloudflare's "Access denied" page before any request reaches the Ubuntu host.

Frontend + backend on a DigitalOcean Ubuntu droplet, the LoRA itinerary model on the developer's Mac, bridged by a Cloudflare tunnel:

```
                  https://trip.23333.info  (Let's Encrypt, HTTP→HTTPS redirect)
                            │
                            ▼
      Ubuntu 24.04  (159.223.182.62, 2 vCPU / 4 GB)
      ┌────────────────────────────────────────────────┐
      │ nginx :80/:443                                  │
      │   /        →  Next.js :3000 (systemd unit)     │
      │   /api/    →  FastAPI :8000  (systemd unit)    │
      │              └─ orch:  Cerebras gpt-oss-120b   │
      │              └─ itin:  https://taxagent.23333.info  ← cloudflare tunnel
      └────────────────────────────────────────────────┘
                            │
                            ▼  (cloudflared)
      MacOS Apple Silicon
      ┌────────────────────────────────────────────────┐
      │ oMLX :5620   →  tripwise-mlx-bf16 (LoRA-merged) │
      └────────────────────────────────────────────────┘
```

Components:
- **Public domain**: `trip.23333.info` is fronted by **Cloudflare** (proxied A-record → 159.223.182.62), so traffic flows through Cloudflare's edge before reaching the Ubuntu host. This is what enables the Access-gated login.
- **Cloudflare Access policy**: a self-hosted application bound to `trip.23333.info` with an **email-domain rule** allowing only `*@columbia.edu`. Identity provider is Cloudflare's built-in **One-Time PIN**, so no Google/GitHub login is required — just a Columbia email that can receive a 6-digit code.
- **Nginx** terminates TLS (Let's Encrypt, auto-renewing via `certbot.timer`), proxies `/api/` to FastAPI with SSE-friendly buffering disabled, and `/` to Next.js.
- **Cloudflare tunnel** runs on the Mac, exposing `localhost:5620` (oMLX) at `https://taxagent.23333.info`. The Ubuntu backend's `ITIN_BASE_URL` points at this URL.
- **Two systemd services** (`planningagent-backend`, `planningagent-frontend`) restart automatically on reboot.

## Setup

For a local-only dev setup you only need the four pieces below. Skip the cloudflare tunnel and run oMLX directly at `http://localhost:5620/v1`.

```
   ┌──── frontend (Next.js)   localhost:3000
   ├──── backend  (FastAPI)   localhost:8000
   ├──── orchestrator LLM     api.cerebras.ai     gpt-oss-120b (or any OpenAI-compat)
   └──── itinerary LLM        localhost:5620      tripwise-mlx-bf16 on oMLX (Mac)
                                                  or vLLM/serve_lora.py on a CUDA box
```

Plus a Tavily API key for the web-search tool used by the Research, Budget and Arrival agents.

### 1. Orchestrator LLM — Cerebras `gpt-oss-120b`

Get an API key from https://cloud.cerebras.ai. Cerebras serves `gpt-oss-120b` at OpenAI-compatible `/v1/chat/completions` with tool calling and `reasoning_effort` support. The default `backend/.env` looks like:

```
ORCH_BASE_URL=https://api.cerebras.ai/v1
ORCH_API_KEY=csk-…
ORCH_MODEL=gpt-oss-120b
```

You can swap to OpenAI / OpenRouter / Anthropic / Together / a local vLLM serving any open model — anything OpenAI-compatible with tool calling works. Just change the three vars. `backend/llm.py` auto-detects Qwen vs reasoning-model variants and adjusts `extra_body` accordingly:

- Qwen models get `chat_template_kwargs.enable_thinking = false`.
- Other (gpt-oss-style) reasoning models get `reasoning_effort = "low"` so they don't burn the entire token budget thinking.
- Cerebras-style providers reject `response_format` alongside `tools`, so the tool-calling loop omits it; the system prompt + `_extract_json` handle JSON extraction.

### 2. Itinerary LLM — `tripwise-mlx-bf16` on oMLX (recommended)

The fine-tuned LoRA was merged into Qwen2.5-7B-Instruct and converted to MLX bf16. The merged model lives on Hugging Face:

> [`aquaqua/tripwise-7b-merged-bf16`](https://huggingface.co/aquaqua/tripwise-7b-merged-bf16) (private repo, ~14 GB across 3 safetensors shards)

**On any Apple Silicon Mac:**

```bash
# 1. Install oMLX (https://github.com/aquaqua/oMLX) — runs an MLX server
#    with an OpenAI-compatible /v1 surface at :5620.

# 2. Download the merged model:
HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download aquaqua/tripwise-7b-merged-bf16 \
  --local-dir ~/Models/tripwise-mlx-bf16

# 3. Register it with oMLX — edit ~/.omlx/settings.json:
#    "model": { "model_dirs": [ "/Users/.../Models/tripwise-mlx-bf16" ] }
#    Quit oMLX and relaunch.

# 4. Verify:
curl localhost:5620/v1/models | jq '.data[].id'
# should include "tripwise-mlx-bf16"
```

`backend/.env` for the local oMLX serve:

```
ITIN_BASE_URL=http://localhost:5620/v1
ITIN_API_KEY=omlx-…           # the oMLX sub-key
ITIN_MODEL=tripwise-mlx-bf16
```

For the production deployment described above, replace `localhost:5620` with the cloudflared tunnel hostname (e.g. `https://taxagent.23333.info`).

**Alternative — load the unmerged adapter on a CUDA host (vLLM / `serve_lora.py`):** see [How to use the LoRA in your own project](#how-to-use-the-lora-in-your-own-project) below.

### 3. Tavily

Free key (1000 searches/month) at https://tavily.com — paste it into `backend/.env` as `TAVILY_API_KEY`. Used by Research, Budget, Arrival, and the candidate-detail endpoint.

### 4. Backend (FastAPI)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env       # then edit keys/URLs
uvicorn backend.server:app --port 8000     # run from repo root
```

`backend/.env` knobs:

| Var | Default | Notes |
|---|---|---|
| `ORCH_BASE_URL` | `https://api.cerebras.ai/v1` | OpenAI-compat orchestrator endpoint |
| `ORCH_API_KEY` | `csk-…` | Cerebras key (or other provider) |
| `ORCH_MODEL` | `gpt-oss-120b` | model id at the orchestrator endpoint |
| `ITIN_BASE_URL` | `http://localhost:5620/v1` | oMLX (or vLLM, or cloudflared tunnel) |
| `ITIN_API_KEY` | `omlx-…` | the itinerary endpoint's auth header |
| `ITIN_MODEL` | `tripwise-mlx-bf16` | model id served at ITIN_BASE_URL |
| `TAVILY_API_KEY` | `tvly-…` | Tavily key |

### 5. Frontend (Next.js)

```bash
cd frontend
cp .env.example .env.local                  # set NEXT_PUBLIC_API_URL if needed
npm install
npm run dev                                 # http://localhost:3000
```

For local dev, leave `NEXT_PUBLIC_API_URL` unset — `frontend/lib/api.ts` falls back to the `/api/*` rewrite proxy in `next.config.mjs`. For a deployed nginx-fronted setup, leave it unset too so calls go same-origin through `/api/`.

## Architecture

```
Browser  ─────────►  Next.js (3000)  ──fetch──►  FastAPI (8000)
                                                      │
                                                      ▼
                                          ┌──────────────────────────┐
                                          │ orchestrator state       │
                                          │ machine (3 phases):      │
                                          │   /destinations          │
                                          │   /research              │
                                          │   /build                 │
                                          │ + /revise (SSE) with     │
                                          │   smart routing          │
                                          └──────────────────────────┘
                                                      │
                          ┌────────────────┬──────────┴──────────┬──────────────────┐
                          ▼                ▼                     ▼                  ▼
                Cerebras gpt-oss-120b   Tavily web      Python subprocess     Qwen2.5-7B + LoRA
                /v1 (orchestration)     search          (python_exec tool)    merged → MLX bf16
                reasoning_effort=low                                          on oMLX :5620
                                                                              (itinerary only)
```

### Multi-agent pipeline

| Phase | Agent | Tools | Output |
|---|---|---|---|
| Phase 0 — `/destinations` | `preference_agent` | none | structured prefs (origin, destinations[], budget, pace, interests, …) |
| | `missing_info_agent` | pure Python | required-field checklist |
| | `destination_suggester_agent` (only if user gave only a country/region) | none | candidate cities + recommended day split |
| Phase 1 — `/research` | `arrival_agent` (only if origin given) | tavily | `{outbound_options, return_options}` per class |
| | `research_agent` (per leg, parallel) | tavily | `{places, restaurants, hotels}` with descriptions, tagged by city |
| Phase 2 — `/build` | `route_agent` (per leg) | none | `{route_groups, meal_plan, transit_notes, day_schedule}` — selects subset by pace + priority |
| | `budget_agent` | tavily | `{buckets, airfare, …}`; backend sums into daily/total |
| | `itinerary_agent` (per leg) | **fine-tuned LoRA** | day-by-day JSON in trained schema |
| | `tips_agent` | none | replaces LoRA's templated tips with itinerary-specific ones |
| | `critic_agent` | none | `{score 0-10, passed, issues, suggested_revisions}` (sees `budget` too) |
| | replan loop | rerun route + itinerary + critic | up to 2 retries when score < 7 |
| `/revise` (SSE) | `revision_router` | none | classifies change as `text` / `structural` / `budget` (+ inferred new tier) |
| | path-specific agents | varies | see [Smart streaming /revise](#smart-streaming-revise) |
| `/candidate-detail` | tavily detailed | tavily (with images) | extra info + photos for one candidate |

### Per-step dependency table

| Step | Depends on | LLM | Tools |
|---|---|---|---|
| `preference_agent` | user request | Cerebras gpt-oss-120b | — |
| `missing_info_agent` | preference output | (pure Python) | — |
| `destination_suggester_agent` | preference output (when destinations empty) | Cerebras | — |
| `arrival_agent` | preference (origin), legs (first/last city) | Cerebras | tavily |
| `research_agent` | preference, one leg | Cerebras | tavily |
| `route_agent` | research, hotel pick, pace, budget, feedback? | Cerebras | — |
| `budget_agent` | route output, arrival (computed_airfare) | Cerebras | tavily |
| `itinerary_agent` | route_groups + selected places + city_prefs | **tripwise-mlx-bf16** (LoRA-merged Qwen2.5-7B on oMLX) | — |
| `tips_agent` | finished itinerary, prefs | Cerebras | — |
| `critic_agent` | itinerary, prefs, **budget** | Cerebras | — |
| replan loop | critic feedback | re-runs route + itinerary + critic | — |
| `revision_router` | itinerary summary, change request | Cerebras | — |
| `revision_agent` | existing itinerary + change request | Cerebras | — |
| `tavily_search_detailed` | candidate name + city | — | tavily (advanced + images) |

### Multi-destination support

Trip can span multiple cities:

- `"3 days in Tokyo"` → single leg.
- `"5 days in Japan: 3 Tokyo, 2 Kyoto"` → two legs in the order given.
- `"5 days in Japan"` → preference agent leaves `destinations: []`, sets `country_or_region: "Japan"`; the destination suggester proposes candidates (4-7 cities) and a default split; the user picks order + days in the picker; the **first** picked city becomes the outbound flight destination, the **last** picked city becomes the return flight origin.

Research + route + itinerary run **once per leg**, then concatenate with day-offset numbering. Each day card shows `Day N · City`.

### Smart streaming `/revise`

The post-completion revision endpoint is now SSE-streaming and **smart-routed**. A `revision_router` agent classifies the user's change request and the orchestrator runs only the agents that actually need to re-execute. Each path emits agent step events in the same SSE format the planning pipeline uses; the frontend's `RevisionProgress` panel shows the path, reason, and a per-agent pill with elapsed timer.

| Category | Triggers | Agents that run | Typical latency |
|---|---|---|---|
| `text` | rewording, swap a single attraction in the same area, evening tweak | `revision_router → revision` (Cerebras edit) | ~1 s |
| `structural` | re-shape day plan, add/remove attractions, pace change, time windows | `revision_router → route → itinerary (LoRA) → tips → critic` (with replan loop, up to 2 retries) | 25-60 s |
| `budget` | tier shift ("under $100/day", "upgrade to luxury", "private jet") | `revision_router → research × N legs → route → itinerary (LoRA) → tips → budget → critic` | 50-90 s |

The **budget** path is the heaviest because a tier shift means the actual hotels/restaurants/places change — recomputing totals over the existing candidates would still price the same Park Hyatt. So the router infers the new `budget_level` (low/medium/high/luxury) from the change, re-researches every leg at that tier, and only then re-routes + regenerates the itinerary.

The **structural** path runs the same critic-driven replan loop as initial planning: if `critic.score < 7`, the orchestrator feeds the critic's issues back into `route_agent` as feedback and re-runs route + itinerary + critic up to 2 more times.

### Resume in-progress trip planning

The frontend persists the latest successful checkpoint to `localStorage` (`tripwise.session`) with a UUID and a 7-day TTL. Three checkpoints:
- after `destinations_complete`
- after `research_complete`
- after `complete` (and after each revision)

On page mount, if a saved session exists, a modal pops up showing the original request and last phase ("Last activity: 12 min ago · picking places"). "Continue" rehydrates state and jumps to that phase; "Start fresh" clears the slot. Resetting via the header logo or starting a new plan also clears it.

Robustness on the LoRA call: `itin_complete` has a **90 s per-attempt timeout with 2 retries** on timeout/connection errors so a single Cloudflare-tunnel idle drop self-heals instead of leaving the UI hung forever.

## API surface

| Method + Path | Body | Streams | Purpose |
|---|---|---|---|
| `GET /health` | — | no | liveness |
| `POST /destinations` | `{request}` | SSE | preference + missing_info + (suggester if country/region) |
| `POST /research` | `{preferences, destinations}` | SSE | arrival + per-leg research |
| `POST /build` | `{preferences, research, arrival, selections}` | SSE | route + budget + itinerary + tips + critic (+ replan loop) |
| `POST /plan` | `{request}` | SSE | one-shot full pipeline (no pickers) |
| `POST /revise` | `{result, change}` | **SSE** | smart-routed revision (text / structural / budget) |
| `POST /candidate-detail` | `{name, city, category}` | no | deep Tavily lookup w/ images for one candidate |

SSE events:

```
{event:"started", payload:{...}}
{event:"step", payload:{name, status:"running"|"done", output?, is_retry?, retry_round?}}
{event:"destinations_complete", payload:{preferences, destinations, needs_resolution, suggester}}
{event:"research_complete", payload:{preferences, destinations, arrival, research}}
{event:"complete", payload:PlanResult | ReviseCompletePayload}
{event:"incomplete", payload:{missing_fields, preferences}}
{event:"error", payload:{type, message, trace}}
```

## The LoRA

### What it does

Given structured trip preferences + selected places + per-day route groups, produce a JSON itinerary in this exact schema:

```json
{
  "trip_summary": "…",
  "daily_itinerary": [
    {
      "day": 1,
      "theme": "…",
      "morning": "…",
      "afternoon": "…",
      "evening": "…",
      "estimated_cost": "$X-$Y per person",
      "transportation_note": "…",
      "feasibility_note": "…"
    }
  ],
  "budget_summary": "…",
  "backup_options": [ "…" ],
  "travel_tips": [ "…" ]
}
```

The fine-tune was scoped narrowly: **JSON shape consistency**. Hotels, real budget, lunch/dinner names, transit chains, multi-destination day numbering, itinerary-specific tips, and budget tiering are all produced by orchestrator agents and stitched on top in the UI — the LoRA's only job is the day-by-day narrative skeleton in the right structure.

### Training environment

Provisioned on Vast.ai for training only; the trained adapter is portable and the merged model is on HF Hub.

| | |
|---|---|
| GPUs | 2× NVIDIA RTX 5090, 32 GB GDDR7 each (sm_120 Blackwell) |
| GPU driver | 580.x (any CUDA 12.8+ driver works) |
| CUDA runtime | 12.8 (`torch==2.10.0+cu128`) |
| Python | 3.12 (Vast.ai PyTorch image, `/venv/main`) |
| Libraries | `transformers==4.57.6`, `peft==0.19.1`, `trl==1.2.0`, `datasets==4.8.4`, `accelerate==1.13.0` |
| HF cache | `HF_HOME=/workspace/.hf_home` (Vast.ai's writable workspace volume; the overlay `/` is only 32 GB so the 14 GB base model goes to workspace) |
| `ulimit -n` | bumped to `65536` before launch (Tavily/HF tokenizer use Rayon and exhaust the default 1024 fd limit under DDP) |
| Distributed launch | `torchrun --nproc_per_node=2` (DDP, no DeepSpeed) |
| Wall-clock end-to-end | **~63 seconds** of training + ~30 s setup |

We chose **plain bf16 LoRA** instead of QLoRA: the 7B base in bf16 is ~14 GB, which fits comfortably on a single 32 GB 5090 alongside the LoRA adapter, gradients, and activations. QLoRA's 4-bit quantization adds noise without saving meaningful memory at this scale.

### Training data

`travel_finetune_examples_1_200.jsonl` — 199 chat-format examples mapping structured trip constraints + selected places + route groups to the JSON itinerary above.

Each line is one chat-completion training example with three messages: `system` (the locked itinerary-agent prompt), `user` (a JSON dict of preferences + selected_places + route_groups), and `assistant` (the target itinerary JSON).

The assistant outputs in this dataset are **templated** (recurring boilerplate phrasing across days). The LoRA learns the schema cleanly but inherits the prose template; this is the dataset bottleneck, not a training one. The orchestrator's `tips_agent` was added partly to compensate — it replaces the LoRA's templated tips with itinerary-specific ones (referencing actual days, places, and constraints).

### Training process

`train.py` is the single end-to-end training script. Per epoch it:

1. **Loads the base model** `Qwen/Qwen2.5-7B-Instruct` in bf16 onto each rank's GPU (`device_map={"": LOCAL_RANK}` under DDP, `"auto"` standalone).
2. **Loads the dataset**, shuffles with `seed=42`, and splits 189 train / 10 eval. The 10 held-out examples are also dumped to `tripwise-itinerary-lora/eval_examples.jsonl` so `eval.py` uses an identical split.
3. **Wraps the model with PEFT LoRA** (`r=16`, `alpha=32`, `dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`, `bias="none"`).
4. **Runs SFT via `trl.SFTTrainer`** with `SFTConfig`:
   - `num_train_epochs=3`
   - `per_device_train_batch_size=2`, `gradient_accumulation_steps=4` → effective batch **16** (= 2×4×2 GPUs)
   - `gradient_checkpointing=True` (use_reentrant=False) — fits with bf16 + LoRA in <30 GB/GPU
   - `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`
   - `max_length=2048`, `packing=False`
   - `logging_steps=2`, `eval_strategy="steps"`, `eval_steps=10`, `save_steps=20`
5. **Saves the LoRA** (only the adapter weights — ~40 MB safetensors, not the full model) and tokenizer to `./tripwise-itinerary-lora/` along with `trainer_state.json` (loss curves) and a custom `metadata.json` (hyperparams + elapsed time).
6. **Total wall-clock: ~63 s** for 3 epochs over 189 examples.

#### Loss trajectory

| step | train_loss | eval_loss |
|---:|---:|---:|
| 2 | 1.960 | — |
| 10 | 1.237 | **1.068** |
| 20 | 0.555 | **0.471** |
| 30 | 0.267 | **0.275** |
| 36 | 0.263 | **0.260** |

Token-accuracy on train climbs from 66% → 94% over 36 steps. No overfitting (eval loss tracks train loss closely). Full curve in `report.md`.

### Evaluation

`eval.py` runs the LoRA against the same 10 held-out examples and compares to the unmodified base model.

#### Methodology

1. **Reload** `Qwen/Qwen2.5-7B-Instruct` in bf16, then attach the trained LoRA via `PeftModel.from_pretrained`.
2. For each held-out example:
   - Apply the chat template (`apply_chat_template`) with the same locked system prompt + the example's user JSON.
   - Generate with **greedy decoding** (`do_sample=False`, `max_new_tokens=1024`) — schema metrics need to be deterministic.
   - Extract the first balanced `{...}` block from the output (handles models that prepend or append text around the JSON).
3. Score the parsed object on five hard metrics + latency.
4. **Tear down** the model, free GPU memory, **reload the base alone** (no adapter), and repeat the loop. Same prompts, same decoding, same scoring — so the comparison is apples-to-apples.
5. Save `eval_output/eval_results.json` with per-example outputs (gold, generated, all flags) and aggregate metrics for both runs.

`report.py` then reads `tripwise-itinerary-lora/trainer_state.json` and `eval_output/eval_results.json` to render `report.md`.

#### Metric definitions

| Metric | Definition |
|---|---|
| `json_valid` | the extracted block parses as JSON (no syntax errors) |
| `schema_complete` | all of `{trip_summary, daily_itinerary, budget_summary, backup_options, travel_tips}` present at the top level |
| `day_count_correct` | `len(daily_itinerary) == trip_length_days` |
| `per_day_complete` | every day dict has `{day, theme, morning, afternoon, evening, estimated_cost, transportation_note, feasibility_note}` |
| `no_extra_places_in_themes` | day themes only mention places from the input `selected_places` (heuristic — capitalized-word match) |
| `avg_latency_s` | mean greedy generation time per example |

#### Results

10 held-out examples, greedy decoding, seed=42:

| Metric | Fine-tuned | Base (Qwen2.5-7B-Instruct) |
|---|---:|---:|
| `json_valid` | 1.000 | 0.900 |
| `schema_complete` | **1.000** | **0.000** |
| `day_count_correct` | **1.000** | **0.000** |
| `per_day_complete` | 1.000 | 0.900 |
| `no_extra_places_in_themes` | 0.900 | 0.900 |
| `avg_latency_s` | ~11 s | ~7 s |

The fine-tune is doing the one thing it was scoped to: **schema and day-count compliance**. Base Qwen produces a perfectly fine-sounding travel plan in *its own* schema (e.g. `{itinerary: {Day 1: {morning: {activity, description}, …}}}`); the LoRA produces *our* schema reliably so downstream code can render it without parser hacks.

Full per-example outputs (gold + base + fine-tuned generations) in [`eval_output/eval_results.json`](eval_output/eval_results.json), full human-readable writeup in [`report.md`](report.md).

### How to retrain

On a host with the environment described above:

```bash
cd /path/to/repo
pip install -r requirements.txt
ulimit -n 65536
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/.hf_home

# train (~63 s on 2× RTX 5090):
torchrun --nproc_per_node=2 train.py
# adapter, tokenizer, eval split, metadata, trainer_state.json all land in
# ./tripwise-itinerary-lora/

# evaluate fine-tuned vs base on 10 held-out examples:
python eval.py
# writes eval_output/eval_results.json

# render the markdown report:
python report.py
# writes report.md
```

If single-GPU (no DDP), drop `torchrun` — the script auto-detects `WORLD_SIZE` and falls back to standalone with `device_map="auto"`.

### How to use the LoRA in your own project

You don't have to use TripWise — the adapter works as a generic "Qwen2.5-7B → JSON itinerary" adapter for the same input shape. Four serving paths:

1. **Pre-merged MLX bf16 (Apple Silicon, recommended for Mac):** download the published merged model and load with mlx_lm:
   ```bash
   hf download aquaqua/tripwise-7b-merged-bf16 --local-dir ./tripwise-mlx-bf16
   ```
   Then serve with oMLX, mlx_lm.server, or any OpenAI-compatible MLX server pointing at `./tripwise-mlx-bf16`.

2. **Load adapter at runtime with `peft`** (most flexible; what `infer.py` does):
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", device_map="auto"
   )
   model = PeftModel.from_pretrained(model, "./tripwise-itinerary-lora")
   ```

3. **Merge adapter into the base** (single fused model; works in any HF/vLLM/Ollama pipeline afterwards):
   ```python
   model = model.merge_and_unload()
   model.save_pretrained("./tripwise-merged", safe_serialization=True)
   ```
   To convert to MLX bf16 afterwards on Apple Silicon:
   ```bash
   pip install 'mlx-lm>=0.20'
   python -m mlx_lm convert --hf-path ./tripwise-merged \
       --mlx-path ./tripwise-mlx-bf16 --dtype bfloat16
   ```

4. **vLLM with multi-LoRA** (lowest latency, batched serving on CUDA):
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
       --lora-modules tripwise=/path/to/tripwise-itinerary-lora
   # then call /v1/chat/completions with model="tripwise"
   ```

Critical: use the exact training-time system prompt (it's in `backend/llm.py:ITINERARY_SYSTEM_PROMPT`) and pass the user message as JSON with the keys the LoRA was trained on (destination, trip_length_days, travelers, budget_level, interests, pace, constraints, selected_places, route_groups). Use greedy decoding (`temperature=0`) for deterministic schema compliance.

## Privacy / secrets

- `backend/.env` and `frontend/.env.local` are gitignored. Never commit API keys.
- The fine-tuned LoRA adapter is committed (~40 MB) since it has no PII.
- The `travel_finetune_examples_1_200.jsonl` dataset is committed.
- The merged HF Hub repo (`aquaqua/tripwise-7b-merged-bf16`) is **private** — set your HF token when cloning.
- SSH keys, `.env` files, `node_modules`, `.next`, training checkpoints, and `.venv` are all excluded.
