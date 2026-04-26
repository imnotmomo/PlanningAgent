# PlanningAgent — TripWise

A multi-agent travel planning assistant. Free-form trip request goes in (e.g. "5 days in Japan from San Francisco, 2 friends, medium budget, anime + food + city views"); a structured day-by-day itinerary, real candidate places/restaurants/hotels, round-trip flight options, and an all-in budget come out.

The current setup uses **two Qwen models** running side by side:

| Role | Model | Where it runs | Used by |
|---|---|---|---|
| **Orchestrator** | `Qwen3.5-9B-MLX-4bit` | Local MLX server on the Mac at `http://localhost:5620/v1` | Preference, Missing-info, Destination-suggester, Arrival, Research, Route, Budget, Critic, Revision agents (everything except Itinerary) |
| **Itinerary** | `Qwen/Qwen2.5-7B-Instruct` + `tripwise` LoRA (fine-tuned in this repo) | vLLM / `serve_lora.py` on a CUDA host, OpenAI-compatible at `http://localhost:8001/v1` (SSH-tunneled if remote) | Itinerary agent only |

Orchestration handles everything that needs world knowledge or tool use (Tavily web search, Python arithmetic). The fine-tuned itinerary model is scoped narrowly to one job: produce schema-conformant JSON for the day-by-day plan. The split lets us train a small, fast LoRA for structure consistency while still getting high-quality reasoning on the agentic side from a general MLX model — without paying any API.

## What's in this repo

```
.
├── backend/                          FastAPI multi-agent backend
│   ├── server.py                     /destinations, /research, /build, /revise, /candidate-detail, /health
│   ├── orchestrator.py               three-phase async pipeline (destinations → research → build)
│   ├── agents.py                     8 agent functions + system prompts
│   ├── llm.py                        AsyncOpenAI clients (orch + itinerary), tool-calling loop
│   ├── tools.py                      tavily_search, tavily_search_detailed, python_exec
│   ├── run.py                        end-to-end CLI runner
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                         Next.js 16 + Tailwind 3 (App Router)
│   ├── app/page.tsx                  state machine: idle → resolving → choosing_destinations → researching → picking → planning → done
│   ├── components/
│   │   ├── TripForm.tsx              free-form request box
│   │   ├── AgentProgress.tsx         live SSE pipeline progress with replan badges
│   │   ├── DestinationPicker.tsx     when user gives only a country/region
│   │   ├── CandidatePicker.tsx       blocks with descriptions, modal, round-trip flight chooser
│   │   ├── ItineraryView.tsx         day cards + hotel + budget + critic score
│   │   └── RevisionForm.tsx          freeform "make it more relaxed" / chip suggestions
│   ├── lib/api.ts                    SSE clients (streamDestinations / streamResearch / streamBuild)
│   ├── DESIGN.md                     design language reference
│   └── …config (Next, Tailwind, TS, PostCSS)
│
├── train.py                          LoRA fine-tuning script (DDP-aware, runs on Vast.ai)
├── eval.py                           held-out eval: fine-tuned vs base, JSON validity / schema / day-count / per-day completeness
├── report.py                         renders training+eval results into report.md
├── infer.py                          standalone inference script (transformers + peft)
├── serve_lora.py                     OpenAI-compatible HTTP wrapper around transformers+peft (fallback if vLLM doesn't cooperate with the host driver)
├── travel_finetune_examples_1_200.jsonl   199 training examples (chat format)
├── tripwise-itinerary-lora/          ⚙️ TRAINED LoRA ADAPTER (40 MB)
│   ├── adapter_model.safetensors     LoRA weights
│   ├── adapter_config.json           r=16, alpha=32, target_modules=[q,k,v,o]_proj
│   ├── tokenizer.json + chat_template.jinja
│   ├── eval_examples.jsonl           the 10 held-out examples used in eval
│   └── metadata.json                 hyperparams + train_size/eval_size + elapsed_seconds
├── eval_output/eval_results.json     full per-example fine-tuned vs base outputs and metrics
├── report.md                         human-readable training + evaluation report
├── tripwise_project_plan.md          original multi-agent design doc
└── tripwise_fine_tuning_plan.md      original fine-tuning plan + dataset spec
```

## Setup

The tested setup is:

```
   ┌──── frontend (Next.js)   localhost:3000
   ├──── backend  (FastAPI)   localhost:8000
   ├──── orchestrator LLM     localhost:5620      Qwen3.5-9B-MLX-4bit (MLX server, local Mac)
   └──── itinerary LLM        localhost:8001      Qwen2.5-7B-Instruct + tripwise LoRA (vLLM/serve_lora.py)
```

Plus a Tavily API key for the web-search tool used by the Research, Budget and Arrival agents.

### 1. Orchestrator LLM — `Qwen3.5-9B-MLX-4bit` on MLX

Apple Silicon Mac, local. Use any MLX-LM-compatible server that exposes
OpenAI's `/v1/chat/completions` and supports tool calling. The default
`backend/.env` assumes the server is at `http://localhost:5620/v1` and the
model id is `Qwen3.5-9B-MLX-4bit`.

The orchestrator handles **all non-itinerary agents**: preference extraction,
arrival lookup, research (Tavily-grounded), route + meal + transit planning,
budget bucket math, critic scoring, replan, and revision. It also drives the
two tools (`tavily_search`, `python_exec`).

You can swap to OpenAI / OpenRouter / Anthropic / Together / a local vLLM
serving an open model — anything OpenAI-compatible with tool calling works.
Just change `ORCH_BASE_URL`, `ORCH_API_KEY`, `ORCH_MODEL` in `backend/.env`.

### 2. Itinerary LLM — `Qwen2.5-7B-Instruct` + `tripwise` LoRA

The adapter in `tripwise-itinerary-lora/` is a bf16 LoRA on
`Qwen/Qwen2.5-7B-Instruct`. You serve the base + adapter behind an
OpenAI-compatible endpoint at `http://localhost:8001/v1` (port is in `.env`).

**Option A — vLLM (fastest, ~50 tok/s):**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --host 127.0.0.1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules tripwise=/path/to/tripwise-itinerary-lora
```

**Option B — `serve_lora.py` (any CUDA driver, ~5–10 tok/s):**
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/some/dir \
  python serve_lora.py --port 8001
```
A small FastAPI wrapper around `transformers + peft` that exposes the same
`/v1/chat/completions` shape vLLM does. Use this when vLLM's prebuilt CUDA
kernels don't match the host's driver (we hit this on a Vast.ai box with
driver 570 vs vLLM 0.19's CUDA-12.9 kernels — `serve_lora.py` was the
painless fallback).

If the GPU is remote, tunnel the port from your Mac:
```bash
ssh -L 8001:127.0.0.1:8001 user@gpu-host
```

### 3. Tavily

Free key (1000 searches/month) at https://tavily.com — paste it into
`backend/.env` as `TAVILY_API_KEY`. Used by Research, Budget, Arrival, and
the candidate-detail endpoint.

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
| `ORCH_BASE_URL` | `http://localhost:5620/v1` | the local MLX endpoint |
| `ORCH_API_KEY` | (server-token) | required header for the MLX server |
| `ORCH_MODEL` | `Qwen3.5-9B-MLX-4bit` | model id the MLX server exposes |
| `ITIN_BASE_URL` | `http://localhost:8001/v1` | vLLM / serve_lora.py endpoint |
| `ITIN_API_KEY` | `dummy` | not validated (local) |
| `ITIN_MODEL` | `tripwise` | LoRA module name (vLLM `--lora-modules tripwise=…`) |
| `TAVILY_API_KEY` | `tvly-…` | Tavily key |

### 5. Frontend (Next.js)

```bash
cd frontend
cp .env.example .env.local                  # set NEXT_PUBLIC_API_URL if needed
npm install
npm run dev                                 # http://localhost:3000
```

`frontend/lib/api.ts` uses `NEXT_PUBLIC_API_URL` for direct CORS calls when
set; otherwise it falls back to the `/api/*` rewrite proxy in
`next.config.mjs`. Backend CORS allows `localhost:3000` by default.

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
                                          └──────────────────────────┘
                                                      │
                          ┌────────────────┬──────────┴──────────┬──────────────────┐
                          ▼                ▼                     ▼                  ▼
                Qwen3.5-9B-MLX-4bit   Tavily web      Python subprocess     Qwen2.5-7B-Instruct
                MLX server :5620      search          (python_exec tool)    + tripwise LoRA
                (orchestration)                                              vLLM/serve_lora.py :8001
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
| Phase 2 — `/build` | `route_agent` (per leg) | none | `{route_groups, meal_plan, transit_notes}` — selects subset by pace + priority |
| | `budget_agent` | tavily | `{buckets, airfare, …}`; backend sums into daily/total |
| | `itinerary_agent` (per leg) | **fine-tuned LoRA** | day-by-day JSON in trained schema |
| | `critic_agent` | none | `{score 0-10, passed, issues, suggested_revisions}` |
| | replan loop | rerun route + itinerary + critic | up to 2 retries when score < 7 |
| `/revise` | `revision_agent` | none | applies a user "make it more relaxed"-style edit |
| `/candidate-detail` | tavily detailed | tavily (with images) | extra info + photos for one candidate |

### Dependency graph

Solid arrows are required inputs; dotted arrows are conditional (the step
only runs when its precondition is met). Each agent box shows its target
LLM and the tools it can call.

```
                                user_request
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ preference_agent             │  → Qwen3.5-9B-MLX-4bit  ·  no tools
                       └──────────────┬───────────────┘
                                      │ prefs (destinations[], country_or_region,
                                      │        origin, trip_length_days,
                                      │        budget_level, pace, interests)
                                      ▼
                       ┌──────────────────────────────┐
                       │ missing_info_agent           │  pure Python
                       └──────────────┬───────────────┘
                          missing? ───┴── yes ──► incomplete event, STOP
                                      │ no
                                      ▼
              ┌──────────────────────────────────────────────────┐
              │ destination_suggester_agent (only if              │ → Qwen3.5-9B-MLX-4bit
              │   destinations==[] and country_or_region set)     │   no tools
              └──────────────┬───────────────────────────────────┘
                  candidates  │  default_split
                              ▼
                       [picker UI: user picks 1+ cities OR uses default_split]
                              │ legs = [{city, country, days}, …]
                              │
                  ┌───────────┴───────────────────────────┐
                  │                                       │
                  ▼                                       ▼
         ┌──────────────────┐               ┌─────────────────────────┐
         │ arrival_agent    │ → Qwen3.5-9B-MLX-4bit         │ research_agent (per leg │ → Qwen3.5-9B-MLX-4bit
         │ only if          │   tavily      │ in parallel via         │   tavily
         │ prefs.origin     │               │ asyncio.gather)         │
         └────────┬─────────┘               └────────────┬────────────┘
            outbound_options                  places, restaurants, hotels
            return_options                    (with descriptions, tagged by city)
            (origin → first_city,                          │
             last_city → origin)                           │
                  │                                        │
                  └────────────────┬───────────────────────┘
                                   ▼
                       [picker UI: candidate selection +
                        flight choice (or "decide later")]
                                   │
                                   │ selections.{places,restaurants,hotels,
                                   │             arrival_choices.{outbound,return}}
                                   ▼
                  ┌────────────────────────────────────────┐
                  │ route_agent (per leg)                  │ → Qwen3.5-9B-MLX-4bit  ·  no tools
                  │ inputs:                                │
                  │   - attractions {priority, optional}   │
                  │   - restaurants {priority, optional}   │
                  │   - hotel_name (from selections)       │
                  │   - pace, interests, budget_level      │
                  │   - feedback (only on replan)          │
                  └────────────┬───────────────────────────┘
                       route_groups (attraction stops/day)
                       meal_plan {Day N: {lunch, dinner}}
                       transit_notes {Day N: "hotel→stops→hotel"}
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │ budget_agent                           │ → Qwen3.5-9B-MLX-4bit  ·  tavily
                  │ inputs:                                │
                  │   - prefs, selected_places             │
                  │   - arrival (with computed_airfare      │
                  │     from user's flight picks, if any)  │
                  │ output buckets {hotel,transit,         │
                  │   meals,attractions}, airfare {low,high}│
                  │ → backend sums into                    │
                  │   daily_estimate, airfare_estimate,    │
                  │   total_estimate                       │
                  └────────────┬───────────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │ itinerary_agent (per leg)              │ → vLLM
                  │ inputs:                                │   Qwen2.5-7B-Instruct
                  │   - city_prefs (per-leg)               │   + tripwise LoRA
                  │   - selected (places + meal_names)     │   no tools
                  │   - route_groups (this leg)            │
                  │ output: {trip_summary, daily_itinerary,│
                  │   budget_summary, backup_options,      │
                  │   travel_tips}                         │
                  └────────────┬───────────────────────────┘
                       per-leg results truncated to leg.days
                       day numbers re-offset across legs
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │ critic_agent                           │ → Qwen3.5-9B-MLX-4bit  ·  no tools
                  │ inputs: full itinerary, prefs          │
                  │ output: {score 0-10, passed,           │
                  │   issues, suggested_revisions}         │
                  └────────────┬───────────────────────────┘
                       │
                       ├── score >= 7 ──► complete event
                       │
                       └── score < 7 (and retries < 2) ──► replan loop:
                                rerun route_agent (with critic feedback)
                                rerun itinerary_agent
                                rerun critic_agent
                                back to score-check

           ┌──────────────────────────────────────────────────────┐
           │ /revise (separate endpoint, post-completion)         │ → Qwen3.5-9B-MLX-4bit
           │ revision_agent: takes itinerary + change_request     │   no tools
           │ Returns revised itinerary in same schema.            │
           └──────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────────────────────┐
           │ /candidate-detail (separate endpoint, picker modal)  │ → tavily
           │ tavily_search_detailed: name + city → images +       │   (advanced
           │ summary + sources                                    │    + images)
           └──────────────────────────────────────────────────────┘
```

### Per-step dependency table

| Step | Depends on | LLM | Tools |
|---|---|---|---|
| `preference_agent` | user request | Qwen3.5-9B-MLX-4bit | — |
| `missing_info_agent` | preference output | (pure Python) | — |
| `destination_suggester_agent` | preference output (when destinations empty) | Qwen3.5-9B-MLX-4bit | — |
| `arrival_agent` | preference (origin), legs (first/last city) | Qwen3.5-9B-MLX-4bit | tavily |
| `research_agent` | preference, one leg | Qwen3.5-9B-MLX-4bit | tavily |
| `route_agent` | research, hotel pick, pace, budget, feedback? | Qwen3.5-9B-MLX-4bit | — |
| `budget_agent` | route output, arrival (computed_airfare) | Qwen3.5-9B-MLX-4bit | tavily |
| `itinerary_agent` | route_groups + selected places + city_prefs | Qwen2.5-7B + tripwise LoRA | — |
| `critic_agent` | itinerary, prefs | Qwen3.5-9B-MLX-4bit | — |
| replan loop | critic feedback | re-runs route + itinerary + critic | — |
| `revision_agent` | existing itinerary + change request | Qwen3.5-9B-MLX-4bit | — |
| `tavily_search_detailed` | candidate name + city | — | tavily (advanced + images) |

### Multi-destination support

Trip can span multiple cities. Examples:

- `"3 days in Tokyo"` → single leg.
- `"5 days in Japan: 3 Tokyo, 2 Kyoto"` → two legs in the order given.
- `"5 days in Japan"` → preference agent leaves `destinations: []`, sets `country_or_region: "Japan"`; the destination suggester proposes candidates (4-7 cities) and a default split; the user picks order + days in the picker; the **first** picked city becomes the outbound flight destination, the **last** picked city becomes the return flight origin.

The orchestrator runs research + route + itinerary **once per leg**, then concatenates with day-offset numbering. Each day card shows `Day N · City`.

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

The fine-tune was scoped narrowly: **JSON shape consistency**. Hotels, real budget, lunch/dinner names, transit chains, multi-destination day numbering, and budget tiering are all produced by orchestrator agents and stitched on top in the UI — the LoRA's only job is the day-by-day narrative skeleton in the right structure.

### Training environment

Provisioned on Vast.ai for training only; the trained adapter is portable.

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

We chose **plain bf16 LoRA** instead of QLoRA: the 7B base in bf16 is ~14 GB,
which fits comfortably on a single 32 GB 5090 alongside the LoRA adapter,
gradients, and activations. QLoRA's 4-bit quantization adds noise without
saving meaningful memory at this scale.

### Training data

`travel_finetune_examples_1_200.jsonl` — 199 chat-format examples mapping
structured trip constraints + selected places + route groups to the JSON
itinerary above.

Each line is one chat-completion training example with three messages:
`system` (the locked itinerary-agent prompt), `user` (a JSON dict of
preferences + selected_places + route_groups), and `assistant` (the
target itinerary JSON).

The assistant outputs in this dataset are **templated** (recurring
boilerplate phrasing across days). The LoRA learns the schema cleanly but
inherits the prose template; this is the dataset bottleneck, not a training
one. Re-training on the same data would reproduce the same templated prose.

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

Token-accuracy on train climbs from 66% → 94% over 36 steps. No overfitting
(eval loss tracks train loss closely). Full curve in `report.md`.

### Evaluation

`eval.py` runs the LoRA against the same 10 held-out examples and compares
to the unmodified base model.

#### Methodology

1. **Reload** `Qwen/Qwen2.5-7B-Instruct` in bf16, then attach the trained
   LoRA via `PeftModel.from_pretrained`.
2. For each held-out example:
   - Apply the chat template (`apply_chat_template`) with the same locked
     system prompt + the example's user JSON.
   - Generate with **greedy decoding** (`do_sample=False`,
     `max_new_tokens=1024`) — schema metrics need to be deterministic.
   - Extract the first balanced `{...}` block from the output (handles
     models that prepend or append text around the JSON).
3. Score the parsed object on five hard metrics + latency.
4. **Tear down** the model, free GPU memory, **reload the base alone** (no
   adapter), and repeat the loop. Same prompts, same decoding, same scoring
   — so the comparison is apples-to-apples.
5. Save `eval_output/eval_results.json` with per-example outputs (gold,
   generated, all flags) and aggregate metrics for both runs.

`report.py` then reads `tripwise-itinerary-lora/trainer_state.json` and
`eval_output/eval_results.json` to render `report.md`.

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

The fine-tune is doing the one thing it was scoped to: **schema and
day-count compliance**. Base Qwen produces a perfectly fine-sounding travel
plan in *its own* schema (e.g. `{itinerary: {Day 1: {morning: {activity,
description}, …}}}`); the LoRA produces *our* schema reliably so downstream
code can render it without parser hacks.

Full per-example outputs (gold + base + fine-tuned generations) in
[`eval_output/eval_results.json`](eval_output/eval_results.json), full
human-readable writeup in [`report.md`](report.md).

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

If single-GPU (no DDP), drop `torchrun` — the script auto-detects
`WORLD_SIZE` and falls back to standalone with `device_map="auto"`.

### How to use the LoRA in your own project

You don't have to use TripWise — the adapter works as a generic "Qwen2.5-7B → JSON itinerary" adapter for the same input shape. Three serving paths:

1. **Load adapter at runtime with `peft`** (most flexible; what `infer.py` does):
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", device_map="auto")
   model = PeftModel.from_pretrained(model, "./tripwise-itinerary-lora")
   ```

2. **Merge adapter into the base** (single fused model; works in any HF/vLLM/Ollama pipeline afterwards):
   ```python
   model = model.merge_and_unload()
   model.save_pretrained("./tripwise-merged", safe_serialization=True)
   ```

3. **vLLM with multi-LoRA** (lowest latency, batched serving):
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
       --lora-modules tripwise=/path/to/tripwise-itinerary-lora
   # then call /v1/chat/completions with model="tripwise"
   ```

Critical: use the exact training-time system prompt (it's in `backend/llm.py:ITINERARY_SYSTEM_PROMPT`) and pass the user message as JSON with the keys the LoRA was trained on (destination, trip_length_days, travelers, budget_level, interests, pace, constraints, selected_places, route_groups). Use greedy decoding (`temperature=0`) for deterministic schema compliance.

## API surface

| Method + Path | Body | Streams | Purpose |
|---|---|---|---|
| `GET /health` | — | no | liveness |
| `POST /destinations` | `{request}` | SSE | preference + missing_info + (suggester if country/region) |
| `POST /research` | `{preferences, destinations}` | SSE | arrival + per-leg research |
| `POST /build` | `{preferences, research, arrival, selections}` | SSE | route + budget + itinerary + critic (+ replan loop) |
| `POST /plan` | `{request}` | SSE | one-shot full pipeline (no pickers) |
| `POST /revise` | `{itinerary, change}` | no | apply a user-requested change |
| `POST /candidate-detail` | `{name, city, category}` | no | deep Tavily lookup w/ images for one candidate |

SSE events:

```
{event:"started", payload:{...}}
{event:"step", payload:{name, status:"running"|"done", output?, is_retry?, retry_round?}}
{event:"destinations_complete", payload:{preferences, destinations, needs_resolution, suggester}}
{event:"research_complete", payload:{preferences, destinations, arrival, research}}
{event:"complete", payload:PlanResult}
{event:"incomplete", payload:{missing_fields, preferences}}
{event:"error", payload:{type, message, trace}}
```

## Privacy / secrets

- `backend/.env` and `frontend/.env.local` are gitignored. Never commit API keys.
- The fine-tuned LoRA adapter is committed (~40 MB) since it has no PII.
- The `travel_finetune_examples_1_200.jsonl` dataset is committed.
- SSH keys, `.env` files, `node_modules`, `.next`, training checkpoints, and `.venv` are all excluded.
