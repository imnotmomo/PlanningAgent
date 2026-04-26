# PlanningAgent — TripWise

A multi-agent travel planning assistant. Free-form trip request goes in (e.g. "5 days in Japan from San Francisco, 2 friends, medium budget, anime + food + city views"); a structured day-by-day itinerary, real candidate places/restaurants/hotels, round-trip flight options, and an all-in budget come out.

The itinerary-generation step is a **fine-tuned LoRA** on top of Qwen 2.5-7B-Instruct, scoped to one job: produce schema-conformant JSON for the day-by-day plan. Everything else (preference extraction, web-grounded research, route + meal + transit planning, budget arithmetic, multi-destination handling, critic-driven replanning) is done by orchestrator agents calling a general LLM (local MLX server or any OpenAI-compatible API).

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

## Quick start

You'll need three things running locally:

1. **An OpenAI-compatible "orchestrator" model** (used by every non-itinerary agent).
2. **The fine-tuned LoRA served behind an OpenAI-compatible endpoint** (used by the Itinerary agent).
3. **A Tavily API key** for web search grounding (Research, Budget, Arrival agents).

Plus the backend (FastAPI) and the frontend (Next.js).

### 1. Start the orchestrator LLM

Two convenient options:

- **Local MLX server on Apple Silicon** — run any Qwen 3-class model (e.g. `Qwen3.5-9B-MLX-4bit`). The default `.env` assumes this is at `http://localhost:5620/v1`.
- **OpenAI / OpenRouter / Anthropic / Together / etc.** — point `ORCH_BASE_URL` at any OpenAI-compatible API. Tool calling must be supported.

Examples in `backend/.env.example`.

### 2. Serve the fine-tuned LoRA

The adapter in `tripwise-itinerary-lora/` is an LoRA on `Qwen/Qwen2.5-7B-Instruct`. You need the base model loaded with the adapter, behind an OpenAI-compatible endpoint.

**Option A — vLLM (fastest, ~50 tok/s):**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --host 127.0.0.1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules tripwise=/path/to/tripwise-itinerary-lora
```

**Option B — `serve_lora.py` (works on any CUDA driver, ~5–10 tok/s):**
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/some/dir \
  python serve_lora.py --port 8001
```
This is a small FastAPI wrapper around `transformers` + `peft` that exposes the same `/v1/chat/completions` shape vLLM does. Use it when vLLM's prebuilt kernels don't match the host's CUDA toolchain (we hit this on a Vast.ai box with driver 570 vs vLLM 0.19's 12.9 kernels — `serve_lora.py` was the painless fallback).

If your GPU is remote (e.g. Vast.ai), tunnel the port:
```bash
ssh -L 8001:127.0.0.1:8001 user@gpu-host
```

### 3. Backend

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env       # then edit keys/URLs
uvicorn backend.server:app --port 8000     # run from repo root
```

`backend/.env` has placeholders for:

| Var | Purpose |
|---|---|
| `ORCH_BASE_URL` / `ORCH_API_KEY` / `ORCH_MODEL` | the orchestration LLM |
| `ITIN_BASE_URL` / `ITIN_API_KEY` / `ITIN_MODEL` | the LoRA-loaded model (defaults `http://localhost:8001/v1` and `tripwise`) |
| `TAVILY_API_KEY` | web search for Research / Budget / Arrival agents |

### 4. Frontend

```bash
cd frontend
cp .env.example .env.local                  # set NEXT_PUBLIC_API_URL if you want
npm install
npm run dev                                 # http://localhost:3000
```

The frontend's lib/api.ts uses `NEXT_PUBLIC_API_URL` for direct CORS calls when set; otherwise falls back to the `/api/*` rewrite proxy in `next.config.mjs`. Backend CORS allows `localhost:3000` by default.

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
                  Orchestrator LLM   Tavily web       Python subprocess     Itinerary LLM
                  (MLX / OpenAI /    search                                  Qwen2.5-7B
                   any compat API)                                          + tripwise LoRA
                                                                            (vLLM or serve_lora.py)
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

### Training

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Method:** bf16 LoRA (no QLoRA / no 4-bit), `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=[q_proj, k_proj, v_proj, o_proj]`
- **Optimizer:** AdamW, `lr=2e-4`, cosine schedule, 5% warmup
- **Batch:** per_device=2 × grad_accum=4 × world_size=2 = effective 16
- **Epochs:** 3 (~36 update steps over 189 train + 10 eval)
- **Hardware:** 2× RTX 5090 (DDP via `torchrun --nproc_per_node=2`)
- **Wall-clock:** **63.1 s**
- **Final losses:** train **0.26**, eval **0.26** (started 1.96 / 1.07)

### Data

`travel_finetune_examples_1_200.jsonl` — 199 chat-format examples mapping structured trip constraints + selected places + route groups to the JSON itinerary above. Filename says 200 but the file has 199 lines (one missing — kept as-is).

The assistant outputs in this dataset are **templated** (recurring boilerplate phrasing across days). The LoRA learns the schema cleanly but inherits the prose template; this is the dataset bottleneck, not a training one. Re-training on the same data would reproduce the same templated prose.

### How to retrain

```bash
# from a host with 2× GPUs, 32GB+ each, CUDA 12.x driver
cd /path/to/repo
pip install -r requirements.txt          # torch, transformers, peft, trl, datasets, accelerate
torchrun --nproc_per_node=2 train.py     # uses train.py:MODEL_NAME and DATA_FILE
# outputs land in ./tripwise-itinerary-lora/
python eval.py                           # held-out 10-example eval (fine-tuned vs base)
python report.py                         # writes report.md from trainer_state + eval_results
```

`train.py` env-var hooks: set `RAYON_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, raise `ulimit -n` to avoid Rayon thread-pool issues on small `nofile` limits.

### Evaluation

Held-out 10 examples (deterministic split, seed=42), greedy decoding (`do_sample=False`):

| Metric | Fine-tuned | Base (Qwen2.5-7B-Instruct) |
|---|---:|---:|
| `json_valid` | 1.000 | 0.900 |
| `schema_complete` | **1.000** | **0.000** |
| `day_count_correct` | **1.000** | **0.000** |
| `per_day_complete` | 1.000 | 0.900 |
| `no_extra_places_in_themes` | 0.900 | 0.900 |
| avg generation latency | ~11 s | ~7 s |

The fine-tune is doing the one thing it was scoped to: **schema and day-count compliance**. Base Qwen produces a perfectly fine-sounding travel plan in *its own* schema; the LoRA produces our schema reliably so downstream code can render it.

Full per-example outputs in `eval_output/eval_results.json`, full writeup in `report.md`.

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

## Original design docs

For background and rationale see [`tripwise_project_plan.md`](tripwise_project_plan.md) and [`tripwise_fine_tuning_plan.md`](tripwise_fine_tuning_plan.md).
