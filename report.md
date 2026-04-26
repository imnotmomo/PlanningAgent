# TripWise Itinerary LoRA — Training and Evaluation Report

## 1. Method

**Goal.** Fine-tune the Itinerary Generation Agent so that, given structured travel constraints + selected places + route groups, it produces a clean, schema-conformant day-by-day JSON itinerary.

**Setup.**

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: bf16 LoRA (no quantization), PEFT on attention projections only
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- Optimizer: AdamW, lr=0.0002, scheduler=cosine, warmup_ratio=0.05
- Batch: per_device=2, grad_accum=4, world_size=2, effective_batch=16
- Epochs: 3, max_seq_length=2048, precision=bf16
- Hardware: 2× RTX 5090 (DDP via torchrun)
- Data: `travel_finetune_examples_1_200.jsonl` → train=190 / held-out eval=10 (seed=42)

## 2. Training

- Wall-clock time: **63.1s**
- Logged train steps: 18
- Logged eval steps: 4

### Training loss

| step | loss |
|---:|---:|
| 2 | 1.9595 |
| 4 | 1.8449 |
| 6 | 1.4451 |
| 8 | 1.2930 |
| 10 | 1.2371 |
| 12 | 1.0425 |
| 14 | 0.9259 |
| 16 | 0.7935 |
| 18 | 0.6441 |
| 20 | 0.5547 |
| 22 | 0.4563 |
| 24 | 0.3749 |
| 26 | 0.3280 |
| 28 | 0.2977 |
| 30 | 0.2665 |
| 32 | 0.2480 |
| 34 | 0.2540 |
| 36 | 0.2629 |

### Eval loss (held-out 10)

| step | eval_loss |
|---:|---:|
| 10 | 1.0684 |
| 20 | 0.4711 |
| 30 | 0.2750 |
| 36 | 0.2601 |

## 3. Evaluation

Evaluated on the same 10 held-out examples used during training (deterministic split, seed=42). Generation: greedy decoding (`do_sample=False`), `max_new_tokens=1024`. Output JSON is extracted by finding the first balanced `{...}` block. Each example is scored on:

- **json_valid:** the extracted block parses as JSON
- **schema_complete:** all of {trip_summary, daily_itinerary, budget_summary, backup_options, travel_tips} are present
- **day_count_correct:** `len(daily_itinerary) == trip_length_days`
- **per_day_complete:** every day dict has {day, theme, morning, afternoon, evening, estimated_cost, transportation_note, feasibility_note}
- **no_extra_places_in_themes:** day themes do not name places outside `selected_places`
- **avg_latency_s:** mean greedy generation latency per example

### Metric comparison

| metric | fine-tuned | base |
|---|---:|---:|
| json_valid | 1.000 | 0.900 |
| schema_complete | 1.000 | 0.000 |
| day_count_correct | 1.000 | 0.000 |
| per_day_complete | 1.000 | 0.900 |
| no_extra_places_in_themes | 0.900 | 0.900 |
| avg_latency_s | 11.256 | 6.951 |

### Sample output (held-out example #1, fine-tuned)

**Input:**

```json
{
  "destination": "Innsbruck",
  "trip_length_days": 2,
  "travelers": "family",
  "budget_level": "medium",
  "interests": [
    "nature",
    "photography",
    "easy sightseeing"
  ],
  "pace": "packed",
  "constraints": [
    "keep the schedule realistic",
    "do not add places outside the provided list"
  ],
  "selected_places": [
    "Golden Roof",
    "Nordkette",
    "Old Town",
    "Ambras Castle"
  ],
  "route_groups": {
    "Day 1": [
      "Golden Roof",
      "Old Town"
    ],
    "Day 2": [
      "Nordkette",
      "Ambras Castle"
    ]
  }
}
```

**Generated:**

```
{"trip_summary": "A 2-day Innsbruck itinerary focused on nature, photography, and easy sightseeing.", "daily_itinerary": [{"day": 1, "theme": "Golden Roof + Old Town", "morning": "Start with Golden Roof and keep the pace comfortable.", "afternoon": "Continue with Old Town and have lunch nearby.", "evening": "Have dinner at a local restaurant and leave time for transportation.", "estimated_cost": "$50-$180 per person", "transportation_note": "Use the most practical local transportation and include a buffer time.", "feasibility_note": "The day is realistic because the main stops are limited and grouped by area."}, {"day": 2, "theme": "Nordkette + Ambras Castle", "morning": "Start with Nordkette and keep the pace comfortable.", "afternoon": "Continue with Ambras Castle and have lunch nearby.", "evening": "Have dinner at a local restaurant and leave time for transportation.", "estimated_cost": "$50-$180 per person", "transportation_note": "Use the most practical local transportation and include a buffer time.", "feasibility_note": "The day is realistic because the main stops are limited and grouped by area."}], "budget_summary": "This plan is designed for a medium budget excluding flights and major lodging changes.", "backup_options": ["Indoor museum option", "Flexible neighborhood walk"], "travel_tips": ["Check current opening hours before visiting.", "Leave buffer time for transportation.", "Book popular activities in advance when needed."]}
```

### Sample output (held-out example #1, base model)

```
```json
{
  "itinerary": {
    "Day 1": {
      "morning": {
        "activity": "Visit Golden Roof",
        "description": "Explore the famous Golden Roof, a landmark in Innsbruck known for its gilded roof tiles."
      },
      "afternoon": {
        "activity": "Stroll through Old Town",
        "description": "Take a leisurely walk through the historic center of Innsbruck, admiring the architecture and local shops."
      },
      "evening": {
        "activity": "Photography Session in Old Town",
        "description": "Capture the beauty of the old town with your camera, especially during sunset."
      }
    },
    "Day 2": {
      "morning": {
        "activity": "Hike Nordkette",
        "description": "Enjoy a moderate hike on Nordkette, offering scenic views of the Alps and the surrounding valleys."
      },
      "afternoon": {
        "activity": "Visit Ambras Castle",
        "description": "Discover the historical Ambras Castle, which houses a rich collection of art and artifacts."
      },
      "evening": {
        "activity": "Relax and Reflect",
        "description": "Spend some time relaxing or reflecting on your experiences from the day."
      }
    }
  }
}
```
```

## 4. Notes and caveats

- The training set's assistant outputs are heavily templated (recurring boilerplate phrases across all days). The fine-tuned model is therefore expected to gain on schema/format metrics rather than on prose quality or travel reasoning.
- Held-out set is only 10 examples; metric differences within ~10 percentage points are not statistically meaningful at this size.
- Evaluation is deterministic (greedy). Sampling-based generation may produce different validity numbers; rerun with sampling if comparing to API serving.
- `no_extra_places_in_themes` is heuristic (capitalized-word match against `selected_places`); treat as a sanity check, not a hard constraint metric.
