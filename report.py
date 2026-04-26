import json
import os
from pathlib import Path

OUTPUT_DIR = Path("./tripwise-itinerary-lora")
EVAL_DIR = Path("./eval_output")
REPORT_PATH = Path("./report.md")


def find_trainer_state():
    candidates = list(OUTPUT_DIR.glob("trainer_state.json"))
    candidates += list(OUTPUT_DIR.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def fmt(v, prec=3):
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def metric_row(name, ft, base):
    return f"| {name} | {fmt(ft.get(name))} | {fmt(base.get(name))} |"


def main():
    meta_path = OUTPUT_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    ts_path = find_trainer_state()
    ts = json.loads(ts_path.read_text()) if ts_path else {}
    log_history = ts.get("log_history", [])

    train_losses = [(r["step"], r["loss"]) for r in log_history if "loss" in r]
    eval_losses = [(r["step"], r["eval_loss"]) for r in log_history if "eval_loss" in r]

    eval_path = EVAL_DIR / "eval_results.json"
    eval_data = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    ft = eval_data.get("fine_tuned", {}).get("aggregate", {})
    base = eval_data.get("base", {}).get("aggregate", {})

    lines = []
    lines.append("# TripWise Itinerary LoRA — Training and Evaluation Report\n")

    lines.append("## 1. Method\n")
    lines.append("**Goal.** Fine-tune the Itinerary Generation Agent so that, given structured travel constraints + selected places + route groups, it produces a clean, schema-conformant day-by-day JSON itinerary.\n")
    train_meta = meta.get("training", {})
    data_meta = meta.get("data", {})
    lora_meta = meta.get("lora", {})
    lines.append("**Setup.**\n")
    lines.append(f"- Base model: `{meta.get('base_model','?')}`")
    lines.append(f"- Method: bf16 LoRA (no quantization), PEFT on attention projections only")
    lines.append(f"- LoRA: r={lora_meta.get('r','?')}, alpha={lora_meta.get('alpha','?')}, dropout={lora_meta.get('dropout','?')}, target_modules={lora_meta.get('target_modules','?')}")
    lines.append(f"- Optimizer: AdamW, lr={train_meta.get('lr','?')}, scheduler={train_meta.get('scheduler','?')}, warmup_ratio={train_meta.get('warmup_ratio','?')}")
    lines.append(f"- Batch: per_device={train_meta.get('per_device_batch_size','?')}, grad_accum={train_meta.get('grad_accum','?')}, world_size={train_meta.get('world_size','?')}, effective_batch={train_meta.get('effective_batch','?')}")
    lines.append(f"- Epochs: {train_meta.get('epochs','?')}, max_seq_length={train_meta.get('max_seq_length','?')}, precision={train_meta.get('precision','?')}")
    lines.append(f"- Hardware: 2× RTX 5090 (DDP via torchrun)")
    lines.append(f"- Data: `{data_meta.get('file','?')}` → train={data_meta.get('train_size','?')} / held-out eval={data_meta.get('eval_size','?')} (seed={data_meta.get('seed','?')})")
    lines.append("")

    lines.append("## 2. Training\n")
    lines.append(f"- Wall-clock time: **{meta.get('elapsed_seconds', float('nan')):.1f}s**")
    lines.append(f"- Logged train steps: {len(train_losses)}")
    lines.append(f"- Logged eval steps: {len(eval_losses)}")
    lines.append("")
    if train_losses:
        lines.append("### Training loss\n")
        lines.append("| step | loss |")
        lines.append("|---:|---:|")
        for s, l in train_losses:
            lines.append(f"| {s} | {l:.4f} |")
        lines.append("")
    if eval_losses:
        lines.append("### Eval loss (held-out 10)\n")
        lines.append("| step | eval_loss |")
        lines.append("|---:|---:|")
        for s, l in eval_losses:
            lines.append(f"| {s} | {l:.4f} |")
        lines.append("")

    lines.append("## 3. Evaluation\n")
    lines.append("Evaluated on the same 10 held-out examples used during training (deterministic split, seed=42). Generation: greedy decoding (`do_sample=False`), `max_new_tokens=1024`. Output JSON is extracted by finding the first balanced `{...}` block. Each example is scored on:\n")
    lines.append("- **json_valid:** the extracted block parses as JSON")
    lines.append("- **schema_complete:** all of {trip_summary, daily_itinerary, budget_summary, backup_options, travel_tips} are present")
    lines.append("- **day_count_correct:** `len(daily_itinerary) == trip_length_days`")
    lines.append("- **per_day_complete:** every day dict has {day, theme, morning, afternoon, evening, estimated_cost, transportation_note, feasibility_note}")
    lines.append("- **no_extra_places_in_themes:** day themes do not name places outside `selected_places`")
    lines.append("- **avg_latency_s:** mean greedy generation latency per example\n")

    lines.append("### Metric comparison\n")
    lines.append("| metric | fine-tuned | base |")
    lines.append("|---|---:|---:|")
    for k in ["json_valid", "schema_complete", "day_count_correct",
              "per_day_complete", "no_extra_places_in_themes", "avg_latency_s"]:
        lines.append(metric_row(k, ft, base))
    lines.append("")

    ft_per = eval_data.get("fine_tuned", {}).get("per_example", [])
    base_per = eval_data.get("base", {}).get("per_example", [])
    if ft_per:
        lines.append("### Sample output (held-out example #1, fine-tuned)\n")
        lines.append("**Input:**\n")
        lines.append("```json")
        lines.append(json.dumps(ft_per[0].get("input", {}), indent=2))
        lines.append("```\n")
        lines.append("**Generated:**\n")
        lines.append("```")
        lines.append(ft_per[0].get("generated", "")[:2000])
        lines.append("```\n")
    if base_per:
        lines.append("### Sample output (held-out example #1, base model)\n")
        lines.append("```")
        lines.append(base_per[0].get("generated", "")[:2000])
        lines.append("```\n")

    lines.append("## 4. Notes and caveats\n")
    lines.append("- The training set's assistant outputs are heavily templated (recurring boilerplate phrases across all days). The fine-tuned model is therefore expected to gain on schema/format metrics rather than on prose quality or travel reasoning.")
    lines.append("- Held-out set is only 10 examples; metric differences within ~10 percentage points are not statistically meaningful at this size.")
    lines.append("- Evaluation is deterministic (greedy). Sampling-based generation may produce different validity numbers; rerun with sampling if comparing to API serving.")
    lines.append("- `no_extra_places_in_themes` is heuristic (capitalized-word match against `selected_places`); treat as a sanity check, not a hard constraint metric.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH}")
    print(f"  fine_tuned aggregate: {json.dumps(ft, indent=2)}")
    print(f"  base aggregate:       {json.dumps(base, indent=2)}")


if __name__ == "__main__":
    main()
