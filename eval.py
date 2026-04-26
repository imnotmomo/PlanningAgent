import json
import os
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./tripwise-itinerary-lora"
EVAL_FILE = os.path.join(ADAPTER, "eval_examples.jsonl")
OUT_DIR = "./eval_output"

REQUIRED_TOP = ["trip_summary", "daily_itinerary", "budget_summary",
                "backup_options", "travel_tips"]
REQUIRED_DAY = ["day", "theme", "morning", "afternoon", "evening",
                "estimated_cost", "transportation_note", "feasibility_note"]


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def extract_json(text: str):
    """Find first balanced JSON object in text."""
    start = text.find("{")
    if start == -1:
        return None, "no opening brace"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob), None
                except Exception as e:
                    return None, f"parse error: {e}"
    return None, "unbalanced braces"


def score(generated: str, user_input: dict) -> dict:
    res = {
        "json_valid": False,
        "schema_complete": False,
        "day_count_correct": False,
        "per_day_complete": False,
        "no_extra_places_in_themes": False,
    }
    obj, err = extract_json(generated)
    if obj is None:
        res["parse_error"] = err
        return res
    res["json_valid"] = True
    res["schema_complete"] = all(k in obj for k in REQUIRED_TOP)

    daily = obj.get("daily_itinerary", [])
    if isinstance(daily, list):
        res["day_count_correct"] = len(daily) == user_input.get("trip_length_days")
        res["per_day_complete"] = all(
            isinstance(d, dict) and all(k in d for k in REQUIRED_DAY)
            for d in daily
        )

    selected_norm = {normalize(p) for p in user_input.get("selected_places", [])}
    themes = " ".join(d.get("theme", "") for d in daily if isinstance(d, dict))
    theme_words = re.findall(r"[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*", themes)
    extras = []
    for w in theme_words:
        n = normalize(w)
        if len(n) < 4:
            continue
        if not any(n in s or s in n for s in selected_norm):
            extras.append(w)
    res["extra_places_in_themes"] = extras
    res["no_extra_places_in_themes"] = len(extras) == 0
    return res


def evaluate(model, tokenizer, examples):
    results = []
    for i, ex in enumerate(examples):
        sys_msg, user_msg = ex["messages"][0], ex["messages"][1]
        gold = ex["messages"][2]["content"]
        try:
            user_input = json.loads(user_msg["content"])
        except Exception:
            user_input = {}
        prompt_msgs = [sys_msg, user_msg]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        latency = time.time() - t0
        s = score(gen, user_input)
        s["latency_s"] = latency
        s["generated"] = gen
        s["gold"] = gold
        s["input"] = user_input
        results.append(s)
        print(f"  [{i+1}/{len(examples)}] valid={s['json_valid']} "
              f"schema={s['schema_complete']} days={s['day_count_correct']} "
              f"perday={s['per_day_complete']} t={latency:.1f}s")
    return results


def aggregate(results):
    keys = ["json_valid", "schema_complete", "day_count_correct",
            "per_day_complete", "no_extra_places_in_themes"]
    n = len(results)
    agg = {k: sum(int(r[k]) for r in results) / n for k in keys}
    agg["avg_latency_s"] = sum(r["latency_s"] for r in results) / n
    agg["n"] = n
    return agg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"loaded {len(examples)} held-out examples")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Fine-tuned (base + LoRA adapter) ===")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    ft_results = evaluate(model, tokenizer, examples)
    ft_agg = aggregate(ft_results)
    print("fine-tuned aggregate:", json.dumps(ft_agg, indent=2))

    del model
    torch.cuda.empty_cache()

    print("\n=== Base model (no adapter) ===")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    base_results = evaluate(model, tokenizer, examples)
    base_agg = aggregate(base_results)
    print("base aggregate:", json.dumps(base_agg, indent=2))

    out = {
        "fine_tuned": {"per_example": ft_results, "aggregate": ft_agg},
        "base": {"per_example": base_results, "aggregate": base_agg},
    }
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR}/eval_results.json")


if __name__ == "__main__":
    main()
