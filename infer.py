import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./tripwise-itinerary-lora"

SYSTEM = (
    "You are an itinerary generation agent. Generate realistic day-by-day "
    "travel itineraries in valid JSON only. Follow the user's constraints "
    "and do not add places that are not provided."
)

EXAMPLE = {
    "destination": "Lisbon",
    "trip_length_days": 3,
    "travelers": "2 adults",
    "budget_level": "medium",
    "interests": ["food", "viewpoints", "history"],
    "pace": "relaxed",
    "constraints": ["prefer public transit"],
    "selected_places": ["Alfama", "Belem Tower", "LX Factory", "Time Out Market"],
    "route_groups": {
        "Day 1": ["Alfama"],
        "Day 2": ["Belem Tower"],
        "Day 3": ["LX Factory", "Time Out Market"],
    },
}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(EXAMPLE)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(text)


if __name__ == "__main__":
    main()
