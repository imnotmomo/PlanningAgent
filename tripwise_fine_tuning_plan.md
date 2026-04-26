# Fine-Tuning Plan: Travel Itinerary Generation Agent

## 1. Fine-Tuning Goal

The project should not fine-tune the whole multi-agent system. Instead, fine-tuning should focus on one clear subtask:

> Given structured travel constraints, selected places, and route groups, generate a clean, realistic, day-by-day itinerary in valid JSON.

This fine-tuned model becomes the **Itinerary Generation Agent**.

## 2. Why Fine-Tune This Component?

The base LLM can already write travel plans, but it may have problems:

1. Inconsistent formatting
2. Invalid JSON
3. Too much extra text
4. Adding places that were not provided
5. Ignoring pace or budget constraints
6. Creating unrealistic days

Fine-tuning helps the model learn the project’s desired output style and structure.

## 3. Recommended Model Size

Use a **7B–8B instruct model**.

Recommended choices:

```text
Qwen2.5-7B-Instruct
Llama-3.1-8B-Instruct
Mistral-7B-Instruct
```

Best recommendation:

```text
Qwen2.5-7B-Instruct
```

Reason:

- Strong structured output ability
- Good JSON following
- Practical for LoRA / QLoRA
- Feasible for a class project

## 4. Fine-Tuning Method

Use **LoRA** or **QLoRA** instead of full fine-tuning.

```text
Pretrained base model
   ↓
Travel itinerary dataset
   ↓
LoRA / QLoRA fine-tuning
   ↓
Small adapter weights
   ↓
Fine-tuned itinerary generation model
```

QLoRA is recommended if GPU memory is limited.

## 5. Training Data Format

Use JSONL chat format. Each line is one training example.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an itinerary generation agent. Generate realistic day-by-day travel itineraries in valid JSON only. Follow the user's constraints and do not add places that are not provided."
    },
    {
      "role": "user",
      "content": "{\"destination\":\"Kyoto\",\"trip_length_days\":4,\"travelers\":\"2 adults\",\"budget_level\":\"medium\",\"interests\":[\"temples\",\"matcha\",\"traditional streets\"],\"pace\":\"relaxed\",\"constraints\":[\"avoid rushed schedule\"],\"selected_places\":[\"Kiyomizu-dera\",\"Gion\",\"Arashiyama Bamboo Grove\"],\"route_groups\":{\"Day 1\":[\"Kiyomizu-dera\",\"Gion\"],\"Day 2\":[\"Arashiyama Bamboo Grove\"]}}"
    },
    {
      "role": "assistant",
      "content": "{\"trip_summary\":\"...\",\"daily_itinerary\":[...],\"budget_summary\":\"...\",\"backup_options\":[...],\"travel_tips\":[...]}"
    }
  ]
}
```

## 6. Input Schema

Each input should include:

```json
{
  "destination": "Kyoto",
  "trip_length_days": 4,
  "travelers": "2 adults",
  "budget_level": "medium",
  "interests": ["temples", "matcha", "traditional streets", "photography"],
  "pace": "relaxed",
  "constraints": ["avoid rushed schedule", "prefer public transit"],
  "selected_places": [
    "Fushimi Inari Taisha",
    "Kiyomizu-dera",
    "Gion",
    "Arashiyama Bamboo Grove",
    "Nishiki Market",
    "Uji"
  ],
  "route_groups": {
    "Day 1": ["Kiyomizu-dera", "Gion"],
    "Day 2": ["Arashiyama Bamboo Grove"],
    "Day 3": ["Fushimi Inari Taisha", "Uji"],
    "Day 4": ["Nishiki Market"]
  }
}
```

## 7. Output Schema

The model should always output valid JSON:

```json
{
  "trip_summary": "A relaxed 4-day Kyoto itinerary focused on temples, matcha, traditional streets, and photography.",
  "daily_itinerary": [
    {
      "day": 1,
      "theme": "Higashiyama temples and Gion streets",
      "morning": "Visit Kiyomizu-dera early and spend time taking photos around the temple area.",
      "afternoon": "Walk through nearby traditional streets and stop for matcha or tea snacks.",
      "evening": "Explore Gion at a relaxed pace and have dinner nearby.",
      "estimated_cost": "$60-$90 per person",
      "transportation_note": "Use local bus or taxi between Kyoto Station and Higashiyama.",
      "feasibility_note": "This day is realistic because the main stops are close together."
    }
  ],
  "budget_summary": "This itinerary fits a medium budget excluding hotel and flight costs.",
  "backup_options": ["Kyoto National Museum", "Philosopher's Path"],
  "travel_tips": ["Start popular temple visits early.", "Check current opening hours before visiting."]
}
```

## 8. Dataset Size

Recommended class project size:

| Dataset Size | Use Case |
|---:|---|
| 50 examples | Demo only |
| 200 examples | Basic fine-tuning test |
| 500 examples | Strong class project |
| 1000+ examples | Better generalization |

Recommended:

```text
500 total examples
400 train
50 validation
50 test
```

The project currently has a generated dataset file for the first 200 examples. More examples can be added later.

## 9. Dataset Diversity

The dataset should cover:

### Destination Types

| Type | Examples |
|---|---|
| Big cities | Tokyo, Paris, London, New York, Seoul |
| Culture/history | Kyoto, Rome, Istanbul, Beijing |
| Nature | Yellowstone, Banff, Iceland, Swiss Alps |
| Beach | Hawaii, Bali, Phuket, Cancun |
| Food-focused | Osaka, Taipei, Bangkok, Barcelona |
| Family trips | Orlando, San Diego, Singapore |
| Budget trips | Mexico City, Lisbon, Prague |
| Luxury trips | Maldives, Seychelles, Swiss Alps |

### Trip Length

```text
1 day
2 days
3 days
4 days
5 days
7 days
10 days
```

### Budget Levels

```text
low
medium
high
luxury
```

### Pace Levels

```text
relaxed
medium
packed
```

### Constraints

Examples:

```text
no car
family with kids
elderly traveler
rainy weather
limited walking
avoid expensive restaurants
public transit only
must include beach time
must include one museum
avoid nightlife
```

## 10. Negative / Correction Examples

The dataset should include examples where the user asks for too much.

Example input:

```json
{
  "destination": "Paris",
  "trip_length_days": 1,
  "pace": "relaxed",
  "selected_places": [
    "Louvre Museum",
    "Eiffel Tower",
    "Versailles",
    "Montmartre",
    "Notre-Dame",
    "Latin Quarter"
  ],
  "constraints": ["avoid rushed schedule"]
}
```

Good output behavior:

```json
{
  "trip_summary": "A relaxed 1-day Paris itinerary cannot realistically include all selected places.",
  "daily_itinerary": [
    {
      "day": 1,
      "theme": "Central Paris highlights",
      "morning": "Visit the Louvre Museum.",
      "afternoon": "Walk around the Seine and Notre-Dame exterior area.",
      "evening": "End with the Eiffel Tower area.",
      "estimated_cost": "$80-$130 per person",
      "transportation_note": "Use the Metro between major areas.",
      "feasibility_note": "Versailles and Montmartre are excluded because including them would make the day too rushed."
    }
  ],
  "excluded_places": [
    {
      "place": "Versailles",
      "reason": "Too far for a relaxed 1-day central Paris plan."
    }
  ]
}
```

This teaches the model not to force unrealistic plans.

## 11. Training Setup

Install dependencies:

```bash
pip install transformers datasets peft accelerate bitsandbytes trl
```

Recommended training approach:

```text
Base model: Qwen2.5-7B-Instruct
Fine-tuning: QLoRA
Epochs: 2-3
Learning rate: 2e-4
Max sequence length: 2048
Batch size: 1-2 per GPU
Gradient accumulation: 8-16
```

## 12. Example Fine-Tuning Code

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"

dataset = load_dataset("json", data_files="travel_finetune_examples_1_200.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir="./travel-itinerary-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    max_seq_length=2048,
    bf16=True
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args
)

trainer.train()
trainer.save_model("./travel-itinerary-lora")
```

## 13. Inference Plan

After training, load the base model and LoRA adapter:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(model, "./travel-itinerary-lora")
```

The multi-agent system should call this fine-tuned model only after the previous agents produce:

```text
structured preferences
selected places
route groups
budget level
constraints
```

## 14. Evaluation

Evaluate the fine-tuned model against the base model.

### Evaluation Metrics

| Metric | Measurement |
|---|---|
| Valid JSON rate | Percentage of outputs that parse as JSON |
| Format consistency | Whether all required fields are present |
| Constraint following | Whether budget, pace, interests, and constraints are followed |
| No extra places | Whether model avoids adding unprovided places |
| Feasibility | Whether each day is realistic |
| Human preference | Human rating from 1-5 |
| Revision quality | Whether updated plan follows user feedback |

## 15. Experiment Design

Compare:

```text
Base model only
vs.
Base model + prompt engineering
vs.
Fine-tuned LoRA model
```

Expected outcome:

| System | Expected Result |
|---|---|
| Base model | Good language, inconsistent format |
| Prompt-engineered model | Better format, still sometimes inconsistent |
| Fine-tuned model | Best JSON consistency and style control |

## 16. Final Fine-Tuning Story for Report

Use this wording:

> We fine-tuned a 7B pretrained instruction model with LoRA for the itinerary generation agent. The training data maps structured travel constraints and route groups to valid JSON itineraries. This makes the final itinerary generation more consistent, personalized, and easier for the multi-agent system to use.

## 17. Final Role in the Full System

The fine-tuned model is not responsible for research, search, or route optimization.

It only handles:

```text
Input:
structured user preferences
+ selected places
+ route groups
+ budget level
+ constraints

Output:
valid JSON itinerary
+ daily plan
+ transportation notes
+ feasibility notes
+ budget summary
+ backup options
+ travel tips
```

This makes the fine-tuning scope clear, practical, and easy to evaluate.
