import json
import os
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "travel_finetune_examples_1_200.jsonl"
OUTPUT_DIR = "./tripwise-itinerary-lora"
SEED = 42

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_DDP = WORLD_SIZE > 1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=DATA_FILE, split="train")
    split = ds.shuffle(seed=SEED).train_test_split(test_size=10, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    if LOCAL_RANK == 0:
        print(f"train={len(train_ds)} eval={len(eval_ds)} ddp={IS_DDP} world_size={WORLD_SIZE}")
        eval_path = os.path.join(OUTPUT_DIR, "eval_examples.jsonl")
        with open(eval_path, "w") as f:
            for ex in eval_ds:
                f.write(json.dumps(ex) + "\n")
        print(f"saved held-out eval set to {eval_path}")

    device_map = {"": LOCAL_RANK} if IS_DDP else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        max_length=2048,
        bf16=True,
        packing=False,
        seed=SEED,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=training_args,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    if LOCAL_RANK == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        meta = {
            "base_model": MODEL_NAME,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "training": {
                "epochs": 3,
                "per_device_batch_size": 2,
                "grad_accum": 4,
                "world_size": WORLD_SIZE,
                "effective_batch": 2 * 4 * WORLD_SIZE,
                "lr": 2e-4,
                "warmup_ratio": 0.05,
                "scheduler": "cosine",
                "max_seq_length": 2048,
                "precision": "bf16",
            },
            "data": {
                "file": DATA_FILE,
                "train_size": len(train_ds),
                "eval_size": len(eval_ds),
                "seed": SEED,
            },
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
