"""
Minimal OpenAI-compatible server for the tripwise LoRA on Qwen2.5-7B-Instruct.

Replaces vLLM where the host driver doesn't ship a CUDA toolchain new enough
for vLLM's prebuilt kernels. Slower (~5-10 tok/s on a 5090) but reliable —
uses the same transformers+peft stack the LoRA was trained on.

Usage:
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/workspace/.hf_home \\
    /venv/main/bin/python serve_lora.py \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --adapter /root/travel-agent/tripwise-itinerary-lora \\
    --port 8001
"""
from __future__ import annotations

import argparse
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--adapter", default="/root/travel-agent/tripwise-itinerary-lora")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--model-name", default="tripwise")
ARGS = parser.parse_args()


print(f"[serve_lora] loading {ARGS.base_model}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(ARGS.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    ARGS.base_model,
    dtype=torch.bfloat16,
    device_map="cuda",
)
print(f"[serve_lora] loading LoRA from {ARGS.adapter}", flush=True)
model = PeftModel.from_pretrained(model, ARGS.adapter)
model.eval()
print(f"[serve_lora] ready, listening on {ARGS.host}:{ARGS.port}", flush=True)


app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": ARGS.model_name, "object": "model", "owned_by": "local"},
            {"id": ARGS.base_model, "object": "model", "owned_by": "local"},
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _generate(messages: list[dict], temperature: float, max_tokens: int, top_p: float):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, int(inputs["input_ids"].shape[1]), int(new_tokens.shape[0])


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.model not in (ARGS.model_name, ARGS.base_model):
        raise HTTPException(status_code=400, detail=f"unknown model: {req.model}")
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming not implemented")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text, prompt_toks, completion_toks = _generate(
        messages, req.temperature, req.max_tokens, req.top_p
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": prompt_toks + completion_toks,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")
