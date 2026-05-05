"""
LLM client routing for the TripWise multi-agent system.

Two clients, configured via environment variables:

- `orchestrator` — used by every agent EXCEPT Itinerary. Currently points at an
  OpenAI-compatible hosted API; switch to self-hosted vLLM later by changing
  ORCH_BASE_URL / ORCH_API_KEY / ORCH_MODEL in .env. No code changes needed.

- `itinerary` — used ONLY by the Itinerary agent. Points at a self-hosted vLLM
  (or serve_lora.py wrapper) loading Qwen/Qwen2.5-7B-Instruct with the
  tripwise-itinerary-lora adapter. The system prompt below MUST match the one
  used during training; do not modify without retraining.

Helpers:
- orch_complete: single-shot chat completion (no tools).
- orch_complete_with_tools: tool-calling loop. Pass tool schemas + a dispatch
  callable; the loop terminates when the model returns a tool-call-free reply
  or `max_steps` is exhausted.
"""
from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

ORCH_BASE_URL = os.environ.get("ORCH_BASE_URL", "https://api.openai.com/v1")
ORCH_API_KEY = os.environ.get("ORCH_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
ORCH_MODEL = os.environ.get("ORCH_MODEL", "gpt-4o")

ITIN_BASE_URL = os.environ.get("ITIN_BASE_URL", "http://localhost:8001/v1")
ITIN_API_KEY = os.environ.get("ITIN_API_KEY", "dummy")
ITIN_MODEL = os.environ.get("ITIN_MODEL", "tripwise")

orchestrator = AsyncOpenAI(base_url=ORCH_BASE_URL, api_key=ORCH_API_KEY)
itinerary = AsyncOpenAI(base_url=ITIN_BASE_URL, api_key=ITIN_API_KEY)


# Qwen3 thinking-mode suppression. The MLX server accepts it via extra_body;
# Cerebras and similar strict providers reject unknown fields, so only send
# it when the orchestrator model is actually a Qwen variant.
_IS_QWEN_ORCH = ORCH_MODEL.lower().startswith("qwen")
NO_THINK_EXTRA: dict[str, Any] = (
    {"chat_template_kwargs": {"enable_thinking": False}} if _IS_QWEN_ORCH else {}
)


async def orch_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format_json: bool = False,
) -> str:
    kwargs: dict[str, Any] = {"extra_body": NO_THINK_EXTRA}
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    r = await orchestrator.chat.completions.create(
        model=ORCH_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return r.choices[0].message.content or ""


async def orch_complete_with_tools(
    system: str,
    user: str,
    *,
    tools: list[dict],
    run_tool: Callable[[str, str], Awaitable[str]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_steps: int = 8,
    response_format_json: bool = False,
) -> str:
    """Tool-calling loop with inline result feedback.

    `run_tool(name, raw_args_json) -> result_json` is the user-supplied
    dispatch (e.g. backend.tools.run_tool). Terminates when the model returns
    text without tool calls, or `max_steps` rounds are spent.

    Quirk: many "OpenAI-compatible" servers (notably the local MLX server we
    use for orchestration) don't honor the standard `role: "tool"` reply
    messages — the model never sees the result and re-issues the same call.
    We fix this by feeding tool results back as a `role: "user"` turn with
    framed content. This format also works on OpenAI, vLLM, and Anthropic
    via SDK shim, so it's the universal default.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    final_text = ""
    for _ in range(max_steps):
        # Cerebras (and some other strict providers) reject `response_format`
        # alongside `tools`. The system prompt instructs JSON output and
        # `_extract_json` handles parsing, so omit response_format here.
        kwargs: dict[str, Any] = {
            "tools": tools,
            "tool_choice": "auto",
            "extra_body": NO_THINK_EXTRA,
        }
        r = await orchestrator.chat.completions.create(
            model=ORCH_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        msg = r.choices[0].message
        final_text = msg.content or ""

        if not msg.tool_calls:
            return final_text

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Run all tool calls in this round
        results: list[tuple[str, str, str]] = []
        for tc in msg.tool_calls:
            result_json = await run_tool(tc.function.name, tc.function.arguments)
            results.append((tc.function.name, tc.function.arguments, result_json))

        # Inline results as a user turn (MLX-compatible format)
        summary = "\n\n".join(
            f"Result of `{name}({args})`:\n{res}" for name, args, res in results
        )
        messages.append({
            "role": "user",
            "content": (
                f"Tool execution complete.\n\n{summary}\n\nUse these results to "
                "produce your final answer in the JSON format requested in the "
                "system message. Do not call any more tools."
            ),
        })

    return final_text


ITINERARY_SYSTEM_PROMPT = (
    "You are an itinerary generation agent. Generate realistic day-by-day "
    "travel itineraries in valid JSON only. Follow the user's constraints "
    "and do not add places that are not provided."
)


async def itin_complete(user_input: dict, *, max_tokens: int = 1024) -> str:
    """Call the fine-tuned itinerary model. `user_input` matches the training
    schema (destination, trip_length_days, travelers, budget_level, interests,
    pace, constraints, selected_places, route_groups)."""
    r = await itinerary.chat.completions.create(
        model=ITIN_MODEL,
        messages=[
            {"role": "system", "content": ITINERARY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_input)},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content or ""
