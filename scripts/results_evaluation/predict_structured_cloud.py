#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud VLM structured prediction through GPTsAPI tutorial-compatible endpoints.

Models kept for this experiment:
  1) gpt-4.1-mini
  2) gemini-2.5-flash

Default mode:
  --api-mode chat
  Uses GPTsAPI OpenAI-compatible endpoint:
    https://api.gptsapi.net/v1/chat/completions

Optional Gemini native mode:
  --api-mode gemini-native
  Uses GPTsAPI Gemini endpoint style:
    https://api.gptsapi.net/v1beta/models/gemini-2.5-flash:generateContent

Input JSONL examples:
  {"id", "image_name", "image_path", "split", "prompt", "gold"}
  {"messages", "images"}

Output JSONL:
  {"sample_id", "image_path", "gold", "raw_output", "pred", "json_valid", ...}
"""

import argparse
import ast
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


FIELDS = ["Smoke", "Fire", "Fire_Size", "Fire_Hotspots"]

ALLOWED_VALUES = {
    "Smoke": {"yes", "no"},
    "Fire": {"yes", "no"},
    "Fire_Size": {"small", "large", "cannot_determine", "no_fire"},
    "Fire_Hotspots": {"one_hotspot", "multiple_hotspots", "cannot_determine", "no_fire"},
}

UNIFIED_PROMPT = """Analyze this wildfire image. Output only a valid JSON object with exactly these fields: Smoke, Fire, Fire_Size, Fire_Hotspots.
Allowed values:
Smoke: yes, no
Fire: yes, no
Fire_Size: small, large, cannot_determine, no_fire
Fire_Hotspots: one_hotspot, multiple_hotspots, cannot_determine, no_fire
Rules:
1. If Smoke is no and Fire is no, set Fire_Size and Fire_Hotspots to no_fire.
2. If Smoke is yes and Fire is no, set Fire_Size and Fire_Hotspots to cannot_determine.
3. Output JSON only. Do not output markdown or explanations."""

DEFAULT_TEST_JSONL = "/mnt/d/Abiyelunwen/new_fire_vlm/data/eval/test_eval.jsonl"
DEFAULT_BASE_URL = "https://api.gptsapi.net"
SUPPORTED_MODELS = {"gpt-4.1-mini", "gemini-2.5-flash"}

# Prices copied from the GPTsAPI page shown by the user / previous script.
# They are used only for rough local estimation. Always trust the website bill for final cost.
PRICE_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


def windows_to_wsl_path(path_str: str) -> str:
    if path_str is None:
        return path_str
    s = str(path_str).strip().replace("\\", "/")
    if len(s) >= 3 and s[1] == ":" and s[2] == "/":
        drive = s[0].lower()
        return f"/mnt/{drive}/{s[3:]}"
    return s


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def extract_balanced_json(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
    return None


def parse_json_like(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                obj = parse_json_like(part.get("text"))
                if obj is not None:
                    return obj
        return None
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    candidates = [s]
    block = extract_balanced_json(s)
    if block and block not in candidates:
        candidates.append(block)

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def normalize_value(field: str, value: Any) -> Optional[str]:
    if value is None:
        return None

    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "n": "no",
        "false": "no",
        "0": "no",
        "none": "no_fire",
        "no_fire_visible": "no_fire",
        "no_forest_fire_visible": "no_fire",
        "no_wildfire": "no_fire",
        "no_fire_detected": "no_fire",
        "cannot_be_determined": "cannot_determine",
        "cannot_determine_from_image": "cannot_determine",
        "unknown": "cannot_determine",
        "uncertain": "cannot_determine",
        "not_clear": "cannot_determine",
        "single_hotspot": "one_hotspot",
        "single": "one_hotspot",
        "one": "one_hotspot",
        "multiple": "multiple_hotspots",
        "multi": "multiple_hotspots",
        "many": "multiple_hotspots",
        "several": "multiple_hotspots",
    }

    v = aliases.get(v, v)
    return v if v in ALLOWED_VALUES[field] else None


def normalize_prediction(obj: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, str]], List[str]]:
    if obj is None:
        return None, ["not_parseable"]

    if "Fire" not in obj and "Flames" in obj:
        obj["Fire"] = obj["Flames"]

    errors = []
    out = {}

    for field in FIELDS:
        if field not in obj:
            errors.append(f"missing:{field}")
            continue

        v = normalize_value(field, obj[field])
        if v is None:
            errors.append(f"invalid:{field}={obj[field]!r}")
        else:
            out[field] = v

    if errors:
        return None, errors

    return out, []


def _extract_image_from_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None

    for m in messages:
        if not isinstance(m, dict):
            continue

        content = m.get("content")

        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image":
                    return part.get("image") or part.get("path") or part.get("url")

    return None


def _extract_gold_from_messages(messages: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(messages, list):
        return None

    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return parse_json_like(m.get("content"))

    return None


def get_gold_and_image(item: Dict[str, Any], fallback_id: str) -> Tuple[str, str, Optional[Dict[str, str]]]:
    sample_id = str(
        item.get("sample_id")
        or item.get("id")
        or item.get("image_name")
        or fallback_id
    )

    image_path = item.get("image_path") or item.get("image")

    if not image_path and isinstance(item.get("images"), list) and item["images"]:
        image_path = item["images"][0]

    if not image_path:
        image_path = _extract_image_from_messages(item.get("messages"))

    if not image_path:
        raise ValueError(f"Sample {sample_id} has no image path")

    image_path = windows_to_wsl_path(image_path)

    gold_obj = None

    if isinstance(item.get("label_dict"), dict):
        gold_obj = item["label_dict"]
    elif isinstance(item.get("gold"), dict):
        gold_obj = item["gold"]
    elif item.get("target_text"):
        gold_obj = parse_json_like(item["target_text"])
    elif isinstance(item.get("messages"), list):
        gold_obj = _extract_gold_from_messages(item["messages"])

    gold, _ = normalize_prediction(gold_obj)
    return sample_id, image_path, gold


def make_output_record(
    model_name: str,
    backend: str,
    api_mode: str,
    sample_id: str,
    image_path: str,
    gold: Optional[Dict[str, str]],
    raw_output: str,
    error: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
    usage: Optional[Dict[str, Any]] = None,
    estimated_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    pred_obj = parse_json_like(raw_output)
    pred, parse_errors = normalize_prediction(pred_obj)

    return {
        "model_name": model_name,
        "backend": backend,
        "api_mode": api_mode,
        "adapter": None,
        "sample_id": sample_id,
        "image_path": image_path,
        "gold": gold,
        "raw_output": raw_output,
        "pred": pred,
        "json_valid": pred is not None,
        "parse_errors": parse_errors,
        "error": error,
        "elapsed_sec": elapsed_sec,
        "usage": usage or {},
        "estimated_cost_usd": estimated_cost_usd,
    }


def image_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def image_to_base64_and_mime(image_path: str) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def extract_text_from_chat_response(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
        return "\n".join(parts).strip()
    return ""


def extract_usage_from_chat_response(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = data.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def call_gptsapi_chat(
    model_name: str,
    prompt: str,
    image_path: str,
    max_output_tokens: int,
    api_key: str,
    base_url: str,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    """Call GPTsAPI Chat mode: POST /v1/chat/completions."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    image_url = image_to_data_url(image_path)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
    }

    r = requests.post(url, headers=build_headers(api_key), json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data, ensure_ascii=False)[:1000]}")

    return extract_text_from_chat_response(data), extract_usage_from_chat_response(data)


def extract_text_from_gemini_response(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def extract_usage_from_gemini_response(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = data.get("usageMetadata") or data.get("usage_metadata") or {}
    return {
        "input_tokens": usage.get("promptTokenCount") or usage.get("prompt_token_count"),
        "output_tokens": usage.get("candidatesTokenCount") or usage.get("candidates_token_count"),
        "total_tokens": usage.get("totalTokenCount") or usage.get("total_token_count"),
    }


def call_gptsapi_gemini_native(
    model_name: str,
    prompt: str,
    image_path: str,
    max_output_tokens: int,
    api_key: str,
    base_url: str,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    """Call GPTsAPI Gemini native non-stream mode: POST /v1beta/models/{model}:generateContent."""
    if not model_name.startswith("gemini-"):
        raise ValueError("--api-mode gemini-native only supports Gemini models")

    url = f"{base_url.rstrip('/')}/v1beta/models/{model_name}:generateContent"
    image_b64, mime = image_to_base64_and_mime(image_path)

    # Header includes both auth styles to match GPTsAPI tutorial compatibility notes.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
        },
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data, ensure_ascii=False)[:1000]}")

    return extract_text_from_gemini_response(data), extract_usage_from_gemini_response(data)


def estimate_cost(model_name: str, usage: Dict[str, Any]) -> Optional[float]:
    if model_name not in PRICE_PER_1M:
        return None

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0

    price = PRICE_PER_1M[model_name]

    cost = (
        input_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"]
    )

    return round(cost, 8)


def run_cloud_generation(args: argparse.Namespace) -> None:
    test_path = Path(args.test_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("GPTSAPI_API_KEY") or os.environ.get("GPTSAPI_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Set GPTSAPI_API_KEY or pass --api-key.")

    items = list(read_jsonl(test_path))

    if args.limit and args.limit > 0:
        items = items[:args.limit]

    print(f"[INFO] backend=gptsapi")
    print(f"[INFO] api_mode={args.api_mode}")
    print(f"[INFO] base_url={args.base_url}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] test_jsonl={test_path}")
    print(f"[INFO] samples={len(items)}")
    print(f"[INFO] output={out_path}")
    print("[INFO] prompt=UNIFIED_PROMPT")

    n_ok = 0
    n_json = 0
    total_cost = 0.0
    t0_all = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(items, start=1):
            sample_id, image_path, gold = get_gold_and_image(item, fallback_id=f"sample_{idx:06d}")

            raw_output = ""
            err = None
            usage = {}
            cost = None
            t0 = time.time()

            try:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"image not found: {image_path}")

                if args.api_mode == "chat":
                    raw_output, usage = call_gptsapi_chat(
                        args.model,
                        args.prompt,
                        image_path,
                        args.max_new_tokens,
                        api_key,
                        args.base_url,
                        args.timeout,
                    )
                elif args.api_mode == "gemini-native":
                    raw_output, usage = call_gptsapi_gemini_native(
                        args.model,
                        args.prompt,
                        image_path,
                        args.max_new_tokens,
                        api_key,
                        args.base_url,
                        args.timeout,
                    )
                else:
                    raise ValueError(f"Unsupported api_mode: {args.api_mode}")

                cost = estimate_cost(args.model, usage)

                if cost is not None:
                    total_cost += cost

                n_ok += 1

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raw_output = ""

            rec = make_output_record(
                model_name=args.model,
                backend="gptsapi",
                api_mode=args.api_mode,
                sample_id=sample_id,
                image_path=image_path,
                gold=gold,
                raw_output=raw_output,
                error=err,
                elapsed_sec=round(time.time() - t0, 4),
                usage=usage,
                estimated_cost_usd=cost,
            )

            if rec["json_valid"]:
                n_json += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if idx == 1 or idx % args.print_every == 0 or idx == len(items):
                print(
                    f"[{idx}/{len(items)}] ok={n_ok}, json_valid={n_json}, "
                    f"last_sample={sample_id}, cost_usd={cost}, err={err}"
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("[DONE]")
    print(f"samples={len(items)}, generated={n_ok}, json_valid={n_json}")
    print(f"estimated_total_cost_usd={round(total_cost, 6)}")
    print(f"elapsed_total_sec={round(time.time() - t0_all, 2)}")
    print(f"output={out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Kept optional for old command compatibility. It is ignored because this script only uses GPTsAPI.
    parser.add_argument("--backend", default="gptsapi", choices=["gptsapi", "openai", "gemini"])
    parser.add_argument("--api-mode", default="chat", choices=["chat", "gemini-native"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = UNIFIED_PROMPT

    if args.backend in {"openai", "gemini"}:
        print("[WARN] --backend is kept only for compatibility. This script uses GPTsAPI endpoints.")

    if args.api_mode == "gemini-native" and not args.model.startswith("gemini-"):
        raise ValueError("--api-mode gemini-native can only be used with gemini-2.5-flash")

    return args


def main() -> None:
    args = parse_args()
    run_cloud_generation(args)


if __name__ == "__main__":
    main()
