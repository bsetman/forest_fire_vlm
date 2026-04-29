#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate structured forest-fire predictions for Qwen3-VL and InternVL3.5.

Current evaluation setting:
  - Use ONE unified prompt only: the same prompt used for Qwen3-VL / InternVL3.5 training.
  - Default test JSONL: /mnt/d/Abiyelunwen/new_fire_vlm/data/test_eval.jsonl

Supported backends:
  1) qwen3vl    : Qwen/Qwen3-VL-2B-Instruct, optional PEFT/LoRA adapter
  2) internvl35 : OpenGVLab/InternVL3_5-2B-HF, optional PEFT/LoRA adapter

Input gold file examples:
  - New test JSONL:
    {"id", "image_name", "image_path", "split", "prompt", "gold"}
  - Qwen-style JSONL:
    {"messages", "images"}
  - InternVL/Axolotl-style JSONL:
    {"messages": [{"role": "user", "content": [{"type": "image", "path": ...}, ...]}, ...]}

Output predictions JSONL:
  {"sample_id", "image_path", "gold", "raw_output", "pred", "json_valid", ...}

Examples:
  # Qwen3-VL LoRA
  python scripts/predict_structured_fire_unified.py \
    --backend qwen3vl \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --adapter /mnt/d/Abiyelunwen/new_fire_vlm/models/qwen3vl_2b_lora_best \
    --test-jsonl /mnt/d/Abiyelunwen/new_fire_vlm/data/test_eval.jsonl \
    --out /mnt/d/Abiyelunwen/new_fire_vlm/outputs/predictions/qwen3vl_lora_predictions.jsonl

  # InternVL3.5 HF LoRA / QLoRA adapter
  python scripts/predict_structured_fire_unified.py \
    --backend internvl35 \
    --model /home/swh/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-2B-HF/snapshots/<snapshot_id> \
    --adapter /mnt/d/Abiyelunwen/new_fire_vlm/outputs/internvl35_qlora/final \
    --test-jsonl /mnt/d/Abiyelunwen/new_fire_vlm/data/test_eval.jsonl \
    --out /mnt/d/Abiyelunwen/new_fire_vlm/outputs/predictions/internvl35_lora_predictions.jsonl
"""

import argparse
import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FIELDS = ["Smoke", "Fire", "Fire_Size", "Fire_Hotspots"]

ALLOWED_VALUES = {
    "Smoke": {"yes", "no"},
    "Fire": {"yes", "no"},
    "Fire_Size": {"small", "large", "cannot_determine", "no_fire"},
    "Fire_Hotspots": {"one_hotspot", "multiple_hotspots", "cannot_determine", "no_fire"},
}

# This is the exact training prompt used in the uploaded Qwen3-VL JSONL and InternVL3.5 Axolotl JSONL.
# Do not keep a separate BLIP2 prompt anymore.
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


def windows_to_wsl_path(path_str: str) -> str:
    """Convert D:\\... or D:/... to /mnt/d/...; leave Linux paths unchanged."""
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
    """Extract the first balanced {...} block from a text response."""
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
                    return text[start : i + 1]
    return None


def parse_json_like(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        # InternVL/Axolotl-style assistant content can be: [{"type": "text", "text": "{...}"}]
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

    # Remove common markdown fences.
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

    # Allow Flames as an alias for Fire, if some model uses the paper's field name.
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
    """Extract first image path from Qwen-style or InternVL/Axolotl-style messages."""
    if not isinstance(messages, list):
        return None
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            # Qwen training format keeps image path in top-level images, so string content has no path.
            continue
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
        raise ValueError(f"Sample {sample_id} has no image_path/image/images/messages image")
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
    adapter: Optional[str],
    sample_id: str,
    image_path: str,
    gold: Optional[Dict[str, str]],
    raw_output: str,
    error: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    pred_obj = parse_json_like(raw_output)
    pred, parse_errors = normalize_prediction(pred_obj)
    return {
        "model_name": model_name,
        "backend": backend,
        "adapter": adapter,
        "sample_id": sample_id,
        "image_path": image_path,
        "gold": gold,
        "raw_output": raw_output,
        "pred": pred,
        "json_valid": pred is not None,
        "parse_errors": parse_errors,
        "error": error,
        "elapsed_sec": elapsed_sec,
    }


def build_bnb_4bit_config(torch: Any) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def predict_qwen3vl(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if args.use_4bit:
        model_kwargs = {"quantization_config": build_bnb_4bit_config(torch), "device_map": "auto"}
    else:
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    print(f"[INFO] Loading Qwen3-VL processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[INFO] Loading Qwen3-VL model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)

    if args.adapter:
        from peft import PeftModel

        print(f"[INFO] Loading LoRA/QLoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Attach helper to processor for the shared generation loop.
    processor._qwen_process_vision_info = process_vision_info  # type: ignore[attr-defined]
    run_generation_loop(args, model_name=args.model, model=model, processor=processor, device=device, backend="qwen3vl")


def predict_internvl35(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.use_4bit:
        model_kwargs = {
            "quantization_config": build_bnb_4bit_config(torch),
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

    print(f"[INFO] Loading InternVL3.5 processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"[INFO] Loading InternVL3.5 model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)

    if args.adapter:
        from peft import PeftModel

        print(f"[INFO] Loading LoRA/QLoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_generation_loop(args, model_name=args.model, model=model, processor=processor, device=device, backend="internvl35")


def run_generation_loop(
    args: argparse.Namespace,
    model_name: str,
    model: Any,
    processor: Any,
    device: str,
    backend: str,
) -> None:
    import torch
    from PIL import Image

    test_path = Path(args.test_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(read_jsonl(test_path))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    print(f"[INFO] backend={backend}")
    print(f"[INFO] model={model_name}")
    print(f"[INFO] test_jsonl={test_path}")
    print(f"[INFO] samples={len(items)}")
    print(f"[INFO] output={out_path}")
    print("[INFO] prompt=UNIFIED_PROMPT")

    n_ok = 0
    n_json = 0
    t0_all = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(items, start=1):
            sample_id, image_path, gold = get_gold_and_image(item, fallback_id=f"sample_{idx:06d}")
            raw_output = ""
            err = None
            t0 = time.time()
            try:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"image not found: {image_path}")

                if backend == "qwen3vl":
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": args.prompt},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = processor._qwen_process_vision_info(messages)  # type: ignore[attr-defined]
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                        )
                    generated_ids_trimmed = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    raw_output = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()

                elif backend == "internvl35":
                    # Native Transformers / HF-format InternVL3.5 inference.
                    # Use the same multimodal chat-template style as training/evaluation.
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": args.prompt},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    # Move tensors to the first model device. This works for both normal and device_map="auto" loading.
                    target_device = getattr(model, "device", None) or device
                    inputs = inputs.to(target_device)
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                        )
                    raw_output = processor.decode(
                        generated_ids[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).strip()

                else:
                    raise ValueError(f"Unsupported backend: {backend}")

                n_ok += 1
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raw_output = ""

            rec = make_output_record(
                model_name=model_name,
                backend=backend,
                adapter=args.adapter,
                sample_id=sample_id,
                image_path=image_path,
                gold=gold,
                raw_output=raw_output,
                error=err,
                elapsed_sec=round(time.time() - t0, 4),
            )
            if rec["json_valid"]:
                n_json += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if idx == 1 or idx % args.print_every == 0 or idx == len(items):
                print(
                    f"[{idx}/{len(items)}] ok={n_ok}, json_valid={n_json}, "
                    f"last_sample={sample_id}, err={err}"
                )

    print("[DONE]")
    print(f"samples={len(items)}, generated={n_ok}, json_valid={n_json}")
    print(f"elapsed_total_sec={round(time.time() - t0_all, 2)}")
    print(f"output={out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["qwen3vl", "internvl35"])
    parser.add_argument("--model", default=None, help="Base model name or local path")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA/QLoRA adapter path")
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional override. By default, the script uses the unified Qwen3/InternVL training prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Use a small number for testing; 0 means full test set")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4bit loading")
    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = UNIFIED_PROMPT

    if args.model is None:
        if args.backend == "qwen3vl":
            args.model = "Qwen/Qwen3-VL-2B-Instruct"
        elif args.backend == "internvl35":
            args.model = "OpenGVLab/InternVL3_5-2B-HF"

    args.use_4bit = not args.no_4bit
    return args


def main() -> None:
    args = parse_args()
    if args.backend == "qwen3vl":
        predict_qwen3vl(args)
    elif args.backend == "internvl35":
        predict_internvl35(args)
    else:
        raise ValueError(args.backend)


if __name__ == "__main__":
    main()
