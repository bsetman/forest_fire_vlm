#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image inference for forest-fire VLM adapters.

Purpose:
  - Run one image at a time after the program starts.
  - Use the same unified prompt and label constraints as predict_structured_new.py.
  - Load a base VLM and an optional LoRA/QLoRA adapter.
  - Print only five final fields:
      Smoke
      Fire
      Fire_Size
      Fire_Hotspots
      description

Supported backends:
  - qwen3vl
  - internvl35
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


FIELDS = ["Smoke", "Fire", "Fire_Size", "Fire_Hotspots"]

ALLOWED_VALUES = {
    "Smoke": {"yes", "no"},
    "Fire": {"yes", "no"},
    "Fire_Size": {"small", "large", "cannot_determine", "no_fire"},
    "Fire_Hotspots": {"one_hotspot", "multiple_hotspots", "cannot_determine", "no_fire"},
}

# Keep this prompt identical to the unified training/evaluation prompt.
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


def windows_to_wsl_path(path_str: str) -> str:
    """Convert D:\\... or D:/... to /mnt/d/...; leave Linux paths unchanged."""
    if path_str is None:
        return path_str
    s = str(path_str).strip().strip('"').strip("'").replace("\\", "/")
    if len(s) >= 3 and s[1] == ":" and s[2] == "/":
        drive = s[0].lower()
        return f"/mnt/{drive}/{s[3:]}"
    return s


def build_bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_processor(
    backend: str,
    model_name: str,
    adapter_path: Optional[str] = None,
    load_4bit: bool = False,
):
    model_kwargs: Dict[str, Any]
    if load_4bit:
        model_kwargs = {
            "quantization_config": build_bnb_4bit_config(),
            "device_map": "auto",
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

    if backend == "internvl35":
        model_kwargs["trust_remote_code"] = True

    print(f"[INFO] Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=(backend == "internvl35"),
    )

    print(f"[INFO] Loading base model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    if adapter_path:
        print(f"[INFO] Loading LoRA/QLoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    model.eval()

    if backend == "qwen3vl":
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL backend requires qwen-vl-utils. Install it with: pip install qwen-vl-utils"
            ) from e
        processor._qwen_process_vision_info = process_vision_info  # type: ignore[attr-defined]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, processor, device


def extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced {...} block from a model response."""
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
    """Parse JSON-like model output, including markdown-fenced JSON and Python dict style."""
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


def extract_field(raw_text: str, field_name: str) -> Optional[str]:
    """Fallback extraction if the model did not return valid JSON."""
    patterns = [
        rf'"{field_name}"\s*:\s*"([^"]+)"',
        rf"'{field_name}'\s*:\s*'([^']+)'",
        rf"{field_name}\s*:\s*([A-Za-z_\- ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def parse_json_or_fields(raw_output: str) -> Optional[Dict[str, Any]]:
    obj = parse_json_like(raw_output)
    if obj is not None:
        return obj

    fallback = {}
    for field in FIELDS:
        value = extract_field(raw_output, field)
        if value is not None:
            fallback[field] = value
    return fallback if fallback else None


def normalize_value(field: str, value: Any) -> Optional[str]:
    if value is None:
        return None

    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        # yes/no fields
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "visible": "yes",
        "present": "yes",
        "detected": "yes",
        "n": "no",
        "false": "no",
        "0": "no",
        "absent": "no",
        "not_visible": "no",
        "not_detected": "no",

        # no-fire aliases
        "none": "no_fire",
        "no": "no_fire" if field in {"Fire_Size", "Fire_Hotspots"} else "no",
        "no_fire_visible": "no_fire",
        "no_forest_fire_visible": "no_fire",
        "no_wildfire": "no_fire",
        "no_fire_detected": "no_fire",

        # cannot determine aliases
        "cannot_be_determined": "cannot_determine",
        "cannot_determine_from_image": "cannot_determine",
        "unknown": "cannot_determine",
        "uncertain": "cannot_determine",
        "not_clear": "cannot_determine",
        "unclear": "cannot_determine",

        # hotspot aliases
        "single_hotspot": "one_hotspot",
        "single": "one_hotspot",
        "one": "one_hotspot",
        "one_fire_hotspot": "one_hotspot",
        "multiple": "multiple_hotspots",
        "multi": "multiple_hotspots",
        "many": "multiple_hotspots",
        "several": "multiple_hotspots",
        "multiple_hotspot": "multiple_hotspots",
    }

    v = aliases.get(v, v)
    return v if v in ALLOWED_VALUES[field] else None


def normalize_prediction(obj: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], List[str]]:
    """
    Normalize model output according to the same constraints used in predict_structured_new.py.

    Final allowed values:
      Smoke: yes/no
      Fire: yes/no
      Fire_Size: small/large/cannot_determine/no_fire
      Fire_Hotspots: one_hotspot/multiple_hotspots/cannot_determine/no_fire
    """
    errors: List[str] = []

    if obj is None:
        errors.append("not_parseable")
        obj = {}

    # Allow Flames as an alias for Fire.
    if "Fire" not in obj and "Flames" in obj:
        obj["Fire"] = obj["Flames"]

    out: Dict[str, str] = {}

    for field in FIELDS:
        if field not in obj:
            errors.append(f"missing:{field}")
            out[field] = "no" if field in {"Smoke", "Fire"} else "cannot_determine"
            continue

        v = normalize_value(field, obj[field])
        if v is None:
            errors.append(f"invalid:{field}={obj[field]!r}")
            out[field] = "no" if field in {"Smoke", "Fire"} else "cannot_determine"
        else:
            out[field] = v

    # Strict logical repair using the same rules as the training/evaluation prompt.
    if out["Smoke"] == "no" and out["Fire"] == "no":
        out["Fire_Size"] = "no_fire"
        out["Fire_Hotspots"] = "no_fire"
    elif out["Smoke"] == "yes" and out["Fire"] == "no":
        out["Fire_Size"] = "cannot_determine"
        out["Fire_Hotspots"] = "cannot_determine"
    elif out["Fire"] == "yes":
        # If fire is visible, no_fire is contradictory. Do not invent small/large;
        # use cannot_determine when the model did not clearly specify the extent/hotspots.
        if out["Fire_Size"] == "no_fire":
            out["Fire_Size"] = "cannot_determine"
        if out["Fire_Hotspots"] == "no_fire":
            out["Fire_Hotspots"] = "cannot_determine"

    return out, errors


def build_russian_description(pred: Dict[str, str]) -> str:
    smoke = pred["Smoke"]
    fire = pred["Fire"]
    size = pred["Fire_Size"]
    hotspots = pred["Fire_Hotspots"]

    size_map = {
        "small": "масштаб возгорания выглядит небольшим",
        "large": "масштаб возгорания выглядит значительным",
        "cannot_determine": "масштаб возгорания по изображению нельзя надёжно определить",
        "no_fire": "признаки открытого огня отсутствуют",
    }

    hotspot_map = {
        "one_hotspot": "виден один основной очаг возгорания",
        "multiple_hotspots": "видны несколько очагов возгорания или протяжённая линия огня",
        "cannot_determine": "количество очагов возгорания по изображению нельзя надёжно определить",
        "no_fire": "очаги возгорания не обнаружены",
    }

    if smoke == "no" and fire == "no":
        return (
            "На изображении не наблюдаются явные признаки дыма или открытого огня. "
            "Визуальные признаки лесного пожара не обнаружены."
        )

    if smoke == "yes" and fire == "no":
        return (
            "На изображении видны признаки дыма, однако открытое пламя явно не определяется. "
            "Согласно принятой схеме разметки, размер пожара и количество очагов в такой ситуации "
            "нельзя надёжно определить только по изображению. Необходима дополнительная проверка участка."
        )

    if smoke == "no" and fire == "yes":
        return (
            "На изображении видны признаки открытого огня, при этом дым выражен слабо или не наблюдается. "
            f"{size_map[size].capitalize()}, {hotspot_map[hotspots]}. "
            "Изображение содержит визуальные признаки активного возгорания."
        )

    return (
        "На изображении наблюдаются признаки дыма и открытого огня. "
        f"{size_map[size].capitalize()}, {hotspot_map[hotspots]}. "
        "Совместное наличие дыма и пламени указывает на активность лесного пожара."
    )


def generate_qwen3vl(model, processor, image_path: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": UNIFIED_PROMPT},
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
    )

    target_device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(target_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_internvl35(model, processor, image_path: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": UNIFIED_PROMPT},
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

    target_device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(target_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    return processor.decode(
        generated_ids[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def generate_raw_output(model, processor, image_path: str, backend: str, max_new_tokens: int) -> str:
    if backend == "qwen3vl":
        return generate_qwen3vl(model, processor, image_path, max_new_tokens)
    if backend == "internvl35":
        return generate_internvl35(model, processor, image_path, max_new_tokens)
    raise ValueError(f"Unsupported backend: {backend}")


def print_text_result(pred: Dict[str, str], description: str) -> None:
    print(f"Smoke: {pred['Smoke']}")
    print(f"Fire: {pred['Fire']}")
    print(f"Fire_Size: {pred['Fire_Size']}")
    print(f"Fire_Hotspots: {pred['Fire_Hotspots']}")
    print(f"description: {description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["qwen3vl", "internvl35"])
    parser.add_argument("--model", required=True, help="Base model name or local path")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA/QLoRA adapter path")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--show-raw", action="store_true", help="Debug only: print raw model output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, processor, _device = load_model_and_processor(
        backend=args.backend,
        model_name=args.model,
        adapter_path=args.adapter,
        load_4bit=args.load_4bit,
    )

    while True:
        image_input = input("\nВведите путь к изображению. Для выхода нажмите q: ").strip()
        if image_input.lower() in {"q", "quit", "exit"}:
            print("Программа завершена.")
            break

        image_path = windows_to_wsl_path(image_input)
        if not Path(image_path).exists():
            print(f"[ERROR] Изображение отсутствует: {image_path}")
            continue

        try:
            raw_output = generate_raw_output(
                model=model,
                processor=processor,
                image_path=image_path,
                backend=args.backend,
                max_new_tokens=args.max_new_tokens,
            )

            if args.show_raw:
                print("\n[RAW MODEL OUTPUT]")
                print(raw_output)
                print("[/RAW MODEL OUTPUT]\n")

            obj = parse_json_or_fields(raw_output)
            pred, parse_errors = normalize_prediction(obj)
            description = build_russian_description(pred)
            print_text_result(pred, description)

            if args.show_raw and parse_errors:
                print(f"[DEBUG] parse_errors: {parse_errors}")

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
