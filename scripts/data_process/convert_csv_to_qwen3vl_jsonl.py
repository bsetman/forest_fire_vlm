#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROMPT = (
    "<image>\n"
    "Analyze this wildfire image. Output only a valid JSON object with exactly these fields: "
    "Smoke, Fire, Fire_Size, Fire_Hotspots.\n"
    "Allowed values:\n"
    "Smoke: yes, no\n"
    "Fire: yes, no\n"
    "Fire_Size: small, large, cannot_determine, no_fire\n"
    "Fire_Hotspots: one_hotspot, multiple_hotspots, cannot_determine, no_fire\n"
    "Rules:\n"
    "1. If Smoke is no and Fire is no, set Fire_Size and Fire_Hotspots to no_fire.\n"
    "2. If Smoke is yes and Fire is no, set Fire_Size and Fire_Hotspots to cannot_determine.\n"
    "3. Output JSON only. Do not output markdown or explanations."
)


REQUIRED_COLUMNS = [
    "image_name",
    "image_path",
    "split",
    "Fire",
    "Smoke",
    "Fire_Size",
    "Fire_Hotspots",
]


FIELD_ORDER = [
    "Smoke",
    "Fire",
    "Fire_Size",
    "Fire_Hotspots",
]


ALLOWED_SPLITS = {
    "train",
    "val",
    "test",
}


ALLOWED_VALUES = {
    "Smoke": {"yes", "no"},
    "Fire": {"yes", "no"},
    "Fire_Size": {"small", "large", "cannot_determine", "no_fire"},
    "Fire_Hotspots": {"one_hotspot", "multiple_hotspots", "cannot_determine", "no_fire"},
}


def windows_to_wsl_path(path_str: str) -> str:
    """
    Convert:
        D:\\Abiyelunwen\\new_fire_vlm\\data\\train\\train_0001.jpg

    To:
        /mnt/d/Abiyelunwen/new_fire_vlm/data/train/train_0001.jpg
    """
    s = str(path_str).strip()
    s = s.replace("\\", "/")

    if len(s) >= 3 and s[1] == ":" and s[2] == "/":
        drive = s[0].lower()
        rest = s[3:]
        return f"/mnt/{drive}/{rest}"

    return s


def normalize_yes_no(value: Any, field: str, row_no: int) -> str:
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    mapping = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",

        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }

    if v not in mapping:
        raise ValueError(
            f"Row {row_no}: illegal {field} value: {value!r}. Allowed: yes/no"
        )

    return mapping[v]


def normalize_size(value: Any, row_no: int) -> str:
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    mapping = {
        "small": "small",
        "large": "large",

        "cannot_determine": "cannot_determine",
        "cannot_be_determined": "cannot_determine",
        "unknown": "cannot_determine",
        "uncertain": "cannot_determine",
        "not_clear": "cannot_determine",

        "no_fire": "no_fire",
        "no_fire_visible": "no_fire",
        "no_forest_fire_visible": "no_fire",
        "none": "no_fire",
        "no": "no_fire",
    }

    if v not in mapping:
        raise ValueError(
            f"Row {row_no}: illegal Fire_Size value: {value!r}. "
            f"Allowed: small/large/cannot_determine/no_fire"
        )

    return mapping[v]


def normalize_hotspots(value: Any, row_no: int) -> str:
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    mapping = {
        "one_hotspot": "one_hotspot",
        "single_hotspot": "one_hotspot",
        "one": "one_hotspot",
        "single": "one_hotspot",
        "1": "one_hotspot",

        "multiple_hotspots": "multiple_hotspots",
        "multiple": "multiple_hotspots",
        "multi": "multiple_hotspots",
        "many": "multiple_hotspots",
        "several": "multiple_hotspots",

        "cannot_determine": "cannot_determine",
        "cannot_be_determined": "cannot_determine",
        "unknown": "cannot_determine",
        "uncertain": "cannot_determine",
        "not_clear": "cannot_determine",

        "no_fire": "no_fire",
        "no_fire_visible": "no_fire",
        "no_forest_fire_visible": "no_fire",
        "none": "no_fire",
        "no": "no_fire",
    }

    if v not in mapping:
        raise ValueError(
            f"Row {row_no}: illegal Fire_Hotspots value: {value!r}. "
            f"Allowed: one_hotspot/multiple_hotspots/cannot_determine/no_fire"
        )

    return mapping[v]


def enforce_logic(label: Dict[str, str], row_no: int, changes: List[str]) -> Dict[str, str]:
    """
    为了和你之前 CSV -> BLIP2 JSONL 的逻辑保持一致，这里默认执行逻辑修正。

    Rules:
    1. Smoke=no and Fire=no -> Fire_Size=no_fire and Fire_Hotspots=no_fire
    2. Smoke=yes and Fire=no -> Fire_Size/Fire_Hotspots cannot_determine
    3. Fire=yes -> Fire_Size/Fire_Hotspots 不能是 no_fire
    """
    smoke = label["Smoke"]
    fire = label["Fire"]

    if smoke == "no" and fire == "no":
        if label["Fire_Size"] != "no_fire":
            changes.append(
                f"Row {row_no}: Smoke=no and Fire=no, Fire_Size {label['Fire_Size']} -> no_fire"
            )
            label["Fire_Size"] = "no_fire"

        if label["Fire_Hotspots"] != "no_fire":
            changes.append(
                f"Row {row_no}: Smoke=no and Fire=no, Fire_Hotspots {label['Fire_Hotspots']} -> no_fire"
            )
            label["Fire_Hotspots"] = "no_fire"

    elif smoke == "yes" and fire == "no":
        if label["Fire_Size"] == "no_fire":
            changes.append(
                f"Row {row_no}: Smoke=yes and Fire=no, Fire_Size no_fire -> cannot_determine"
            )
            label["Fire_Size"] = "cannot_determine"

        if label["Fire_Hotspots"] == "no_fire":
            changes.append(
                f"Row {row_no}: Smoke=yes and Fire=no, Fire_Hotspots no_fire -> cannot_determine"
            )
            label["Fire_Hotspots"] = "cannot_determine"

    elif fire == "yes":
        if label["Fire_Size"] == "no_fire":
            changes.append(
                f"Row {row_no}: Fire=yes, Fire_Size no_fire -> cannot_determine"
            )
            label["Fire_Size"] = "cannot_determine"

        if label["Fire_Hotspots"] == "no_fire":
            changes.append(
                f"Row {row_no}: Fire=yes, Fire_Hotspots no_fire -> cannot_determine"
            )
            label["Fire_Hotspots"] = "cannot_determine"

    return label


def build_qwen_record(image_path: str, target_text: str) -> Dict[str, Any]:
    """
    生成和原 convert_blip2_jsonl_to_qwen3vl_jsonl.py 完全一致的结构。
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            },
            {
                "role": "assistant",
                "content": target_text,
            },
        ],
        "images": [image_path],
    }


def convert_csv_to_qwen3vl(
    csv_path: Path,
    out_dir: Path,
    convert_to_wsl: bool = True,
    no_logic: bool = False,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    value_counts: Dict[str, Counter] = defaultdict(Counter)
    logic_changes: List[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. Found: {reader.fieldnames}"
            )

        for row_no, row in enumerate(reader, start=2):
            split = str(row["split"]).strip().lower()

            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Row {row_no}: illegal split={split!r}. Allowed: train/val/test"
                )

            image_name = str(row["image_name"]).strip()
            image_path = str(row["image_path"]).strip()

            if not image_name:
                raise ValueError(f"Row {row_no}: image_name is empty")

            if not image_path:
                raise ValueError(f"Row {row_no}: image_path is empty")

            if convert_to_wsl:
                image_path = windows_to_wsl_path(image_path)

            label = {
                "Smoke": normalize_yes_no(row["Smoke"], "Smoke", row_no),
                "Fire": normalize_yes_no(row["Fire"], "Fire", row_no),
                "Fire_Size": normalize_size(row["Fire_Size"], row_no),
                "Fire_Hotspots": normalize_hotspots(row["Fire_Hotspots"], row_no),
            }

            if not no_logic:
                label = enforce_logic(label, row_no, logic_changes)

            ordered_label = {field: label[field] for field in FIELD_ORDER}

            for field, value in ordered_label.items():
                if value not in ALLOWED_VALUES[field]:
                    raise ValueError(
                        f"Row {row_no}: normalized illegal {field}={value!r}"
                    )
                value_counts[field][value] += 1

            value_counts["split"][split] += 1

            # 关键：这里必须是紧凑 JSON，才能和之前 qwen3vl jsonl 完全一致
            target_text = json.dumps(
                ordered_label,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            qwen_record = build_qwen_record(
                image_path=image_path,
                target_text=target_text,
            )

            split_records[split].append(qwen_record)

    output_map = {
        "train": out_dir / "train_qwen3vl.jsonl",
        "val": out_dir / "val_qwen3vl.jsonl",
        "test": out_dir / "test_qwen3vl.jsonl",
    }

    for split, records in split_records.items():
        out_path = output_map[split]

        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[OK] {split}: {len(records)} samples -> {out_path}")

    stats = {
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "split_counts": {k: len(v) for k, v in split_records.items()},
        "value_counts": {k: dict(v) for k, v in value_counts.items()},
        "logic_changes_count": len(logic_changes),
        "logic_changes_preview": logic_changes[:50],
        "field_order": FIELD_ORDER,
        "prompt": PROMPT,
    }

    stats_path = out_dir / "qwen3vl_dataset_stats.json"

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] stats -> {stats_path}")

    print("\n[SUMMARY]")
    print(json.dumps(stats["split_counts"], ensure_ascii=False, indent=2))

    print("\n[VALUE_COUNTS]")
    print(json.dumps(stats["value_counts"], ensure_ascii=False, indent=2))

    if logic_changes:
        print(
            f"\n[INFO] Applied {len(logic_changes)} logic corrections. "
            f"See {stats_path} for preview."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly convert annotations_all.csv to Qwen3-VL JSONL files."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="/mnt/d/Abiyelunwen/new_fire_vlm/data/annotations_all.csv",
        help="Path to annotations_all.csv",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/d/Abiyelunwen/new_fire_vlm/data/qwen3vl",
        help="Output directory for train_qwen3vl.jsonl / val_qwen3vl.jsonl / test_qwen3vl.jsonl",
    )

    parser.add_argument(
        "--keep-windows-path",
        action="store_true",
        help="Keep original Windows paths instead of converting them to WSL paths.",
    )

    parser.add_argument(
        "--no-logic",
        action="store_true",
        help=(
            "Disable automatic logical corrections. "
            "Use this only if you want to preserve CSV labels exactly."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_csv_to_qwen3vl(
        csv_path=Path(args.csv).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        convert_to_wsl=not args.keep_windows_path,
        no_logic=args.no_logic,
    )


if __name__ == "__main__":
    main()