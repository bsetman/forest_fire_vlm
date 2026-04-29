import argparse
import json
from pathlib import Path

import pandas as pd


TEXT_PROMPT = (
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


ALLOWED_VALUES = {
    "Smoke": {"yes", "no"},
    "Fire": {"yes", "no"},
    "Fire_Size": {"small", "large", "cannot_determine", "no_fire"},
    "Fire_Hotspots": {"one_hotspot", "multiple_hotspots", "cannot_determine", "no_fire"},
}


def normalize_value(x):
    return str(x).strip().lower()


def win_path_to_wsl(path):
    path = str(path).strip()

    if len(path) >= 3 and path[1] == ":" and path[2] in ["\\", "/"]:
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    return path.replace("\\", "/")


def validate_row(row, idx):
    errors = []

    for col in REQUIRED_COLUMNS:
        if col not in row:
            errors.append(f"missing column: {col}")

    for field, allowed in ALLOWED_VALUES.items():
        value = normalize_value(row.get(field, ""))
        if value not in allowed:
            errors.append(
                f"invalid {field}={value!r}, allowed={sorted(allowed)}"
            )

    fire = normalize_value(row.get("Fire", ""))
    fire_size = normalize_value(row.get("Fire_Size", ""))
    hotspots = normalize_value(row.get("Fire_Hotspots", ""))

    if fire == "no":
        if fire_size != "no_fire":
            errors.append(f"Fire=no but Fire_Size={fire_size}")
        if hotspots != "no_fire":
            errors.append(f"Fire=no but Fire_Hotspots={hotspots}")

    if fire == "yes":
        if fire_size == "no_fire":
            errors.append("Fire=yes but Fire_Size=no_fire")
        if hotspots == "no_fire":
            errors.append("Fire=yes but Fire_Hotspots=no_fire")

    if errors:
        image_name = row.get("image_name", "UNKNOWN")
        raise ValueError(
            f"Row {idx}, image={image_name}: " + "; ".join(errors)
        )


def build_test_eval_jsonl(csv_path, output_path, split_name="test", convert_to_wsl=True):
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    split_name = split_name.lower()

    test_df = df[df["split"] == split_name].copy()

    if len(test_df) == 0:
        raise ValueError(f"No rows found with split == {split_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for idx, row in test_df.iterrows():
        row_dict = row.to_dict()
        validate_row(row_dict, idx)

        image_name = str(row_dict["image_name"]).strip()
        image_path = str(row_dict["image_path"]).strip()

        if convert_to_wsl:
            image_path = win_path_to_wsl(image_path)

        gold = {
            "Smoke": normalize_value(row_dict["Smoke"]),
            "Fire": normalize_value(row_dict["Fire"]),
            "Fire_Size": normalize_value(row_dict["Fire_Size"]),
            "Fire_Hotspots": normalize_value(row_dict["Fire_Hotspots"]),
        }

        record = {
            "id": Path(image_name).stem,
            "image_name": image_name,
            "image_path": image_path,
            "split": split_name,
            "prompt": TEXT_PROMPT,
            "gold": gold,
        }

        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("[DONE] Unified test evaluation JSONL generated")
    print(f"Input CSV:    {csv_path}")
    print(f"Output JSONL: {output_path}")
    print(f"Samples:      {len(records)}")
    print("=" * 80)

    print("\nLabel distribution:")
    for field in ["Smoke", "Fire", "Fire_Size", "Fire_Hotspots"]:
        print(f"\n[{field}]")
        counts = test_df[field].astype(str).str.strip().str.lower().value_counts()
        for label, count in counts.items():
            ratio = count / len(test_df) * 100
            print(f"  {label:20s} {count:5d}  {ratio:6.2f}%")

    print("\nFirst sample:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="/mnt/d/Abiyelunwen/new_fire_vlm/data/annotations_all.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/d/Abiyelunwen/new_fire_vlm/data/eval/test_eval.jsonl",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--keep-windows-path",
        action="store_true",
    )

    args = parser.parse_args()

    build_test_eval_jsonl(
        csv_path=args.csv,
        output_path=args.output,
        split_name=args.split,
        convert_to_wsl=not args.keep_windows_path,
    )


if __name__ == "__main__":
    main()