import argparse
import csv
import re
from pathlib import Path
from uuid import uuid4


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def build_name(pattern: str, index: int, ext: str) -> str:
    stem = pattern.format(index=index)
    return f"{stem}{ext.lower()}"


def find_next_index(directory: Path, pattern: str) -> int:
    name_re = build_pattern_regex(pattern)

    max_index = -1
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = name_re.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def build_pattern_regex(pattern: str) -> re.Pattern:
    sample = pattern.format(index=123456789)
    escaped = re.escape(sample).replace("123456789", r"(\d+)")
    return re.compile(rf"^{escaped}\.[^.]+$", re.IGNORECASE)


def is_already_renamed(path: Path, pattern: str) -> bool:
    return build_pattern_regex(pattern).match(path.name) is not None


def collect_images(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


def collect_unrenamed_images(directory: Path, recursive: bool, pattern: str) -> list[Path]:
    return [
        path
        for path in collect_images(directory, recursive)
        if not is_already_renamed(path, pattern)
    ]


def select_segment(images: list[Path], offset: int, count: int | None) -> list[Path]:
    if offset < 0:
        raise ValueError("--offset must be greater than or equal to 0")
    if count is None:
        return images[offset:]
    if count < 0:
        raise ValueError("--count must be greater than or equal to 0")
    return images[offset : offset + count]


def rename_images(
    directory: Path,
    output_csv: Path,
    pattern: str,
    start: int | None,
    offset: int,
    count: int | None,
    recursive: bool,
    dry_run: bool,
):
    if "{index" not in pattern:
        raise ValueError('Pattern must contain "{index}", for example "train_{index:04d}"')

    images = collect_unrenamed_images(directory, recursive, pattern)
    selected = select_segment(images, offset, count)
    current_index = find_next_index(directory, pattern) if start is None else start

    planned = []
    used_targets = set()
    for source in selected:
        target = source.with_name(build_name(pattern, current_index, source.suffix))
        current_index += 1

        if target in used_targets:
            raise FileExistsError(f"Duplicate target name planned: {target}")
        used_targets.add(target)

        if source.resolve() != target.resolve() and target.exists():
            raise FileExistsError(f"Target file already exists: {target}")

        planned.append((source, target))

    if not dry_run:
        temp_pairs = []
        for source, target in planned:
            temp = source.with_name(f".rename_tmp_{uuid4().hex}{source.suffix}")
            source.rename(temp)
            temp_pairs.append((source, temp, target))

        for _source, temp, target in temp_pairs:
            temp.rename(target)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["old_name", "new_name", "old_path", "new_path"])
        for source, target in planned:
            writer.writerow([source.name, target.name, str(source), str(target)])

    return planned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename image files to a unified numbered format and create a CSV mapping."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Image directory. Default: current directory.",
    )
    parser.add_argument(
        "--pattern",
        default="train_{index:04d}",
        help='Output filename pattern without extension. Default: "train_{index:04d}".',
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index. If omitted, continue after the largest existing matching filename.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many images before renaming. Useful for segmented batches.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Rename only this many images in the current segment.",
    )
    parser.add_argument(
        "--csv",
        default="rename_mapping.csv",
        help="CSV output path. Default: rename_mapping.csv.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create the CSV preview; do not rename files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    directory = Path(args.directory).resolve()
    output_csv = Path(args.csv).resolve()

    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {directory}")

    planned = rename_images(
        directory=directory,
        output_csv=output_csv,
        pattern=args.pattern,
        start=args.start,
        offset=args.offset,
        count=args.count,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )

    action = "Previewed" if args.dry_run else "Renamed"
    print(f"{action} {len(planned)} image(s). CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
