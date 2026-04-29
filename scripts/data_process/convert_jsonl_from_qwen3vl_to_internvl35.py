#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Qwen3-VL training JSONL to InternVL3.5-compatible datasets.

It creates two formats:
1) Axolotl multimodal chat_template format:
   {"messages": [{"role":"user","content":[{"type":"image","path":"..."},{"type":"text","text":"..."}]}, ...]}
2) InternVL official conversations format:
   {"id":"...","image":"...","conversations":[{"from":"human","value":"<image>\n..."}, ...]}

Recommended for this project:
- Use Axolotl format for QLoRA .
- Keep prompt and target text identical to Qwen as much as possible.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def windows_to_wsl_path(path_str: str) -> str:
    s = str(path_str).strip().replace('\\', '/')
    if len(s) >= 3 and s[1] == ':' and s[2] == '/':
        return f"/mnt/{s[0].lower()}/{s[3:]}"
    return s


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def extract_pair(item: Dict[str, Any]) -> Tuple[str, str]:
    """Return user prompt and assistant target from Qwen messages."""
    messages = item.get('messages')
    if not isinstance(messages, list):
        raise ValueError('Missing messages list')

    user_text = None
    assistant_text = None
    for m in messages:
        role = m.get('role')
        content = m.get('content', '')
        if role == 'user' and user_text is None:
            user_text = str(content)
        elif role == 'assistant' and assistant_text is None:
            assistant_text = str(content)

    if user_text is None or assistant_text is None:
        raise ValueError('Missing user or assistant message')

    return user_text, assistant_text


def extract_image(item: Dict[str, Any]) -> str:
    images = item.get('images')
    if isinstance(images, list) and images:
        return windows_to_wsl_path(images[0])
    image = item.get('image') or item.get('image_path')
    if image:
        return windows_to_wsl_path(image)
    raise ValueError('Missing image path')


def strip_image_token(text: str) -> str:
    return text.replace('<image>\n', '').replace('<image>', '').strip()


def ensure_image_token(text: str) -> str:
    text = text.strip()
    if '<image>' in text:
        return text
    return '<image>\n' + text


def convert_one(item: Dict[str, Any], idx: int):
    user_text, assistant_text = extract_pair(item)
    image_path = extract_image(item)

    sample_id = item.get('id') or item.get('sample_id') or Path(image_path).stem or f'sample_{idx:06d}'

    # Axolotl: explicit multimodal content list.
    axolotl = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'path': image_path},
                    {'type': 'text', 'text': strip_image_token(user_text)},
                ],
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': assistant_text},
                ],
            },
        ]
    }

    # Official InternVL: conversations + <image> token in the human text.
    official = {
        'id': str(sample_id),
        'image': image_path,
        'conversations': [
            {'from': 'human', 'value': ensure_image_token(strip_image_token(user_text))},
            {'from': 'gpt', 'value': assistant_text},
        ],
    }

    return axolotl, official


def convert_file(src: Path, axolotl_out: Path, official_out: Path) -> int:
    axolotl_out.parent.mkdir(parents=True, exist_ok=True)
    official_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with axolotl_out.open('w', encoding='utf-8') as fax, official_out.open('w', encoding='utf-8') as fof:
        for idx, item in enumerate(read_jsonl(src), 1):
            ax, off = convert_one(item, idx)
            fax.write(json.dumps(ax, ensure_ascii=False) + '\n')
            fof.write(json.dumps(off, ensure_ascii=False) + '\n')
            n += 1
    return n


def write_meta(project_root: Path, train_official: Path, val_official: Path, out_dir: Path, train_len: int, val_len: int, max_dynamic_patch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_train = {
        'fire_structured_train': {
            'root': '/',
            'annotation': str(train_official),
            'data_augment': False,
            'max_dynamic_patch': max_dynamic_patch,
            'repeat_time': 1,
            'length': train_len,
        }
    }
    meta_val = {
        'fire_structured_val': {
            'root': '/',
            'annotation': str(val_official),
            'data_augment': False,
            'max_dynamic_patch': max_dynamic_patch,
            'repeat_time': 1,
            'length': val_len,
        }
    }
    (out_dir / 'internvl35_train_meta.json').write_text(json.dumps(meta_train, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'internvl35_val_meta.json').write_text(json.dumps(meta_val, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen-train', default='/mnt/d/Abiyelunwen/new_fire_vlm/data/qwen3vl/train_qwen3vl.jsonl')
    parser.add_argument('--qwen-val', default='/mnt/d/Abiyelunwen/new_fire_vlm/data/qwen3vl/val_qwen3vl.jsonl')
    parser.add_argument('--out-dir', default='/mnt/d/Abiyelunwen/new_fire_vlm/data/internvl35')
    parser.add_argument('--project-root', default='/mnt/d/Abiyelunwen/new_fire_vlm')
    parser.add_argument('--max-dynamic-patch', type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_ax = out_dir / 'train_internvl35_axolotl.jsonl'
    val_ax = out_dir / 'val_internvl35_axolotl.jsonl'
    train_off = out_dir / 'train_internvl35_official.jsonl'
    val_off = out_dir / 'val_internvl35_official.jsonl'

    train_len = convert_file(Path(args.qwen_train), train_ax, train_off)
    val_len = convert_file(Path(args.qwen_val), val_ax, val_off)
    write_meta(Path(args.project_root), train_off, val_off, out_dir, train_len, val_len, args.max_dynamic_patch)

    print('[DONE]')
    print(f'train samples = {train_len}')
    print(f'val samples   = {val_len}')
    print(f'Axolotl train = {train_ax}')
    print(f'Axolotl val   = {val_ax}')
    print(f'Official train= {train_off}')
    print(f'Official val  = {val_off}')
    print(f'Meta train    = {out_dir / "internvl35_train_meta.json"}')
    print(f'Meta val      = {out_dir / "internvl35_val_meta.json"}')


if __name__ == '__main__':
    main()
