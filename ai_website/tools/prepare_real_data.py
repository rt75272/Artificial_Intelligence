#!/usr/bin/env python3
"""
Prepare a real-image dataset for fine-tuning the ultra model.

Usage examples:
  # Create an empty template you can fill with images
  python tools/prepare_real_data.py --create-template --output data_real

  # Split a folder containing class subfolders 'cats' and 'dogs'
  python tools/prepare_real_data.py --input /path/to/real_images --output data_real --val-ratio 0.2

  # Use symlinks instead of copying to save space
  python tools/prepare_real_data.py --input /path/to/real_images --output data_real --symlink

Requirements:
- Input directory structure:
    input/
      cats/  *.jpg|*.png|...
      dogs/  *.jpg|*.png|...
- Output directory created as:
    output/
      train/{cats,dogs}
      validation/{cats,dogs}

Notes:
- By default, copies files. Use --symlink to create symlinks instead.
- Optional --balance trims both classes to the same count (min of the two) before splitting.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CLASSES = ["cats", "dogs"]


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()  # Fast validation
        return True
    except Exception:
        return False


def collect_images(input_dir: Path, cls: str) -> List[Path]:
    cls_dir = input_dir / cls
    if not cls_dir.is_dir():
        raise FileNotFoundError(f"Expected class folder missing: {cls_dir}")
    files = [p for p in cls_dir.iterdir() if is_image_file(p)]
    return files


def ensure_output_structure(output_dir: Path):
    for split in ("train", "validation"):
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


def place(files: List[Path], dst_dir: Path, use_symlink: bool):
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            # Avoid collisions by adding a numeric suffix
            stem, suf = src.stem, src.suffix
            i = 1
            while True:
                cand = dst_dir / f"{stem}_{i}{suf}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1
        if use_symlink:
            if dst.exists():
                continue
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)


def create_template(output_dir: Path):
    for split in ("train", "validation"):
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created template at: {output_dir}")
    print("Fill these folders with your real images:")
    for split in ("train", "validation"):
        for cls in CLASSES:
            print(f"  - {output_dir / split / cls}")


def main():
    ap = argparse.ArgumentParser(description="Prepare real dataset for fine-tuning")
    ap.add_argument("--input", type=str, help="Input directory with 'cats' and 'dogs' subfolders")
    ap.add_argument("--output", type=str, default="data_real", help="Output dataset directory")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Copy files (default)")
    mode.add_argument("--symlink", action="store_true", help="Symlink files instead of copying")
    ap.add_argument("--balance", action="store_true", help="Balance classes by trimming to min count")
    ap.add_argument("--create-template", action="store_true", help="Only create empty output structure")
    args = ap.parse_args()

    output_dir = Path(args.output)

    if args.create_template:
        create_template(output_dir)
        return

    if not args.input:
        raise SystemExit("--input is required unless --create-template is used")

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Gather images
    random.seed(args.seed)
    files_by_cls = {}
    total_valid = 0
    for cls in CLASSES:
        files = collect_images(input_dir, cls)
        valid_files = []
        for f in files:
            if validate_image(f):
                valid_files.append(f)
            else:
                print(f"⚠️ Skipping unreadable image: {f}")
        random.shuffle(valid_files)
        files_by_cls[cls] = valid_files
        total_valid += len(valid_files)

    if args.balance:
        min_count = min(len(files_by_cls[c]) for c in CLASSES)
        for c in CLASSES:
            files_by_cls[c] = files_by_cls[c][:min_count]
        print(f"⚖️ Balanced classes to {min_count} images each")

    ensure_output_structure(output_dir)

    # Split and place
    val_ratio = max(0.0, min(0.9, args.val_ratio))
    use_symlink = True if args.symlink else False

    summary = {}
    for cls in CLASSES:
        files = files_by_cls[cls]
        n = len(files)
        n_val = int(round(n * val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

        place(train_files, output_dir / "train" / cls, use_symlink)
        place(val_files, output_dir / "validation" / cls, use_symlink)

        summary[cls] = {
            "train": len(train_files),
            "validation": len(val_files),
            "total": n,
        }

    # Report
    mode_str = "symlinked" if use_symlink else "copied"
    print("✅ Real dataset prepared")
    print(f"   output: {output_dir} ({mode_str})  val_ratio={val_ratio}")
    for cls in CLASSES:
        s = summary[cls]
        print(f"   {cls:<4}: train={s['train']:4d}  val={s['validation']:4d}  total={s['total']:4d}")
    print("\nNext:")
    print(f"  ULTRA_REAL_DATA_DIR={output_dir} ./AI/bin/python train_ultra_model.py")


if __name__ == "__main__":
    main()
