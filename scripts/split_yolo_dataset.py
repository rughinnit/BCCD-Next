#!/usr/bin/env python3
"""Split a single-split YOLO export into train/valid/test with class stratification.

Roboflow exports sometimes ship every image in ``train/`` even though ``data.yaml``
declares val/test. This moves the existing image/label pairs into three folders,
keeping each class present in every split, and rewrites the ``data.yaml`` split keys.

Usage:
    python scripts/split_yolo_dataset.py datasets/roboflow/models-u4dak__bccd-blw1i__v1
    python scripts/split_yolo_dataset.py <root> --ratios 0.7 0.15 0.15 --seed 0 --dry-run

The script refuses to run when val/ or test/ already contain images, unless --force
is passed. Image/label pairs are moved, never copied, so the dataset does not grow.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if fields:
            classes.append(int(float(fields[0])))
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset root containing train/images + train/labels")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting existing valid/ and test/ contents")
    parser.add_argument("--dry-run", action="store_true", help="report the planned split only")
    args = parser.parse_args()

    root: Path = args.root
    source = root / "train"
    images = sorted(p for p in (source / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return fail(f"no images found in {source / 'images'}")

    if sum(args.ratios) <= 0:
        return fail("ratios must sum to a positive number")

    existing = {s: len(list((root / s / "images").glob("*"))) for s in SPLITS[1:] if (root / s / "images").is_dir()}
    if any(existing.values()) and not args.force:
        return fail(f"valid/ or test/ already populated {existing}; pass --force to re-split")

    class_counts: Counter[int] = Counter()
    per_image: dict[Path, list[int]] = {}
    for image in images:
        classes = label_classes(source / "labels" / f"{image.stem}.txt")
        per_image[image] = classes
        class_counts.update(set(classes))

    # Stratify on the rarest class present in each image so small classes are not
    # swallowed by the RBC majority.
    groups: dict[int | str, list[Path]] = defaultdict(list)
    for image, classes in per_image.items():
        key: int | str = min(classes, key=lambda c: class_counts[c]) if classes else "empty"
        groups[key].append(image)

    rng = random.Random(args.seed)
    assigned: dict[str, list[Path]] = {s: [] for s in SPLITS}
    total = len(images)
    for key in sorted(groups, key=str):
        items = sorted(groups[key])
        rng.shuffle(items)
        n_train = round(len(items) * args.ratios[0])
        n_valid = round(len(items) * args.ratios[1])
        n_train = min(n_train, max(0, len(items) - 2)) if len(items) >= 3 else len(items)
        n_valid = min(n_valid, max(0, len(items) - n_train - 1)) if len(items) >= 3 else 0
        assigned["train"].extend(items[:n_train])
        assigned["valid"].extend(items[n_train:n_train + n_valid])
        assigned["test"].extend(items[n_train + n_valid:])

    print(f"{root.name}: {total} images, {len(class_counts)} classes")
    for split in SPLITS:
        print(f"  {split}: {len(assigned[split])} ({len(assigned[split]) / total:.1%})")

    if args.dry_run:
        per_split_classes = {
            split: Counter(c for image in items for c in set(per_image[image]))
            for split, items in assigned.items()
        }
        for split, counts in per_split_classes.items():
            print(f"  {split} classes: {dict(sorted(counts.items()))}")
        return 0

    for split in SPLITS:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    if args.force:
        for split in SPLITS[1:]:
            for folder in ("images", "labels"):
                for stale in (root / split / folder).glob("*"):
                    stale.unlink()

    moved = 0
    for split in SPLITS:
        for image in assigned[split]:
            label = source / "labels" / f"{image.stem}.txt"
            shutil.move(str(image), str(root / split / "images" / image.name))
            if label.exists():
                shutil.move(str(label), str(root / split / "labels" / label.name))
            moved += 1

    pairs = {
        split: (
            len(list((root / split / "images").glob("*"))),
            len(list((root / split / "labels").glob("*.txt"))),
        )
        for split in SPLITS
    }
    for split, (n_img, n_lbl) in pairs.items():
        status = "ok" if n_img == n_lbl else "MISMATCH"
        print(f"  wrote {split}: {n_img} images, {n_lbl} labels [{status}]")

    update_yaml(root, args.ratios)
    print(f"moved {moved} pairs; data.yaml split keys rewritten")
    return 0 if all(a == b for a, b in pairs.values()) else 1


def update_yaml(root: Path, ratios: list[float]) -> None:
    yaml_path = root / "data.yaml"
    text = yaml_path.read_text()
    lines = text.splitlines()
    wanted = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
    }
    seen = set()
    out = []
    for line in lines:
        match = re.match(r"^(train|val|test)\s*:", line)
        if match:
            key = match.group(1)
            seen.add(key)
            out.append(f"{key}: {wanted[key]}")
        else:
            out.append(line)
    for key in ("train", "val", "test"):
        if key not in seen:
            out.insert(0, f"{key}: {wanted[key]}")
    note = f"# split: stratified {'/'.join(f'{r:.2f}' for r in ratios)} of the original single train split"
    if note not in out:
        out.append("")
        out.append(note)
    yaml_path.write_text("\n".join(out) + "\n")


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
