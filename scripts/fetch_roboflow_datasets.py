#!/usr/bin/env python3
"""Fetch the Roboflow Universe BCCD copies used by this repo.

Downloads YOLOv8 exports from the Roboflow export API into
``datasets/roboflow/<workspace>__<project>__v<version>`` and optionally
re-verifies the extracted image/label counts. Nothing is trained or converted.

Usage:
    export ROBOFLOW_API_KEY=...        # PowerShell: $env:ROBOFLOW_API_KEY = '...'
    python scripts/fetch_roboflow_datasets.py
    python scripts/fetch_roboflow_datasets.py --verify
    python scripts/fetch_roboflow_datasets.py --only roboflow-100/bccd-ouzjz --force

The API key is read from the environment only and is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "datasets" / "roboflow"
ZIP_ROOT = OUT_ROOT / "_zips"
API = "https://api.roboflow.com"

# workspace/project/version -> expected image count after extraction
DATASETS: list[dict[str, object]] = [
    {
        "workspace": "roboflow-100",
        "project": "bccd-ouzjz",
        "version": 2,
        "format": "yolov8",
        "images": 364,
        "license": "CC BY 4.0",
    },
    {
        "workspace": "models-u4dak",
        "project": "bccd-blw1i",
        "version": 1,
        "format": "yolov8",
        "images": 1090,
        "license": "CC BY 4.0",
    },
    {
        "workspace": "quan-c2r6k",
        "project": "bccd-ccnkh",
        "version": 1,
        "format": "yolov8",
        "images": 8599,
        "license": "Public Domain",
    },
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIXES = {".txt"}
README_NAMES = {"readme.dataset.txt", "readme.roboflow.txt"}


def slug(entry: dict[str, object]) -> str:
    return f"{entry['workspace']}__{entry['project']}__v{entry['version']}"


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set.\n"
            "Get a free key at app.roboflow.com -> Settings -> API Keys, then:\n"
            "  PowerShell: $env:ROBOFLOW_API_KEY = '<key>'\n"
            "  bash:       export ROBOFLOW_API_KEY='<key>'"
        )
    return key


def export_link(entry: dict[str, object], key: str) -> str:
    url = (
        f"{API}/{entry['workspace']}/{entry['project']}/{entry['version']}"
        f"/{entry['format']}?api_key={key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        sys.exit(f"{slug(entry)}: HTTP {exc.code} from export API: {body}")
    except urllib.error.URLError as exc:
        sys.exit(f"{slug(entry)}: network error: {exc.reason}")

    link = payload.get("export", {}).get("link")
    if not link:
        sys.exit(f"{slug(entry)}: no export link in response: {payload}")
    return link


def download(entry: dict[str, object], key: str, force: bool) -> Path:
    name = slug(entry)
    target = OUT_ROOT / name
    if target.exists() and not force:
        print(f"{name}: already extracted, skipping (use --force to replace)")
        return target

    ZIP_ROOT.mkdir(parents=True, exist_ok=True)
    archive = ZIP_ROOT / f"{name}.zip"

    print(f"{name}: requesting export link")
    link = export_link(entry, key)
    print(f"{name}: downloading -> {archive.name}")
    with urllib.request.urlopen(link, timeout=900) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)

    size_mb = archive.stat().st_size / (1 << 20)
    print(f"{name}: {size_mb:.1f} MB on disk, extracting")

    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)

    return target


def count_split(split_dir: Path) -> tuple[int, int]:
    images = sum(
        1
        for path in split_dir.glob("images/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sum(
        1
        for path in split_dir.glob("labels/*")
        if path.is_file() and path.suffix.lower() in LABEL_SUFFIXES
        and path.name.lower() not in README_NAMES
    )
    return images, labels


def verify(entry: dict[str, object]) -> bool:
    name = slug(entry)
    target = OUT_ROOT / name
    if not target.exists():
        print(f"{name}: MISSING (run without --verify first)")
        return False

    splits = sorted(
        child.name
        for child in target.iterdir()
        if child.is_dir() and (child / "images").is_dir()
    )
    total_images = 0
    total_labels = 0
    ok = True
    for split in splits:
        images, labels = count_split(target / split)
        total_images += images
        total_labels += labels
        status = "ok" if images == labels else "MISMATCH"
        if images != labels:
            ok = False
        print(f"{name}/{split}: {images} images, {labels} labels [{status}]")

    expected = int(entry["images"])  # type: ignore[arg-type]
    if total_images != expected:
        print(f"{name}: expected {expected} images, found {total_images}")
        ok = False

    print(f"{name}: {total_images} images, {total_labels} labels, license {entry['license']}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="only re-check extracted counts")
    parser.add_argument("--force", action="store_true", help="re-download and replace existing folders")
    parser.add_argument(
        "--only",
        metavar="WORKSPACE/PROJECT",
        help="limit to one project, e.g. roboflow-100/bccd-ouzjz",
    )
    args = parser.parse_args()

    selected = DATASETS
    if args.only:
        wanted = args.only.strip()
        selected = [
            entry
            for entry in DATASETS
            if f"{entry['workspace']}/{entry['project']}" == wanted
        ]
        if not selected:
            sys.exit(f"--only {wanted} does not match any configured dataset")

    if args.verify:
        results = [verify(entry) for entry in selected]
        return 0 if all(results) else 1

    key = api_key()
    for entry in selected:
        download(entry, key, args.force)

    print()
    results = [verify(entry) for entry in selected]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
