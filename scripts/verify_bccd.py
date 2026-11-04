#!/usr/bin/env python3
"""Integrity check for the on-disk BCCD VOC dataset."""
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import cv2

ROOT = "/home/archryzen/bccd/datasets/bccd"
IMG_DIR = os.path.join(ROOT, "JPEGImages")
ANN_DIR = os.path.join(ROOT, "Annotations")
SPLIT_DIR = os.path.join(ROOT, "ImageSets", "Main")
CLASSES = {"RBC", "WBC", "Platelets"}

images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg"))
anns = sorted(f for f in os.listdir(ANN_DIR) if f.lower().endswith(".xml"))
img_stems = {os.path.splitext(f)[0] for f in images}
ann_stems = {os.path.splitext(f)[0] for f in anns}

report = {
    "image_count": len(images),
    "annotation_count": len(anns),
    "images_without_xml": sorted(img_stems - ann_stems),
    "xml_without_image": sorted(ann_stems - img_stems),
    "malformed_xml": [],
    "unknown_classes": [],
    "out_of_bounds": [],
    "zero_or_negative_area": [],
    "size_mismatch": [],
    "total_objects": 0,
    "objects_per_class": Counter(),
}

for f in anns:
    path = os.path.join(ANN_DIR, f)
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        report["malformed_xml"].append((f, repr(e)))
        continue

    stem = os.path.splitext(f)[0]
    img_path = os.path.join(IMG_DIR, stem + ".jpg")
    img = cv2.imread(img_path) if os.path.exists(img_path) else None
    if img is not None:
        ih, iw = img.shape[:2]
    else:
        size = root.find("size")
        iw = int(float(size.findtext("width", "0")))
        ih = int(float(size.findtext("height", "0")))

    size = root.find("size")
    if size is not None:
        xw = int(float(size.findtext("width", "640")))
        xh = int(float(size.findtext("height", "480")))
        if (xw, xh) != (iw, ih):
            report["size_mismatch"].append((f, (xw, xh), (iw, ih)))

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        report["total_objects"] += 1
        report["objects_per_class"][name] += 1
        if name not in CLASSES:
            report["unknown_classes"].append((f, name))
        bb = obj.find("bndbox")
        if bb is None:
            report["zero_or_negative_area"].append((f, name, "missing bndbox"))
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            report["zero_or_negative_area"].append((f, name, "unparsable bndbox"))
            continue
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            report["zero_or_negative_area"].append((f, name, (xmin, ymin, xmax, ymax)))
        if xmin < 0 or ymin < 0 or xmax > iw or ymax > ih:
            report["out_of_bounds"].append((f, name, (xmin, ymin, xmax, ymax), (iw, ih)))

# Split integrity
split_names = {}
for split in ("train", "val", "test"):
    p = os.path.join(SPLIT_DIR, split + ".txt")
    with open(p) as fh:
        entries = [ln.strip() for ln in fh if ln.strip()]
    stems = [os.path.splitext(e)[0] for e in entries]
    split_names[split] = stems

overlaps = {}
seen = defaultdict(list)
for split, stems in split_names.items():
    for s in stems:
        seen[s].append(split)
overlaps = {s: v for s, v in seen.items() if len(v) > 1}

missing_from_splits = sorted(img_stems - set(seen))
split_unknown = {s: sorted(set(stems) - img_stems) for s, stems in split_names.items()}
dupes_within = {s: sorted([k for k, v in Counter(stems).items() if v > 1])
                for s, stems in split_names.items()}

print("=== INTEGRITY ===")
print("images:", report["image_count"])
print("annotations:", report["annotation_count"])
print("images_without_xml:", report["images_without_xml"])
print("xml_without_image:", report["xml_without_image"])
print("malformed_xml:", report["malformed_xml"])
print("unknown_classes:", report["unknown_classes"])
print("out_of_bounds count:", len(report["out_of_bounds"]))
for row in report["out_of_bounds"][:20]:
    print("  OOB", row)
print("zero_or_negative_area count:", len(report["zero_or_negative_area"]))
for row in report["zero_or_negative_area"][:20]:
    print("  BAD", row)
print("size_mismatch count:", len(report["size_mismatch"]))
for row in report["size_mismatch"][:20]:
    print("  SIZE", row)
print("total_objects:", report["total_objects"])
print("objects_per_class:", dict(report["objects_per_class"]))

print("=== SPLITS ===")
for s, stems in split_names.items():
    print(f"{s}: {len(stems)} entries ({len(set(stems))} unique)")
print("cross_split_overlaps:", overlaps)
print("images_not_in_any_split:", missing_from_splits)
print("split_entries_without_image:", split_unknown)
print("duplicates_within_split:", dupes_within)
