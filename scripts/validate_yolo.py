#!/usr/bin/env python3
"""Validate generated YOLO labels against the original VOC boxes."""
import glob
import json
import os
import random
import xml.etree.ElementTree as ET

import cv2

ROOT = "/home/archryzen/bccd/datasets/bccd"
IMG_DIR = os.path.join(ROOT, "JPEGImages")
ANN_DIR = os.path.join(ROOT, "Annotations")
OUT = "/home/archryzen/bccd/datasets/bccd_yolo"
CLASSES = ["RBC", "WBC", "Platelets"]
SAMPLE = 10
random.seed(20261103)


def voc_boxes(stem):
    root = ET.parse(os.path.join(ANN_DIR, stem + ".xml")).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        box = tuple(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if name in CLASSES and box[2] > box[0] and box[3] > box[1]:
            boxes.append((name, box))
    return boxes


def check(stem, split):
    label_path = os.path.join(OUT, "labels", split, stem + ".txt")
    img = cv2.imread(os.path.join(IMG_DIR, stem + ".jpg"))
    ih, iw = img.shape[:2]
    with open(label_path) as fh:
        rows = [ln.split() for ln in fh if ln.strip()]

    recon = []
    for row in rows:
        assert len(row) == 5, f"{stem}: bad row {row}"
        cls = int(row[0])
        vals = [float(v) for v in row[1:]]
        assert cls in range(3), f"{stem}: bad class {cls}"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{stem}: out of range {vals}"
        cx, cy, w, h = vals
        box = ((cx - w / 2) * iw, (cy - h / 2) * ih,
               (cx + w / 2) * iw, (cy + h / 2) * ih)
        recon.append((CLASSES[cls], box))

    orig = voc_boxes(stem)
    assert len(recon) == len(orig), f"{stem}: count {len(recon)} != {len(orig)}"

    # Greedy match each reconstructed box to the nearest unused original box
    # of the same class (center distance), then measure edge deviation.
    unmatched = list(orig)
    max_dev = 0.0
    pairs = []
    for name, box in recon:
        candidates = [o for o in unmatched if o[0] == name]
        assert candidates, f"{stem}: no VOC box left for {name}"
        rc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        best = min(candidates, key=lambda o: (rc[0] - (o[1][0] + o[1][2]) / 2) ** 2
                   + (rc[1] - (o[1][1] + o[1][3]) / 2) ** 2)
        unmatched.remove(best)
        dev = max(abs(a - b) for a, b in zip(box, best[1]))
        max_dev = max(max_dev, dev)
        pairs.append((name, box, best[1], dev))
    assert max_dev <= 1.0, f"{stem}: deviation {max_dev} > 1 px"
    return max_dev, pairs


rng = random.Random(20261103)
all_labels = sorted(glob.glob(os.path.join(OUT, "labels", "*", "*.txt")))
sample = rng.sample(all_labels, SAMPLE)

print(f"=== RANDOM SAMPLE ({SAMPLE} files) ===")
sample_max = 0.0
for lp in sample:
    split = os.path.basename(os.path.dirname(lp))
    stem = os.path.splitext(os.path.basename(lp))[0]
    dev, pairs = check(stem, split)
    sample_max = max(sample_max, dev)
    print(f"{split}/{stem}: {len(pairs)} objects, max dev {dev:.9f} px")

print(f"sample max deviation: {sample_max:.9f} px")

# Full-dataset cross-check as an extra guarantee.
full_max = 0.0
full_files = 0
for lp in all_labels:
    split = os.path.basename(os.path.dirname(lp))
    stem = os.path.splitext(os.path.basename(lp))[0]
    dev, _ = check(stem, split)
    full_max = max(full_max, dev)
    full_files += 1
print(f"=== FULL CHECK ({full_files} label files) ===")
print(f"full max deviation: {full_max:.9f} px")
assert full_max <= 1.0
