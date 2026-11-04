#!/usr/bin/env python3
"""Build a clean YOLO-format dataset from the BCCD VOC annotations."""
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import cv2

ROOT = "/home/archryzen/bccd/datasets/bccd"
IMG_DIR = os.path.join(ROOT, "JPEGImages")
ANN_DIR = os.path.join(ROOT, "Annotations")
SPLIT_DIR = os.path.join(ROOT, "ImageSets", "Main")
OUT = "/home/archryzen/bccd/datasets/bccd_yolo"
CLASSES = ["RBC", "WBC", "Platelets"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
SPLITS = ("train", "val", "test")


def parse_voc(xml_path):
    """Return (image_name, width, height, [(cls_name, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        box = tuple(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((name, *box))
    return filename, width, height, objects


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for split in SPLITS:
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    stats = {s: Counter() for s in SPLITS}
    skipped_images = []
    dropped_objects = []
    image_sizes = {}

    for split in SPLITS:
        with open(os.path.join(SPLIT_DIR, split + ".txt")) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        for entry in names:
            stem = os.path.splitext(entry)[0]
            img_src = os.path.join(IMG_DIR, stem + ".jpg")
            xml_src = os.path.join(ANN_DIR, stem + ".xml")

            img = cv2.imread(img_src)
            ih, iw = img.shape[:2]
            _, xml_w, xml_h, objects = parse_voc(xml_src)
            if (xml_w, xml_h) != (iw, ih):
                raise SystemExit(f"size mismatch for {stem}: xml {(xml_w, xml_h)} vs img {(iw, ih)}")

            lines = []
            for name, xmin, ymin, xmax, ymax in objects:
                if name not in CLASS_ID:
                    dropped_objects.append((stem, name, "unknown class"))
                    continue
                # Guard against degenerate / out-of-range boxes.
                xmin = min(max(xmin, 0.0), iw)
                xmax = min(max(xmax, 0.0), iw)
                ymin = min(max(ymin, 0.0), ih)
                ymax = min(max(ymax, 0.0), ih)
                if xmax - xmin <= 0 or ymax - ymin <= 0:
                    dropped_objects.append((stem, name, "zero/negative area"))
                    continue
                cx = (xmin + xmax) / 2.0 / iw
                cy = (ymin + ymax) / 2.0 / ih
                w = (xmax - xmin) / iw
                h = (ymax - ymin) / ih
                cls = CLASS_ID[name]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[split][name] += 1

            if not lines:
                skipped_images.append((split, stem, "no valid objects"))
                continue

            shutil.copy2(img_src, os.path.join(OUT, "images", split, stem + ".jpg"))
            with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + "\n")
            image_sizes[stem] = (iw, ih)

    # Config
    os.makedirs("/home/archryzen/bccd/configs", exist_ok=True)
    with open("/home/archryzen/bccd/configs/bccd.yaml", "w") as fh:
        fh.write(
            "# BCCD blood-cell detection dataset (YOLO format)\n"
            f"path: {OUT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "nc: 3\n"
            "names: ['RBC', 'WBC', 'Platelets']\n"
        )

    report = {
        "per_split_class_counts": {s: dict(stats[s]) for s in SPLITS},
        "per_split_totals": {s: sum(stats[s].values()) for s in SPLITS},
        "grand_total": sum(sum(stats[s].values()) for s in SPLITS),
        "skipped_images": skipped_images,
        "dropped_objects": dropped_objects,
        "images_written": {s: len(os.listdir(os.path.join(OUT, "images", s))) for s in SPLITS},
        "labels_written": {s: len(os.listdir(os.path.join(OUT, "labels", s))) for s in SPLITS},
    }
    with open("/home/archryzen/bccd/datasets/bccd_yolo/build_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
