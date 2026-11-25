#!/usr/bin/env python3
"""Common-space benchmark: 3-class BCCD vs 7-class Roboflow.

Every model is evaluated on both test sets in one shared 3-class space
(RBC, WBC, Platelets), with identical inference settings and COCOeval.

  - 3-class models: class ids are already canonical.
  - 7-class model: the five WBC subtypes are collapsed to WBC, then a
    per-class NMS merges duplicate subtype boxes in the common space.

Test sets:
  bccd_official : datasets/merged/images/test   (72 images, official BCCD split)
  u4dak         : models-u4dak .../test/images   (165 images, local re-split)

No TTA: plain inference only, so every cell of the matrix uses the same protocol.
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = "/home/archryzen/bccd"
CANON = ["RBC", "WBC", "Platelets"]
C2I = {name: i for i, name in enumerate(CANON)}
WBC_SUBTYPES = {"Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"}
# gfx803 has a multi-model GPU fault in ROCm 5.4.3; CPU is the reliable path.
DEVICE = os.environ.get("BENCH_DEVICE", "cpu")

SETS = {
    "bccd_official": {
        "images": f"{ROOT}/datasets/merged/images/test",
        "labels": f"{ROOT}/datasets/merged/labels/test",
        "names": {0: "RBC", 1: "WBC", 2: "Platelets"},
    },
    "u4dak": {
        "images": f"{ROOT}/datasets/roboflow/models-u4dak__bccd-blw1i__v1/test/images",
        "labels": f"{ROOT}/datasets/roboflow/models-u4dak__bccd-blw1i__v1/test/labels",
        "names": {
            0: "Basophil", 1: "Eosinophil", 2: "Lymphocyte", 3: "Monocyte",
            4: "Neutrophil", 5: "Platelets", 6: "RBC",
        },
    },
}

MODELS = {
    "yolo26n_3c": "/mnt/Ext/runs/merged/v26n/weights/best.pt",
    "yolo11n_3c": "/mnt/Ext/runs/merged/v11n/weights/best.pt",
    "yolo26n_7c": "/mnt/Ext/runs/roboflow/u4dak_yolo26n/weights/best.pt",
}


def canon(name):
    if name in WBC_SUBTYPES:
        return C2I["WBC"]
    return C2I[name]


def build_gt(set_name):
    spec = SETS[set_name]
    images = sorted(glob.glob(f"{spec['images']}/*.jpg"))
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CANON)],
    }
    ann_id = 1
    for i, path in enumerate(images):
        im = cv2.imread(path)
        h, w = im.shape[:2]
        coco["images"].append(
            {"id": i, "file_name": os.path.basename(path), "width": w, "height": h}
        )
        lp = os.path.join(spec["labels"], os.path.splitext(os.path.basename(path))[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp):
            t = line.split()
            if len(t) != 5:
                continue
            cid = canon(spec["names"][int(t[0])])
            cx, cy, bw, bh = map(float, t[1:])
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": cid,
                    "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    "area": bw * w * bh * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco, images


def evaluate(results, gt):
    if not results:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "per_class": {}}
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    p = e.eval["precision"]  # [T, R, K, A, M]

    def mean_ap(i0, i1, k=None):
        s = p[i0:i1, :, :, 0, 2] if k is None else p[i0:i1, :, k, 0, 2]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else 0.0

    per_class = {
        name: {"AP50": mean_ap(0, 1, k), "AP50-95": mean_ap(0, 10, k)}
        for k, name in enumerate(CANON)
    }
    return {"mAP50": mean_ap(0, 1), "mAP50-95": mean_ap(0, 10), "per_class": per_class}


def predict(model, images, names):
    results = []
    for i, path in enumerate(images):
        r = model.predict(
            path, imgsz=640, conf=0.001, iou=0.7, max_det=300,
            device=DEVICE, verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu()
        scores = r.boxes.conf.cpu()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        mapped = np.array([canon(names[int(c)]) for c in cls])

        keep_parts = []
        for k in range(len(CANON)):
            idx = np.where(mapped == k)[0]
            if len(idx) == 0:
                continue
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            keep = nms(boxes[idx_t], scores[idx_t], 0.7)
            keep_parts.append(idx[keep.numpy()])
        if not keep_parts:
            continue

        for j in np.concatenate(keep_parts):
            x1, y1, x2, y2 = boxes[j].tolist()
            results.append(
                {
                    "image_id": i,
                    "category_id": int(mapped[j]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[j]),
                }
            )
    return results


def main():
    gts, images = {}, {}
    for set_name in SETS:
        gts[set_name], images[set_name] = build_gt(set_name)
        print(f"{set_name}: {len(images[set_name])} images, "
              f"{len(gts[set_name]['annotations'])} boxes", flush=True)

    matrix = {}
    for model_name, model_path in MODELS.items():
        model = YOLO(model_path)
        for set_name in SETS:
            t0 = time.time()
            res = predict(model, images[set_name], model.names)
            metrics = evaluate(res, gts[set_name])
            metrics["predictions"] = len(res)
            metrics["seconds"] = round(time.time() - t0, 1)
            matrix[f"{model_name}__{set_name}"] = metrics
            print(
                f"{model_name:12s} {set_name:14s} "
                f"mAP50={metrics['mAP50']:.4f} mAP50-95={metrics['mAP50-95']:.4f} "
                f"preds={len(res)} ({metrics['seconds']}s)",
                flush=True,
            )

    with open(f"{ROOT}/results/benchmark_common.json", "w") as fh:
        json.dump(matrix, fh, indent=2)
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
