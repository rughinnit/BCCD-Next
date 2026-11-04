#!/usr/bin/env python3
"""Count from the segmentation model, with optional mask-IoU NMS.

Box NMS merges touching cells because their rectangles overlap. Two touching cells have
high box IoU but LOW mask IoU, so suppressing on mask overlap instead should keep both.
Thresholds tuned on val, reported on test.
"""
import glob, os, sys, json
import numpy as np
import cv2

sys.path.insert(0, "/home/archryzen/yolo")
from ultralytics import YOLO

ROOT = "/home/archryzen/bccd/datasets/merged"
W = "/mnt/Ext/runs/merged_seg/v26n_seg/weights/best.pt"
NAMES = ["RBC", "WBC", "Platelets"]
CONFS = np.round(np.arange(0.05, 0.96, 0.05), 2)
MASK_IOUS = [0.2, 0.35, 0.5]


def gt(split):
    g = []
    for p in sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg")):
        b = os.path.splitext(os.path.basename(p))[0]
        c = np.zeros(3, int)
        lab = f"{ROOT}/labels/{split}/{b}.txt"
        if os.path.exists(lab):
            for line in open(lab):
                t = line.split()
                if len(t) == 5:
                    c[int(t[0])] += 1
        g.append(c)
    return np.stack(g)


def imgs(split):
    return sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg"))


model = YOLO(W)


def predict(paths):
    """Return per image: boxes(xyxy), scores, classes, binary masks downsampled."""
    out = []
    for p in paths:
        r = model.predict(p, imgsz=640, conf=0.02, iou=0.9, max_det=300,
                          verbose=False, device="cpu")[0]
        if r.boxes is None or len(r.boxes) == 0:
            out.append(None)
            continue
        boxes = r.boxes.xyxy.numpy()
        scores = r.boxes.conf.numpy()
        cls = r.boxes.cls.numpy().astype(int)
        masks = r.masks.data.numpy() if r.masks is not None else None
        if masks is not None:
            masks = (masks > 0.5)
            # downsample masks to 160x120 for fast IoU
            small = np.stack([cv2.resize(m.astype(np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST)
                              for m in masks]).astype(bool)
        else:
            small = None
        out.append((boxes, scores, cls, small))
    return out


def mask_nms(boxes, scores, cls, masks, conf, mask_iou_thr):
    keep = []
    order = np.argsort(-scores)
    for i in order:
        if scores[i] < conf:
            continue
        ok = True
        for j in keep:
            if cls[i] != cls[j]:
                continue
            if masks is not None:
                inter = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                iou = inter / max(union, 1)
            else:  # fall back to box IoU
                a, b = boxes[i], boxes[j]
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                iou = inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1e-9)
            if iou > mask_iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    c = np.zeros(3, int)
    for j in keep:
        c[cls[j]] += 1
    return c


def counts(preds, conf, thr, use_mask=True):
    c = np.zeros((len(preds), 3), int)
    for i, p in enumerate(preds):
        if p is None:
            continue
        boxes, scores, cls, masks = p
        c[i] = mask_nms(boxes, scores, cls, masks if use_mask else None, conf, thr)
    return c


Iv, It = imgs("val"), imgs("test")
Gv, Gt = gt("val"), gt("test")
Pv, Pt = predict(Iv), predict(It)

best = None
for use_mask in (False, True):
    for thr in (MASK_IOUS if use_mask else [0.5]):
        for conf in CONFS:
            mae = np.abs(counts(Pv, conf, thr, use_mask) - Gv).mean()
            if best is None or mae < best[0]:
                best = (mae, use_mask, thr, conf)
val_mae, use_mask, thr, conf = best
P = counts(Pt, conf, thr, use_mask)
E = np.abs(P - Gt)
print(f"seg model {W}")
print(f"val-best: mask_nms={use_mask} thr={thr} conf={conf:.2f} (val MAE {val_mae:.2f})")
for k in range(3):
    print(f"  {NAMES[k]:10s} MAE {E[:,k].mean():5.2f}  median {np.median(E[:,k]):.0f}  "
          f"within±1 {(E[:,k]<=1).mean()*100:5.1f}%  bias {(P[:,k]-Gt[:,k]).mean():+.2f}  "
          f"total {P[:,k].sum()}/{Gt[:,k].sum()}")
print(f"  ALL        MAE {E.sum(1).mean():.2f}  exact {(E.sum(1)==0).mean()*100:.1f}%")
json.dump({"mask_nms": bool(use_mask), "thr": float(thr), "conf": float(conf),
           "test_mae": E.mean(0).tolist(), "totals": [int(x) for x in P.sum(0)],
           "gt": [int(x) for x in Gt.sum(0)]}, open("/mnt/Ext/models/count_seg.json", "w"), indent=2)
