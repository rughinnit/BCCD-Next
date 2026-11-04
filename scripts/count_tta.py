#!/usr/bin/env python3
"""Counting with TTA + WBF fusion (horizontal flip), tuned on val, reported on test.

Duplicate boxes on touching RBCs that are individually confident but only partially
overlapping survive plain NMS; fusing an original and a mirrored pass lets WBF collapse
them.
"""
import glob, os, sys, json
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/home/archryzen/yolo")
from detect import YoloNCNN

ROOT = "/home/archryzen/bccd/datasets/merged"
MODEL = sys.argv[1]
OUT = sys.argv[2]
IOU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
CONFS = np.round(np.arange(0.05, 0.96, 0.05), 2)
NAMES = ["RBC", "WBC", "Platelets"]


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


def fuse(d1, d2, w, h):
    lists_b, lists_s, lists_l = [], [], []
    for dets, flip in ((d1, False), (d2, True)):
        if not len(dets):
            lists_b.append(np.zeros((0, 4))); lists_s.append(np.zeros((0,))); lists_l.append(np.zeros((0,)))
            continue
        b = dets[:, :4].copy()
        if flip:
            x1 = w - dets[:, 2]; x2 = w - dets[:, 0]
            b[:, 0], b[:, 2] = x1, x2
        b = np.clip(b, 0, [w, h, w, h])
        lists_b.append(b / np.array([w, h, w, h]))
        lists_s.append(dets[:, 4]); lists_l.append(dets[:, 5].astype(int))
    bb, ss, ll = weighted_boxes_fusion(lists_b, lists_s, lists_l, iou_thr=0.55, skip_box_thr=0.0)
    if len(bb) == 0:
        return np.zeros((0, 6), np.float32)
    out = np.zeros((len(bb), 6), np.float32)
    out[:, 0] = bb[:, 0] * w; out[:, 1] = bb[:, 1] * h
    out[:, 2] = bb[:, 2] * w; out[:, 3] = bb[:, 3] * h
    out[:, 4] = ss; out[:, 5] = ll
    return out


def run(split, model, iou):
    dets = []
    for p in sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg")):
        im = cv2.imread(p); h, w = im.shape[:2]
        d1 = model(im, 0.02, iou)
        d2 = model(cv2.flip(im, 1), 0.02, iou)
        dets.append(fuse(d1, d2, w, h))
    return dets


def counts(dets, conf):
    c = np.zeros((len(dets), 3), int)
    for i, d in enumerate(dets):
        if len(d):
            k = d[d[:, 4] >= conf]
            for j in range(3):
                c[i, j] = int((k[:, 5] == j).sum())
    return c


model = YoloNCNN(MODEL)
Gv, Gt = gt("val"), gt("test")
dv, dt = run("val", model, IOU), run("test", model, IOU)
best = min(((np.abs(counts(dv, c) - Gv).mean(), c) for c in CONFS), key=lambda t: t[0])
val_mae, bc = best
P = counts(dt, bc)
E = np.abs(P - Gt)
print(f"model {MODEL}  TTA=flip+WBF  iou={IOU}")
print(f"val-best conf {bc:.2f} (val MAE {val_mae:.2f})")
for k in range(3):
    print(f"  test {NAMES[k]:10s} MAE {E[:,k].mean():5.2f}  median {np.median(E[:,k]):.0f}  "
          f"within±1 {(E[:,k]<=1).mean()*100:5.1f}%  bias {(P[:,k]-Gt[:,k]).mean():+.2f}  "
          f"total {P[:,k].sum()}/{Gt[:,k].sum()}")
print(f"  test all-class MAE {E.sum(1).mean():.2f}   exact-image {(E.sum(1)==0).mean()*100:.1f}%")
json.dump({"model": MODEL, "tta": "flip+wbf", "iou": IOU, "conf": float(bc),
           "test_mae": E.mean(0).tolist(), "test_totals": [int(x) for x in P.sum(0)],
           "gt_totals": [int(x) for x in Gt.sum(0)]}, open(OUT, "w"), indent=2)
