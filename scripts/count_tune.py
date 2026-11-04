#!/usr/bin/env python3
"""Tune NMS IoU and confidence for counting on val, then report on test.

Fixes the RBC duplicate problem (touching cells) by searching IoU instead of assuming 0.7.
"""
import glob, os, sys, json
import numpy as np
import cv2

sys.path.insert(0, "/home/archryzen/yolo")
from detect import YoloNCNN

ROOT = "/home/archryzen/bccd/datasets/merged"
NAMES = ["RBC", "WBC", "Platelets"]
CONFS = np.round(np.arange(0.05, 0.96, 0.05), 2)
IOUS = [0.50, 0.60, 0.70, 0.80]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "bccd_v26n_best_ncnn_model"
OUT = sys.argv[2] if len(sys.argv) > 2 else "count_tune.json"


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


def dets_at(m, paths, iou):
    return [m(cv2.imread(p), 0.02, iou) for p in paths]


def counts(dets, conf):
    c = np.zeros((len(dets), 3), int)
    for i, d in enumerate(dets):
        if len(d):
            k = d[d[:, 4] >= conf]
            for j in range(3):
                c[i, j] = int((k[:, 5] == j).sum())
    return c


m = YoloNCNN(MODEL)
Gv, Gt = gt("val"), gt("test")
Iv, It = imgs("val"), imgs("test")
dv = {iou: dets_at(m, Iv, iou) for iou in IOUS}
dt = {iou: dets_at(m, It, iou) for iou in IOUS}

best = None
for iou in IOUS:
    for conf in CONFS:
        mae = np.abs(counts(dv[iou], conf) - Gv).mean()
        if best is None or mae < best[0]:
            best = (mae, iou, conf)
val_mae, bi, bc = best
P = counts(dt[bi], bc)
E = np.abs(P - Gt)
print(f"model {MODEL}")
print(f"val-best: iou={bi:.2f} conf={bc:.2f} (val all-class MAE {val_mae:.2f})")
for k in range(3):
    print(f"  test {NAMES[k]:10s} MAE {E[:,k].mean():5.2f}  median {np.median(E[:,k]):.0f}  "
          f"within±1 {(E[:,k]<=1).mean()*100:5.1f}%  bias {(P[:,k]-Gt[:,k]).mean():+.2f}  "
          f"total {P[:,k].sum()}/{Gt[:,k].sum()}")
print(f"  test all-class MAE {E.sum(1).mean():.2f}")
json.dump({"model": MODEL, "iou": bi, "conf": float(bc), "val_mae": float(val_mae),
           "test_mae": E.mean(0).tolist(), "test_totals": [int(x) for x in P.sum(0)],
           "gt_totals": [int(x) for x in Gt.sum(0)]}, open(OUT, "w"), indent=2)
