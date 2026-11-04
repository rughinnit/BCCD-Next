#!/usr/bin/env python3
"""Per-class fix for counting: choose resolution/threshold per class on val, optional tiling.

Issues being addressed:
  * WBC got worse at 960 (+9.9% aggregate) -> allow a different config per class
  * platelets are 10-20 px -> tiling gives them ~2x more pixels
  * RBC duplicates from touching cells -> cross-tile fusion can collapse them
Everything is tuned on val and reported on test.
"""
import glob, os, sys, json
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion
sys.path.insert(0, "/home/archryzen/yolo")
from detect import YoloNCNN

ROOT = "/home/archryzen/bccd/datasets/merged"
NAMES = ["RBC", "WBC", "Platelets"]
CONFS = np.round(np.arange(0.05, 0.96, 0.05), 2)
M640 = YoloNCNN("bccd_v26n_640_ncnn_model")
M960 = YoloNCNN("bccd_v26n_960_ncnn_model")


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


def tile_boxes(im, model, conf, iou, frac=0.6, steps=(0.0, 0.4)):
    """Run the detector on overlapping tiles; fuse back in image coordinates."""
    h, w = im.shape[:2]
    tw, th = int(w * frac), int(h * frac)
    xs = [int(w * s) for s in steps]
    ys = [int(h * s) for s in steps]
    bl, sl, ll = [], [], []
    for y in ys:
        for x in xs:
            tile = im[y:y + th, x:x + tw]
            d = model(tile, conf, iou)
            if not len(d):
                bl.append(np.zeros((0, 4))); sl.append(np.zeros((0,))); ll.append(np.zeros((0,)))
                continue
            b = d[:, :4].copy()
            b[:, [0, 2]] += x; b[:, [1, 3]] += y
            bl.append(b / np.array([w, h, w, h])); sl.append(d[:, 4]); ll.append(d[:, 5].astype(int))
    bb, ss, ll2 = weighted_boxes_fusion(bl, sl, ll, iou_thr=0.55, skip_box_thr=0.0)
    if not len(bb):
        return np.zeros((0, 6), np.float32)
    out = np.zeros((len(bb), 6), np.float32)
    out[:, 0] = bb[:, 0] * w; out[:, 1] = bb[:, 1] * h
    out[:, 2] = bb[:, 2] * w; out[:, 3] = bb[:, 3] * h
    out[:, 4] = ss; out[:, 5] = ll2
    return out


def config_dets(paths, cfg):
    kind, model, iou = cfg
    out = []
    for p in paths:
        im = cv2.imread(p)
        if kind == "full":
            out.append(model(im, 0.02, iou))
        else:
            out.append(tile_boxes(im, model, 0.02, iou))
    return out


def counts(dets, conf):
    c = np.zeros((len(dets), 3), int)
    for i, d in enumerate(dets):
        if len(d):
            k = d[d[:, 4] >= conf]
            for j in range(3):
                c[i, j] = int((k[:, 5] == j).sum())
    return c


CFGS = {
    "640": ("full", M640, 0.5),
    "960": ("full", M960, 0.5),
    "tile640": ("tile", M640, 0.5),
    "tile960": ("tile", M960, 0.5),
}

Iv, It = imgs("val"), imgs("test")
Gv, Gt = gt("val"), gt("test")
cache = {k: (config_dets(Iv, v), config_dets(It, v)) for k, v in CFGS.items()}

# best (config, conf) per class on val
choice = []
for k in range(3):
    best = None
    for name, (dv, _) in cache.items():
        C = counts(dv, 0.5)  # placeholder, recompute per conf
        for conf in CONFS:
            c = counts(dv, conf)[:, k]
            mae = np.abs(c - Gv[:, k]).mean()
            if best is None or mae < best[0]:
                best = (mae, name, conf)
    choice.append(best)

print("per-class choice on val:")
for k in range(3):
    print(f"  {NAMES[k]:10s} config={choice[k][1]:8s} conf={choice[k][2]:.2f} (val MAE {choice[k][0]:.2f})")

P = np.zeros_like(Gt)
for k in range(3):
    _, name, conf = choice[k]
    P[:, k] = counts(cache[name][1], conf)[:, k]
E = np.abs(P - Gt)
print("\ntest with per-class configs:")
for k in range(3):
    print(f"  {NAMES[k]:10s} MAE {E[:,k].mean():5.2f}  median {np.median(E[:,k]):.0f}  "
          f"within±1 {(E[:,k]<=1).mean()*100:5.1f}%  bias {(P[:,k]-Gt[:,k]).mean():+.2f}  "
          f"total {P[:,k].sum()}/{Gt[:,k].sum()}")
print(f"  ALL        MAE {E.sum(1).mean():.2f}")

json.dump({"choice": [{"class": NAMES[k], "config": choice[k][1], "conf": float(choice[k][2]),
                       "val_mae": float(choice[k][0])} for k in range(3)],
           "test_mae": E.mean(0).tolist(),
           "test_totals": [int(x) for x in P.sum(0)],
           "gt_totals": [int(x) for x in Gt.sum(0)]},
          open("count_fix_all.json", "w"), indent=2)
