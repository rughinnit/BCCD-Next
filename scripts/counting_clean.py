#!/usr/bin/env python3
"""Counting accuracy with a clean protocol:

* class-aware NMS, IoU 0.7 (matches Ultralytics evaluation)
* confidence threshold tuned on the **val** split (87 images)
* reported on the **test** split (72 images)

Detections are run once per image at conf 0.02; thresholds are then swept offline.
"""
import glob, os, sys, json
import numpy as np
import cv2

sys.path.insert(0, "/home/archryzen/yolo")
from detect import YoloNCNN

ROOT = "/home/archryzen/bccd/datasets/merged"
MODELS = {"yolov9t": "bccd_v9t_best_ncnn_model",
          "yolo11n": "bccd_v11n_best_ncnn_model",
          "yolo26n": "bccd_v26n_best_ncnn_model"}
NAMES = ["RBC", "WBC", "Platelets"]
CONFS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def split_data(split):
    imgs = sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg"))
    gts, dets = [], []
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        c = np.zeros(3, int)
        lab = f"{ROOT}/labels/{split}/{base}.txt"
        if os.path.exists(lab):
            for line in open(lab):
                t = line.split()
                if len(t) == 5:
                    c[int(t[0])] += 1
        gts.append(c)
    return imgs, np.stack(gts)


def run_dets(model_dir, imgs):
    m = YoloNCNN(model_dir)
    out = []
    for p in imgs:
        d = m(cv2.imread(p), 0.02, 0.70)
        out.append(d)
    return out


def counts_at(dets, conf):
    c = np.zeros((len(dets), 3), int)
    for i, d in enumerate(dets):
        if not len(d):
            continue
        keep = d[d[:, 4] >= conf]
        for k in range(3):
            c[i, k] = int((keep[:, 5] == k).sum())
    return c


def mae(P, G):
    return np.abs(P - G).mean(0)


def main():
    report = {}
    for name, mdir in MODELS.items():
        imgs_v, Gv = split_data("val")
        imgs_t, Gt = split_data("test")
        dv = run_dets(mdir, imgs_v)
        dt = run_dets(mdir, imgs_t)

        # global threshold from val
        best = None
        for c in CONFS:
            m = np.abs(counts_at(dv, c) - Gv).mean()
            if best is None or m < best[1]:
                best = (c, m)
        cg = best[0]
        # per-class thresholds from val
        cp = []
        for k in range(3):
            bk = None
            for c in CONFS:
                err = np.abs(counts_at(dv, c)[:, k] - Gv[:, k]).mean()
                if bk is None or err < bk[1]:
                    bk = (c, err)
            cp.append(bk[0])

        def evaluate(dets, G, thr, label):
            if isinstance(thr, (list, tuple)):
                P = np.zeros_like(G)
                for k in range(3):
                    P[:, k] = counts_at(dets, thr[k])[:, k]
            else:
                P = counts_at(dets, thr)
            err = np.abs(P - G)
            rel = 100 * err.mean(0) / np.maximum(G.mean(0), 1e-9)
            bias = (P - G).mean(0)
            print(f"  {label}: " + "  ".join(
                f"{NAMES[k]} MAE {err[:,k].mean():.2f} ({rel[k]:.0f}%) bias {bias[k]:+.2f}"
                for k in range(3)) + f"  | all {err.sum(1).mean():.2f}")
            return {"mae": err.mean(0).tolist(), "rel": rel.tolist(),
                    "bias": bias.tolist(), "totals": [int(P.sum(0)[k]) for k in range(3)]}

        print(f"=== {name}  (val-tuned: global conf {cg:.2f}, per-class {cp})")
        print(f"  val GT: {Gv.sum(0).tolist()}  test GT: {Gt.sum(0).tolist()}")
        r_glob = evaluate(dt, Gt, cg, f"test @ conf {cg:.2f}")
        r_pc = evaluate(dt, Gt, cp, f"test @ per-class {cp}")
        print()
        report[name] = {"global_conf": float(cg), "per_class_conf": [float(x) for x in cp],
                        "test_global": r_glob, "test_perclass": r_pc}
    json.dump(report, open("/mnt/Ext/models/counting_clean.json", "w"), indent=2)


if __name__ == "__main__":
    main()
