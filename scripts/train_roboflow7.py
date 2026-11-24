#!/usr/bin/env python3
"""Train yolo26n on the 7-class Roboflow BCCD copy and evaluate on its test split.

Dataset: datasets/roboflow/models-u4dak__bccd-blw1i__v1
  Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil, Platelets, RBC
Splits: 762 train / 163 val / 165 test (stratified, seed 0)

The 7-class numbers are a separate domain from the 3-class BCCD results; do not
compare them with results/RESULTS.md.
"""
import json
import time

from ultralytics import YOLO

DATA = "/home/archryzen/bccd/configs/roboflow_7class.yaml"
PROJ = "/mnt/Ext/runs/roboflow"
WEIGHTS = "yolo26n.pt"
TAG = "u4dak_yolo26n"

if __name__ == "__main__":
    m = YOLO(WEIGHTS)
    t0 = time.time()
    m.train(
        data=DATA, epochs=100, imgsz=640, batch=8, device=0, workers=4,
        amp=False, cache=True, seed=0, plots=False, verbose=False,
        project=PROJ, name=TAG, exist_ok=True,
    )

    best = f"{PROJ}/{TAG}/weights/best.pt"
    mb = YOLO(best)
    out = {}
    for augment in (False, True):
        key = "tta" if augment else "plain"
        t = mb.val(
            data=DATA, split="test", imgsz=640, batch=8, device=0,
            augment=augment, plots=False, verbose=False,
        )
        out[key] = {
            "mAP50": float(t.box.map50),
            "mAP50-95": float(t.box.map),
            "P": float(t.box.mp),
            "R": float(t.box.mr),
            "per_class": {
                t.names[i]: {
                    "AP50": float(t.box.ap50[i]),
                    "AP50-95": float(t.box.ap[i]),
                }
                for i in range(len(t.names))
            },
        }
    out["seconds"] = round(time.time() - t0)
    out["weights"] = best
    with open(f"{PROJ}/{TAG}/eval.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
