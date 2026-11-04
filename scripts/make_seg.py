#!/usr/bin/env python3
"""Build datasets/merged_seg: box labels converted to 4-point polygons for segmentation.

BCCD has no mask annotations, so each box becomes a rectangle polygon. This is a synthetic
mask set - see results/RESULTS.md for why that limits what a seg model can learn here.
"""
import glob, os, shutil, yaml

SRC, DST = "datasets/merged", "datasets/merged_seg"
for s in ["train", "val", "test"]:
    os.makedirs(f"{DST}/images/{s}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{s}", exist_ok=True)
    for p in glob.glob(f"{SRC}/images/{s}/*"):
        shutil.copy(p, f"{DST}/images/{s}/{os.path.basename(p)}")
    for lp in glob.glob(f"{SRC}/labels/{s}/*.txt"):
        out = []
        for line in open(lp):
            t = line.split()
            if len(t) != 5:
                continue
            c, cx, cy, w, h = int(t[0]), *map(float, t[1:])
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            out.append(f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n")
        open(f"{DST}/labels/{s}/{os.path.basename(lp)}", "w").writelines(out)
yaml.safe_dump({"path": os.path.abspath(DST), "train": "images/train", "val": "images/val",
                "test": "images/test", "nc": 3, "names": ["RBC", "WBC", "Platelets"]},
               open("configs/merged_seg.yaml", "w"), sort_keys=False)
