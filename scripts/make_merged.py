#!/usr/bin/env python3
"""Build datasets/merged: BCCD-train + BCD (near-duplicates of the BCCD val/test removed).

Class order everywhere: 0=RBC, 1=WBC, 2=Platelets.
BCD ships as Platelets,RBC,WBC -> remapped to 2,0,1.
"""
import cv2, glob, os, shutil, yaml, numpy as np

ROOT = "datasets/merged"
for sub in ["images/train", "labels/train", "images/val", "labels/val", "images/test", "labels/test"]:
    os.makedirs(f"{ROOT}/{sub}", exist_ok=True)


def dhash(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (17, 16), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


ref = glob.glob("datasets/bccd_yolo/images/test/*.jpg") + glob.glob("datasets/bccd_yolo/images/val/*.jpg")
refh = [dhash(p) for p in ref]


def is_dup(p):
    a = dhash(p)
    if a is None:
        return True
    return min(int(np.count_nonzero(a != b)) for b in refh) <= 4


n = 0
for p in glob.glob("datasets/bccd_yolo/images/train/*.jpg"):
    name = "bccd_" + os.path.basename(p)
    shutil.copy(p, f"{ROOT}/images/train/{name}")
    shutil.copy(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",
                f"{ROOT}/labels/train/{name.rsplit('.', 1)[0]}.txt")
    n += 1

skipped = added = 0
for p in sorted(glob.glob("datasets/bcd/*/images/*")):
    if not p.lower().endswith((".jpg", ".png", ".jpeg")) or is_dup(p):
        skipped += 1
        continue
    d = os.path.dirname(p)
    base = os.path.splitext(os.path.basename(p))[0]
    lab = os.path.join(os.path.dirname(d), "labels", base + ".txt")
    if not os.path.exists(lab):
        continue
    shutil.copy(p, f"{ROOT}/images/train/bcd_{os.path.basename(p)}")
    out = []
    for line in open(lab):
        t = line.split()
        if len(t) != 5:
            continue
        c = {0: 2, 1: 0, 2: 1}.get(int(t[0]))
        if c is not None:
            out.append(f"{c} {' '.join(t[1:])}\n")
    open(f"{ROOT}/labels/train/{base}.txt", "w").writelines(out)
    added += 1

for split in ["val", "test"]:
    for p in glob.glob(f"datasets/bccd_yolo/images/{split}/*.jpg"):
        shutil.copy(p, f"{ROOT}/images/{split}/{os.path.basename(p)}")
        shutil.copy(p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",
                    f"{ROOT}/labels/{split}/{os.path.basename(p).rsplit('.', 1)[0]}.txt")

yaml.safe_dump({"path": os.path.abspath(ROOT), "train": "images/train", "val": "images/val",
                "test": "images/test", "nc": 3, "names": ["RBC", "WBC", "Platelets"]},
               open("configs/merged.yaml", "w"), sort_keys=False)
print(f"bccd train {n}, bcd added {added}, near-dupes skipped {skipped}")
