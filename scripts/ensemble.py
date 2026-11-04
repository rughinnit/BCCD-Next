import glob, json, os, sys
import numpy as np
import cv2
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = "/home/archryzen/bccd/datasets/merged"
IMGS = sorted(glob.glob(f"{ROOT}/images/test/*.jpg"))
LABS = [p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for p in IMGS]
MODELS = {
    "v9t": "/mnt/Ext/runs/merged/v9t/weights/best.pt",
    "v11n": "/mnt/Ext/runs/merged/v11n/weights/best.pt",
    "v12n": "/mnt/Ext/runs/merged/v12n/weights/best.pt",
    "v26n": "/mnt/Ext/runs/merged/v26n/weights/best.pt",
}

# ---- ground truth in COCO form ------------------------------------------------
coco_gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n}
            for i, n in enumerate(["RBC", "WBC", "Platelets"])]}
ann_id = 1
for i, (p, lp) in enumerate(zip(IMGS, LABS)):
    im = cv2.imread(p); h, w = im.shape[:2]
    coco_gt["images"].append({"id": i, "file_name": os.path.basename(p), "width": w, "height": h})
    if not os.path.exists(lp):
        continue
    for line in open(lp):
        t = line.split()
        if len(t) != 5:
            continue
        c, cx, cy, bw, bh = int(t[0]), *map(float, t[1:])
        coco_gt["annotations"].append({
            "id": ann_id, "image_id": i, "category_id": c,
            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
            "area": bw * w * bh * h, "iscrowd": 0})
        ann_id += 1
json.dump(coco_gt, open("/tmp/gt.json", "w"))

def evaluate(results, tag):
    if not results:
        print(f"{tag}: no predictions"); return
    coco_dt = COCO("/tmp/gt.json").loadRes(results)
    e = COCOeval(COCO("/tmp/gt.json"), coco_dt, "bbox")
    e.evaluate(); e.accumulate(); e.summarize()
    print(f"{tag}: mAP50-95={e.stats[0]:.4f} mAP50={e.stats[1]:.2f} mAP75={e.stats[2]:.2f}")

# ---- per-model predictions ----------------------------------------------------
preds = {}
for name, w in MODELS.items():
    m = YOLO(w)
    out = []
    for i, p in enumerate(IMGS):
        r = m.predict(p, imgsz=640, conf=0.001, iou=0.7, max_det=300, device="cpu", verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy(); sc = b.conf.cpu().numpy(); cl = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, sc, cl):
            out.append({"image_id": i, "category_id": int(c),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)})
    preds[name] = out
    evaluate(out, f"single {name}")

# ---- WBF ensemble -------------------------------------------------------------
shapes = []
for p in IMGS:
    im = cv2.imread(p); shapes.append((im.shape[1], im.shape[0]))

COMBOS = [("v9t", "v11n"), ("v9t", "v11n", "v12n"), ("v9t", "v11n", "v12n", "v26n")]
for combo in COMBOS:
    fused = []
    for i, (p, (w, h)) in enumerate(zip(IMGS, shapes)):
        bl, sl, ll = [], [], []
        for name in combo:
            bs, ss, ls = [], [], []
            for r in preds[name]:
                if r["image_id"] != i:
                    continue
                x, y, bw, bh = r["bbox"]
                bs.append([max(0, x / w), max(0, y / h), min(1, (x + bw) / w), min(1, (y + bh) / h)])
                ss.append(r["score"]); ls.append(r["category_id"])
            bl.append(np.array(bs).reshape(-1, 4) if bs else np.zeros((0, 4)))
            sl.append(np.array(ss) if ss else np.zeros((0,)))
            ll.append(np.array(ls) if ls else np.zeros((0,)))
        if all(len(s) == 0 for s in sl):
            continue
        boxes, scores, labels = weighted_boxes_fusion(bl, sl, ll, iou_thr=0.55, skip_box_thr=0.0)
        for bb, s, l in zip(boxes, scores, labels):
            fused.append({"image_id": i, "category_id": int(l),
                          "bbox": [float(bb[0] * w), float(bb[1] * h),
                                   float((bb[2] - bb[0]) * w), float((bb[3] - bb[1]) * h)],
                          "score": float(s)})
    evaluate(fused, "WBF " + "+".join(combo))
