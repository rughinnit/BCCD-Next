import json, time, traceback
from pathlib import Path
from ultralytics import YOLO

DATA = "/home/archryzen/bccd/configs/merged.yaml"
PROJ = "/mnt/Ext/runs/merged"
OUT = Path(f"{PROJ}/sweep_results.json")

# n-tier family sweep (v11n and v26n already trained by train_merged.py)
RUNS = [("yolov8n", "v8n", 8), ("yolov9t", "v9t", 8),
        ("yolov10n", "v10n", 8), ("yolo12n", "v12n", 8)]

out = json.loads(OUT.read_text()) if OUT.is_file() else {}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

for weights, tag, bs in RUNS:
    t0 = time.time()
    try:
        log(f"=== {tag} ({weights}) ===")
        m = YOLO(f"{weights}.pt")
        r = m.train(data=DATA, epochs=100, imgsz=640, batch=bs, device=0, workers=4,
                    amp=False, cache=True, seed=0, plots=False, verbose=False,
                    project=PROJ, name=tag, exist_ok=True)
        best = f"{PROJ}/{tag}/weights/best.pt"
        mb = YOLO(best)
        res = {}
        for aug in (False, True):
            t = mb.val(data=DATA, split="test", imgsz=640, batch=8, device=0,
                       augment=aug, plots=False, verbose=False)
            res["tta" if aug else "plain"] = {
                "mAP50": float(t.box.map50), "mAP50-95": float(t.box.map),
                "P": float(t.box.mp), "R": float(t.box.mr)}
        out[tag] = {"weights": weights, "seconds": round(time.time()-t0),
                    "val": {k: float(v) for k, v in r.results_dict.items()}, "test": res}
        log(f"{tag} DONE plain={res['plain']['mAP50']:.4f} tta={res['tta']['mAP50']:.4f}")
    except Exception:
        out[tag] = {"weights": weights, "error": traceback.format_exc()[-600:]}
        log(f"{tag} FAILED")
    OUT.write_text(json.dumps(out, indent=2))

log("SWEEP DONE")
