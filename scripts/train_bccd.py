import json, time, traceback
from ultralytics import YOLO

DATA = "/home/archryzen/bccd/configs/bccd.yaml"
PROJ = "/mnt/Ext/runs/bccd"
RUNS = [("yolov8n", "v8n"), ("yolo11n", "v11n"), ("yolo26n", "v26n")]
out = {}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for weights, tag in RUNS:
    t0 = time.time()
    try:
        log(f"=== training {tag} ({weights}) ===")
        m = YOLO(f"{weights}.pt")
        r = m.train(data=DATA, epochs=100, imgsz=640, batch=8, device=0, workers=4,
                    amp=False, cache=True, seed=0, plots=False, verbose=False,
                    project=PROJ, name=tag, exist_ok=True)
        val = {k: float(v) for k, v in r.results_dict.items()}

        best = f"{PROJ}/{tag}/weights/best.pt"
        log(f"{tag}: val done, evaluating test split")
        mb = YOLO(best)
        t = mb.val(data=DATA, split="test", imgsz=640, batch=8, device=0,
                   plots=False, verbose=False)
        test = {"mAP50": float(t.box.map50), "mAP50-95": float(t.box.map),
                "precision": float(t.box.mp), "recall": float(t.box.mr)}
        out[tag] = {"weights": weights, "seconds": round(time.time() - t0),
                    "val": val, "test": test}
        log(f"{tag} DONE test mAP50={test['mAP50']:.4f} mAP50-95={test['mAP50-95']:.4f}")
    except Exception:
        out[tag] = {"weights": weights, "error": traceback.format_exc()[-800:]}
        log(f"{tag} FAILED: {out[tag]['error'][-200:]}")
    json.dump(out, open(f"{PROJ}/results.json", "w"), indent=2)

log("ALL DONE")
print(json.dumps({k: v.get("test") or v.get("error", "")[:120] for k, v in out.items()}, indent=2))
