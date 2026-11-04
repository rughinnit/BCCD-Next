import json, time, traceback
from ultralytics import YOLO

DATA = "/home/archryzen/bccd/configs/merged.yaml"
PROJ = "/mnt/Ext/runs/merged"
RUNS = [("yolo11n", "v11n", 8), ("yolo26n", "v26n", 8), ("yolo11s", "v11s", 4)]
out = {}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

for weights, tag, bs in RUNS:
    t0 = time.time()
    try:
        log(f"=== {tag} ({weights}) batch={bs} ===")
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
        out[tag] = {"weights": weights, "batch": bs, "seconds": round(time.time()-t0),
                    "val": {k: float(v) for k, v in r.results_dict.items()}, "test": res}
        log(f"{tag} DONE plain mAP50={res['plain']['mAP50']:.4f} tta={res['tta']['mAP50']:.4f}")
    except Exception:
        out[tag] = {"weights": weights, "error": traceback.format_exc()[-600:]}
        log(f"{tag} FAILED")
    json.dump(out, open(f"{PROJ}/results.json", "w"), indent=2)

log("ALL DONE")
print(json.dumps({k: v.get("test") for k, v in out.items()}, indent=2))
