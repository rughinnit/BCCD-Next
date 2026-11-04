import json, time
from pathlib import Path
from ultralytics import YOLO

RUN = Path("/mnt/Ext/runs/merged/v11s")
RES = Path("/mnt/Ext/runs/merged/results.json")
DATA = "/home/archryzen/bccd/configs/merged.yaml"

last = RUN / "weights/last.pt"
assert last.is_file(), f"no checkpoint at {last}"

print(f"[{time.strftime('%H:%M:%S')}] resuming v11s from {last}", flush=True)
m = YOLO(str(last))
m.train(resume=True)

best = str(RUN / "weights/best.pt")
mb = YOLO(best)
res = {}
for aug in (False, True):
    t = mb.val(data=DATA, split="test", imgsz=640, batch=8, device=0,
               augment=aug, plots=False, verbose=False)
    res["tta" if aug else "plain"] = {
        "mAP50": float(t.box.map50), "mAP50-95": float(t.box.map),
        "P": float(t.box.mp), "R": float(t.box.mr)}

d = json.loads(RES.read_text()) if RES.is_file() else {}
d["v11s"] = {"weights": "yolo11s", "resumed": True, "test": res}
RES.write_text(json.dumps(d, indent=2))
print(f"[{time.strftime('%H:%M:%S')}] v11s DONE plain={res['plain']['mAP50']:.4f}/"
      f"{res['plain']['mAP50-95']:.4f} tta={res['tta']['mAP50']:.4f}/{res['tta']['mAP50-95']:.4f}",
      flush=True)
