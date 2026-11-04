#!/usr/bin/env python3
"""Live training dashboard for the BCCD runs. Refreshes every 2 s; Ctrl-C to quit."""
import csv, glob, os, re, subprocess, sys, time
from datetime import datetime

RUNS = "/mnt/Ext/runs/merged"
TOTAL_EPOCHS = 100
ORDER = ["v8n", "v9t", "v10n", "v11n", "v11s", "v12n", "v26n"]


def bar(frac, width=34):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def last_row(path):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def gpu():
    try:
        out = subprocess.run(["radeontop", "-d", "-", "-l", "1"], capture_output=True,
                             text=True, timeout=3).stdout.strip().splitlines()[-1]
    except Exception:
        return "GPU ?"
    g = re.search(r"gpu ([0-9.]+)%", out)
    v = re.search(r"vram [0-9.]+% ([0-9.]+)mb", out)
    s = re.search(r"sclk [0-9.]+% ([0-9.]+ghz)", out)
    try:
        sens = subprocess.run(["sensors"], capture_output=True, text=True, timeout=3).stdout
        t = re.search(r"edge:\s+\+([0-9.]+)", sens)
        w = re.search(r"PPT:\s+([0-9.]+)", sens)
    except Exception:
        t = w = None
    return (f"GPU {float(g.group(1)) if g else 0:5.1f}%  "
            f"VRAM {int(float(v.group(1))) if v else 0:4d} MB  "
            f"CLK {s.group(1) if s else '?'}  "
            f"{'T ' + t.group(1) + 'C' if t else ''}  "
            f"{'P ' + w.group(1) + 'W' if w else ''}")


def render():
    lines = []
    now = datetime.now().strftime("%H:%M:%S")
    lines.append(f"  BCCD training — {now}")
    lines.append("")

    active = None
    for tag in ORDER:
        p = f"{RUNS}/{tag}/results.csv"
        if not os.path.exists(p):
            continue
        r = last_row(p)
        if not r:
            continue
        ep = int(r.get("epoch", 0)) + 1
        if ep < TOTAL_EPOCHS and os.path.getmtime(p) > time.time() - 900:
            active = (tag, ep, r)
    if active:
        tag, ep, r = active
        t = float(r.get("time", 0))
        eta = (t / ep) * (TOTAL_EPOCHS - ep) if ep else 0
        lines.append(f"  ▶ {tag:<6s} epoch {ep:3d}/{TOTAL_EPOCHS}  [{bar(ep / TOTAL_EPOCHS)}] {100*ep/TOTAL_EPOCHS:5.1f}%")
        lines.append(f"    elapsed {t/60:6.1f} min   eta {eta/60:6.1f} min   {t/max(ep,1):5.1f} s/epoch")
        m50 = r.get("metrics/mAP50(B)") or "-"
        m95 = r.get("metrics/mAP50-95(B)") or "-"
        bl = r.get("train/box_loss") or "-"
        cl = r.get("train/cls_loss") or "-"
        dl = r.get("train/dfl_loss") or r.get("train/l1_loss") or "-"
        try:
            lines.append(f"    mAP50 {float(m50):.4f}   mAP50-95 {float(m95):.4f}   "
                         f"loss box/cls/{'l1' if 'train/dfl_loss' not in r else 'dfl'} {float(bl):.3f}/{float(cl):.3f}/{float(dl):.3f}")
        except Exception:
            lines.append("    (epoch 0 — metrics appear after the first validation)")
    else:
        newest = None
        for tag in ORDER:
            d = f"{RUNS}/{tag}"
            if os.path.isdir(d) and os.path.getmtime(d) > time.time() - 900:
                newest = tag
        if newest:
            rows = 0
            try:
                rows = sum(1 for _ in open(f"{RUNS}/{newest}/results.csv")) - 1
            except Exception:
                pass
            lines.append(f"  ▶ {newest:<6s} epoch {rows+1:3d}/{TOTAL_EPOCHS}  "
                         f"[{bar(rows/TOTAL_EPOCHS)}] {100*rows/TOTAL_EPOCHS:5.1f}%  (epoch in progress)")
        else:
            lines.append("  no run currently active")

    lines.append("")
    lines.append("  completed (test split, TTA):")
    import json
    res = {}
    for f in (f"{RUNS}/results.json", f"{RUNS}/sweep_results.json"):
        if os.path.exists(f):
            try:
                res.update(json.load(open(f)))
            except Exception:
                pass
    for tag in ORDER:
        v = res.get(tag)
        if not v or "test" not in v:
            continue
        t = v["test"]
        pl, tt = t.get("plain", {}), t.get("tta", {})
        lines.append(f"    {tag:<6s} mAP50 {pl.get('mAP50', 0):.4f} → {tt.get('mAP50', 0):.4f} TTA   "
                     f"mAP50-95 {pl.get('mAP50-95', 0):.4f} → {tt.get('mAP50-95', 0):.4f} TTA")

    lines.append("")
    lines.append("  " + gpu())
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        print(render())
        sys.exit(0)
    try:
        while True:
            sys.stdout.write("\033[2J\033[H" + render() + "\n")
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
