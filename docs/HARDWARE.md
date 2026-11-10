# Hardware and stability notes — RX 570 on gfx803

These notes record why long training runs occasionally ended unexpectedly. They are separate
from the detection results: all committed results were produced successfully. Investigation
performed on **2026-11-04**; last power-loss event was **2026-11-03 20:10:52**.

## System

| Component | Detail |
|---|---|
| Motherboard | ASRock AB350 Pro4 (AMD B350, ATX), BIOS AMI **P4.60 (2018-01-15)** |
| CPU | AMD Ryzen 3 1200 — 4C/4T, 3.1 GHz base / 3.4 GHz boost, 65 W |
| RAM | 2 × 4 GB DDR4-2666 single-rank, dual-channel (2 of 4 slots free) |
| GPU | Sapphire **Radeon RX 570 Pulse ITX 4 GB** (Ellesmere, gfx803), 120 W cap |
| GPU link | CPU-attached PCIe 3.0 ×16 (`00:03.1`), Gen3 x16 under load, Gen1 at idle |
| PSU | **XPG Pylon 550 W**, 80+ Bronze, single 12 V rail (~45 A) |
| Storage | 128 GB TEAM SATA SSD (root) + 1 TB Seagate ST1000DM003 (datasets/runs) |
| OS | Arch Linux, kernel 7.2.11 |

Power budget: ~240 W sustained, ~300–350 W peak — well inside a 550 W unit.

## Symptom

Unexpected whole-machine power-offs during long training runs. The failing boot (2026-11-03
12:26 → 20:10) ended with **no shutdown sequence, no kernel panic, no thermal event, no MCE,
and no AER messages** — the log simply stopped, which is the signature of abrupt power loss.

## PCIe AER storm

- Boot 2026-11-03 22:59 → 11:52 logged **112,992 correctable AER messages** — roughly 37,600
  distinct events, because each is reported by the root port, the GPU, and the GPU audio
  function.
- **All correctable:** Bad TLP (`0x40`) + Replay Timer Timeout (`0x1000`). Zero uncorrectable,
  zero fatal, zero GPU resets.
- **Strictly load-correlated:** ~1–7 events/s under GPU compute, **0 at idle**.
- **Link is healthy:** Gen3 x16 (8 GT/s) under load; `ASPM Disabled`; L1 substates disabled.
- **Controlled experiment** (identical 20 s windows): idle **0** errors; 179 GB of host↔device
  copies **3**; 683 dense matmuls at 119 W / 65–70 °C **164**. The errors follow **GPU
  power/clock**, not PCIe traffic.

Correctable errors are replayed by the link, so they do not corrupt data and cause no FPS loss;
they are a symptom, not the failure.

## GPU VM page faults

Two kernel faults, both from `scripts/ensemble.py` during multi-model inference:

```
GPU fault detected: 147 0x05184802   (pid 21503, read from TC4)
GPU fault detected: 147 0x03600802   (pid 21958, read from TC0)
```

Decoded: a **virtual-memory protection fault** — the GPU read a page that was not mapped, from
the texture-cache path. ROCm aborts the process (SIGABRT + coredump). The same job on CPU
succeeds. This is a **ROCm 5.4.3 / gfx803 software bug**, not a hardware fault, and not the
cause of the power-offs (the machine stayed up).

Mitigation: run the ensemble on CPU, or split it into one model per process and fuse the saved
boxes in a separate CPU pass.

## Ruled out

| Suspect | Evidence |
|---|---|
| ASPM | `LnkCtl: ASPM Disabled`, L1 substates disabled |
| Riser / missing lanes | `LnkSta: Speed 8GT/s, Width x16` under load |
| Thermal / MCE / OOM | none in the failing boot; CPU 40–42 °C idle, GPU 65–70 °C loaded |
| VRAM failure | zero uncorrectable AER, zero GPU resets |
| GPU driver | faults are link-layer and follow power/clock, not a specific op |

## Leading hypothesis

**Power delivery.** The 550 W PSU is well above the system's ~250 W draw, so simple undersizing
is not the explanation. Remaining candidates, in order:

1. A faulty or aging PSU (the Pylon line launched ~2020; capacitors degrade).
2. A bad connection — PCIe 8-pin, 24-pin, or EPS — arcing under load.
3. Mains (brownout / outlet / strip).

## Next steps (not yet done)

1. **100 W power-cap diagnostic** — `echo 100000000 > /sys/class/drm/card1/device/hwmon/hwmon0/power1_cap`,
   run a long job, then restore 120 W. If the storm collapses and the machine survives, power
   delivery is confirmed.
2. **Measure +12 V under load** — multimeter on a Molex/SATA 12 V wire, or Windows HWiNFO64 /
   the BIOS monitor. Linux `nct6775` does not expose this board's divider, so its inputs are
   unlabelled and too coarse.
3. **Windows cross-check** — Event Viewer → `WHEA-Logger` (Event ID 17) to confirm whether the
   corrected PCIe errors are OS-independent.
4. **Reseat** the GPU and its 8-pin, use a dedicated PCIe cable, try another outlet/strip.
5. **Swap in a known-good 500 W+ PSU** for one run — the decisive test.
6. **Update the 2018 BIOS (P4.60)** only after the power is stable.
7. **Auto-resume watchdog** — `last.pt` is written each epoch, so a power cut should cost at most
   one epoch, not a whole run.

## Status

- Last power-loss: **2026-11-03 20:10:52** (abrupt, no kernel trace).
- Clean reboots on 2026-11-04; machine offline 2026-11-04 → 2026-11-10.
- Current boot (2026-11-10): **0 AER events**, 0 GPU faults (idle).
- **Unresolved** — power delivery is the leading hypothesis, pending the measurements above.

## Reproducing the diagnostics

```bash
# AER events per boot (needs the systemd-journal group)
journalctl -k -b -1 --no-pager | grep -c 'PCIe Bus Error'
journalctl -k --no-pager | grep -iE 'GPU fault detected|ring .*timeout|GPU reset'

# link state (root for LnkCap/LnkSta)
sudo lspci -vv -s 26:00.0 | grep -E 'LnkCap:|LnkSta:|ASPM|CESta:|UESta:'

# CPU / board sensors
sudo modprobe nct6775 && sensors
```
