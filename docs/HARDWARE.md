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

## Root cause — resolved

The fault was a **loose PCIe connector**, not the PSU. Reseating the GPU in its slot and
re-seating the 8-pin PCIe power connector eliminated both the AER storm and the power-offs.

The corrected picture: a partially-seated connector raised the link's error rate under load
(Bad TLP + replay timeout), and the same contact fault produced the occasional abrupt power cut.
The 550 W PSU was adequate all along — the issue was mechanical contact resistance, not
undersizing.

## If it recurs

1. Power off, reseat the GPU and the 8-pin PCIe connector (both ends).
2. Use a dedicated PCIe cable; do not daisy-chain one cable into two connectors.
3. Try another outlet/strip.
4. If it still happens: measure +12 V under load (multimeter, or BIOS/Windows sensor), swap in a
   known-good PSU, and update the 2018 BIOS.

## Status

- **Resolved 2026-11-10:** reseating the PCIe connector stopped the AER storm and the power-offs.
- Last power-loss before the fix: **2026-11-03 20:10:52**.
- The GPU VM faults in `ensemble.py` remain a separate ROCm/gfx803 software issue; run that job on
  CPU or split it into one model per process.

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
