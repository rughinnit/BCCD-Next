#!/bin/bash
# Compact GPU monitor for RX 570 (no root). Ctrl-C to stop.
S=/sys/class/drm/card1/device
while true; do
  line=$(timeout 2 radeontop -d - -l 1 2>/dev/null | tail -1)
  gpu=$(sed -n 's/.*gpu \([0-9.]*\)%.*/\1/p' <<<"$line")
  vram=$(sed -n 's/.*vram [0-9.]*% \([0-9.]*\)mb.*/\1/p' <<<"$line")
  sclk=$(sed -n 's/.*sclk [0-9.]*% \([0-9.]*ghz\).*/\1/p' <<<"$line")
  temp=$(sensors 2>/dev/null | awk '/edge:/{print $2; exit}')
  watt=$(sensors 2>/dev/null | awk '/PPT:/{print $2"W"; exit}')
  printf "\rGPU %5s%%  VRAM %6s MB  SCLK %-7s  %-6s  %-7s " "${gpu:-?}" "${vram%%.*}" "$sclk" "$temp" "$watt"
  sleep 1
done
