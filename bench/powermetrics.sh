#!/bin/sh
# Cross-check of the rootless energy numbers (mlx/molt_mlx/energy.py) with
# Apple's powermetrics, which needs root.  Samples CPU, GPU and ANE power every
# 100 ms while one benchmark worker runs, then prints the integrated energy.
#
#   sudo sh bench/powermetrics.sh 1.5b-0.5b D open-01
#
# Output: bench/results/powermetrics/<pair>_<cond>_<id>.{plist,txt}
set -eu
PAIR=${1:?pair}; COND=${2:?cond}; ID=${3:?prompt id}
OUT=bench/results/powermetrics; mkdir -p "$OUT"
BASE="$OUT/${PAIR}_${COND}_${ID}"
powermetrics --samplers cpu_power,gpu_power,ane_power -i 100 -f plist -o "$BASE.plist" &
PM=$!
sleep 1
sudo -u "${SUDO_USER:-$USER}" python3 -m bench.worker --pair "$PAIR" --cond "$COND" --ids "$ID" \
    --out "$BASE.worker.json"
kill "$PM"; wait "$PM" 2>/dev/null || true
python3 - "$BASE.plist" > "$BASE.txt" <<'PY'
import plistlib, sys
raw = open(sys.argv[1], "rb").read().split(b"\x00")
cpu = gpu = ane = 0.0; n = 0
for chunk in raw:
    chunk = chunk.strip()
    if not chunk:
        continue
    d = plistlib.loads(chunk)
    p = d.get("processor", {})
    dt = d.get("elapsed_ns", 1e8) / 1e9
    cpu += p.get("cpu_power", 0) / 1000 * dt
    gpu += p.get("gpu_power", 0) / 1000 * dt
    ane += p.get("ane_power", 0) / 1000 * dt
    n += 1
print(f"samples={n} cpu_j={cpu:.2f} gpu_j={gpu:.2f} ane_j={ane:.2f}")
PY
cat "$BASE.txt"
