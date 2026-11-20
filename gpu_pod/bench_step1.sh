#!/bin/bash
# Benchmarks step 1 of Docker/gpu-training.md on the pod: the config
# schnaps_normal_gpu (4,096 tables x 84 plies per update, the CPU network)
# for BENCH_UPDATES updates, then reports the seconds per update, the
# throughput, the GPU's peak memory and what the full 3,000-update run would
# take and cost at PRICE dollars per hour.
#
#   bash /workspace/open_spiel/gpu_pod/bench_step1.sh            # 20 updates
#   BENCH_UPDATES=30 PRICE=0.59 bash .../bench_step1.sh
#
# BENCH_FLAGS adds train flags (a local test at a tiny batch uses it).
# Run pod_setup.sh first. The benchmark's checkpoints go to a scratch dir
# that is wiped at each run; the log stays in /workspace/bench_step1.log.
set -euo pipefail

WS=${WS:-/workspace}
REPO_DIR=${REPO_DIR:-$WS/open_spiel}
VENV=${VENV:-$WS/venv}
BENCH_UPDATES=${BENCH_UPDATES:-20}
FULL_UPDATES=${FULL_UPDATES:-3000}
PRICE=${PRICE:-0.69}
OUT=$WS/bench_step1
LOG=$WS/bench_step1.log
MEM=$WS/bench_step1.gpu_mem

# shellcheck disable=SC1091
. "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR"
rm -rf "$OUT"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$MEM" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

echo "=== $(date '+%F %T') schnaps_normal_gpu, $BENCH_UPDATES updates" | tee "$LOG"
START=$(date +%s)
python -m open_spiel.python.scale.train \
  --config=schnaps_normal_gpu \
  --checkpoint_dir="$OUT" \
  --num_updates="$BENCH_UPDATES" ${BENCH_FLAGS:-} >> "$LOG" 2>&1
END=$(date +%s)
kill $SMI 2>/dev/null || true
echo "=== $(date '+%F %T') done in $((END - START)) s" | tee -a "$LOG"

python - "$LOG" "$MEM" "$FULL_UPDATES" "$PRICE" <<'EOF'
import re, statistics, sys
log, mem, full, price = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
text = open(log).read()
secs = [float(x) for x in re.findall(r"update_seconds=([0-9.]+)", text)]
fps = [float(x) for x in re.findall(r"frames_per_s=([0-9.]+)", text)]
peak = max((int(l) for l in open(mem) if l.strip().isdigit()), default=0)
if len(secs) < 3:
  sys.exit(f"only {len(secs)} updates logged; see {log}")
steady = secs[2:]  # the first updates include compilation
med = statistics.median(steady)
hours = med * full / 3600
print(f"updates logged     {len(secs)}  (first two: {secs[0]:.1f} s, {secs[1]:.1f} s, compiling)")
print(f"seconds per update {med:.2f} median, {min(steady):.2f} to {max(steady):.2f}")
print(f"frames per second  {statistics.median(fps[2:]):,.0f} median")
print(f"GPU memory peak    {peak:,} MiB")
print(f"full run           {full} updates ~ {hours:.1f} h ~ ${hours * price:.2f} at ${price}/h"
      " (training only: add the evals every 100 updates and startup)")
print("laptop reference   ~34 s per update at 128 tables (32x fewer plies)")
EOF
