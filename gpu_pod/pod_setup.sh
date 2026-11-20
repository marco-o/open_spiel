#!/bin/bash
# Sets up a RunPod GPU pod for the schnaps training (Docker/gpu-training.md).
#
# Run on the pod, from any directory:
#   git clone -b schnaps https://github.com/marco-o/open_spiel /workspace/open_spiel
#   bash /workspace/open_spiel/gpu_pod/pod_setup.sh
#
# It installs uv and Python 3.14 (the laptop's version), makes the venv
# /workspace/venv with the laptop's package versions plus JAX for CUDA, and
# checks that JAX sees the GPU and that the step-1 config loads. Training
# needs no C++ build: pyspiel is only used for serving and replays.
# Everything lives under /workspace, the pod's persistent volume, so a
# stopped pod keeps it. Running the script again is safe.
set -euo pipefail

WS=${WS:-/workspace}
REPO_DIR=${REPO_DIR:-$WS/open_spiel}
VENV=${VENV:-$WS/venv}
PY_VERSION=${PY_VERSION:-3.14}
JAX_VERSION=${JAX_VERSION:-0.12.0.1}

say() { echo "=== $(date '+%F %T') $*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v uv >/dev/null; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

say "Python $PY_VERSION and the venv $VENV"
uv python install "$PY_VERSION"
[ -x "$VENV/bin/python" ] || uv venv --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"

say "packages"
if ! uv pip install -r "$REPO_DIR/gpu_pod/requirements.txt"; then
  say "pinned install failed; retrying with tfp-nightly unpinned"
  grep -v '^tfp-nightly' "$REPO_DIR/gpu_pod/requirements.txt" > /tmp/req.txt
  uv pip install -r /tmp/req.txt tfp-nightly
fi
if ! uv pip install "jax[cuda12]==$JAX_VERSION"; then
  say "jax[cuda12] unavailable; trying jax[cuda13]"
  uv pip install "jax[cuda13]==$JAX_VERSION"
fi

export PYTHONPATH="$REPO_DIR"
say "JAX devices"
python - <<'EOF'
import jax
devs = jax.devices()
print(jax.__version__, devs)
assert devs[0].platform == "gpu", "JAX does not see the GPU"
EOF

say "step-1 config test"
cd "$REPO_DIR/open_spiel/python/scale"
python -m pytest -q -p no:cacheprovider configs/tests/schnaps_normal_gpu_test.py

say "setup done; next: bash $REPO_DIR/gpu_pod/bench_step1.sh"
