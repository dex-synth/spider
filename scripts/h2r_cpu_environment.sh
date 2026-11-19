#!/usr/bin/env bash
# Environment of the CPU recipe (spider.h2r.cpu_plan): MuJoCo's CPU batch API, no GPU.
# It is separate from scripts/h2r_environment.sh, whose MJWarp pins MuJoCo 3.7.
set -euo pipefail
prefix=${1:?Usage: bash scripts/h2r_cpu_environment.sh ENV_PREFIX}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ ! -x "$prefix/bin/python" ]; then uv venv --python 3.12.14 "$prefix"; fi
# The packages examples/run_cpu.py and its viewer modules import, plus the tests'.
# matplotlib is missing from upstream's list: the Viser panel colours traces with it.
uv pip install --python "$prefix/bin/python" \
  --constraint "$repo/scripts/h2r_cpu_runtime_constraints.txt" \
  'mujoco==3.17.0' 'torch==2.7.1' numpy scipy pyyaml loguru hydra-core tyro \
  opencv-python-headless 'imageio[ffmpeg]' 'rerun-sdk==0.26.2' viser trimesh \
  matplotlib pytest \
  'h2r-io @ git+ssh://git@github.com/dex-synth/h2r_bench.git@h2r-io-v1#subdirectory=packages/h2r-io'
uv pip install --python "$prefix/bin/python" --no-deps -e "$repo"
