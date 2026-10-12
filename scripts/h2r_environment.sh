#!/usr/bin/env bash
# Native dependencies, constrained to CUDA 12 for the current benchmark host.
set -euo pipefail
prefix=${1:?Usage: bash scripts/h2r_environment.sh ENV_PREFIX}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ ! -x "$prefix/bin/python" ]; then uv venv --python 3.12 "$prefix"; fi
uv pip install --python "$prefix/bin/python" --index-strategy unsafe-best-match \
  --index https://pypi.nvidia.com \
  --default-index https://pypi.org/simple \
  -e "$repo" 'torch==2.7.1' 'pandas>=2,<3'

# ARCTIC's native processor uses manotorch and legacy Chumpy pickles.
# manotorch pins an obsolete OpenCV wheel: use the already-resolved modern one.
uv pip install --python "$prefix/bin/python" 'setuptools<81' pip
uv pip install --python "$prefix/bin/python" --no-build-isolation 'chumpy==0.70'
uv pip install --python "$prefix/bin/python" --no-deps \
  'manotorch @ git+https://github.com/lixiny/manotorch.git@a2a70c591f91551078b7bb2af9b5d9f275b626e0'
