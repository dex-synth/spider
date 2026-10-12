#!/usr/bin/env bash
# Bounded native fast-recipe diagnostic; NOT the articulated common task.
set -euo pipefail
prefix=${1:?ENV_PREFIX ARCTIC_DATA MANO_DIR OUTPUT_DIR}
arctic_data=${2:?}
mano_dir=${3:?}
output=${4:?}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$output/source-layout/unpack/arctic_data" "$output/mano/models"
ln -sfn "$arctic_data" "$output/source-layout/unpack/arctic_data/data"
for side in LEFT RIGHT; do ln -sfn "$mano_dir/MANO_${side}.pkl" "$output/mano/models/MANO_${side}.pkl"; done
cd "$repo"
"$prefix/bin/python" spider/process_datasets/arcticv2.py \
  --arctic-root "$output/source-layout" --mano-assets-root "$output/mano" \
  --dataset-dir "$output/datasets" --subject s01 --sequence box_use_01 --no-show-viewer
common=(--dataset-dir "$output/datasets" --dataset-name arcticv2 --robot-type xhand \
        --embodiment-type bimanual --task s01-box_use_01 --data-id 0)
"$prefix/bin/python" spider/preprocess/decompose_fast.py "${common[@]}"
ln -sfn "$repo/spider/assets/robots" "$output/datasets/processed/arcticv2/assets/robots"
"$prefix/bin/python" spider/preprocess/generate_xml.py "${common[@]}" --no-show-viewer
"$prefix/bin/python" spider/preprocess/ik_fast.py "${common[@]}" --no-show-viewer --no-save-video
"$prefix/bin/python" examples/run_mjwp.py +override=arcticv2_fast \
  dataset_dir="$output/datasets" show_viewer=false save_video=false ++wait_on_finish=false \
  num_samples=32 max_num_iterations=1 max_sim_steps=4 ++use_torch_compile=false \
  ++output_dir="$output/mpc"
