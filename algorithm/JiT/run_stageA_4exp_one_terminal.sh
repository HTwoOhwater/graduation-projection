#!/usr/bin/env bash
set -euo pipefail

# One-terminal launcher for 4 concurrent Stage-A experiments on 4 GPUs.
# Updated per request: doubled batch size and epochs (8->16, 50->100).

ROOT_DIR="/mnt/workspace/Dehaze"
JIT_DIR="$ROOT_DIR/algorithm/JiT"
VENV_ACT="$ROOT_DIR/.venv/bin/activate"
RESULT_DIR="$ROOT_DIR/result"

cd "$JIT_DIR"
source "$VENV_ACT"

COMMON_ARGS=(
  --dataset_mode paired
  --data_path "$ROOT_DIR/datasets"
  --train_split train
  --lq_dirname haze_images
  --gt_dirname original_images
  --pairing_mode stem
  --model JiT-B/16
  --img_size 256
  --batch_size 16
  --epochs 100
  --num_workers 4
  --save_last_freq 1
  --eval_freq 1
  --device cuda
)

start_exp() {
  local gpu="$1"
  local out_name="$2"
  shift 2
  echo "[$(date '+%F %T')] Start $out_name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python main_jit.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$RESULT_DIR/$out_name" \
    "$@" \
    > "$RESULT_DIR/${out_name}.log" 2>&1 &
  echo "$!"
}

PID1=$(start_exp 0 jit_stageA_l1_x2 --loss_type l1)
PID2=$(start_exp 1 jit_stageA_l2_x2 --loss_type l2)
PID3=$(start_exp 2 jit_stageA_l1l2_73_x2 --loss_type l1_l2 --loss_l1_weight 0.7 --loss_l2_weight 0.3)
PID4=$(start_exp 3 jit_stageA_l1l2_37_x2 --loss_type l1_l2 --loss_l1_weight 0.3 --loss_l2_weight 0.7)

echo "Launched PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Logs:"
echo "  $RESULT_DIR/jit_stageA_l1_x2.log"
echo "  $RESULT_DIR/jit_stageA_l2_x2.log"
echo "  $RESULT_DIR/jit_stageA_l1l2_73_x2.log"
echo "  $RESULT_DIR/jit_stageA_l1l2_37_x2.log"

echo "Use the following command to monitor all four logs:"
echo "  tail -f $RESULT_DIR/jit_stageA_l1_x2.log $RESULT_DIR/jit_stageA_l2_x2.log $RESULT_DIR/jit_stageA_l1l2_73_x2.log $RESULT_DIR/jit_stageA_l1l2_37_x2.log"

wait
