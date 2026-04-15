#!/bin/bash
# 一鍵執行：train -> test -> evaluation (AUC/Precision)
#
# 預設使用：
#   experiments/sglatrack/vit_coco_uav123_mala_relu.yaml
#
# 用法：
#   bash python/run_all_mala_relu.sh
#   CONFIG=vit_coco_uav123_mala_relu NUM_GPUS=1 DATASET=uav123 bash python/run_all_mala_relu.sh
#
# 注意：此腳本假設你從 repo root 執行（也可在其他路徑執行，會自動切到專案 root）。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/python"

# ----------------------
# 可調參數（可用環境變數覆蓋）
# ----------------------
CONFIG="${CONFIG:-vit_coco_uav123_mala_relu}"   # 不含 .yaml
SCRIPT="${SCRIPT:-sglatrack}"
SAVE_DIR="${SAVE_DIR:-output}"
MODE="${MODE:-multiple}"                       # single | multiple | multi_node（通常用 single/multiple）
NUM_GPUS="${NUM_GPUS:-1}"                      # multiple 模式下用到

DATASET="${DATASET:-uav123}"                   # test dataset name
THREADS="${THREADS:-8}"
TEST_NUM_GPUS="${TEST_NUM_GPUS:-1}"

# evaluation 畫圖（PDF）
PLOT="${PLOT:-1}"                               # 1=產圖, 0=不產圖

echo "========================================================================"
echo "一鍵流程：train -> test -> evaluation"
echo "========================================================================"
echo "ROOT_DIR      = $ROOT_DIR"
echo "SCRIPT        = $SCRIPT"
echo "CONFIG        = $CONFIG (experiments/$SCRIPT/$CONFIG.yaml)"
echo "SAVE_DIR      = $SAVE_DIR"
echo "MODE          = $MODE"
echo "NUM_GPUS      = $NUM_GPUS"
echo "DATASET       = $DATASET"
echo "THREADS       = $THREADS"
echo "TEST_NUM_GPUS = $TEST_NUM_GPUS"
echo "PLOT          = $PLOT"
echo "========================================================================"
echo ""

cleanup_dir () {
  local label="$1"
  local target_dir="$2"

  echo "[Cleanup] $label"
  echo "  path: $target_dir"

  # 安全檢查：避免誤刪到空路徑或根目錄
  if [[ -z "${target_dir:-}" || "$target_dir" == "/" || "$target_dir" == "." ]]; then
    echo "  ❌ 安全檢查失敗：目標路徑非法"
    exit 1
  fi

  if [[ -d "$target_dir" ]]; then
    echo "  action: remove"
    rm -rf -- "$target_dir"
  else
    echo "  action: skip (not found)"
  fi
  echo ""
}

echo "[0/3] Cleanup..."
cleanup_dir "train checkpoints (same as CONFIG)" "$SAVE_DIR/checkpoints/train/$SCRIPT/$CONFIG"
cleanup_dir "test tracking_results (same as CONFIG)" "$SAVE_DIR/test/tracking_results/$SCRIPT/$CONFIG"
cleanup_dir "evaluation result_plots cache (same as DATASET)" "$SAVE_DIR/test/result_plots/$DATASET"

echo "[1/3] Train..."
if [[ "$MODE" == "single" ]]; then
  python tracking/train.py \
    --script "$SCRIPT" \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    --mode single
else
  python tracking/train.py \
    --script "$SCRIPT" \
    --config "$CONFIG" \
    --save_dir "$SAVE_DIR" \
    --mode multiple \
    --nproc_per_node "$NUM_GPUS"
fi

echo ""
echo "[2/3] Test (generate tracking results)..."
python tracking/test.py "$SCRIPT" "$CONFIG" \
  --dataset_name "$DATASET" \
  --threads "$THREADS" \
  --num_gpus "$TEST_NUM_GPUS"

echo ""
echo "[3/3] Evaluation (AUC / Precision / Norm Precision)..."
if [[ "$PLOT" == "1" ]]; then
  python tracking/calculate_metrics.py \
    --tracker "$SCRIPT" \
    --param "$CONFIG" \
    --dataset "$DATASET" \
    --plot
else
  python tracking/calculate_metrics.py \
    --tracker "$SCRIPT" \
    --param "$CONFIG" \
    --dataset "$DATASET"
fi

echo ""
echo "========================================================================"
echo "全部完成！"
echo "- checkpoints: $SAVE_DIR/checkpoints/train/$SCRIPT/$CONFIG/"
echo "- test results: $SAVE_DIR/test/tracking_results/$SCRIPT/$CONFIG/$DATASET/"
echo "- eval plots: $SAVE_DIR/test/result_plots/$DATASET/"
echo "========================================================================"

echo ""
echo "[Profile] forward_test 速度（MACs / latency / FPS，隨機張量，非完整追蹤 pipeline）"
python tracking/profile_model.py --script "$SCRIPT" --config "$CONFIG"

