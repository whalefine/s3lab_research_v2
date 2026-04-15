#!/bin/bash
# 使用已訓練好的模型，依「清單檔」順序對多個 test dataset 逐一跑 tracking/test.py
# （可選）每個 dataset：test →（可選）calculate_metrics.py
# 全部完成後：預設於終端機印出彙整文字表（含各 dataset 之 AUC/P、平均、profile_model 之 GPU FPS）
#
# 清單檔格式（純文字）：
#   - 每行一個 dataset 名稱（與 --dataset_name 相同，例如 uav123、uavdt）
#   - 以 # 開頭的行視為註解
#   - 空行會略過
#
# 用法（建議在 repo root 執行）：
#   bash python/run_test_datasets.sh
#   bash python/run_test_datasets.sh /path/to/my_datasets.txt
#   CONFIG=vit_coco_uav123_care SCRIPT=sglatrack bash python/run_test_datasets.sh
#
# 環境變數（皆可選，有預設值）：
#   DATASET_LIST   清單檔路徑（若第一個參數有給檔案，則優先於此）
#   SCRIPT         tracker 腳本名，預設 sglatrack
#   CONFIG         實驗設定檔名（不含 .yaml），須與訓練時相同
#   SAVE_DIR       預設 output（與 train/test 輸出目錄一致）
#   THREADS        預設 8（並行序列數；單機常見 4～8）
#   TEST_NUM_GPUS  預設 1（單張 GPU 請維持 1）
#   SUMMARY_TABLE  預設 1：最後執行 tracking/benchmark_summary_table.py（讀 eval_data、跑 profile_model、印出文字表）
#   CALC_FPS       在 SUMMARY_TABLE=1 時：預設 1 會於彙整腳本內跑 profile_model；設 0 則傳 --skip_profile（FPS 欄為 —）
#                  在 SUMMARY_TABLE=0 時：預設 1 則單獨執行 profile_model.py 僅印出 FPS
#   CALC_METRICS   預設 1：再執行 calculate_metrics；設 0 則跳過（彙整表需有 eval_data，請先跑過 metrics）
#   PLOT           僅在 CALC_METRICS=1 時有效；預設 1 會加 --plot

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/python"

SCRIPT="${SCRIPT:-sglatrack}"
CONFIG="${CONFIG:-vit_coco_uav123_care}"
SAVE_DIR="${SAVE_DIR:-output}"
THREADS="${THREADS:-8}"
TEST_NUM_GPUS="${TEST_NUM_GPUS:-1}"
SUMMARY_TABLE="${SUMMARY_TABLE:-1}"
CALC_FPS="${CALC_FPS:-1}"
CALC_METRICS="${CALC_METRICS:-1}"
PLOT="${PLOT:-1}"

if [[ "${1:-}" != "" && -f "${1:-}" ]]; then
  LIST_FILE="$1"
elif [[ -n "${DATASET_LIST:-}" ]]; then
  LIST_FILE="$DATASET_LIST"
else
  LIST_FILE="$ROOT_DIR/python/test_datasets_list.txt"
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "錯誤：找不到清單檔：$LIST_FILE"
  echo "可複製範例： cp \"$ROOT_DIR/python/test_datasets_list.example.txt\" \"$ROOT_DIR/python/test_datasets_list.txt\""
  echo "或指定路徑： DATASET_LIST=/path/to/list.txt bash $0"
  exit 1
fi

echo "========================================================================"
echo "依清單順序測試多個 dataset"
echo "========================================================================"
echo "LIST_FILE     = $LIST_FILE"
echo "SCRIPT        = $SCRIPT"
echo "CONFIG        = $CONFIG (experiments/$SCRIPT/$CONFIG.yaml)"
echo "SAVE_DIR      = $SAVE_DIR"
echo "THREADS       = $THREADS"
echo "TEST_NUM_GPUS = $TEST_NUM_GPUS"
echo "SUMMARY_TABLE = $SUMMARY_TABLE"
echo "CALC_FPS      = $CALC_FPS"
echo "CALC_METRICS  = $CALC_METRICS"
echo "PLOT          = $PLOT"
echo "========================================================================"
echo ""

idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
  # 去掉前後空白
  ds="${line#"${line%%[![:space:]]*}"}"
  ds="${ds%"${ds##*[![:space:]]}"}"

  [[ -z "$ds" ]] && continue
  [[ "$ds" =~ ^# ]] && continue

  idx=$((idx + 1))
  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "[$idx] dataset: $ds"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  python tracking/test.py "$SCRIPT" "$CONFIG" \
    --dataset_name "$ds" \
    --threads "$THREADS" \
    --num_gpus "$TEST_NUM_GPUS"

  if [[ "$CALC_METRICS" == "1" ]]; then
    if [[ "$PLOT" == "1" ]]; then
      python tracking/calculate_metrics.py \
        --tracker "$SCRIPT" \
        --param "$CONFIG" \
        --dataset "$ds" \
        --plot
    else
      python tracking/calculate_metrics.py \
        --tracker "$SCRIPT" \
        --param "$CONFIG" \
        --dataset "$ds"
    fi
  fi

done < "$LIST_FILE"

if [[ "$SUMMARY_TABLE" == "1" && "$idx" -gt 0 ]]; then
  echo ""
  echo "========================================================================"
  echo "[彙整表] benchmark_summary_table.py（AUC / P / Avg. / GPU FPS）"
  echo "========================================================================"
  SUMMARY_EXTRA=()
  if [[ "$CALC_FPS" == "0" ]]; then
    SUMMARY_EXTRA+=(--skip_profile)
  fi
  python tracking/benchmark_summary_table.py \
    --tracker "$SCRIPT" \
    --param "$CONFIG" \
    --dataset_list "$LIST_FILE" \
    --script "$SCRIPT" \
    "${SUMMARY_EXTRA[@]}"
elif [[ "$CALC_FPS" == "1" && "$idx" -gt 0 ]]; then
  echo ""
  echo "========================================================================"
  echo "[FPS] profile_model：隨機張量 forward_test（MACs / latency / FPS，非完整追蹤 pipeline）"
  echo "========================================================================"
  python tracking/profile_model.py --script "$SCRIPT" --config "$CONFIG"
fi

echo ""
echo "========================================================================"
echo "全部 dataset 跑完（共 $idx 個）。"
echo "結果目錄：$SAVE_DIR/test/tracking_results/$SCRIPT/$CONFIG/<dataset>/"
if [[ "$SUMMARY_TABLE" == "1" ]]; then
  echo "彙整表：見上方文字輸出（benchmark_summary_table.py）"
elif [[ "$CALC_FPS" == "1" ]]; then
  echo "FPS：見上方 profile_model 輸出"
fi
if [[ "$CALC_METRICS" == "1" ]]; then
  echo "評估圖表：$SAVE_DIR/test/result_plots/<dataset>/"
fi
echo "========================================================================"
