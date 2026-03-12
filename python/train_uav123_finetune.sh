#!/bin/bash
# UAV123 + COCO Finetune 訓練腳本
# 使用預訓練權重進行 finetune

set -e

echo "========================================================================"
echo "UAV123 + COCO Finetune 訓練"
echo "========================================================================"
echo ""

# 配置（注意：不要加 experiments/ 前綴，train.py 會自動添加）
CONFIG="deit_distilled_uav123_finetune"  # ✓ 使用實際存在的配置檔案
NUM_GPUS=1

# 檢查預訓練權重是否存在
PRETRAIN_WEIGHT="output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
if [ ! -f "$PRETRAIN_WEIGHT" ]; then
    echo "❌ 找不到預訓練權重: $PRETRAIN_WEIGHT"
    echo ""
    echo "請確認您有以下檔案："
    echo "  - $PRETRAIN_WEIGHT"
    exit 1
fi

echo "✓ 找到預訓練權重: $PRETRAIN_WEIGHT"
echo "✓ 配置檔案: experiments/sglatrack/$CONFIG.yaml"
echo "✓ 使用 GPU 數量: $NUM_GPUS"
echo ""

# 顯示配置資訊
echo "訓練設定："
echo "  - Detection Dataset: COCO17"
echo "  - Tracking Dataset: UAV123"
echo "  - 學習率: 0.0001 (finetune 用較小的學習率)"
echo "  - Epoch: 100"
echo "  - Batch Size: 32"
echo "  - 預訓練權重: 從 deit_distilled 載入"
echo ""
echo "開始訓練..."
echo ""

# 執行訓練
# 注意：--config 只需要檔案名稱，不需要完整路徑
python tracking/train.py \
    --script sglatrack \
    --config "$CONFIG" \
    --save_dir output \
    --mode multiple \
    --nproc_per_node $NUM_GPUS \
    --script_prv sglatrack \
    --config_prv deit_distilled

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "訓練完成！"
    echo "========================================================================"
    echo ""
    echo "模型儲存位置: output/checkpoints/train/sglatrack/deit_distilled_uav123_finetune/"
    echo ""
    echo "下一步："
    echo "  1. 測試模型: python tracking/test.py sglatrack deit_distilled_uav123_finetune --dataset_name uav123 --threads 4 --num_gpus 1"
    echo "  2. 計算指標: python tracking/calculate_metrics.py --param deit_distilled_uav123_finetune --dataset uav123 --plot"
else
    echo "訓練失敗！退出代碼: $TRAIN_EXIT_CODE"
    echo "========================================================================"
    echo ""
    echo "請檢查上方的錯誤訊息"
fi
echo ""
