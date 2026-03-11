#!/bin/bash
# SGLATrack 快速啟動腳本

echo "=================================="
echo "SGLATrack 模型還原與測試工具"
echo "=================================="
echo ""

# 檢查 conda 環境
if ! conda env list | grep -q "sgla"; then
    echo "❌ 錯誤: sgla 環境不存在"
    echo "請先執行: conda env create -f environment_sgla1.yml"
    exit 1
fi

echo "✅ Conda 環境 'sgla' 已找到"

# 檢查預訓練模型
if [ ! -f "pretrained_models/deit_tiny_distilled_patch16_224.pth" ]; then
    echo "❌ 錯誤: 找不到預訓練模型"
    echo "請確認檔案存在: pretrained_models/deit_tiny_distilled_patch16_224.pth"
    exit 1
fi

echo "✅ 預訓練模型檔案已找到"

# 啟用環境
echo ""
echo "啟用 conda 環境..."
eval "$(conda shell.bash hook)"
conda activate sgla

echo "✅ 環境已啟用: $(which python)"
echo "Python 版本: $(python --version)"

echo ""
echo "=================================="
echo "請選擇操作:"
echo "=================================="
echo "1) 下載模型權重（用於測試）"
echo "2) 訓練模型（單 GPU）"
echo "3) 訓練模型（多 GPU，4 張）"
echo "4) 測試模型 - UAV123"
echo "5) 測試模型 - UAV123_10fps"
echo "6) 測試模型 - UAVTrack112"
echo "7) 測試模型 - UAVDT"
echo "8) 測試模型 - DTB70"
echo "9) 分析測試結果"
echo "10) 測試模型效能（FLOPs/速度）"
echo "0) 退出"
echo ""
read -p "請輸入選項 [0-10]: " choice

case $choice in
    1)
        echo ""
        echo "📥 請手動下載模型權重："
        echo "連結: https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X?usp=sharing"
        echo ""
        echo "下載後請放置到："
        echo "output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
        echo ""
        echo "建立必要目錄..."
        mkdir -p output/checkpoints/train/sglatrack/deit_distilled/
        echo "✅ 目錄已建立，請將下載的模型放到上述路徑"
        ;;
    2)
        echo ""
        echo "🚀 開始單 GPU 訓練..."
        python tracking/train.py \
            --script sglatrack \
            --config deit_distilled \
            --save_dir ./output \
            --mode single \
            --use_wandb 0
        ;;
    3)
        echo ""
        echo "🚀 開始多 GPU 訓練（4 GPU）..."
        python tracking/train.py \
            --script sglatrack \
            --config deit_distilled \
            --save_dir ./output \
            --mode multiple \
            --nproc_per_node 4 \
            --use_wandb 0
        ;;
    4)
        echo ""
        echo "🧪 測試 UAV123 資料集..."
        python tracking/test.py \
            --tracker_param sglatrack \
            --dataset uav123 \
            --threads 8 \
            --num_gpus 4
        ;;
    5)
        echo ""
        echo "🧪 測試 UAV123_10fps 資料集..."
        python tracking/test.py \
            --tracker_param sglatrack \
            --dataset uav123_10fps \
            --threads 8 \
            --num_gpus 4
        ;;
    6)
        echo ""
        echo "🧪 測試 UAVTrack112 資料集..."
        python tracking/test.py \
            --tracker_param sglatrack \
            --dataset uavtrack112 \
            --threads 8 \
            --num_gpus 4
        ;;
    7)
        echo ""
        echo "🧪 測試 UAVDT 資料集..."
        python tracking/test.py \
            --tracker_param sglatrack \
            --dataset uavdt \
            --threads 8 \
            --num_gpus 4
        ;;
    8)
        echo ""
        echo "🧪 測試 DTB70 資料集..."
        python tracking/test.py \
            --tracker_param sglatrack \
            --dataset dtb70 \
            --threads 8 \
            --num_gpus 4
        ;;
    9)
        echo ""
        echo "📊 分析測試結果..."
        python tracking/analysis_results.py
        ;;
    10)
        echo ""
        echo "⚡ 測試模型效能（FLOPs 和速度）..."
        python tracking/profile_model.py
        ;;
    0)
        echo "退出程式"
        exit 0
        ;;
    *)
        echo "❌ 無效的選項"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "操作完成!"
echo "=================================="
