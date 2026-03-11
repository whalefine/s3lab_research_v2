#!/bin/bash
# SGLATrack 測試命令修正指南

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            SGLATrack 測試命令修正                                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "❌ 錯誤的命令格式:"
echo "   python tracking/test.py --tracker_param sglatrack --dataset uav123 ..."
echo ""

echo "✅ 正確的命令格式:"
echo "   python tracking/test.py <tracker_name> <tracker_param> --dataset_name <dataset> ..."
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "正確的測試命令"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

echo "1️⃣  測試 UAV123 (使用 deit_distilled 配置)"
echo "────────────────────────────────────────────────────────────────────"
echo "python tracking/test.py sglatrack deit_distilled \\"
echo "  --dataset_name uav123 --threads 4 --num_gpus 1"
echo ""

echo "2️⃣  測試 UAV123_10fps"
echo "────────────────────────────────────────────────────────────────────"
echo "python tracking/test.py sglatrack deit_distilled \\"
echo "  --dataset_name uav123_10fps --threads 4 --num_gpus 1"
echo ""

echo "3️⃣  測試 UAVDT"
echo "────────────────────────────────────────────────────────────────────"
echo "python tracking/test.py sglatrack deit_distilled \\"
echo "  --dataset_name uavdt --threads 4 --num_gpus 1"
echo ""

echo "4️⃣  測試單個序列 (例如 bike1)"
echo "────────────────────────────────────────────────────────────────────"
echo "python tracking/test.py sglatrack deit_distilled \\"
echo "  --dataset_name uav123 --sequence bike1 --threads 1 --num_gpus 1"
echo ""

echo "5️⃣  使用其他配置 (如 coco_uav123)"
echo "────────────────────────────────────────────────────────────────────"
echo "python tracking/test.py sglatrack coco_uav123 \\"
echo "  --dataset_name uav123 --threads 4 --num_gpus 1"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "命令參數說明"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "必需參數 (位置參數):"
echo "  tracker_name       - 追蹤器名稱 (固定為 'sglatrack')"
echo "  tracker_param      - 配置檔案名稱 (不含 .yaml)"
echo "                       可用: deit_distilled, coco_only, coco_uav123"
echo ""
echo "可選參數:"
echo "  --dataset_name     - 資料集名稱"
echo "                       可用: uav123, uav123_10fps, uavdt, dtb70 等"
echo "  --threads          - 執行緒數量 (預設: 0)"
echo "  --num_gpus         - GPU 數量 (預設: 1)"
echo "  --sequence         - 測試特定序列"
echo "  --debug            - 除錯等級 (0-3)"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "可用的配置檔案"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "1. deit_distilled   - 原始完整訓練配置"
echo "                      模型位置: output/checkpoints/train/sglatrack/deit_distilled/"
echo ""
echo "2. coco_only        - COCO + UAV123 訓練配置"
echo "                      模型位置: output/checkpoints/train/sglatrack/coco_only/"
echo ""
echo "3. coco_uav123      - COCO + UAV123 組合訓練"
echo "                      模型位置: output/checkpoints/train/sglatrack/coco_uav123/"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "可用的測試資料集"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "✓ uav123         - UAV123 標準測試集"
echo "✓ uav123_10fps   - UAV123 10fps 版本"
echo "○ uavdt          - UAVDT 資料集 (需下載)"
echo "○ dtb70          - DTB70 資料集 (需下載)"
echo "○ uavtrack112    - UAVTrack112 資料集 (需下載)"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "目前可用的模型"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
if [ -f "output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar" ]; then
    echo "✓ deit_distilled/sglatrack_ep0297.pth.tar (可用)"
else
    echo "✗ deit_distilled - 無模型檔案"
fi

if [ -d "output/checkpoints/train/sglatrack/coco_only" ]; then
    echo "✓ coco_only - 目錄存在"
    ls output/checkpoints/train/sglatrack/coco_only/*.pth.tar 2>/dev/null | wc -l | xargs echo "  模型數量:"
else
    echo "○ coco_only - 未訓練"
fi

if [ -d "output/checkpoints/train/sglatrack/coco_uav123" ]; then
    echo "✓ coco_uav123 - 目錄存在"
    ls output/checkpoints/train/sglatrack/coco_uav123/*.pth.tar 2>/dev/null | wc -l | xargs echo "  模型數量:"
else
    echo "○ coco_uav123 - 未訓練"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "快速開始"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "使用現有模型測試 UAV123:"
echo ""
echo "conda activate sgla"
echo "python tracking/test.py sglatrack deit_distilled \\"
echo "  --dataset_name uav123 --threads 4 --num_gpus 1"
echo ""
echo "查看結果:"
echo "ls output/test/tracking_results/sglatrack/deit_distilled/uav123/"
echo ""
