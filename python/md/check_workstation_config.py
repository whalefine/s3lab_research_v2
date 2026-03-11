#!/usr/bin/env python3
"""
驗證工作站配置的腳本
檢查 COCO 和 UAV123 資料集是否正確配置
"""

import os
import sys
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_item(condition, success_msg, fail_msg, fix_msg=None):
    """檢查項目並輸出結果"""
    if condition:
        print(f"{Colors.GREEN}✓{Colors.END} {success_msg}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.END} {fail_msg}")
        if fix_msg:
            print(f"{Colors.YELLOW}  → {fix_msg}{Colors.END}")
        return False

def main():
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}工作站資料集配置驗證{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    all_ok = True
    
    # 1. 檢查 COCO 資料集
    print(f"{Colors.BLUE}[1] COCO 資料集檢查{Colors.END}")
    
    coco_dir = project_root / "data" / "coco"
    if check_item(coco_dir.exists(), f"COCO 目錄存在: {coco_dir}", 
                  f"COCO 目錄不存在: {coco_dir}"):
        
        # 檢查 annotations
        anno_dir = coco_dir / "annotations"
        train_anno = anno_dir / "instances_train2017.json"
        val_anno = anno_dir / "instances_val2017.json"
        
        check_item(train_anno.exists(), 
                   f"訓練標註檔案存在: {train_anno.name} ({train_anno.stat().st_size / 1024 / 1024:.0f} MB)",
                   f"訓練標註檔案不存在: {train_anno}",
                   "執行: cd data/coco && unzip annotations_trainval2017.zip")
        
        check_item(val_anno.exists(), 
                   f"驗證標註檔案存在: {val_anno.name} ({val_anno.stat().st_size / 1024 / 1024:.0f} MB)",
                   f"驗證標註檔案不存在: {val_anno}",
                   "執行: cd data/coco && unzip annotations_trainval2017.zip")
        
        # 檢查 images
        images_dir = coco_dir / "images"
        train_images = images_dir / "train2017"
        val_images = images_dir / "val2017"
        
        if train_images.exists():
            train_count = len(list(train_images.glob("*.jpg")))
            check_item(train_count > 0, 
                      f"訓練圖片: {train_count:,} 張",
                      f"訓練圖片目錄為空: {train_images}")
        else:
            all_ok = False
            print(f"{Colors.RED}✗{Colors.END} 訓練圖片目錄不存在: {train_images}")
        
        if val_images.exists():
            val_count = len(list(val_images.glob("*.jpg")))
            check_item(val_count > 0, 
                      f"驗證圖片: {val_count:,} 張",
                      f"驗證圖片目錄為空: {val_images}")
        else:
            all_ok = False
            print(f"{Colors.RED}✗{Colors.END} 驗證圖片目錄不存在: {val_images}")
    else:
        all_ok = False
    
    # 2. 檢查 UAV123 資料集
    print(f"\n{Colors.BLUE}[2] UAV123 資料集檢查{Colors.END}")
    
    uav123_dir = project_root / "data" / "uav123" / "UAV123"
    if check_item(uav123_dir.exists(), f"UAV123 目錄存在: {uav123_dir}", 
                  f"UAV123 目錄不存在: {uav123_dir}"):
        
        anno_dir = uav123_dir / "anno"
        data_seq_dir = uav123_dir / "data_seq"
        
        if anno_dir.exists():
            anno_count = len(list(anno_dir.glob("*.txt")))
            check_item(anno_count > 0, 
                      f"標註檔案: {anno_count} 個",
                      f"標註目錄為空: {anno_dir}")
        else:
            all_ok = False
            print(f"{Colors.RED}✗{Colors.END} 標註目錄不存在: {anno_dir}")
        
        if data_seq_dir.exists():
            seq_count = len([d for d in data_seq_dir.iterdir() if d.is_dir()])
            check_item(seq_count > 0, 
                      f"影像序列: {seq_count} 個",
                      f"影像序列目錄為空: {data_seq_dir}")
        else:
            all_ok = False
            print(f"{Colors.RED}✗{Colors.END} 影像序列目錄不存在: {data_seq_dir}")
    else:
        all_ok = False
    
    # 3. 檢查 UAV123_10fps (可選)
    print(f"\n{Colors.BLUE}[3] UAV123_10fps 資料集檢查 (可選){Colors.END}")
    
    uav123_10fps_dir = project_root / "data" / "uav123_10fps"
    if uav123_10fps_dir.exists():
        print(f"{Colors.GREEN}✓{Colors.END} UAV123_10fps 目錄存在")
    else:
        print(f"{Colors.YELLOW}○{Colors.END} UAV123_10fps 目錄不存在（非必需）")
    
    # 4. 檢查配置檔案
    print(f"\n{Colors.BLUE}[4] 配置檔案檢查{Colors.END}")
    
    config_files = [
        ("COCO 單獨訓練配置", "experiments/sglatrack/coco_only.yaml"),
        ("原始配置", "experiments/sglatrack/deit_distilled.yaml"),
        ("測試路徑配置", "lib/test/evaluation/local.py"),
        ("訓練路徑配置", "lib/train/admin/local.py"),
    ]
    
    for name, path in config_files:
        file_path = project_root / path
        check_item(file_path.exists(), f"{name}: {path}", f"{name} 不存在: {path}")
    
    # 5. 測試 Python 載入
    print(f"\n{Colors.BLUE}[5] Python 套件載入測試{Colors.END}")
    
    try:
        from lib.train.admin.local import EnvironmentSettings
        env = EnvironmentSettings()
        print(f"{Colors.GREEN}✓{Colors.END} 成功載入環境設定")
        print(f"  COCO 路徑: {env.coco_dir}")
        
        # 驗證路徑
        if os.path.exists(env.coco_dir):
            print(f"{Colors.GREEN}✓{Colors.END} COCO 路徑有效")
        else:
            print(f"{Colors.RED}✗{Colors.END} COCO 路徑無效: {env.coco_dir}")
            all_ok = False
            
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} 無法載入環境設定: {e}")
        all_ok = False
    
    try:
        from lib.test.evaluation.local import local_env_settings
        settings = local_env_settings()
        print(f"{Colors.GREEN}✓{Colors.END} 成功載入測試設定")
        print(f"  UAV123 路徑: {settings.uav123_path}")
        
        # 驗證路徑
        if os.path.exists(settings.uav123_path):
            print(f"{Colors.GREEN}✓{Colors.END} UAV123 路徑有效")
        else:
            print(f"{Colors.RED}✗{Colors.END} UAV123 路徑無效: {settings.uav123_path}")
            all_ok = False
            
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} 無法載入測試設定: {e}")
        all_ok = False
    
    # 6. 測試 COCO 資料集載入
    print(f"\n{Colors.BLUE}[6] COCO 資料集載入測試{Colors.END}")
    
    try:
        import json
        from lib.train.admin.local import EnvironmentSettings
        
        env = EnvironmentSettings()
        anno_file = os.path.join(env.coco_dir, 'annotations', 'instances_train2017.json')
        
        if os.path.exists(anno_file):
            print(f"{Colors.GREEN}✓{Colors.END} 開始載入 COCO 標註檔案...")
            with open(anno_file, 'r') as f:
                coco_data = json.load(f)
            
            print(f"{Colors.GREEN}✓{Colors.END} COCO 標註載入成功!")
            print(f"  圖片數量: {len(coco_data['images']):,}")
            print(f"  標註數量: {len(coco_data['annotations']):,}")
            print(f"  類別數量: {len(coco_data['categories'])}")
        else:
            print(f"{Colors.RED}✗{Colors.END} COCO 標註檔案不存在: {anno_file}")
            all_ok = False
            
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} COCO 資料集載入失敗: {e}")
        all_ok = False
    
    # 總結
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    if all_ok:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ 所有檢查通過! 可以開始訓練{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ 部分檢查未通過，請修正後再訓練{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # 使用說明
    if all_ok:
        print(f"{Colors.BOLD}下一步: 開始訓練{Colors.END}\n")
        
        print(f"{Colors.BLUE}【使用 COCO 資料集訓練】{Colors.END}")
        print("單 GPU:")
        print("  python tracking/train.py --script sglatrack --config coco_only \\")
        print("    --save_dir ./output --mode single --use_wandb 0\n")
        
        print("多 GPU (4 張):")
        print("  python tracking/train.py --script sglatrack --config coco_only \\")
        print("    --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0\n")
        
        print(f"{Colors.BLUE}【測試 UAV123】{Colors.END}")
        print("  python tracking/test.py --tracker_param sglatrack \\")
        print("    --dataset uav123 --threads 8 --num_gpus 1\n")
        
        print(f"{Colors.YELLOW}提示: 訓練前建議先用小的 epoch 數測試 (例如 5 epochs){Colors.END}")
        print(f"{Colors.YELLOW}修改 experiments/sglatrack/coco_only.yaml 中的 EPOCH: 5{Colors.END}\n")

if __name__ == "__main__":
    main()
