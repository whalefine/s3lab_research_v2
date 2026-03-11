#!/usr/bin/env python3
"""
驗證所有資料集路徑配置
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lib.train.admin.local import EnvironmentSettings
from lib.test.evaluation.local import local_env_settings

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_path(name, path, check_files=None):
    """檢查路徑是否存在，並可選檢查內部檔案"""
    exists = os.path.exists(path)
    
    if exists:
        status = f"{Colors.GREEN}✓{Colors.END}"
        msg = f"{name}: {path}"
        
        # 檢查內部檔案/目錄
        if check_files:
            missing = []
            for file in check_files:
                if not os.path.exists(os.path.join(path, file)):
                    missing.append(file)
            
            if missing:
                status = f"{Colors.YELLOW}⚠{Colors.END}"
                msg += f"\n    {Colors.YELLOW}缺少: {', '.join(missing)}{Colors.END}"
    else:
        status = f"{Colors.RED}✗{Colors.END}"
        msg = f"{name}: {path} {Colors.RED}(不存在){Colors.END}"
    
    print(f"{status} {msg}")
    return exists

def main():
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}資料集路徑驗證{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # 訓練路徑設定
    print(f"{Colors.BLUE}[1] 訓練路徑設定 (lib/train/admin/local.py){Colors.END}")
    train_env = EnvironmentSettings()
    
    train_paths = [
        ("COCO", train_env.coco_dir, ['annotations', 'images']),
        ("UAV123 (訓練)", train_env.uav123_dir, ['anno', 'data_seq']),
        ("LaSOT", train_env.lasot_dir, None),
        ("GOT-10k", train_env.got10k_dir, None),
        ("TrackingNet", train_env.trackingnet_dir, None),
    ]
    
    train_ok = 0
    for name, path, check in train_paths:
        if check_path(name, path, check):
            train_ok += 1
    
    print()
    
    # 測試路徑設定
    print(f"{Colors.BLUE}[2] 測試路徑設定 (lib/test/evaluation/local.py){Colors.END}")
    test_env = local_env_settings()
    
    test_paths = [
        ("UAV123", test_env.uav123_path, ['anno/UAV123', 'data_seq/UAV123']),
        ("UAV123_10fps", test_env.uav123_10fps_path, ['anno/UAV123_10fps', 'data_seq/UAV123_10fps']),
        ("UAVDT", test_env.uavdt_path, None),
        ("DTB70", test_env.dtb70_path, None),
        ("UAVTrack112", test_env.uavtrack_path, None),
        ("GOT-10k", test_env.got10k_path, None),
        ("LaSOT", test_env.lasot_path, None),
        ("TrackingNet", test_env.trackingnet_path, None),
    ]
    
    test_ok = 0
    for name, path, check in test_paths:
        if check_path(name, path, check):
            test_ok += 1
    
    print()
    
    # 詳細檢查現有資料集
    print(f"{Colors.BLUE}[3] 現有資料集詳細檢查{Colors.END}")
    
    # COCO
    if os.path.exists(train_env.coco_dir):
        anno_train = os.path.join(train_env.coco_dir, 'annotations', 'instances_train2017.json')
        anno_val = os.path.join(train_env.coco_dir, 'annotations', 'instances_val2017.json')
        img_train = os.path.join(train_env.coco_dir, 'images', 'train2017')
        img_val = os.path.join(train_env.coco_dir, 'images', 'val2017')
        
        print(f"\n{Colors.BOLD}COCO 資料集:{Colors.END}")
        check_path("  訓練標註", anno_train)
        check_path("  驗證標註", anno_val)
        check_path("  訓練圖片", img_train)
        check_path("  驗證圖片", img_val)
        
        if os.path.exists(img_train):
            train_count = len([f for f in os.listdir(img_train) if f.endswith('.jpg')])
            print(f"    訓練圖片數量: {train_count:,}")
    
    # UAV123
    if os.path.exists(test_env.uav123_path):
        anno_dir = os.path.join(test_env.uav123_path, 'anno', 'UAV123')
        data_dir = os.path.join(test_env.uav123_path, 'data_seq', 'UAV123')
        
        print(f"\n{Colors.BOLD}UAV123 資料集:{Colors.END}")
        if os.path.exists(anno_dir):
            anno_count = len([f for f in os.listdir(anno_dir) if f.endswith('.txt')])
            print(f"{Colors.GREEN}✓{Colors.END}   標註檔案: {anno_count} 個")
        else:
            print(f"{Colors.RED}✗{Colors.END}   標註目錄不存在: {anno_dir}")
        
        if os.path.exists(data_dir):
            seq_count = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            print(f"{Colors.GREEN}✓{Colors.END}   視訊序列: {seq_count} 個")
        else:
            print(f"{Colors.RED}✗{Colors.END}   序列目錄不存在: {data_dir}")
    
    # UAV123_10fps
    if os.path.exists(test_env.uav123_10fps_path):
        anno_dir = os.path.join(test_env.uav123_10fps_path, 'anno', 'UAV123_10fps')
        data_dir = os.path.join(test_env.uav123_10fps_path, 'data_seq', 'UAV123_10fps')
        
        print(f"\n{Colors.BOLD}UAV123_10fps 資料集:{Colors.END}")
        if os.path.exists(anno_dir):
            anno_count = len([f for f in os.listdir(anno_dir) if f.endswith('.txt')])
            print(f"{Colors.GREEN}✓{Colors.END}   標註檔案: {anno_count} 個")
        else:
            print(f"{Colors.RED}✗{Colors.END}   標註目錄不存在: {anno_dir}")
        
        if os.path.exists(data_dir):
            seq_count = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            print(f"{Colors.GREEN}✓{Colors.END}   視訊序列: {seq_count} 個")
        else:
            print(f"{Colors.RED}✗{Colors.END}   序列目錄不存在: {data_dir}")
    
    # 總結
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    total_train = len(train_paths)
    total_test = len(test_paths)
    
    if train_ok == total_train and test_ok == total_test:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ 所有路徑配置正確!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ 部分路徑需要確認{Colors.END}")
        print(f"  訓練路徑: {train_ok}/{total_train} 正確")
        print(f"  測試路徑: {test_ok}/{total_test} 正確")
    
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    # 顯示當前可用的資料集
    print(f"{Colors.BOLD}當前可用於訓練的資料集:{Colors.END}")
    available = []
    if os.path.exists(train_env.coco_dir):
        available.append("COCO17")
    if os.path.exists(train_env.uav123_dir):
        available.append("UAV123")
    if os.path.exists(train_env.lasot_dir):
        available.append("LASOT")
    if os.path.exists(train_env.got10k_dir):
        available.append("GOT10K_vottrain")
    if os.path.exists(train_env.trackingnet_dir):
        available.append("TRACKINGNET")
    
    if available:
        for ds in available:
            print(f"  {Colors.GREEN}✓{Colors.END} {ds}")
    else:
        print(f"  {Colors.YELLOW}無可用訓練資料集{Colors.END}")
    
    print(f"\n{Colors.BOLD}當前可用於測試的資料集:{Colors.END}")
    test_available = []
    if os.path.exists(test_env.uav123_path):
        test_available.append("uav123")
    if os.path.exists(test_env.uav123_10fps_path):
        test_available.append("uav123_10fps")
    if os.path.exists(test_env.uavdt_path):
        test_available.append("uavdt")
    if os.path.exists(test_env.dtb70_path):
        test_available.append("dtb70")
    
    if test_available:
        for ds in test_available:
            print(f"  {Colors.GREEN}✓{Colors.END} {ds}")
    else:
        print(f"  {Colors.YELLOW}無可用測試資料集{Colors.END}")
    
    print()

if __name__ == "__main__":
    main()
