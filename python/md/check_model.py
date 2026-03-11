#!/usr/bin/env python3
"""
SGLATrack 模型檢查與驗證工具
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

def check_item(condition, success_msg, fail_msg):
    """檢查項目並輸出結果"""
    if condition:
        print(f"{Colors.GREEN}✓{Colors.END} {success_msg}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.END} {fail_msg}")
        return False

def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SGLATrack 模型檢查與驗證工具{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    all_checks_passed = True
    
    # 1. 檢查 Python 版本
    print(f"{Colors.BLUE}[1] Python 環境檢查{Colors.END}")
    python_version = sys.version_info
    is_python37 = python_version.major == 3 and python_version.minor == 7
    check_item(
        is_python37,
        f"Python 版本正確: {python_version.major}.{python_version.minor}.{python_version.micro}",
        f"Python 版本不符: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 3.7.x)"
    )
    if not is_python37:
        all_checks_passed = False
        print(f"{Colors.YELLOW}  → 請啟用 sgla 環境: conda activate sgla{Colors.END}")
    
    # 2. 檢查關鍵套件
    print(f"\n{Colors.BLUE}[2] 套件檢查{Colors.END}")
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            if package == 'torch':
                import torch
                version = torch.__version__
                cuda_available = torch.cuda.is_available()
                cuda_msg = f" (CUDA: {Colors.GREEN}可用{Colors.END})" if cuda_available else f" (CUDA: {Colors.RED}不可用{Colors.END})"
                print(f"{Colors.GREEN}✓{Colors.END} {name} 已安裝 (v{version}){cuda_msg}")
            else:
                print(f"{Colors.GREEN}✓{Colors.END} {name} 已安裝")
        except ImportError:
            print(f"{Colors.RED}✗{Colors.END} {name} 未安裝")
            all_checks_passed = False
    
    # 3. 檢查預訓練模型
    print(f"\n{Colors.BLUE}[3] 預訓練模型檢查{Colors.END}")
    pretrained_model = project_root / "pretrained_models" / "deit_tiny_distilled_patch16_224.pth"
    if check_item(
        pretrained_model.exists(),
        f"預訓練模型已下載: {pretrained_model.name} ({pretrained_model.stat().st_size / 1024 / 1024:.1f} MB)",
        f"預訓練模型不存在: {pretrained_model}"
    ):
        pass
    else:
        all_checks_passed = False
        print(f"{Colors.YELLOW}  → 下載連結: https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth{Colors.END}")
    
    # 4. 檢查訓練好的模型權重
    print(f"\n{Colors.BLUE}[4] 訓練模型權重檢查{Colors.END}")
    trained_model = project_root / "output" / "checkpoints" / "train" / "sglatrack" / "deit_distilled" / "sglatrack_ep0297.pth.tar"
    
    if trained_model.exists():
        check_item(
            True,
            f"訓練模型已存在: {trained_model.name} ({trained_model.stat().st_size / 1024 / 1024:.1f} MB)",
            ""
        )
        print(f"{Colors.GREEN}  → 可以直接進行測試{Colors.END}")
    else:
        check_item(
            False,
            "",
            f"訓練模型不存在（如需測試請下載）"
        )
        print(f"{Colors.YELLOW}  → 選項 1: 從 Google Drive 下載預訓練模型{Colors.END}")
        print(f"{Colors.YELLOW}     https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X{Colors.END}")
        print(f"{Colors.YELLOW}  → 選項 2: 自行訓練模型（需要準備訓練資料集）{Colors.END}")
    
    # 5. 檢查配置檔案
    print(f"\n{Colors.BLUE}[5] 配置檔案檢查{Colors.END}")
    config_files = [
        ("實驗配置", "experiments/sglatrack/deit_distilled.yaml"),
        ("訓練路徑配置", "lib/train/admin/local.py"),
        ("測試路徑配置", "lib/test/evaluation/local.py"),
        ("測試參數配置", "lib/test/parameter/sglatrack.py"),
    ]
    
    for name, path in config_files:
        file_path = project_root / path
        check_item(
            file_path.exists(),
            f"{name}: {path}",
            f"{name} 不存在: {path}"
        )
    
    # 6. 檢查資料集目錄
    print(f"\n{Colors.BLUE}[6] 資料集目錄檢查{Colors.END}")
    data_dir = project_root / "data"
    
    if data_dir.exists():
        datasets = {
            'lasot': 'LaSOT (訓練用)',
            'got10k': 'GOT-10k (訓練用)',
            'coco': 'COCO (訓練用)',
            'trackingnet': 'TrackingNet (訓練用)',
            'UAV123': 'UAV123 (測試用)',
            'UAV123_10fps': 'UAV123_10fps (測試用)',
            'uavdt': 'UAVDT (測試用)',
            'V4RFlight112': 'UAVTrack112 (測試用)',
            'DTB70': 'DTB70 (測試用)',
        }
        
        found_datasets = []
        for dataset_name, description in datasets.items():
            dataset_path = data_dir / dataset_name
            if dataset_path.exists():
                print(f"{Colors.GREEN}✓{Colors.END} {description}: {dataset_name}/")
                found_datasets.append(dataset_name)
            else:
                print(f"{Colors.YELLOW}○{Colors.END} {description}: 未找到")
        
        if not found_datasets:
            print(f"{Colors.YELLOW}  → 尚未準備資料集{Colors.END}")
            print(f"{Colors.YELLOW}  → 訓練資料: 需要 LaSOT, GOT-10k, COCO, TrackingNet{Colors.END}")
            print(f"{Colors.YELLOW}  → 測試資料下載: https://pan.baidu.com/s/1MaeGLRcAUbJxksbF_CrOeQ?pwd=5vbv{Colors.END}")
    else:
        print(f"{Colors.YELLOW}○{Colors.END} data/ 目錄不存在")
        print(f"{Colors.YELLOW}  → 建立目錄: mkdir -p data{Colors.END}")
    
    # 7. 檢查輸出目錄
    print(f"\n{Colors.BLUE}[7] 輸出目錄檢查{Colors.END}")
    output_dir = project_root / "output"
    check_item(
        output_dir.exists(),
        f"輸出目錄已建立: output/",
        f"輸出目錄不存在"
    )
    
    # 總結
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    if all_checks_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ 所有必要項目檢查通過!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ 部分項目需要設定{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    # 下一步建議
    print(f"{Colors.BOLD}下一步建議:{Colors.END}")
    
    if not trained_model.exists():
        print(f"\n{Colors.BLUE}【測試模型】{Colors.END}")
        print("1. 下載預訓練模型權重（推薦用於測試）")
        print("   https://drive.google.com/drive/folders/1sHL7aFVZFwkPy6js48x-EKfoZC7oJc9X")
        print(f"   放置到: {trained_model}")
        print("\n2. 下載測試資料集")
        print("   https://pan.baidu.com/s/1MaeGLRcAUbJxksbF_CrOeQ?pwd=5vbv")
        print(f"   解壓到: {data_dir}/")
        print("\n3. 執行測試")
        print("   python tracking/test.py --tracker_param sglatrack --dataset uav123 --threads 8 --num_gpus 4")
    else:
        print(f"\n{Colors.GREEN}模型已就緒，可以開始測試!{Colors.END}")
        print("執行測試:")
        print("  python tracking/test.py --tracker_param sglatrack --dataset uav123 --threads 8 --num_gpus 4")
    
    print(f"\n{Colors.BLUE}【訓練模型】{Colors.END}")
    print("1. 準備訓練資料集（LaSOT, GOT-10k, COCO, TrackingNet）")
    print("2. 執行訓練:")
    print("   python tracking/train.py --script sglatrack --config deit_distilled --save_dir ./output --mode single --use_wandb 0")
    
    print(f"\n{Colors.BLUE}【使用快速啟動腳本】{Colors.END}")
    print("執行: ./quick_start.sh")
    print("或: bash quick_start.sh")
    
    print()

if __name__ == "__main__":
    main()
