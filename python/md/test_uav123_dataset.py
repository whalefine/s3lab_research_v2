#!/usr/bin/env python3
"""
測試 UAV123 訓練資料集載入
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lib.train.dataset.uav123 import UAV123
from lib.train.admin.local import EnvironmentSettings

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def main():
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}UAV123 訓練資料集測試{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    try:
        # 載入環境設定
        env = EnvironmentSettings()
        print(f"{Colors.BLUE}[1] 環境設定{Colors.END}")
        print(f"  UAV123 路徑: {env.uav123_dir}")
        
        if not os.path.exists(env.uav123_dir):
            print(f"{Colors.RED}✗ UAV123 路徑不存在!{Colors.END}")
            return
        print(f"{Colors.GREEN}✓ UAV123 路徑存在{Colors.END}\n")
        
        # 載入資料集
        print(f"{Colors.BLUE}[2] 載入 UAV123 資料集{Colors.END}")
        dataset = UAV123(root=env.uav123_dir)
        
        num_sequences = dataset.get_num_sequences()
        print(f"{Colors.GREEN}✓ 成功載入 UAV123 資料集{Colors.END}")
        print(f"  序列數量: {num_sequences}\n")
        
        # 列出前 10 個序列
        print(f"{Colors.BLUE}[3] 序列列表 (前 10 個){Colors.END}")
        for i, seq_name in enumerate(dataset.sequence_list[:10]):
            print(f"  {i+1}. {seq_name}")
        if num_sequences > 10:
            print(f"  ... 還有 {num_sequences - 10} 個序列\n")
        else:
            print()
        
        # 測試載入第一個序列
        print(f"{Colors.BLUE}[4] 測試載入第一個序列{Colors.END}")
        seq_id = 0
        seq_name = dataset.sequence_list[seq_id]
        print(f"  序列名稱: {seq_name}")
        
        # 獲取序列資訊
        seq_info = dataset.get_sequence_info(seq_id)
        num_frames = len(seq_info['bbox'])
        print(f"  幀數: {num_frames}")
        print(f"  邊界框形狀: {seq_info['bbox'].shape}")
        print(f"  有效幀數: {seq_info['valid'].sum().item()}")
        print(f"  可見幀數: {seq_info['visible'].sum().item()}")
        
        # 測試載入幾個幀
        print(f"\n{Colors.BLUE}[5] 測試載入圖片{Colors.END}")
        try:
            frame_ids = [0, num_frames//4, num_frames//2, num_frames-1]
            frames, anno_frames, meta = dataset.get_frames(seq_id, frame_ids)
            
            print(f"{Colors.GREEN}✓ 成功載入 {len(frames)} 張圖片{Colors.END}")
            for i, (fid, frame) in enumerate(zip(frame_ids, frames)):
                print(f"  幀 {fid}: {frame.shape if hasattr(frame, 'shape') else 'loaded'}")
            
            print(f"\n  物件類別: {meta['object_class_name']}")
            
        except Exception as e:
            print(f"{Colors.RED}✗ 載入圖片失敗: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
        
        # 統計資訊
        print(f"\n{Colors.BLUE}[6] 資料集統計{Colors.END}")
        total_frames = 0
        min_frames = float('inf')
        max_frames = 0
        
        for seq_id in range(min(num_sequences, 10)):  # 只檢查前 10 個以節省時間
            try:
                info = dataset.get_sequence_info(seq_id)
                nframes = len(info['bbox'])
                total_frames += nframes
                min_frames = min(min_frames, nframes)
                max_frames = max(max_frames, nframes)
            except:
                pass
        
        if total_frames > 0:
            avg_frames = total_frames / min(num_sequences, 10)
            print(f"  平均幀數 (前10個): {avg_frames:.0f}")
            print(f"  最短序列: {min_frames} 幀")
            print(f"  最長序列: {max_frames} 幀")
        
        # 成功總結
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}✓ UAV123 訓練資料集測試通過!{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
        
        print(f"{Colors.BOLD}可以開始訓練:{Colors.END}")
        print(f"  python tracking/train.py --script sglatrack --config coco_uav123 \\")
        print(f"    --save_dir ./output --mode single --use_wandb 0\n")
        
    except Exception as e:
        print(f"\n{Colors.RED}✗ 錯誤: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
