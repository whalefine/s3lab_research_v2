"""
計算追蹤結果的平均 FPS
Usage: python tracking/calculate_fps.py --tracker sglatrack --param deit_distilled --dataset uav123
"""
import os
import sys
import argparse
import numpy as np

# 將專案根目錄加入 Python 路徑（因為腳本在 tracking/ 目錄下）
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.environment import env_settings


def calculate_fps(tracker_name, param_name, dataset_name, run_id=None):
    """計算指定 tracker 在指定 dataset 上的平均 FPS"""
    
    env = env_settings()
    results_path = env.results_path
    
    # 構建結果路徑
    if run_id is None:
        tracker_dir = os.path.join(results_path, tracker_name, param_name, dataset_name)
    else:
        tracker_dir = os.path.join(results_path, tracker_name, f"{param_name}_{run_id:03d}", dataset_name)
    
    if not os.path.exists(tracker_dir):
        print(f"❌ 錯誤: 找不到結果路徑 {tracker_dir}")
        print(f"   請先執行測試: python tracking/test.py {tracker_name} {param_name} --dataset_name {dataset_name}")
        return None
    
    # 讀取所有 *_time.txt 檔案
    time_files = [f for f in os.listdir(tracker_dir) if f.endswith('_time.txt')]
    
    if len(time_files) == 0:
        print(f"❌ 錯誤: 在 {tracker_dir} 中找不到時間檔案")
        return None
    
    print(f"找到 {len(time_files)} 個序列的時間檔案")
    print("=" * 70)
    
    all_times = []
    sequence_fps = []
    total_frames = 0
    
    for time_file in sorted(time_files):
        seq_name = time_file.replace('_time.txt', '')
        time_path = os.path.join(tracker_dir, time_file)
        
        try:
            times = np.loadtxt(time_path)
            if times.ndim == 0:
                times = np.array([times])
            
            total_time = np.sum(times)
            num_frames = len(times)
            fps = num_frames / total_time if total_time > 0 else 0
            
            all_times.extend(times.tolist())
            sequence_fps.append((seq_name, fps, num_frames, total_time))
            total_frames += num_frames
            
        except Exception as e:
            print(f"⚠️  讀取 {time_file} 時發生錯誤: {e}")
    
    if len(sequence_fps) == 0:
        print("❌ 沒有成功讀取任何序列")
        return None
    
    # 計算統計資料
    total_time = sum(all_times)
    avg_fps_weighted = total_frames / total_time
    avg_fps_mean = np.mean([x[1] for x in sequence_fps])
    min_fps = min([x[1] for x in sequence_fps])
    max_fps = max([x[1] for x in sequence_fps])
    
    # 輸出結果
    print(f"\n📊 {tracker_name} ({param_name}) on {dataset_name}")
    print("=" * 70)
    print(f"總序列數:        {len(sequence_fps)}")
    print(f"總幀數:          {total_frames:,}")
    print(f"總執行時間:      {total_time:.2f} 秒")
    print(f"\n平均 FPS:        {avg_fps_weighted:.2f} (總幀數/總時間) ⭐ 推薦")
    print(f"平均 FPS:        {avg_fps_mean:.2f} (各序列平均)")
    print(f"最快 FPS:        {max_fps:.2f}")
    print(f"最慢 FPS:        {min_fps:.2f}")
    print("=" * 70)
    
    # 顯示前5快和後5慢的序列
    sequence_fps_sorted = sorted(sequence_fps, key=lambda x: x[1], reverse=True)
    
    print("\n🏆 最快的 5 個序列:")
    for i, (seq_name, fps, num_frames, seq_time) in enumerate(sequence_fps_sorted[:5], 1):
        print(f"  {i}. {seq_name:35s}: {fps:7.2f} FPS ({num_frames:4d} frames, {seq_time:.2f}s)")
    
    print("\n🐌 最慢的 5 個序列:")
    for i, (seq_name, fps, num_frames, seq_time) in enumerate(sequence_fps_sorted[-5:][::-1], 1):
        print(f"  {i}. {seq_name:35s}: {fps:7.2f} FPS ({num_frames:4d} frames, {seq_time:.2f}s)")
    
    print("\n" + "=" * 70)
    
    return {
        'num_sequences': len(sequence_fps),
        'total_frames': total_frames,
        'total_time': total_time,
        'avg_fps_weighted': avg_fps_weighted,
        'avg_fps_mean': avg_fps_mean,
        'min_fps': min_fps,
        'max_fps': max_fps
    }


def main():
    parser = argparse.ArgumentParser(description='計算追蹤結果的平均 FPS')
    parser.add_argument('--tracker', type=str, default='sglatrack', help='Tracker 名稱')
    parser.add_argument('--param', type=str, default='deit_distilled', help='參數檔名稱')
    parser.add_argument('--dataset', type=str, default='uav123', help='資料集名稱')
    parser.add_argument('--runid', type=int, default=None, help='Run ID (可選)')
    
    args = parser.parse_args()
    
    calculate_fps(args.tracker, args.param, args.dataset, args.runid)


if __name__ == '__main__':
    main()
