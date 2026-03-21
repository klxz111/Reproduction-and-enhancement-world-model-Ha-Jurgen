import os
import glob
import numpy as np
from pathlib import Path
from datetime import timedelta

# ===================== 核心配置（与特征提取代码完全对齐） =====================
DATA_DIR = "/mnt/e/WorldModel-Mamba2-CarRacing/01_data/raw_rollouts"
FEAT_DIR = "/mnt/e/WorldModel-Mamba2-CarRacing/02_features/dinov2_small"

def count_total_frames():
    """统计所有 rollouts_*.npy 特征文件的总帧数"""
    feat_dir = Path(FEAT_DIR)
    # 匹配特征文件：rollouts_0000.npy ~ rollouts_0099.npy
    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "rollouts_*.npy")))
    
    if not feat_files:
        raise FileNotFoundError(f"未找到特征文件，请检查路径: {FEAT_DIR}")
    
    total_frames = 0
    file_count = len(feat_files)
    
    print("=" * 60)
    print("📊 正在统计 DINOv2 特征文件帧数...")
    print("=" * 60)
    
    for idx, file_path in enumerate(feat_files, 1):
        file_name = os.path.basename(file_path)
        try:
            # 内存映射加载，避免大文件占内存
            feat = np.load(file_path, mmap_mode="r")
            frame_num = feat.shape[0]
            total_frames += frame_num
            print(f"[{idx}/{file_count}] {file_name:<20} | 帧数: {frame_num:,} | 维度: {feat.shape[1]}")
        except Exception as e:
            print(f"❌ 处理 {file_name} 失败: {str(e)[:50]}")
    
    return total_frames, file_count

def estimate_running_time(file_count, avg_time_per_file=7.5):
    """
    预估总运行时间
    :param file_count: 已处理的文件总数
    :param avg_time_per_file: 单文件平均耗时（分钟），从日志看约 7~8 分钟
    """
    total_min = file_count * avg_time_per_file
    total_sec = int(total_min * 60)
    td = timedelta(seconds=total_sec)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return total_min, hours, minutes, seconds

def main():
    try:
        total_frames, file_count = count_total_frames()
    except Exception as e:
        print(f"统计帧数失败: {e}")
        return
    
    # 预估运行时间（可根据实际单文件耗时调整，比如 7.0 或 8.0）
    total_min, hours, minutes, seconds = estimate_running_time(file_count, avg_time_per_file=7.5)
    
    print("\n" + "=" * 60)
    print("🎉 DINOv2 特征提取统计完成 | 汇总报告")
    print("=" * 60)
    print(f"📁 已处理文件总数: {file_count} 个")
    print(f"🎞️  特征总帧数: {total_frames:,} 帧")
    print(f"📏 单文件平均帧数: {total_frames // file_count:,} 帧")
    print(f"⏱️  单文件平均耗时: 7.5 分钟（可修改脚本中 avg_time_per_file 调整）")
    print(f"⏳ 预估总运行时间: {hours} 小时 {minutes} 分钟 {seconds} 秒")
    print(f"   （总计: {total_min:.1f} 分钟 / {total_min/60:.1f} 小时）")
    print("=" * 60)

if __name__ == "__main__":
    main()
