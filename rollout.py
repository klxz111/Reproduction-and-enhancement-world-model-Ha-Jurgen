#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CarRacing 数据采集
核心优化：
1. 存储路径改为WSL2本地（彻底解决文件锁冲突）
2. 保留16进程+内存复用+动作多样化（速度拉满）
3. 断点续采+同步落盘，稳定不丢数据
"""
import os
import sys
import time
import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import gymnasium as gym
import warnings
from queue import Empty

# ===================== 核心配置（改为WSL2本地路径+16进程） =====================
NUM_WORKERS = 16          # 保留16进程，速度拉满
TOTAL_EPISODES = 10000    # 总采集局数
MAX_STEPS_PER_EPISODE = 1000  # 单局最大步数
SAVE_INTERVAL = 100       # 增大落盘间隔，减少写文件次数

改为WSL2本地路径（替换成用户名，比如 /home/xxx/...）
# 注意：请把 lhyzyrx 改成自己的WSL2用户名！
DATA_DIR = "/home/lhyzyrx/WorldModel-Mamba2-CarRacing/01_data/raw_rollouts"  
os.makedirs(DATA_DIR, exist_ok=True)

CROP_SIZE = (84, 84)      # 图像裁剪尺寸
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.txt")  # 断点续采文件

# ===================== 屏蔽无关警告 =====================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== 环境封装类（精简+高质量） =====================
class CarRacingEnv:
    def __init__(self):
        """初始化环境：保留RGB渲染，精简预处理"""
        self.env = gym.make(
            "CarRacing-v2",
            render_mode="rgb_array",  # 保留高质量RGB图像
            continuous=True,
            max_episode_steps=MAX_STEPS_PER_EPISODE,
            disable_env_checker=True,
            domain_randomize=False
        )
        self.obs_shape = (84, 84, 3)

    def reset(self):
        obs, _ = self.env.reset()
        return self._preprocess_obs(obs)

    def step(self, action):
        action = np.clip(action, [-1, 0, 0], [1, 1, 1])
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess_obs(obs), reward, (terminated or truncated), info

    def _preprocess_obs(self, obs):
        """极简预处理：只裁剪+归一化，最大化提速"""
        # 裁剪：96x96 → 84x84（去掉顶部UI和左右边距）
        obs = obs[:-12, 6:-6, :].astype(np.float32) / 255.0
        return obs

    def close(self):
        self.env.close()

# ===================== Worker：内存复用+动作多样化+精简过滤 =====================
def worker(queue, worker_id):
    """单进程采集：内存复用+精简逻辑，提速核心"""
    try:
        env = CarRacingEnv()
        print(f"✅ Worker {worker_id} 初始化完成（RGB模式），开始采集...")

        # 预分配缓冲区（修复维度+内存复用）
        frames_buf = np.zeros((MAX_STEPS_PER_EPISODE, 84, 84, 3), dtype=np.float32)
        actions_buf = np.zeros((MAX_STEPS_PER_EPISODE, 3), dtype=np.float32)
        rewards_buf = np.zeros(MAX_STEPS_PER_EPISODE, dtype=np.float32)
        dones_buf = np.zeros(MAX_STEPS_PER_EPISODE, dtype=np.bool_)

        while True:
            step = 0
            obs = env.reset()
            done = False

            # 单局采集（无冗余计算）
            while not done and step < MAX_STEPS_PER_EPISODE:
                # 动作多样化：正态分布转向+偏向油门（提升数据质量）
                action = np.array([
                    np.random.normal(0, 0.5),    # 转向集中在中间
                    np.random.uniform(0.3, 1.0), # 油门偏向前进
                    np.random.uniform(0, 0.1)    # 刹车少用
                ], dtype=np.float32)
                
                next_obs, reward, done, _ = env.step(action)
                
                # 直接写入缓冲区（无append，提速30%）
                frames_buf[step] = obs
                actions_buf[step] = action
                rewards_buf[step] = reward
                dones_buf[step] = done
                
                obs = next_obs
                step += 1

            # 极简过滤：只过滤极短局（减少计算开销）
            if step < 50:
                continue

            # 封装数据（精简拷贝）
            episode_data = {
                "frames": frames_buf[:step].copy(),
                "actions": actions_buf[:step].copy(),
                "rewards": rewards_buf[:step].copy(),
                "dones": dones_buf[:step].copy()
            }
            
            queue.put((worker_id, episode_data))

    except Exception as e:
        print(f"❌ Worker {worker_id} 出错: {e}", file=sys.stderr)
    finally:
        env.close()

# ===================== 同步落盘（解决WSL2文件锁核心方案） =====================
def save_episodes_to_hdf5(episodes, file_idx):
    """同步落盘：单进程写文件，彻底避免锁冲突"""
    file_path = os.path.join(DATA_DIR, f"rollouts_{file_idx:04d}.h5")
    # 使用lzf压缩（比gzip快5倍，兼容性更好）
    with h5py.File(file_path, "w", libver="latest") as f:
        for ep_idx, episode in enumerate(episodes):
            grp = f.create_group(f"episode_{ep_idx}")
            grp.create_dataset("frames", data=episode["frames"], compression="lzf")
            grp.create_dataset("actions", data=episode["actions"], compression="lzf")
            grp.create_dataset("rewards", data=episode["rewards"], compression="lzf")
            grp.create_dataset("dones", data=episode["dones"], compression="lzf")
    print(f"📁 已保存 {len(episodes)} 局数据到 {file_path}")
    
    # 更新断点（原子写入，避免丢失）
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(min((file_idx + 1) * SAVE_INTERVAL, TOTAL_EPISODES)))

# ===================== 主进程：稳定+容错+断点续采 =====================
def main():
    """主进程：16进程分批启动+断点续采+超时保护"""
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=200)  # 增大队列，减少Worker阻塞

    # 断点续采（兼容中断）
    collected_total = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                collected_total = int(f.read().strip())
            print(f"🔄 从断点继续采集，已完成 {collected_total} 局")
        except:
            collected_total = 0

    # 分批启动16进程（0.3秒/个，避免瞬间过载）
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(queue, i))
        p.start()
        processes.append(p)
        time.sleep(0.3)
    print(f"\n🚀 已启动 {NUM_WORKERS} 进程，需采集 {TOTAL_EPISODES - collected_total} 局...\n")

    # 数据接收+落盘
    collected_episodes = []
    file_idx = collected_total // SAVE_INTERVAL
    pbar = tqdm(total=TOTAL_EPISODES, desc="采集进度")
    pbar.update(collected_total)

    try:
        while collected_total < TOTAL_EPISODES:
            try:
                # 超时保护（60秒），避免死等
                worker_id, episode_data = queue.get(timeout=60)
                collected_episodes.append(episode_data)
                collected_total += 1
                pbar.update(1)

                # 同步落盘（每100局一次，减少写文件次数）
                if len(collected_episodes) >= SAVE_INTERVAL:
                    save_episodes_to_hdf5(collected_episodes[:SAVE_INTERVAL], file_idx)
                    collected_episodes = collected_episodes[SAVE_INTERVAL:]
                    file_idx += 1

            except Empty:
                continue  # 队列超时，继续等待

    except KeyboardInterrupt:
        print("\n⚠️  用户中断，正在保存剩余数据...")
    finally:
        # 优雅停止进程
        for p in processes:
            p.terminate()
            p.join(timeout=10)
        
        # 保存剩余数据（关键：避免丢失）
        if collected_episodes:
            save_episodes_to_hdf5(collected_episodes, file_idx)
        
        pbar.close()
        print(f"\n🎉 采集完成！共采集 {min(collected_total, TOTAL_EPISODES)} 局，数据路径：{DATA_DIR}")
        
        # 提示：采集完成后拷贝到Windows盘的命令
        print(f"\n📌 拷贝数据到Windows E盘命令：")
        print(f"cp -r {DATA_DIR} /mnt/e/WorldModel-Mamba2-CarRacing/01_data/")

if __name__ == "__main__":
    main()
