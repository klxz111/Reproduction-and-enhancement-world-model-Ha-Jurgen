import os
import glob
import torch
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ===================== 核心配置 =====================
DATA_DIR = "/mnt/e/WorldModel-Mamba2-CarRacing/01_data/raw_rollouts"
OUTPUT_DIR = "/mnt/e/WorldModel-Mamba2-CarRacing/02_features/dinov2_small"

MODEL_NAME = "facebook/dinov2-small"
RESIZE_SIZE = 224
BATCH_SIZE = 2  # 8GB 显存，若OOM改为1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 国内镜像源（兜底）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/mnt/e/huggingface_cache"

# ===================== 加载模型（兼容硬件） =====================
def init_model():
    print(f"📥 从国内镜像拉取 {MODEL_NAME} 模型...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir="/mnt/e/huggingface_cache")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # 改用 float16，兼容你的 4070 Laptop
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        cache_dir="/mnt/e/huggingface_cache"
    ).to(DEVICE).eval()
    used_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2
    print(f"✅ 模型加载完成 | 已占用显存: {used_mem:.1f} MB")
    return processor, model

# ===================== 处理 H5 文件（适配数据结构） =====================
def process_h5_file(h5_path, processor, model):
    all_features = []
    try:
        with h5py.File(h5_path, 'r') as f:
            episode_keys = [k for k in f.keys() if k.startswith("episode_")]
            print(f"\n📂 处理 {os.path.basename(h5_path)} | 共 {len(episode_keys)} 个 episode")

            for ep_key in tqdm(episode_keys, desc=f"[{os.path.basename(h5_path)}] 遍历 episode"):
                ep_group = f[ep_key]
                if 'frames' in ep_group:
                    images_np = ep_group['frames'][:]
                else:
                    print(f"⚠️  {ep_key} 未找到 'frames'，跳过")
                    continue

                for i in range(0, len(images_np), BATCH_SIZE):
                    batch_imgs = images_np[i:i+BATCH_SIZE]
                    batch_pil = [Image.fromarray(img.astype('uint8')).convert("RGB") for img in batch_imgs]
                    # 兼容旧版 transformers：用 size 替代 resize_size
                    inputs = processor(
                        images=batch_pil,
                        return_tensors="pt",
                        size={"height": RESIZE_SIZE, "width": RESIZE_SIZE}
                    ).to(DEVICE, dtype=torch.float16)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    cls_feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_features.append(cls_feat)

                    del inputs, outputs, cls_feat, batch_pil
                    torch.cuda.empty_cache()

        if all_features:
            all_feat = np.concatenate(all_features, axis=0)
            out_name = os.path.basename(h5_path).replace(".h5", ".npy")
            np.save(os.path.join(OUTPUT_DIR, out_name), all_feat)
            print(f"✅ 保存特征: {out_name} | 总帧数: {all_feat.shape[0]} | 特征维度: {all_feat.shape[1]}")
        else:
            print(f"❌ {os.path.basename(h5_path)} 未提取到任何特征")

    except Exception as e:
        print(f"❌ 处理失败 {h5_path}: {str(e)[:200]}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h5_files = sorted(glob.glob(os.path.join(DATA_DIR, "rollouts_*.h5")))
    print(f"🔍 找到 {len(h5_files)} 个待处理文件")

    if len(h5_files) == 0:
        print("❌ 没有找到任何文件，请检查 DATA_DIR 路径是否正确！")
        print(f"当前路径: {DATA_DIR}")
        os.system(f"ls {DATA_DIR} | head -10")
        exit(1)

    processor, model = init_model()
    for h5_file in h5_files:
        process_h5_file(h5_file, processor, model)

    print("\n🎉 所有文件特征提取完成！特征保存路径：", OUTPUT_DIR)
