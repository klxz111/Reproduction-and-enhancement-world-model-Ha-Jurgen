# Mamba-based World Model on CarRacing

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-12-8-0-download-archive)
[![CarRacing-v0](https://img.shields.io/badge/Env-CarRacing--v0-orange.svg)](https://gymnasium.farama.org/environments/box2d/car_racing/)

> **当前进度**：Phase2 构建mamba模型（模型架构验证已跑通 目前着手于训练调优和推理）
本项目是世界模型（World Model）复现工程，一个基于 Mamba 的 CarRacing 世界模型复现项目，重点关注在消费级 GPU 约束下的训练、推理与控制链路实现。当前已完成基础架构搭建与阶段性验证，仓库主要展示数据处理、模型组织、实验配置和可复现工程流程。


## 📋 测试环境
| 硬件模块 | 核心参数 | 
|----------|----------|
| 处理器 | Intel i7-14700HX（14P+6E核，20线程） |
| 显卡 | NVIDIA RTX4070 Laptop（8GB GDDR6） | 
| 内存 | 16GB DDR5 5600MT/s |
| 存储 | 1TB NVMe SSD（E盘451GB可用） |

## 🛠️ 环境搭建

### 1. 系统环境准备
- 确认NVIDIA驱动版本≥560.00（适配CUDA 12.8）
- 关闭显存占用冗余进程（AI绘图、浏览器硬件加速等），保证空闲显存≥7.5GB


### 2. 隔离Python环境创建
```bash
# 创建conda环境
conda create -n worldmodel-mamba2 python=3.12 -y
conda activate worldmodel-mamba2

# 安装PyTorch（适配CUDA 12.8）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 手动安装Mamba2依赖组件（预编译版本）
# 下载对应CUDA 12.8 + Python 3.12的预编译包
# causal-conv1d: https://github.com/Dao-AILab/causal-conv1d/releases
# mamba-ssm: https://github.com/state-spaces/mamba/releases
pip install <下载的causal-conv1d包路径>
pip install <下载的mamba-ssm包路径>

# 安装其他核心依赖
pip install gym[box2d]==0.26.2 swig h5py tqdm
pip install transformers accelerate
pip install tensorboard matplotlib
```

### 3. 环境校验
```python
import torch
print("CUDA可用:", torch.cuda.is_available())
print("显卡名称:", torch.cuda.get_device_name(0))
print("PyTorch CUDA版本:", torch.version.cuda)

# 验证Mamba2安装
from mamba_ssm import Mamba2
test_mamba = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2).cuda()
test_input = torch.randn(2, 1000, 256).cuda()
test_output = test_mamba(test_input)
print("Mamba2前向传播正常，输出shape:", test_output.shape)
```
✅ 验收标准：无报错，所有校验项正常输出

### 4. 项目目录结构
```
E:/WorldModel-Mamba2-CarRacing/
├── 01_data/                # 数据集存储
│   ├── raw_rollouts/       # 原始环境帧（HDF5格式）
│   └── latent_features/    # 128维视觉隐状态
├── 02_checkpoints/         # 模型权重（仅保存最优checkpoint）
├── 03_logs/                # 训练/监控日志
├── 04_src/                 # 核心源码
│   ├── data_pipeline/      # 数据采集/特征提取
│   ├── model/              # Mamba2动力学模型
│   ├── controller/         # MPC控制器
│   └── utils/              # 工具函数/硬件优化
├── train.py                # 训练主脚本
├── rollout.py              # 数据采集主脚本
├── extract_features.py     # 特征提取主脚本
├── run_control.py          # 闭环控制测试
└── README.md               # 项目文档
```

## 🚀 复现步骤

### Phase 0: 硬件性能基线测试（前置）
```bash
# 测试CPU多核性能
python 04_src/utils/test_cpu_perf.py

# 测试SSD读写性能
python 04_src/utils/test_ssd_perf.py

# 测试GPU显存/算力
python 04_src/utils/test_gpu_perf.py
```

### Phase 1: 数据采集与视觉降维（2-3天）
#### 1. 多核并行数据采集
```bash
# 启动12进程并行采集10000个Episode
python rollout.py \
  --num_episodes 10000 \
  --num_workers 12 \
  --save_dir 01_data/raw_rollouts/ \
  --batch_save 100
数据规模：100 条场景轨迹，总计 8,242,152 帧 视觉数据
处理效率：单文件平均耗时 7.5 分钟，单卡总耗时 12.5 小时，实现全量数据离线处理
稳定性：0 运行报错，8GB 显存下峰值占用 ≤6GB，本地操作无卡顿
特征规格：输出 384 维 DINOv2 视觉特征，维度固定、时序对齐，可直接用于 Mamba2 动力学模型训练
```
🔧 **Debug记录**：
- 初始进程数设为16导致系统卡顿，调整为12（绑定P核）后稳定
- 单局小于50步的异常数据需过滤，避免污染数据集
- HDF5分块大小设为4K匹配SSD块大小，读取速度提升30%
```

#### 2. 离线视觉特征提取（DINOv2）
```bash
# 8GB显存适配：Batch Size=2，FP16推理（当前使用配置）
python extract_features.py \
  --raw_data_dir 01_data/raw_rollouts/ \
  --save_dir 01_data/latent_features/ \
  --batch_size 2 \
  --fp16 True


🔧 **Debug记录（核心问题&解决方案）**：
| 问题类型 | 具体现象 | 分析过程 | 解决方案 | 验证结果 |
|----------|----------|----------|----------|----------|
| 文件名匹配错误 | 匹配到0个h5文件 | 用ls命令验证路径，发现文件名是rollouts_*.h5（多s） | 把rollout_*.h5改为rollouts_*.h5 | 成功匹配100个文件 |
| 数据结构不匹配 | 找不到images字段 | 编写脚本查看H5结构，发现图片存在frames字段 | 读取episode下的frames字段 | 成功提取图片数据 |
| transformers版本兼容 | resize_size参数报错 | 查transformers文档，旧版不支持该参数 | 改用size={"height":224,"width":224} | 预处理正常运行 |
| 精度不兼容 | BFloat16不支持 | 硬件（RTX4070）不兼容该精度 | 改为Float16，验证显存占用 | 模型加载正常，无OOM |
| 显存占用过高 | 8GB显存OOM | 初始Batch Size=64导致显存溢出 | 降至Batch Size=2，FP16推理 | 单批次显存≤2GB |

#### 3. 数据集校验
```bash
python 04_src/data_pipeline/validate_dataset.py
```
✅ 验收标准：有效Episode≥10000，总帧数≥816万，批量加载耗时＜10ms

### Phase 2: Mamba2动力学模型训练（3-4天）
#### 1. 模型训练
```bash
python train.py \
  --data_dir 01_data/latent_features/ \
  --ckpt_dir 02_checkpoints/ \
  --log_dir 03_logs/ \
  --batch_size 8 \
  --accum_steps 4 \
  --seq_len 1000 \
  --epochs 50 \
  --fp16 True \
  --gradient_checkpointing True
```

#### 2. 训练监控
```bash
tensorboard --logdir 03_logs/
```
访问 `http://localhost:6006` 查看Loss曲线、显存占用、训练速度

#### 3. 模型验证
```bash
python 04_src/model/validate_model.py \
  --ckpt_path 02_checkpoints/best_model.pth \
  --val_data_dir 01_data/latent_features/
```
✅ 验收标准：
- 训练/验证Loss持续收敛，无过拟合
- 状态预测MSE＜0.05，奖励预测MAE＜0.1
- 单步推理耗时＜1ms，批量推理500条轨迹＜10ms

### Phase 3: MPC控制器与闭环验证（2天）
#### 1. MPC控制测试
```bash
# 实时闭环控制
python run_control.py \
  --model_ckpt 02_checkpoints/best_model.pth \
  --num_candidates 500 \
  --horizon 50 \
  --fp16_infer True \
  --record_video True
```


#### 2. 性能指标（预期）
| 指标 | 数值 |
|------|------|
| 单步控制耗时 | ＜10ms（满足实时性） |
| 单局平均奖励 | ＞900（满分1000） |
| 赛道完成率 | ＞95% |
| 显存占用（推理） | ≤1GB |




## 🐞 常见问题与解决方案
| 问题 | 解决方案 |
|------|----------|
| Mamba安装报错 | 确保CUDA≥12.8，从预编译包安装：<br>1. causal-conv1d: https://github.com/Dao-AILab/causal-conv1d/releases<br>2. mamba-ssm: https://github.com/state-spaces/mamba/releases |
| 数据采集进程卡死 | 降低进程数至10，检查环境是否正常初始化 |
| 训练时OOM | 启用梯度检查点，降低Batch Size至4，关闭不必要的后台程序 |
| 推理速度慢 | 启用FP16/INT8量化，关闭TensorBoard监控 |
| MPC控制不稳定 | 增加候选轨迹数至800，缩短规划时域至40步 |

## 📝 引用与参考
1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces (ICLR 2024)
2. DINOv2: Learning Robust Visual Features without Supervision (Meta AI)
3. World Models (ICML 2018)
4. Model Predictive Control for Reinforcement Learning (NeurIPS 2020)


## 📞 问题反馈
如有复现过程中的问题，可提交Issue，包含：
1. 硬件配置详情
2. 完整报错日志
3. 执行的命令参数
4. 已尝试的解决方案
