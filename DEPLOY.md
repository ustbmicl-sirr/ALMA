# ALMA 部署和运行指南

## 快速开始

### 1. 在远程服务器上克隆/同步代码

```bash
# 方式1: 从 GitHub 克隆
git clone https://github.com/ustbmicl-visHMARL/gd-papers.git
cd gd-papers/references/codes/ALMA

# 方式2: 使用 rsync 同步
rsync -avz --progress /path/to/ALMA user@server:/path/to/destination/
```

### 2. 设置环境

```bash
# 安装 StarCraft II 和 SMAC maps
chmod +x scripts/*.sh
./scripts/setup.sh

# 设置环境变量
export SC2PATH=$(pwd)/3rdparty/StarCraftII
echo 'export SC2PATH='$SC2PATH >> ~/.bashrc
source ~/.bashrc
```

### 3. 构建 Docker 镜像

```bash
# 构建 Docker 镜像 (使用更新的 CUDA 11.8)
./scripts/build_docker.sh

# 如果需要使用 Weights & Biases 记录实验
export WANDB_API_KEY=your_api_key
./scripts/build_docker.sh
```

### 4. 运行训练

```bash
# SaveTheCity 环境 + QMIX (基础测试)
./scripts/train.sh 0 ff qmix_atten

# ALMA 方法 (SaveTheCity)
./scripts/train.sh 0 ff qmix_atten \
    --agent.subtask_cond=mask \
    --hier_agent.task_allocation=aql

# ALMA + EA 优化
./scripts/train.sh 0 ff qmix_atten \
    --agent.subtask_cond=mask \
    --hier_agent.task_allocation=aql \
    --ea.enabled=True

# StarCraft 环境 (需要 SC2 安装)
./scripts/train.sh 0 sc2multiarmy refil \
    --scenario=6-8sz_maxsize4_maxarmies3_symmetric \
    --agent.subtask_cond=mask \
    --hier_agent.task_allocation=aql
```

### 5. 评估模型

```bash
./scripts/evaluate.sh 0 ff qmix_atten <checkpoint_name>
```

## 方法参数说明

| 方法 | 参数 |
|------|------|
| ALMA | `--agent.subtask_cond=mask --hier_agent.task_allocation=aql` |
| ALMA (No Mask) | `--agent.subtask_cond=full_obs --hier_agent.task_allocation=aql` |
| Heuristic | `--agent.subtask_cond=mask --hier_agent.task_allocation=heuristic` |
| COPA | `--hier_agent.copa=True` |

## 环境说明

| 环境 | 配置 | 算法 |
|------|------|------|
| SaveTheCity (ff) | `--env-config=ff` | `qmix_atten` |
| StarCraft II | `--env-config=sc2multiarmy` | `refil` |

## EA 参数

```bash
--ea.enabled=True           # 启用 EA
--ea.pop_size=5            # 种群大小
--ea.elite_size=2          # 精英数量
--ea.mutation_rate=0.3     # 变异率
--ea.crossover_rate=0.8    # 交叉率
--ea.eval_episodes=3       # 每个基因组评估的 episode 数
--ea.interval=5000         # EA 更新间隔
```

## 故障排除

### Docker GPU 问题
```bash
# 检查 NVIDIA Docker
nvidia-smi
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### StarCraft II 问题
```bash
# 确保 SC2PATH 正确设置
echo $SC2PATH
ls $SC2PATH
```

### 内存不足
```bash
# 减小 batch size
--batch_size=16
--hier_agent.max_bs=200
```
