# ATA: ViTTT 上的分类 TTA 实验

一个紧凑的科研代码目录，支持 ViTTT Tiny / Small / Base、干净 ImageNet / ImageNet-C 评测，以及 Source、Tent、InfoMax、EATA、SAR、DeYO、Lambda、LambdaW0。

Agent 接手先阅读 [AGENTS.md](AGENTS.md)，其中记录方法、评测约定和运行上下文，与代码一起提交到 Git。数据、权重和实验输出独立保存；4090 与临时 H20 机器共用现有配置，路径自动检测，需要复现实验时另外同步对应的实验目录。

## 模型保持原样

`models/vittt.py` 和 `models/ttt_block.py` 从 [ViTTT 官方仓库](https://github.com/LeapLabTHU/ViTTT) **逐字节复制**，上游版本和 SHA256 保存在 `models/UPSTREAM.json`，检查中会校验。Source、Tent、InfoMax 使用原始架构和内部 TTT 更新规则。`models/__init__.py` 提供模型选择、严格权重加载和下载；`models/lambda_ttt.py` 只为 Lambda 替换内层导数，继承原始 forward，复用加载好的原始权重。

当前只包含普通 ViTTT，不包含 H-ViTTT。Source 保持原模型完整前向（包括内部 TTT 更新），只是不做外部 TTA 优化。模型采用原始 224×224 输入和 1000 类分类头。

```text
ATA/
├── run.py                 # 配置 -> 数据 -> 适应/评测 -> 记录/保存
├── data.py                # ImageNet / ImageNet-C、样本列表和预处理
├── methods.py             # 各方法的更新、参数选择和状态重置
├── sam.py                 # SAR 官方的 SAM，两步更新
├── utils.py               # YAML 配置、随机种子、实验文件、checkpoint 工具
├── summarize.py           # 汇总不同运行到 comparison.csv
├── models/                # 原始模型、构建/加载入口、Lambda 内层更新
├── configs/               # base、干净/损坏数据设置、各方法配置
├── weights/pretrained/    # 源模型的预训练权重
├── outputs/               # 每次运行一个独立目录，权重保存默认关闭
└── tests/                 # 数据、状态、原模型一致性、Lambda 梯度等关键检查
```

## 环境与权重

已有 `vittt` conda 环境可以直接使用，当前无需额外安装依赖：

```bash
conda activate vittt
# 在当前机器的 ATA 项目根目录运行后续命令，目录位置见 AGENTS.md。
```

已验证 Python 3.10、torch 2.5.1+cu124、torchvision 0.20.1+cu124、timm 0.4.12，其余版本见 `requirements.txt`。新建环境时先安装 GPU 对应的 PyTorch，再安装其余依赖，例如：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

下载官方预训练权重（已存在时检查后复用；不覆盖已有文件）：

```bash
python run.py --download tiny
# 按需下载另外两个尺寸
python run.py --download small base
```

Tiny 的默认路径是 `weights/pretrained/vittt_tiny.pth`。也支持在配置中指定自己的权重。支持原始 state_dict，以及 `model`、`state_dict`、`state_dict_ema` 等外层 key；通过 `model.checkpoint_key` 明确选择，加载采用 `strict=True`。官方训练文件包含 pickled 配置，因此仅加载可信的官方权重或自己的 checkpoint。

### 两台机器的路径兼容

配置文件保留 4090 机器的默认路径。代码优先使用指定路径，不存在时在两台机器的已知路径之间回退，两个方向均支持；两处均不存在时在执行阶段报错。`--dry-run` 仍可查看不可用路径，自定义路径不被替换。相对路径按项目根目录解析。

| 用途 | 4090 路径 | H20 路径 |
|---|---|---|
| ImageNet-C | `/mnt/bit/data/imagenet-c` | `/data1/buaa/datasets/imagenet-c` |
| clean ImageNet / EATA Fisher | `/mnt/bit/data/imagenet/images/val` | `/data1/buaa/datasets/imagenet/val` |
| holder Python 脚本 | 相邻目录 `../g/g.py` | 相邻目录 `../gpu-holder/gpu_hold.py` |
| holder 状态目录 | `/run/user/<uid>/gpu-holder` | `/tmp/gpu-holder-<uid>` |

`GPU_HOLDER_STATE_DIR` 若已设置则优先使用。`sweep.py` 默认使用已有 holder 进程，`--no-holder` 则直接运行，不检测 holder 脚本或状态目录。tmux 只用于保持长任务运行。实验保存的 `config.yaml` 记录检测后的实际数据路径。

三个权重统一存放在项目相对目录 `weights/pretrained/`，文件名为 `vittt_tiny.pth`、`vittt_small.pth`、`vittt_base.pth`，不做机器路径回退。换机时准备对应权重即可。

## 干净 ImageNet 验证集

通过 `data.name=imagenet` 选择干净 ImageNet。`data.root` 直接指向包含 1000 个 synset 类别文件夹的验证集目录，本机路径为 `/mnt/bit/data/imagenet/images/val`：

```text
val/<ImageNet synset>/<image>.JPEG
```

提供两份配置：`configs/imagenet_5k.yaml`（固定种子、每类 5 张）和 `configs/imagenet_50k.yaml`（完整 50,000 张）。这里的图片数指整个干净验证集。数据作为一个 `clean` 任务评测，`data.group=clean`，`data.severities=[]`；记录中的 severity 为 null / CSV 空值。

```bash
# Source 在干净 ImageNet 全量验证集上的 top-1 / top-5
python run.py --config configs/imagenet_50k.yaml

# 先用 5K 检查，或加相同的 TTA 方法配置
python run.py --config configs/imagenet_5k.yaml
python run.py --config configs/imagenet_5k.yaml configs/tent.yaml
python run.py --config configs/imagenet_5k.yaml configs/infomax.yaml

# 查看执行计划；更换机器时覆盖验证集路径
python run.py --config configs/imagenet_50k.yaml --dry-run
python run.py --config configs/imagenet_50k.yaml --set data.root=/path/to/imagenet/val
```

干净 ImageNet 和 ImageNet-C 共用采样/顺序控制、方法接口、权重保存和恢复流程；预处理按下面的默认配置分别设置。`samples.json` 也能复用；跨干净/损坏数据复用时必须具有相同的类别和相对文件名，缺失图片会报错。汇总 CSV 新增 `dataset` 列，明确区分 `imagenet` 和 `imagenet_c`；已有 ImageNet-C 记录与 checkpoint 仍可读取。

## ImageNet-C：15 个测试域与 4 个验证域

`data.name=imagenet_c` 为默认设置，数据根目录是本机已有的 `/mnt/bit/data/imagenet-c`，目录格式：

```text
imagenet-c/<corruption>/<severity>/<ImageNet synset>/<image>.JPEG
```

标准 15 域：`gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression`。

extra 4 域：`speckle_noise, gaussian_blur, spatter, saturate`。

| 配置 | 域 | 每个域、每个 severity 的图片数 |
|---|---|---|
| `configs/test_5k.yaml` | 标准 15 域 | 5,000 |
| `configs/test_50k.yaml` | 标准 15 域 | 50,000 |
| `configs/val_5k.yaml` | extra 4 域 | 5,000 |
| `configs/val_50k.yaml` | extra 4 域 | 50,000 |

这四个配置默认都使用 severity 5。设置 `data.severities=[1,2,3,4,5]` 可运行全部等级。例如 15 域 × 5 个等级 × 50K，总计 3,750,000 次图片评测。

`data.group=standard` 和 `extra` 严格区分；`data.domains` 可进一步选择组内的部分域。缺少请求的域、类别或选中的图片会报错，不会静默跳过。

5K 默认采用固定种子、每类 5 张的均衡采样；50K 每类 50 张。**5K 是本项目明确指定的采样协议，不代表某个外部论文的固定 5K 子集。** 可以设置 `data.sampling=random` 改为全局随机无放回抽样；复现外部子集时提供 `data.manifest`。

第一次从一个任务建立 `samples.json`，然后在所有域/等级复用相同的相对文件名、固定标签映射和顺序。**只配置一个 `run.seed`，同时控制选图、输入顺序，以及 Python / NumPy / PyTorch 的运行随机数。** 比如 `--set run.seed=42` 会一起改变这三部分，数据配置中不再单独设置种子。不同方法使用相同的 seed 和协议即可保持图片与顺序一致；seed 也记录在 `metadata.json` 和样本清单中。

显式设置 `data.manifest` 时，以指定清单中的图片与顺序为准，`run.seed` 仍控制运行随机数。50K 全量评测使用全部图片，改变 seed 会影响打乱顺序。

默认预处理在配置中明确列出：

- **ImageNet-C（标准 15 域及 extra 4 域）**：`CenterCrop(224) → ToTensor → Normalize`，`data.resize: null`，不执行 Resize。
- **干净 ImageNet**：`Resize(256, bicubic) → CenterCrop(224) → ToTensor → Normalize`，由干净数据配置设置 `data.resize: 256`。

`data.center_crop`、`data.mean`、`data.std` 分别记录裁剪大小及归一化参数。所有正式配置默认 `batch_size: 64`。默认打乱样本顺序；确需类别连续流时设置 `data.shuffle=false`。旧结果保留原配置，重跑时可读取旧 `config.yaml` 并只覆盖 `data.resize=null`。

## 运行实验

配置层次很简单：`base.yaml` → `--config` 指定的文件，从左到右覆盖 → `--set`。未知字段会报错。数据/权重/输出等配置路径相对 ATA 根目录解析；配置文件参数本身相对当前工作目录。

```bash
# 先查看最终配置、任务数和总样本量，不执行评测
python run.py --config configs/test_5k.yaml configs/tent.yaml --dry-run

# Source：标准 15 域，每域 5K，severity 5
python run.py --config configs/test_5k.yaml

# Tent：标准 15 域，每域 50K
python run.py --config configs/test_50k.yaml configs/tent.yaml

# InfoMax：extra 4 域验证，每域 5K
python run.py --config configs/val_5k.yaml configs/infomax.yaml

# extra 4 域、50K、全部 severity
python run.py --config configs/val_50k.yaml configs/tent.yaml \
  --set 'data.severities=[1,2,3,4,5]' run.name=tent_extra_all

# Small；切换尺寸时同时指定相应 checkpoint
python run.py --config configs/test_5k.yaml configs/tent.yaml \
  --set model.name=small model.checkpoint=weights/pretrained/vittt_small.pth run.name=tent_small

# 调参使用 extra 域；确定设置后，在 standard 域固定评测
python run.py --config configs/val_5k.yaml configs/tent.yaml \
  --set method.optimizer.lr=0.0001 run.name=tent_extra_lr1e4

# 只跑一个域、16 张的快速检查（任意数量用 random 采样）
python run.py --config configs/tent.yaml \
  --set 'data.domains=[gaussian_noise]' data.samples=16 data.sampling=random \
  data.batch_size=8 data.workers=0 run.name=smoke_tent

# 使用其他方法实验的完全相同子集和顺序
python run.py --config configs/test_5k.yaml configs/infomax.yaml \
  --set data.manifest=outputs/<run_id>/samples.json
```

所有方法目前用 FP32；默认超参数是可运行起点，不代表调优结果。批量扫参直接写 shell 循环即可；每个进程对应一个完整数据流，多张卡并行不同实验，例如 `CUDA_VISIBLE_DEVICES=1 python run.py ...`，进程内仍使用 `run.device=cuda:0`。

`sweep.py` 提供多卡排队入口。受 Git 跟踪的 YAML 描述模型、方法、数据和参数网格，GPU 在启动时通过 `--gpus` 选择：`--gpus all` 使用全部可见显卡，`--gpus 0 1 2` 使用指定显卡。编号遵循 `CUDA_VISIBLE_DEVICES`，不根据显存占用自动筛选；使用 holder 时保持相同的可见设备设置。旧运行配置中的 `gpus` 列表仍兼容，命令行选择优先。

`configs/sweep_50k.yaml` 包含 B 的 Tent / InfoMax / Lambda、S 的 Lambda，每组 3 个学习率 × 3 个 seed × test/val，共 72 次；都是每域 50K、severity 5、batch 64、SGD。各卡按预计耗时均衡分配，按 seed 顺序独立执行，当前任务完成后立即进入下一项。`configs/sweep_baselines_val_50k.yaml` 是 S/B 的 EATA / SAR / DeYO 验证网格，共 36 次。

```bash
python sweep.py --gpus 0 1 2 --dry-run
python -u sweep.py --gpus 0 1 2 --output outputs/sweep_50k --controller-pid <当前holder的PID>

# GPU 无需 holder 释放时直接运行；S/B × EATA/SAR/DeYO × 6 个学习率 × seed 0。
# extra 4 域各 50K，severity 5、batch 64；也可把 all 换成指定卡号。
python -u sweep.py --config configs/sweep_baselines_val_50k.yaml --gpus all --no-holder
```

在 tmux 中启动队列即可离开。队列目录的 `sweep.yaml` 保存实际采用的网格和 GPU 列表，`status.json` 记录每项任务的命令、GPU、状态、退出码和结果路径，`logs/` 保留启动日志，`runs/` 下每次实验仍使用独立目录。可以随时执行 `python summarize.py outputs/sweep_50k/runs` 汇总当前结果；各卡队列结束时也会刷新汇总。Ctrl-C / SIGTERM 会停止排队，holder 模式下自动恢复原占用。

`--no-holder` 会跳过 holder 脚本和状态目录检测，直接启动各实验，也不会恢复 GPU 占用。队列的 `status.json` 用 `use_holder` 记录运行方式；单项失败后继续其余任务。其他模型、方法、数据和重置协议保持相同。

一次实验的卡号、holder PID 和临时参数选择只保存在运行命令及输出目录中。需要调整参数网格时，可把临时 YAML 放在 `outputs/` 下，通过 `--config` 传入；仅启动实验无需修改仓库配置或创建 Git 提交。

## 方法和状态协议

- **Source**：不做外部适应，原始 TTT 计算照常执行。
- **Tent**：熵最小化，默认仅更新 LayerNorm 的 weight/bias。这是 ViTTT 上的 LN 版本；[官方 Tent](https://github.com/DequanWang/tent/blob/master/tent.py) 使用 BatchNorm2d，不能直接称为原版 BN 实现。默认保持 eval 模式，关闭 dropout/stochastic depth，同时为选中参数开启梯度；LayerNorm 的计算不因 train/eval 改变。
- **InfoMax**：每个 batch 优化 `mean(H(p(y|x))) - diversity_weight * H(mean(p(y|x)))`，默认同样更新 LN。它是信息最大化目标的 TTA 基线，**不包含完整 SHOT 的伪标签聚类等步骤**。batch size / 类别分布会影响这个目标，需作为实验设置记录。
- **Lambda**：外层使用相同的 InfoMax，只更新内层 loss 的 θ，冻结原始模型全部权重，包括所有 LN。详见下面的配置。

Tent / InfoMax 可用 `method.parameters=all` 显式启用全部参数适应；Lambda 始终只更新 θ。默认 `steps=1`。`method.prediction=before_update` 返回最后一次适应前向的预测，发生在最后一次优化器更新之前，匹配 Tent 的返回方式；多步时已经包含前面步骤的更新。`after_update` 则在完成所有更新后再做一次前向并评分。真实标签仅由评测器计算指标，不进入 `method.step(images)`，代码不根据测试标签选择最佳权重。

### Lambda：适应内层 loss

每个输出通道使用独立的 `lambda = lambda_max * tanh(theta)`，θ 初始为 0。内层 loss 为 `scale/N * sum(-v_hat*v + 0.5*lambda*v_hat²)`，生产代码不显式求 loss，而是把原来的导数 `-v/N*scale` 替换为 `(lambda*v_hat-v)/N*scale`。SwiGLU 的 N 是 token 数，卷积分支的 N 是空间位置数；原始归一化、内层步长 1 和两路前向保持原样。λ 可以为正或负，其范围为 `(-lambda_max, lambda_max)`。

`configs/lambda.yaml` 提供方法配置，数据和模型配置照常叠加：

```yaml
method:
  name: lambda
  parameters: theta             # Lambda 始终只更新 theta
  lambda_max: 1.0
  branches:
    swiglu: true
    conv: true
  layers: all                    # 根据实际模型层数选择；也可用 [0, 1, 11]
  diversity_weight: 1.0
```

`branches` 控制在哪条分支上加入 Lambda，不删除原模型分支。关闭某一路时，这一路继续使用原始 TTT 更新，不添加 θ；两路至少开启一路。`layers` 使用从 0 开始的 block 编号，未选中的 block 使用原始 TTT 更新。默认全部层、两路开启；ViTTT-S 共 5,376 个可训练标量（SwiGLU 每层 6×64 个，卷积每层 64 个，共 12 层）。θ=0 时与 Source 的输出完全一致；适应重置同时恢复 θ=0 和初始优化器状态。

```bash
# ViTTT-S / ImageNet-C 50K；先检查最终配置
python run.py --config configs/test_50k.yaml configs/lambda.yaml \
  --set model.name=small model.checkpoint=weights/pretrained/vittt_small.pth \
  run.name=lambda_small --dry-run

# 仅在最后两层的 SwiGLU 上适应，lambda_max=0.5
python run.py --config configs/test_50k.yaml configs/lambda.yaml \
  --set model.name=small model.checkpoint=weights/pretrained/vittt_small.pth \
  method.branches.conv=false 'method.layers=[10,11]' method.lambda_max=0.5
```

优化器沿用 `base.yaml` 的 SGD（lr=0.001、momentum=0.9、weight_decay=0），学习率只是可运行起点，可用 `method.optimizer.lr` 覆盖。默认 batch size 64、每个 batch 先计分后更新、每个任务重新从 θ=0 开始；默认不保存 checkpoint。显式开启 checkpoint 后，θ 和优化器动量一起保存，使用 `--resume` 恢复。

`protocol.reset`：

- `task`：每个 domain+severity 重置，默认值。
- `domain`：切换 corruption 时重置，同一 corruption 的多个 severity 连续适应。
- `batch`：每个 batch 重置。
- `never`：完整运行期间连续适应。

重置恢复源模型参数、buffer 和初始优化器状态。domain 顺序按 `data.py` 中的定义或显式 `data.domains` 列表；severity 顺序按配置。DataLoader 不丢弃最后一个 batch。

继续添加方法时，在 `methods.py` 实现 `step(images)`，返回 `(logits, metrics)`，并实现 `reset/state_dict/load_state_dict`，加入 `METHODS`，再提供一个 YAML 配置即可。可以继承现有方法复用目标函数、更新、保存和重置。

## LambdaW0：同时适应 loss 与快权重初值

`configs/lambda_w0.yaml` 的方法名是 `lambda_w0`。外层目标仍为 InfoMax，同时更新 θ 和选中 TTT 分支的快权重初值 W0：SwiGLU 为原模型的 `w1/w2`，卷积为 `w3`。W0 从官方预训练权重开始，θ 从 0 开始；QKV、投影、MLP、全部 LN 和分类头冻结。内层更新仍使用原来的步长和归一化。

两个参数组共用一个优化器，分别配置学习率：

```yaml
method:
  name: lambda_w0
  optimizer:
    name: sgd
    lr: 0.3           # theta 学习率
  w0_lr: 0.00001      # 快权重初值学习率
  layers: all
  branches:
    swiglu: true
    conv: true
```

`layers` 和 `branches` 同时确定 Lambda 与 W0 的适应范围：关闭卷积适应时 `w3` 保持冻结；未选中的 block 保持原样。每个任务重置 θ=0、W0=源权重及两组优化器动量。以上学习率只是起点，需在 extra 域选择。

```bash
python run.py --config configs/val_50k.yaml configs/lambda_w0.yaml \
  --set model.name=small model.checkpoint=weights/pretrained/vittt_small.pth \
  method.optimizer.lr=0.3 method.w0_lr=0.00001 run.name=lambda_w0_small
```

## EATA、SAR、DeYO

三个方法按作者官方代码移植，固定提交、文件哈希、接口调整和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。生产代码只保留必要的更新和依赖，官方 checkout 独立存放。

| 方法 | ViTTT 上的更新参数 | S 标量数 | B 标量数 |
|---|---|---:|---:|
| EATA | 全部 block 的 LN 和最终 LN | 19,200 | 38,400 |
| SAR / DeYO | blocks.0～8 的 norm1/norm2；冻结 blocks.9～11 和最终 norm | 13,824 | 27,648 |

所有方法统一 `eval()`，适应时开启梯度；当前模型 Dropout=0、DropPath=Identity，LN 的 train/eval 计算相同。所有 block 仍完整前向。数据预处理、batch 64、先计分后更新、每任务重置和默认不保存 checkpoint 的设置沿用原协议。

**EATA** 的更新顺序：整批前向 → `H < 0.4*ln(C)` 熵筛选 → 与历史概率向量的绝对余弦相似度 `< 0.05` 的非冗余筛选 → 更新概率 EMA（0.9）→ 加权熵加 Fisher 正则 → SGD。loss 为 `mean(H * exp(E0-stopgrad(H))) + 2000*sum(F*(phi-phi0)^2)`，权重不再归一化。最终没有样本时跳过更新，Fisher 正则也不单独触发更新。

EATA 初始化从 `method.eata.fisher_root` 的干净 ImageNet 随机选 2000 张，使用模型 argmax 伪标签的交叉熵。按官方累加每个 batch 的 LN 梯度平方，再除以 batch 数，不执行权重更新。Fisher batch 默认 64；预处理为 `Resize(256,bilinear) → CenterCrop(224) → ToTensor → Normalize`。`run.seed` 控制清单和实际 batch 顺序，保存到 `fisher_samples.json`。每个实验只计算一次，在内存中供各 corruption 复用。

**SAR** 的更新顺序：第一次前向和熵筛选（`0.4*ln(C)`）→ 第一次反传 → SAM 扰动 → 第二次完整前向 → 在首轮入选样本中再次熵筛选 → 第二次反传 → 恢复扰动前参数并做 SGD 更新。SAM `rho=0.05`、`adaptive=false`，基础 SGD momentum=0.9。第二次筛选后熵的 EMA 系数为 0.9，低于 0.2 时触发恢复。返回第一次前向预测。

SAR 保留官方内部恢复行为：复位源参数和 SAM 自身状态，保留基础 SGD 动量及本次 EMA；外部任务重置则完整清空动量和 EMA。官方空筛选路径仍执行零梯度和已有动量的更新，日志中的未定义 loss 记为 null。对应行为已与官方代码核对。

**DeYO** 的更新顺序：整批前向 → 熵 `< 0.5*ln(C)` 筛选 → 对入选图像做结构扰动并无梯度前向 → `PLPD > 0.2` 筛选 → 原图熵加权 → SGD。PLPD 为原图预测类别在原图、扰动图上的概率之差；权重为 `exp(0.4*ln(C)-stopgrad(H)) + exp(stopgrad(PLPD))`。两项相加，只反传原图熵；空筛选跳过更新。

DeYO 默认 `patch` 扰动，`patch_len=4` 是每行/列四块，共 16 块。保留官方独立逐图 CPU 随机排列、einops 重排和 Resize；也支持 `pixel` 和 `occ`。遮挡默认大小 112，起点行/列均为 56，用图像通道均值填充。熵/PLPD 筛选和两项加权默认全部开启。

```bash
python run.py --config configs/val_50k.yaml configs/eata.yaml \
  --set model.name=base model.checkpoint=weights/pretrained/vittt_base.pth \
  method.optimizer.lr=0.001 run.name=eata_base
# SAR / DeYO 分别替换为 configs/sar.yaml / configs/deyo.yaml。
```

除学习率外，默认沿用官方参数；学习率用 extra 4 域的多个 seed 均值选择，再固定到 standard 15 域。逐 batch 记录筛选数量、loss、前向/反向次数与实际更新数，SAR 另记熵 EMA 和恢复次数，DeYO 另记 PLPD 均值。筛选/调用计数也汇总到域结果和 summary。

EATA / SAR 的扩展 checkpoint 保存恢复和旧 checkpoint 兼容留待后续；当前使用默认 `run.save_weights=none`，显式调用这两种方法的 checkpoint 保存会提示尚未实现。

## 实验记录与保存权重

```text
outputs/<时间_名字_唯一标识>/
  config.yaml              # 最终配置
  metadata.json            # 环境、源权重 hash、样本 hash、参数选择、运行状态
  samples.json             # 固定类别映射 + 有序图片列表
  code/                    # 当次 Python/配置文件快照
  run.log
  metrics.jsonl            # 逐 batch 准确率、适应损失和耗时
  results.csv              # 逐域/逐 severity 结果
  summary.json             # 均值、错误率、top5、显存和完成状态
  top1.txt                 # 空格分隔的 Top-1 数字行，每个两位小数
  checkpoints/             # 仅显式开启权重保存时才会创建
    last.pt                # run.save_weights=last 或 task 时保存
    <domain>_s<level>.pt    # run.save_weights=task 时额外保留；干净数据为 clean.pt
```

**默认 `run.save_weights=none`，不调用权重序列化，也不创建 `checkpoints/` 目录。** 配置、日志和指标照常记录，已有预训练权重保持独立存放。

需要时显式设置 `run.save_weights=last`，每完成一个任务覆盖 `last.pt`；`task` 额外保存每个任务的模型。checkpoint 包含模型、优化器、方法计数、随机数状态、任务位置和已完成结果。保存的是该任务结束时的实际状态；`reset=batch` 时自然是最后一个 batch 的状态。

```bash
# 显式开启 checkpoint 保存
python run.py --config configs/test_5k.yaml configs/tent.yaml \
  --set run.save_weights=last

# 从已保存的任务边界恢复；新运行目录仍默认不保存 checkpoint
python run.py --config configs/test_5k.yaml configs/tent.yaml \
  --resume outputs/<run_id>/checkpoints/last.pt

# 汇总实验，包含失败/未完成状态，不混作完整结果
python summarize.py
```

恢复时模型/方法/数据/协议设置必须一致，并检查源权重、样本身份和顺序的 hash。中断中的任务重新运行，不从 batch 中间续跑。可以修改设备、worker 数、输出目录、运行名和日志频率。实际使用的代码快照分别保留在每次运行中；恢复不自动替换当前代码。

实验成功结束后，终端和 `run.log` 最后会输出一行纯数字，以空格分隔、每个保留两位小数，同时保存为 `top1.txt`。ImageNet-C 按配置的 corruption 顺序输出各域 Top-1，最后追加整体平均值；多个 severity 时先对每个域求平均。行前会标明顺序。干净 ImageNet 只输出一个 Top-1 数字。JSON / CSV 保留原始精度。

准确率和错误率单位为百分比。summary 的 `mean_top1` 是任务平均准确率，**不是 mCE**；这里没有套用 AlexNet 归一化。15 测试域与 4 验证域分别运行、分别汇总。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -B -m unittest discover -s tests -v
```

检查覆盖：单一 seed 对选图/顺序的控制、干净 ImageNet 单任务及配置隔离、15/4 域隔离、样本数量和复用、缺失图片报错、标签映射、可训练参数范围、重置优化器、预测时机、权重严格加载、任务恢复、旧配置恢复兼容，以及存在原仓库时的前向/反向逐项一致性。Lambda 另外核对两路手推更新及 θ 梯度与显式 loss 的自动微分结果、零初始化的 Source 等价性、分支和层选择、仅 θ 更新及状态恢复。

数据依据：[ImageNet-C 官方代码和数据说明](https://github.com/hendrycks/robustness)。模型许可证和原始署名保留在 `LICENSE` 及模型文件中。
