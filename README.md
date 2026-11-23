# ATA: ViTTT 上的分类 TTA 实验

一个紧凑的科研代码目录，支持 ViTTT Tiny / Small / Base、干净 ImageNet / ImageNet-C 评测、Source、Tent、InfoMax，以及针对 TTT 内层 loss 的 Lambda 方法。

Agent 接手先阅读 [AGENTS.md](AGENTS.md)，其中记录方法、评测约定和运行上下文，与代码一起提交到 Git。数据、权重和实验输出独立保存；换设备后安装环境、下载权重并配置数据路径，需要复现实验时另外同步对应的实验目录。

## 模型保持原样

`models/vittt.py` 和 `models/ttt_block.py` 从 [ViTTT 官方仓库](https://github.com/LeapLabTHU/ViTTT) **逐字节复制**，上游版本和 SHA256 保存在 `models/UPSTREAM.json`，检查中会校验。Source、Tent、InfoMax 使用原始架构和内部 TTT 更新规则。`models/__init__.py` 提供模型选择、严格权重加载和下载；`models/lambda_ttt.py` 只为 Lambda 替换内层导数，继承原始 forward，复用加载好的原始权重。

当前只包含普通 ViTTT，不包含 H-ViTTT。Source 保持原模型完整前向（包括内部 TTT 更新），只是不做外部 TTA 优化。模型采用原始 224×224 输入和 1000 类分类头。

```text
ATA/
├── run.py                 # 配置 -> 数据 -> 适应/评测 -> 记录/保存
├── data.py                # ImageNet / ImageNet-C、样本列表和预处理
├── methods.py             # Source / Tent / InfoMax / Lambda
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
cd /mnt/bit/users/liyuanxi/projects/ATA
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

所有方法目前用 FP32；默认超参数是可运行起点，尚未在 extra 域调优。批量扫参直接写 shell 循环即可；每个进程对应一个完整数据流，多张卡并行不同实验，例如 `CUDA_VISIBLE_DEVICES=1 python run.py ...`，进程内仍使用 `run.device=cuda:0`。

`sweep.py` 提供多卡排队入口，网格保存在 `configs/sweep_50k.yaml`。当前包含 B 的 Tent / InfoMax / Lambda、S 的 Lambda，每组 3 个学习率 × 3 个 seed × test/val，共 72 次；都是每域 50K、severity 5、batch 64、SGD。六条队列按预计耗时均衡分配，各卡按 seed 0 → 1 → 2 独立执行，当前任务完成后立即进入下一项。每项经 holder 自动释放和恢复 GPU，单项失败会记录并继续后续任务。

```bash
python sweep.py --dry-run
python -u sweep.py --output outputs/sweep_50k --controller-pid <当前holder的PID>
```

在 tmux 中启动队列即可离开。队列目录的 `status.json` 记录每项任务的命令、GPU、状态、退出码和结果路径，`logs/` 保留启动日志，`runs/` 下每次实验仍使用独立目录。可以随时执行 `python summarize.py outputs/sweep_50k/runs` 汇总当前结果；六条队列各自结束时也会刷新汇总。Ctrl-C / SIGTERM 会停止排队并交给 holder 恢复占用。

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
