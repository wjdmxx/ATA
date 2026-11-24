# ATA 项目上下文

## 项目与代码

本项目研究分类任务中针对 TTT 模型的 TTA：通过外层适应来调整 TTT 的内层 loss。当前使用普通 ViTTT（Tiny / Small / Base），主要实验使用 ViTTT-S。

保持适合科研的简单结构，优先复用现有模块，按语义分段，代码简洁易读。

- `run.py`：实验入口，组织评测、重置、记录和 checkpoint。
- `data.py`：ImageNet / ImageNet-C、固定图片清单、预处理。
- `methods.py`：Source、Tent、InfoMax、Lambda、LambdaW0、EATA、SAR、DeYO；`sam.py` 为 SAR 官方两步优化器。
- `models/vittt.py`、`models/ttt_block.py`：官方原文件，SHA256 记录在 `models/UPSTREAM.json`。保持原始架构和计算规则。
- `models/lambda_ttt.py`：Lambda 专用内层导数，复用源模型权重并继承原始 forward。
- `utils.py`：配置、两台机器的路径回退、随机数、运行目录、代码快照、状态保存。
- `configs/`：数据和方法 YAML；`summarize.py` 汇总实验。

## 方法约定

- Source 保留模型内层 TTT，只是不进行外层 TTA。
- Tent 最小化预测熵；InfoMax 使用 `mean(H(p)) - H(mean(p))`。两者默认只更新全部 LayerNorm 的 weight / bias，包含最终 LN。
- Lambda 外层同样使用 InfoMax，始终只更新 θ，全部原始参数和 LN 冻结。
- Lambda 内层 loss 为 `scale/N * sum(-v_hat*v + 0.5*lambda*v_hat²)`；导数为 `(lambda*v_hat-v)/N*scale`，保持原始梯度归一化和内层步长 1。
- `lambda = lambda_max * tanh(theta)`，θ 初始为 0，`lambda_max` 默认 1。θ=0 时输出与 Source 完全一致。
- `method.branches.swiglu/conv` 独立控制在哪一路加入 Lambda，默认均开启；关闭后该路仍执行原始 TTT。
- `method.layers: all` 按实际模型层数选择，也可指定从 0 开始的 block 编号。SwiGLU 每头每输出通道独立 θ，卷积每通道独立 θ；ViTTT-S 默认共 5,376 个可训练标量。
- `lambda_w0` 在同一 InfoMax 目标下更新 θ 与选中层/分支的快权重初值：SwiGLU 的 w1/w2、卷积的 w3。`method.optimizer.lr` 控制 θ，`method.w0_lr` 独立控制 W0；其余参数冻结。任务重置恢复 θ=0、预训练 W0 和两组动量。
- EATA 更新全部 LN，使用熵/非冗余筛选和 Fisher 正则；启动时以 2000 张随机 clean 图片及模型伪标签计算 batch 梯度平方平均，Fisher 仅驻内存，清单写入 fisher_samples.json。
- SAR、DeYO 更新 blocks.0～8 的 LN，冻结 blocks.9～11 及最终 norm，全部 block 仍参与前向。SAR 使用官方 SAM+SGD；内部恢复保留官方 SGD 动量与新 EMA，外部跨域重置完整清空。DeYO 使用熵/PLPD 筛选与相加的两项权重，默认 4×4 patch 重排。
- 全部方法统一 eval；当前模型 Dropout=0、DropPath=Identity。EATA / SAR 的扩展 checkpoint 和旧 checkpoint 兼容后置，当前按默认不保存 checkpoint 使用。官方版本、移植细节和许可证见 THIRD_PARTY.md，完整 loss/超参数见 README。

## 数据与评测

- ImageNet-C：标准 15 corruption 正式评测，extra 4 corruption（speckle_noise、gaussian_blur、spatter、saturate）用于验证。默认 severity 5，支持 5K / 50K，每个域、每个 severity 分别计算图片数。
- ImageNet-C 预处理：`CenterCrop(224) → ToTensor → ImageNet Normalize`，`data.resize: null`。
- 干净 ImageNet：`Resize(256, bicubic) → CenterCrop(224) → ToTensor → ImageNet Normalize`；作为一个 clean 任务。
- 正式实验 batch size 64、FP32、eval 模式；适应前向保留梯度。标签只用于计分。
- 一个 `run.seed` 同时控制选图、打乱顺序和运行随机数。每个实验生成 `samples.json`，各域沿用同一份清单；`data.manifest` 可跨实验复用清单。
- 默认每个 batch 先预测计分，再更新一步；每个 domain+severity 重置源状态及优化器动量，Lambda 同时回到 θ=0。最后一个 batch 保留。
- 本机 ViTTT-S Source 参考：clean 50K Top-1 为 81.72%；ImageNet-C 标准 15 域、severity 5、50K 平均为 46.0952%。
- 已跑 Tent SGD lr=1e-5、InfoMax SGD lr=1e-3、momentum=0.9，seed 1 / 2；这些实验通过命令行覆盖学习率，方法 YAML 没有固化调参结果。Lambda 当前沿用 base 的 lr=0.001，尚未调优。

## 运行与记录

配置顺序：`configs/base.yaml` → `--config` 文件从左到右覆盖 → `--set key=value`。默认模型是 Tiny；切换 Small / Base 时同时指定模型名和相应权重。

```bash
conda activate vittt
python run.py --config configs/test_50k.yaml configs/lambda.yaml \
  --set model.name=small model.checkpoint=weights/pretrained/vittt_small.pth \
  run.name=lambda_small --dry-run
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -B -m unittest discover -s tests -v
```

去掉 `--dry-run` 即执行实验。环境安装和权重下载见 `README.md`、`requirements.txt`；下载入口为 `python run.py --download small base`。

批量实验入口为 `sweep.py --config configs/sweep_50k.yaml`，当前网格是 B 的 Tent / InfoMax / Lambda 和 S 的 Lambda，各 3 个学习率、seed 0/1/2、test/val 两组，共 72 次。队列按预计耗时均衡分配到 GPU 2–7；各卡按 seed 顺序独立执行，完成当前任务就开始下一项，不设跨卡 seed 分界。每项通过 holder 的 exec 自动释放和恢复显卡；失败记录后继续剩余任务。队列目录保存 `status.json`、每项启动日志及 `runs/` 下的完整实验目录。

`configs/sweep_baselines_val_50k.yaml` 是 EATA / SAR / DeYO 的首轮学习率搜索：S/B、extra 4 域每域 50K、batch 64、seed 0，学习率 [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]，共 36 次，配置使用 GPU 4–7。其他方法超参数保持官方默认，SAR 使用 SAM+SGD；EATA 的 Fisher 默认 clean 2000 张。沿用同一个 sweep.py 排队。

每次实验写入 `outputs/<时间_名字_唯一标识>/`：最终配置、环境和参数信息、样本清单、代码快照、逐 batch 指标、逐域结果、汇总和日志。`run.save_weights: none` 为默认值，不保存 checkpoint；显式开启后保存模型、θ、优化器和恢复状态。

实验结束后，终端、`run.log` 和 `top1.txt` 输出空格分隔、两位小数的 Top-1 数字行：各 corruption 后接均值；clean 只有一个数字。`python summarize.py` 汇总到 `outputs/comparison.csv`。

检查覆盖原模型一致性、数据协议、冻结范围、状态重置与恢复，以及两路 Lambda 手推更新和 θ 梯度与显式 loss 自动微分的一致性。

## 当前机器

4090 机器项目位于 `/mnt/bit/users/liyuanxi/projects/ATA`，环境 Python 为 `/mnt/bit/users/liyuanxi/miniconda3/envs/vittt/bin/python`。ImageNet-C 位于 `/mnt/bit/data/imagenet-c`，clean 验证集位于 `/mnt/bit/data/imagenet/images/val`。

临时 H20 机器项目位于 `/data1/buaa/liyuanxi/projects/ATA`，环境 Python 为 `/data1/conda_envs/liyuanxi/vittt/bin/python`，ImageNet-C 位于 `/data1/buaa/datasets/imagenet-c`，clean 验证集位于 `/data1/buaa/datasets/imagenet/val`。`utils.resolve_machine_path` 优先使用配置中的 4090 路径，不存在时回退到 H20 路径，覆盖评测数据和 EATA Fisher；自定义路径不回退，实际执行时找不到路径则报错。两台机器切换无需改 YAML。权重统一使用项目相对路径 `weights/pretrained/vittt_{tiny,small,base}.pth`；H20 上已从相邻 ViTTT 项目复制三个权重。

GPU 占用工具优先使用相邻的 `../g/g.py`，不存在时回退到 `../gpu-holder/gpu_hold.py`。跑 GPU 实验时使用它的 `exec --gpus ... -- <实验命令>`，自动完成释放、实验、重新占用；实验的 `run.device` 要对应所选 GPU。`sweep.py` 自动检测脚本和状态目录；状态目录优先 `/run/user/$(id -u)/gpu-holder`，不存在时回退 `/tmp/gpu-holder-$(id -u)`，显式 `GPU_HOLDER_STATE_DIR` 优先。多实例时用当前 PID 指定 `--controller-pid`。

长实验放在 `tmux` 的 `ata` session 中，启动后通过日志查看进度。当前机器的 GPU、holder PID 和 tmux 状态在运行前查询。

## Git 与接手

远程仓库为 `git@github.com:wjdmxx/ATA.git`（https://github.com/wjdmxx/ATA），主分支为 `main`。

Git 跟踪代码、配置、依赖、README、本文件和上游来源信息。数据、预训练权重、实验输出及机器专用配置独立存放；换设备时从 Git 恢复代码，再下载权重、配置数据路径。要复现某次实验，另外同步该实验目录中的配置和 `samples.json`。

完成一组相关修改后，运行适用检查，提交代码和相关文档。方法、协议或运行方式变化时同步更新本文件，详细使用说明放在 README；本文件保留当前事实和接手需要的信息。
