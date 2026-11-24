# 官方方法来源与移植约定

## 固定版本

| 方法 | 官方仓库 | Commit | 参考文件 |
|---|---|---|---|
| EATA 原始实现 | https://github.com/mr-eggplant/EATA | f739b3668cc7617e9b9f1979c1a358497a3472c3 | eata.py、main.py |
| EATA 的作者 LN 版本、SAR | https://github.com/mr-eggplant/SAR | 20f6e24b17525f34503510afccedc0629b67b7c4 | eata.py、sar.py、sam.py、main.py |
| DeYO | https://github.com/Jhyun17/DeYO | 3d7d2897d864907571ef75edf31f0028d53f1432 | methods/deyo.py、config.py、exp_deyo.sh |

官方文件 SHA256：

```text
SAR/eata.py          ab549bd7e0a764aaa264a6320fdd464f77987482d2a9928de984383769fc6dc0
SAR/sar.py           0553a395ac2bc087049720f1f162789079fe5ee25aa6a1cf82b2ea6286d66eff
SAR/sam.py           b0569de29015016996feae257d30be6cc36f80d42d0f51d4a201eb39d2491712
DeYO/methods/deyo.py f1510bd283e96e159c051dd5ce835a87fb7b3c7dab9f05f99a822e0d04cc14d7
```

## 接入 ATA 的部分

- `methods.py` 保留官方筛选顺序、loss、权重 detach 和更新公式，接入 `step(images)` 与本项目日志。EATA 采用作者 SAR 仓库中的 LN 支持。
- 所有方法统一 eval；当前 ViTTT 的 Dropout=0、DropPath=Identity，LN 的 train/eval 计算相同。
- SAR / DeYO 的官方排除层映射为 `blocks.9/10/11` 和最终 `norm`；只为实际选中的 LN 开启梯度。全部 block 仍参与前向。
- DeYO 保留官方 CPU 随机排列、einops patch/pixel 重排及 torchvision Resize。省去未使用的绘图和外部日志依赖，真实标签仅由评测器使用。
- EATA 使用入口设置 e_margin=0.4*ln(C)、d_margin=0.05、fisher_alpha=2000。Fisher 为 2000 张随机 clean 图片的伪标签 CE batch 梯度平方的 batch 平均，不更新源权重。clean 选择及实际 batch 顺序由本项目 run.seed 固定，写入 fisher_samples.json；预处理为 Resize(256,bilinear)、CenterCrop(224)、ImageNet Normalize。
- EATA 每个任务恢复源权重与优化器并清空概率 EMA，Fisher 保留复用，对应官方逐域新建方法的评测方式。
- `sam.py` 从 SAR 官方版本原样复制。SAR 内部恢复保留官方的基础 SGD 动量和新 EMA；跨域 reset 则按 ATA 协议完整清空动量。官方空筛选路径仍执行两次反传和动量更新；日志把未定义 loss 记为 null。
- EATA / SAR 的扩展 checkpoint 保存恢复后置，当前使用默认 run.save_weights=none。

以下为移植代码的原始许可证和署名。

## EATA License

```text
MIT License

Copyright (c) 2023 Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang,
Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SAR License

```text
BSD 3-Clause License

Copyright (c) niushuaicheng 2023,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## DeYO License

```text
MIT License

Copyright (c) 2024 Jonghyun Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
