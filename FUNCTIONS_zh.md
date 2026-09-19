# ComfyDL 节点参考

> [English version / 英文版](FUNCTIONS.md)

本文档详细说明 ComfyDL 中的每一个自定义节点：功能、输入与输出。节点集起初源于 d2l（动手学深度学习）教材代码库的映射，并持续扩展更多自主开发的节点。

ComfyDL 以 **GPL-3.0** 授权（见 [LICENSE](LICENSE)），同时也随其自有的 GUI 版本 [ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI) 一起分发——在那里，下文所有节点都作为内置节点注册。

---

## 数据类型

ComfyDL 节点通过以下 ComfyUI 类型槽传递结构化数据：

| 类型名 | Python 类型 | 说明 |
|-----------|-------------|------|
| `TENSOR` | `torch.Tensor` | 任意形状的 PyTorch 张量 —— **ComfyUI 核心类型**，与内置 `Network & Layers` 节点共用（原名 `cdlTensor`） |
| `BBOX` | `torch.Tensor [N,4]` | 边界框张量，格式为 `(x1, y1, x2, y2)` —— **ComfyUI 核心类型**（原名 `cdlBbox`） |
| `cdlModel` | `nn.Module` | PyTorch 模型实例 —— ComfyDL 专有，核心无同义类型 |
| `cdlVocab` | `dict` | 词表字典，包含 `idx_to_token` 和 `token_to_idx` —— ComfyDL 专有 |
| `cdlDataloader` | `torch.utils.data.DataLoader` | PyTorch 数据加载器 —— ComfyDL 专有 |

`TENSOR` 与 `BBOX` 定义在 ComfyUI 核心中（`comfy/comfy_types/node_typing.py` 与
`comfy_api/latest/_io.py`），因此 ComfyDL 节点可与核心节点（如 `Network & Layers` 系列）在同一插槽上
直接连线。`cdlModel` / `cdlVocab` / `cdlDataloader` 在核心中没有等价类型，保持 ComfyDL 专有；
旧名 `cdlTensor` / `cdlBbox` 仍作为兼容别名从 `nodes/__init__.py` 导出。

直接使用的 ComfyUI 标准类型：
- `IMAGE` — 图像批次，`torch.Tensor [B, H, W, C]`
- `MASK` — 掩码，`torch.Tensor [H, W]` 或 `[B, C, H, W]`
- `LATENT` — 潜变量字典 `{"samples": ..., "noise_mask"?, "batch_index"?, "type"?}`
- `AUDIO` — 音频字典 `{"waveform": ..., "sampler_rate": ...}`
- `SIGMAS` — 噪声调度，`torch.Tensor [N]`
- `LORA_MODEL` — 张量集合 `dict[str, torch.Tensor]`（预留，见 §18）
- `LOSS_MAP` — 张量集合 `{"loss": [torch.Tensor, ...]}`（预留，见 §18）
- `ARRAY` — 普通 Python `list`，用于 §18 的平行元数据槽
- `MODEL`、`CLIP`、`VAE`、`CONDITIONING` — ComfyUI 核心的模型协议类型，由 §19 恢复
- `INT`、`FLOAT`、`STRING`、`BOOLEAN` — 基本标量类型

---

## 1. d2l / Device Utils（3 个节点）

### Device Info
- **类名**：`CdlDeviceInfo`
- **d2lcore 函数**：`num_gpus()`
- **功能**：查询当前环境的 GPU 数量和 CUDA 可用性。
- **输入**：无
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `num_gpus` | `INT` | 可用 GPU 数量 |
  | `has_cuda` | `INT` | CUDA 是否可用（1=是，0=否） |

### Try GPU
- **类名**：`CdlTryGpu`
- **d2lcore 函数**：`try_gpu(i)`
- **功能**：尝试获取索引为 `gpu_index` 的 GPU 设备名。若该 GPU 不可用则回退到 CPU。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `gpu_index` | `INT` | 0 | GPU 索引（0~16） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `device_str` | `STRING` | 设备字符串，如 `"cuda:0"` 或 `"cpu"` |

### Try All GPUs
- **类名**：`CdlTryAllGpus`
- **d2lcore 函数**：`try_all_gpus()`
- **功能**：返回所有可用 GPU 设备名的逗号分隔字符串。若无 GPU 可用则返回 `"cpu"`。
- **输入**：无
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `device_str` | `STRING` | 如 `"cuda:0,cuda:1,cuda:2,cuda:3"` 或 `"cpu"` |

---

## 2. d2l / CV Models（5 个节点）

### Corr2D
- **类名**：`CdlCorr2d`
- **d2lcore 函数**：`corr2d(X, K)`
- **功能**：对输入张量执行二维互相关运算——这是卷积操作的基础。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `input_tensor` | `TENSOR` | 输入二维张量 |
  | `kernel` | `TENSOR` | 核二维张量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 互相关结果 |

### LeNet
- **类名**：`CdlLeNet`
- **d2lcore 函数**：`LeNet(lr, num_classes)`
- **功能**：构建经典的 LeNet-5 卷积神经网络。使用 `LazyConv2d` 和 `LazyLinear`——输入形状在首次前向传播时自动推断。
- **架构**：`LazyConv2d(6,5) → Sigmoid → AvgPool2d(2,2) → LazyConv2d(16,5) → Sigmoid → AvgPool2d(2,2) → Flatten → LazyLinear(120) → Sigmoid → LazyLinear(84) → Sigmoid → LazyLinear(num_classes)`
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_classes` | `INT` | 10 | 输出类别数（1~1000） |
  | `lr` | `FLOAT` | 0.1 | 学习率（预留参数） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | LeNet-5 模型实例 |

### ResNet-18
- **类名**：`CdlResNet18`
- **d2lcore 函数**：`resnet18(num_classes, in_channels)`
- **功能**：构建修改版 ResNet-18 模型（使用更小的卷积核/步幅/填充，不含最大池化）。包含 4 个残差块组（每组 2 个残差块），通道数分别为 64、128、256、512，后接全局平均池化和全连接层。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_classes` | `INT` | 10 | 输出类别数（1~10000） |
  | `in_channels` | `INT` | 1 | 输入通道数（1~1024） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | ResNet-18 模型实例 |

### Residual Block
- **类名**：`CdlResidual`
- **d2lcore 函数**：`Residual(num_channels, use_1x1conv, strides)`
- **功能**：创建单个 ResNet 残差块。包含两个卷积层（Conv2d + BatchNorm + ReLU）和一个可选的 1×1 快捷卷积。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_channels` | `INT` | 64 | 输出通道数（1~2048） |
  | `use_1x1conv` | `BOOLEAN` | False | 启用 1×1 快捷卷积 |
  | `strides` | `INT` | 1 | 卷积步幅（1~4） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `block` | `cdlModel` | 残差块模块实例 |

### ResNeXt Block
- **类名**：`CdlResNeXtBlock`
- **d2lcore 函数**：`ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv, strides)`
- **功能**：创建单个 ResNeXt 块，使用分组卷积实现多分支结构。包含瓶颈结构（1×1 降维 → 3×3 分组卷积 → 1×1 升维）+ 可选快捷连接。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_channels` | `INT` | 64 | 输出通道数（1~2048） |
  | `groups` | `INT` | 32 | 分组卷积的组数（1~1024） |
  | `bot_mul` | `FLOAT` | 0.5 | 瓶颈通道倍数（0.125~2.0） |
  | `use_1x1conv` | `BOOLEAN` | False | 启用快捷卷积 |
  | `strides` | `INT` | 1 | 卷积步幅（1~4） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `block` | `cdlModel` | ResNeXt 块模块实例 |

---

## 3. d2l / GAN（2 个节点）

### Update Discriminator
- **类名**：`CdlUpdateD`
- **d2lcore 函数**：`update_D(X, Z, net_D, net_G, loss, trainer_D)`
- **功能**：执行一次 GAN 判别器的训练更新。使用真实数据 `X` 和生成器 `net_G` 从噪声 `Z` 生成的假数据，计算 BCE 损失并反向传播以更新判别器参数。使用 SGD 优化器（lr=0.01）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `TENSOR` | 真实数据批次（可选） |
  | `Z` | `TENSOR` | 噪声输入（可选） |
  | `net_D` | `cdlModel` | 判别器模型（可选） |
  | `net_G` | `cdlModel` | 生成器模型（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `loss_D` | `FLOAT` | 判别器损失；若任一输入缺失则返回 0.0 |

### Update Generator
- **类名**：`CdlUpdateG`
- **d2lcore 函数**：`update_G(Z, net_D, net_G, loss, trainer_G)`
- **功能**：执行一次 GAN 生成器的训练更新。从噪声 `Z` 生成假数据，尝试欺骗判别器，计算 BCE 损失并反向传播以更新生成器参数。使用 SGD 优化器（lr=0.01）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `Z` | `TENSOR` | 噪声输入（可选） |
  | `net_D` | `cdlModel` | 判别器模型（可选） |
  | `net_G` | `cdlModel` | 生成器模型（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `loss_G` | `FLOAT` | 生成器损失；若任一输入缺失则返回 0.0 |
  | `fake_X` | `TENSOR` | 生成器产生的假数据 |

---

## 4. ComfyUI / utilities（4 个节点）

已并入 ComfyUI 核心分类 `utilities`（前端分组：实用工具）。类名保留 `Cdl` 前缀。

### MessageBox
- **类名**：`CdlMessageBox`
- **功能**：通过 ctypes 调用 `user32.dll` 中的 `MessageBoxW` 弹出原生 Windows 消息对话框。支持阻塞和非阻塞两种模式。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `title` | `STRING` | `"ComfyDL"` | 对话框标题 |
  | `text` | `STRING` | `"Hello from ComfyDL!"` | 对话框消息文本（多行） |
  | `button_type` | `COMBO` | `MB_OK` | 按钮类型：MB_OK / MB_OKCANCEL / MB_ABORTRETRYIGNORE / MB_YESNOCANCEL / MB_YESNO / MB_RETRYCANCEL |
  | `icon_type` | `COMBO` | `MB_ICONINFORMATION` | 图标类型：MB_ICONINFORMATION / MB_ICONWARNING / MB_ICONERROR / MB_ICONQUESTION |
  | `block` | `BOOLEAN` | True | True=阻塞模式（等待用户关闭），False=非阻塞模式 |
  | `any_input` | `*` | — | 通配输入（可选），用于触发节点执行 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `result` | `STRING` | 用户点击的按钮名称（如 `"2 (IDCANCEL)"`）；在非 Windows 系统上返回占位字符串 |

### NoOp
- **类名**：`CdlNoOp`
- **功能**：空操作节点——接受任意输入但不执行任何计算。等同于 Python 的 ``pass`` 或汇编的 ``NOP``。可作为任意数据类型的空接收器、工作流构建时的占位节点，或调试时的旁路工具。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `any_input` | `*` | — | 通配输入（可选），接受任意类型数据——直接丢弃 |
- **输出**：无

### Timer（计时基准）
- **类名**：`CdlTimer`
- **功能**：对张量操作进行基准测试——在预热 3 次后运行 `num_iters` 次并报告总耗时与平均耗时。支持 `sum`、`mean`、`abs`、`sqrt`、`neg`。当张量位于 GPU 时，计时前后会执行 CUDA 同步。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `operation` | `COMBO` | `sum` | 计时操作：sum / mean / abs / sqrt / neg |
  | `num_iters` | `INT` | 10 | 计时迭代次数（1~100000） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `report` | `STRING` | 人类可读的计时报告 |
  | `avg_seconds` | `FLOAT` | 平均每次迭代秒数 |

### ?
- **类名**：`CdlWhat`
- **功能**：试一试就知道了~
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `OMG` | `BOOLEAN` | False | 试一试就知道了~ |
  | `any_input` | `*` | — | 任意通配输入（可选）——被忽略 |
- **输出**：无（输出节点）

---

## 5. d2l / NLP Utils（5 个节点）

### Tokenize
- **类名**：`CdlTokenize`
- **d2lcore 函数**：`tokenize(lines, token)`
- **功能**：将输入文本按行分词。支持词级别（按空白符分割）和字符级别两种分词模式。每行视为一个句子；行内 token 以逗号分隔输出，行间以换行符分隔。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `text` | `STRING` | `"the quick brown fox\njumps over the lazy dog"` | 输入文本，每行一个句子（多行） |
  | `token_mode` | `COMBO` | `word` | 分词模式：`word`（按空白符分词）/ `char`（字符级分词） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tokens_str` | `STRING` | 分词结果字符串；每行 token 以逗号分隔，行间以换行符分隔 |

### Get Tokens & Segments
- **类名**：`CdlGetTokensAndSegments`
- **d2lcore 函数**：`get_tokens_and_segments(tokens_a, tokens_b)`
- **功能**：为 BERT 模型准备输入。将 A 段和 B 段的 token 与 `[CLS]` 和 `[SEP]` 标记拼接，并生成段 ID（A 段为 0，B 段为 1）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tokens_a` | `STRING` | `"the,quick,brown,fox"` | A 段，逗号分隔的 token（多行） |
  | `tokens_b` | `STRING` | `"jumps,over"` | B 段，逗号分隔的 token（可选，多行） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tokens` | `STRING` | 拼接后的 token 序列（含 `[CLS]` 和 `[SEP]`），逗号分隔 |
  | `segments` | `STRING` | 段 ID 序列，逗号分隔（0=A，1=B） |

### Vocab Build
- **类名**：`CdlVocabBuild`
- **d2lcore 函数**：`Vocab(tokens, min_freq, reserved_tokens)`
- **功能**：从 token 文本构建词表。统计 token 频率，保留出现次数 ≥ `min_freq` 的 token，并添加保留 token（如 `<pad>`、`<bos>`、`<eos>`）。`<unk>` token 自动包含在内。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tokens_text` | `STRING` | `"the quick\nbrown fox\nthe lazy dog"` | Token 文本，每行一个或逗号分隔（多行） |
  | `min_freq` | `INT` | 1 | 最小频率阈值（1~100000） |
  | `reserved_tokens` | `STRING` | `"<pad>,<bos>,<eos>"` | 保留 token，逗号分隔 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `vocab` | `cdlVocab` | 词表字典 `{"idx_to_token": [...], "token_to_idx": {...}}` |
  | `vocab_size` | `INT` | 词表大小 |

### Vocab Encode
- **类名**：`CdlVocabEncode`
- **d2lcore 函数**：`Vocab.__getitem__(tokens)`
- **功能**：使用词表将 token 字符串转换为索引张量。词表中不存在的 token 映射为 `<unk>` 索引。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `vocab` | `cdlVocab` | — | 词表字典 |
  | `tokens` | `STRING` | `"the,quick,brown"` | 逗号分隔的 token 序列（多行） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `indices` | `TENSOR` | 编码后的索引张量（`torch.int64`） |

### Vocab Decode
- **类名**：`CdlVocabDecode`
- **d2lcore 函数**：`Vocab.to_tokens(indices)`
- **功能**：使用词表将索引张量转换回 token 字符串。超出范围的索引映射为 `<unk>`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `vocab` | `cdlVocab` | 词表字典 |
  | `indices` | `TENSOR` | 索引张量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tokens_str` | `STRING` | 解码后的 token 字符串，逗号分隔 |

---

## 6. d2l / NLP Models（12 个节点）

NLP 模型构建节点包装 d2lcore 的 RNN/GRU/RNNLM、注意力/Transformer 与 Seq2Seq 构件。所有构建器都返回 `cdlModel`，可接入 `CdlModelForward` / `CdlModelInfo` / `CdlModelSave` 等节点进行查看与推理。RNN/GRU 前向输入为时间优先 `(num_steps, batch_size, num_inputs)`；注意力模块与 Transformer 编码器为批次优先。

### RNN（从头实现）
- **类名**：`CdlRNNScratch`
- **d2lcore 函数**：`RNNScratch(num_inputs, num_hiddens, sigma)`
- **功能**：从头构建 RNN（tanh 单元、手动创建参数）。前向输入形状 `(num_steps, batch_size, num_inputs)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_inputs` | `INT` | 32 | 输入特征维度（语言模型中为词表大小）（1~100000） |
  | `num_hiddens` | `INT` | 64 | 隐藏单元数（1~4096） |
  | `sigma` | `FLOAT` | 0.01 | 随机参数初始化标准差（0.0001~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | RNNScratch 实例 |

### RNN（高层封装）
- **类名**：`CdlRNN`
- **d2lcore 函数**：`RNN(num_inputs, num_hiddens)`
- **功能**：使用 PyTorch 高层 `nn.RNN` 构建 RNN。前向输入 `(num_steps, batch_size, num_inputs)`，返回 `(output, h_n)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_inputs` | `INT` | 32 | 输入特征维度（1~100000） |
  | `num_hiddens` | `INT` | 64 | 隐藏单元数（1~4096） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | RNN 实例 |

### GRU
- **类名**：`CdlGRU`
- **d2lcore 函数**：`GRU(num_inputs, num_hiddens, num_layers, dropout)`
- **功能**：使用 PyTorch 高层 `nn.GRU` 构建多层 GRU。前向输入 `(num_steps, batch_size, num_inputs)`，返回 `(output, h_n)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_inputs` | `INT` | 32 | 输入特征维度（1~100000） |
  | `num_hiddens` | `INT` | 64 | 每层隐藏单元数（1~4096） |
  | `num_layers` | `INT` | 1 | GRU 层数（1~20） |
  | `dropout` | `FLOAT` | 0.0 | 层间 dropout（0~0.9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | GRU 实例 |

### RNN 语言模型（从头实现）
- **类名**：`CdlRNNLMScratch`
- **d2lcore 函数**：`RNNLMScratch(rnn, vocab_size, lr)`
- **功能**：将 RNN/GRU `cdlModel`（要求 `num_inputs == vocab_size`）包装为从头实现的语言模型，输出投影到 `vocab_size` 个类别。前向输入索引张量 `(batch_size, num_steps)`，返回 logits `(num_steps, batch_size, vocab_size)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `rnn` | `cdlModel` | — | 满足 `num_inputs == vocab_size` 的 RNN/GRU cdlModel |
  | `vocab_size` | `INT` | 32 | 输出投影的词表大小（2~100000） |
  | `lr` | `FLOAT` | 0.01 | 训练时使用的学习率（0.0001~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | RNNLMScratch 实例 |

### RNN 语言模型（高层封装）
- **类名**：`CdlRNNLM`
- **d2lcore 函数**：`RNNLM(rnn, vocab_size, lr)`
- **功能**：将 RNN/GRU `cdlModel` 包装为使用高层 `LazyLinear` 头的语言模型，接口与从头版本一致。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `rnn` | `cdlModel` | — | 满足 `num_inputs == vocab_size` 的 RNN/GRU cdlModel |
  | `vocab_size` | `INT` | 32 | 输出投影的词表大小（2~100000） |
  | `lr` | `FLOAT` | 0.01 | 训练时使用的学习率（0.0001~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | RNNLM 实例 |

### RNN LM 预测
- **类名**：`CdlRNNLMScratchPredict`
- **d2lcore 函数**：`RNNLMScratch.predict(prefix, num_preds, vocab, device)`
- **功能**：给定起始前缀，用 RNN 语言模型生成文本。接受 `cdlVocab` 字典（来自 `CdlVocabBuild`），返回前缀加上 `num_preds` 个预测 token。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `model` | `cdlModel` | — | RNNLMScratch / RNNLM cdlModel |
  | `vocab` | `cdlVocab` | — | 来自 `CdlVocabBuild` 的词表字典 |
  | `prefix` | `STRING` | `"the "` | 起始 token，如 `"the "` |
  | `num_preds` | `INT` | 10 | 前缀之后预测的 token 数（1~1000） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `prediction` | `STRING` | 前缀加上预测出的 token |

### 点积注意力
- **类名**：`CdlDotProductAttention`
- **d2lcore 函数**：`DotProductAttention(dropout)`
- **功能**：构建缩放点积注意力层。前向 `(queries, keys, values, valid_lens)`，批次优先张量 `(batch, seq, num_hiddens)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `dropout` | `FLOAT` | 0.0 | 注意力权重上的 dropout（0~0.9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 注意力层 |

### 加性注意力
- **类名**：`CdlAdditiveAttention`
- **d2lcore 函数**：`AdditiveAttention(num_hiddens, dropout)`
- **功能**：构建加性（Bahdanau）注意力层。前向 `(queries, keys, values, valid_lens)`，批次优先张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_hiddens` | `INT` | 8 | 加性评分函数的隐藏单元数（1~4096） |
  | `dropout` | `FLOAT` | 0.0 | 注意力权重上的 dropout（0~0.9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 注意力层 |

### 多头注意力
- **类名**：`CdlMultiHeadAttention`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/NLP Models`）
- **替代**：在纯 `TENSOR` 图上改用核心注意力节点（`Network & Layers/Attention`）：`AttentionMultihead`（q/k/v 分开接线）、`AttentionSelf` / `AttentionCross`（常用形态封装）或组装好的 `TransformerEncoderBlock`。本节点构建的是模块级 `cdlModel`（而非 `TENSOR`），功能未变，`cdlModel` 流水线仍可继续使用。
- **d2lcore 函数**：`MultiHeadAttention(num_hiddens, num_heads, dropout, bias)`
- **功能**：构建多头注意力层。`num_hiddens` 必须能被 `num_heads` 整除。前向 `(queries, keys, values, valid_lens)`，批次优先张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_hiddens` | `INT` | 8 | 模型宽度；需能被 `num_heads` 整除（1~4096） |
  | `num_heads` | `INT` | 4 | 并行注意力头数（1~64） |
  | `dropout` | `FLOAT` | 0.0 | 注意力权重上的 dropout（0~0.9） |
  | `use_bias` | `BOOLEAN` | False | Q/K/V/O 投影是否使用偏置 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 多头注意力层 |

### 位置编码
- **类名**：`CdlPositionalEncoding`
- **d2lcore 函数**：`PositionalEncoding(num_hiddens, dropout, max_len)`
- **功能**：构建正弦位置编码层。前向 `X` 形状 `(batch, seq, num_hiddens)`，为输入添加位置信息。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_hiddens` | `INT` | 16 | 模型宽度（特征维度）（1~4096） |
  | `dropout` | `FLOAT` | 0.0 | 添加编码后的 dropout（0~0.9） |
  | `max_len` | `INT` | 1000 | 最大支持序列长度（1~100000） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 位置编码层 |

### 位置逐元素 FFN
- **类名**：`CdlPositionWiseFFN`
- **d2lcore 函数**：`PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)`
- **功能**：构建位置逐元素前馈网络（两个全连接层 + ReLU），对每个位置独立应用。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `ffn_num_hiddens` | `INT` | 64 | 内层全连接的隐藏单元数（1~16384） |
  | `ffn_num_outputs` | `INT` | 16 | 输出单元数，通常等于 `num_hiddens`（1~16384） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | FFN 模块 |

### 残差 + 层归一化（Add & Norm）
- **类名**：`CdlAddNorm`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/NLP Models`）
- **替代**：在纯 `TENSOR` 图上用 `NormalizationLayerNorm` 接在 `BasicAdd` 之后即可——`LayerNorm(dropout(Y) + X)` 两者完全等价。本节点操作的是模块级 `cdlModel`（而非 `TENSOR`），功能未变，`cdlModel` 流水线仍可继续使用。
- **d2lcore 函数**：`AddNorm(norm_shape, dropout)`
- **功能**：构建残差连接后接层归一化：`LayerNorm(dropout(Y) + X)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `norm_shape` | `INT` | 16 | LayerNorm 的特征维度（1~16384） |
  | `dropout` | `FLOAT` | 0.0 | 残差分支上的 dropout（0~0.9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | Add & Norm 模块 |

### Transformer 编码器块
- **类名**：`CdlTransformerEncoderBlock`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/NLP Models`）
- **替代**：在纯 `TENSOR` 图上由核心层节点组合即可（多头注意力 + `BasicAdd` 后接 `NormalizationLayerNorm` + 由 `BasicLinear` 搭出的位置逐元素 FFN）。本节点构建的是模块级 `cdlModel`（而非 `TENSOR`），功能未变，`cdlModel` 流水线仍可继续使用。
- **d2lcore 函数**：`TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias)`
- **功能**：构建单个 Transformer 编码器块（多头注意力 + FFN，含残差与层归一化）。前向 `(X, valid_lens)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_hiddens` | `INT` | 8 | 模型宽度；需能被 `num_heads` 整除（1~4096） |
  | `ffn_num_hiddens` | `INT` | 64 | 位置逐元素 FFN 的隐藏单元数（1~16384） |
  | `num_heads` | `INT` | 4 | 注意力头数（1~64） |
  | `dropout` | `FLOAT` | 0.0 | dropout 概率（0~0.9） |
  | `use_bias` | `BOOLEAN` | False | 注意力投影是否使用偏置 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | Transformer 编码器块 |

### Transformer 编码器
- **类名**：`CdlTransformerEncoder`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/NLP Models`）
- **替代**：在纯 `TENSOR` 图上由核心层节点组合即可（嵌入 + 位置编码 + 堆叠编码器块）。本节点构建的是模块级 `cdlModel`（而非 `TENSOR`），功能未变，`cdlModel` 流水线仍可继续使用。
- **d2lcore 函数**：`TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias)`
- **功能**：构建完整 Transformer 编码器（嵌入 + 位置编码 + `num_blks` 个堆叠块）。前向 `(X, valid_lens)`，`X` 为 token 索引 `(batch, seq)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `vocab_size` | `INT` | 32 | 输入嵌入的词表大小（2~100000） |
  | `num_hiddens` | `INT` | 8 | 模型宽度；需能被 `num_heads` 整除（1~4096） |
  | `ffn_num_hiddens` | `INT` | 64 | 位置逐元素 FFN 的隐藏单元数（1~16384） |
  | `num_heads` | `INT` | 4 | 注意力头数（1~64） |
  | `num_blks` | `INT` | 2 | 堆叠编码器块数（1~50） |
  | `dropout` | `FLOAT` | 0.0 | dropout 概率（0~0.9） |
  | `use_bias` | `BOOLEAN` | False | 注意力投影是否使用偏置 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | Transformer 编码器 |

### Seq2Seq 编码器
- **类名**：`CdlSeq2SeqEncoder`
- **d2lcore 函数**：`Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)`
- **功能**：构建序列到序列学习的 RNN 编码器（嵌入 + 多层 GRU）。前向 `X` 为 token 索引 `(batch, num_steps)`，返回 `(outputs, state)`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `vocab_size` | `INT` | 32 | 源嵌入的词表大小（2~100000） |
  | `embed_size` | `INT` | 16 | 嵌入维度（1~4096） |
  | `num_hiddens` | `INT` | 16 | 每层 GRU 的隐藏单元数（1~4096） |
  | `num_layers` | `INT` | 2 | GRU 层数（1~20） |
  | `dropout` | `FLOAT` | 0.0 | GRU 层间 dropout（0~0.9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | Seq2Seq 编码器 |

### 初始化 Seq2Seq 权重
- **类名**：`CdlInitSeq2Seq`
- **d2lcore 函数**：`init_seq2seq(module)`
- **功能**：原地应用 Xavier 均匀分布权重初始化。`nn.Linear` 与 `nn.GRU` 层会被初始化，其余层保持不变。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意待初始化的 `nn.Module` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 原地初始化后的同一模型 |

---

## 7. d2l / Tensor Basic（5 个节点）

### Tensor → String
- **类名**：`CdlTensorToStr`
- **功能**：将张量格式化为人类可读的字符串。显示张量的形状、数据类型、设备信息和数值内容。超过 `max_elems` 的大张量会被截断（显示前半部分 + 后半部分）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `max_elems` | `INT` | 100 | 最大显示元素数（10~10000） |
  | `precision` | `INT` | 6 | 数值显示精度（1~16 位） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `text` | `STRING` | 格式化字符串（形状、数据类型、设备、数值） |

### String → Tensor
- **类名**：`CdlStrToTensor`
- **功能**：将字符串解析为 PyTorch 张量。支持标准的 Python 列表字面量格式，如 `"[[1,2],[3,4]]"`、`"[1,2,3,4,5]"`。自动处理空白符和尾随逗号。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `text` | `STRING` | `""` | 张量的字符串表示（多行），如 `"[[1,2],[3,4]]"` |
  | `error_strategy` | `COMBO` | `empty_tensor` | 错误处理策略：`empty_tensor`=返回空张量；`zero_tensor`=返回 `[0.]`；`raise_error`=抛出异常 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tensor` | `TENSOR` | 解析后的张量（`torch.float32`） |

### Conv2D
- **类名**：`CdlConv2d`
- **功能**：对输入张量执行二维卷积，使用指定的卷积核。封装 ``torch.nn.functional.conv2d``，支持 stride 和 padding 参数。自动将 2-D/3-D 输入扩展为 4-D ``(N, C, H, W)``，结果去掉 batch 维度返回。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `input_tensor` | `TENSOR` | — | 输入张量 |
  | `kernel` | `TENSOR` | — | 卷积核 |
  | `stride` | `INT` | 1 | 卷积步幅（1~4） |
  | `padding` | `INT` | 0 | 零填充（0~10） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 卷积结果 |

### Transpose
- **类名**：`CdlTranspose`
- **功能**：交换张量的两个维度。封装 ``torch.transpose``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `dim0` | `INT` | 0 | 第一个要交换的维度（0~5） |
  | `dim1` | `INT` | 1 | 第二个要交换的维度（0~5） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 转置后的张量 |

### Broadcast
- **类名**：`CdlBroadcast`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/Tensor Basic`）
- **替代**：核心节点 `BasicBroadcast`（`Network & Layers/Basic`），同样的 `torch.broadcast_to`，且直接操作 `TENSOR`。
- **功能**：将张量广播到目标形状。封装 ``torch.broadcast_to``。目标形状以逗号分隔字符串输入（如 ``"3,1,4"``）。解析错误时返回原张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `target_shape` | `STRING` | `""` | 目标形状，逗号分隔（如 ``"3,1,4"``） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 广播后的张量 |

### Reshape
- **类名**：`CdlReshape`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/Tensor Basic`）
- **替代**：核心节点 `BasicReshape`（`Network & Layers/Basic`），同样的 `torch.reshape`，且直接操作 `TENSOR`。
- **功能**：将张量变形为新的形状。封装 ``torch.reshape``。目标形状以逗号分隔字符串输入（如 ``"2,8"``、``"4,-1"``）。解析错误时返回原张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `target_shape` | `STRING` | `""` | 目标形状，逗号分隔（如 ``"2,8"``） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 变形后的张量 |

### Activation
- **类名**：`CdlActivation`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/Tensor Basic`）
- **替代**：14 个核心激活节点（`Network & Layers/Activation`）覆盖本节点的全部激活，并额外多出 `selu`、`mish`、`relu6`、`hardswish`、`identity` 五种，每个算子一个节点。
- **功能**：对张量逐元素应用激活函数。通过下拉菜单选择：``relu``、``sigmoid``、``tanh``、``leaky_relu``、``elu``、``gelu``、``silu``、``softmax``、``softplus``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量 |
  | `func` | `COMBO` | `relu` | 激活函数：relu / sigmoid / tanh / leaky_relu / elu / gelu / silu / softmax / softplus |
  | `dim` | `INT` | -1 | softmax 的维度（-4~4） |
  | `negative_slope` | `FLOAT` | 0.01 | leaky_relu 的负斜率（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 激活后的张量 |

### Random Tensor（随机张量）
- **类名**：`CdlRandomTensor`
- **功能**：按所选分布（`normal`、`uniform`、`randint`）生成随机张量。`seed >= 0` 固定随机种子以保证结果可复现；`-1` 表示不固定。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `shape` | `STRING` | `"4,4"` | 输出形状，逗号分隔的维度（如 `"4,4"`） |
  | `dist` | `COMBO` | `normal` | 分布：normal / uniform / randint |
  | `mean` | `FLOAT` | 0.0 | `normal` 的均值 |
  | `std` | `FLOAT` | 1.0 | `normal` 的标准差 |
  | `low` | `FLOAT` | 0.0 | `uniform` / `randint` 的下界（包含） |
  | `high` | `FLOAT` | 1.0 | `uniform` / `randint` 的上界（不包含） |
  | `seed` | `INT` | -1 | 随机种子（>= 0 固定 RNG，-1 = 随机） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tensor` | `TENSOR` | 指定形状的随机张量 |

---

## 8. d2l / TorchOps（10 个节点）

### Linear Regression
- **类名**：`CdlLinReg`
- **d2lcore 函数**：`linreg(X, w, b)`
- **功能**：线性回归前向计算：\( \hat{y} = X w + b \)
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `TENSOR` | 输入特征矩阵 |
  | `w` | `TENSOR` | 权重向量 |
  | `b` | `TENSOR` | 偏置向量/标量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `TENSOR` | 预测值 |

### Squared Loss
- **类名**：`CdlSquaredLoss`
- **d2lcore 函数**：`squared_loss(y_hat, y)`
- **功能**：计算平方损失：\( \frac{1}{2}(\hat{y} - y)^2 \)（注意：不做平均）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `TENSOR` | 预测值 |
  | `y` | `TENSOR` | 真实值（自动重塑为与 `y_hat` 相同的形状） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `loss` | `TENSOR` | 逐元素损失 |

### Masked Softmax
- **类名**：`CdlMaskedSoftmax`
- **d2lcore 函数**：`masked_softmax(X, valid_lens)`
- **功能**：在最后一个维度上执行带掩码的 softmax。使用 `valid_lens` 指定每条序列的有效长度；超出有效长度的位置在 softmax 前被设为一个极大的负值（使概率趋近于 0）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `TENSOR` | 输入张量 |
  | `valid_lens` | `TENSOR` | 有效长度张量（可选；若未提供则执行普通 softmax） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | 掩码 softmax 结果 |

### Sequence Mask
- **类名**：`CdlSequenceMask`
- **d2lcore 函数**：`sequence_mask(X, valid_len, value)`
- **功能**：将序列中超出有效长度的位置替换为指定值。常用于将填充位置置零。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `X` | `TENSOR` | — | 输入序列张量 |
  | `valid_len` | `TENSOR` | — | 每条序列的有效长度 |
  | `mask_value` | `FLOAT` | 0.0 | 掩码填充值（-1e9~1e9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `masked` | `TENSOR` | 掩码后的张量 |

### Accuracy
- **类名**：`CdlAccuracy`
- **d2lcore 函数**：`accuracy(y_hat, y)`
- **功能**：计算分类任务的正确预测数量。若预测为多类别 logits，先取 argmax，再与标签比较。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `TENSOR` | 预测值（logits 或类别索引） |
  | `y` | `TENSOR` | 真实标签 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `accuracy` | `FLOAT` | 正确预测数量（浮点） |
  | `count` | `INT` | 正确预测数量（整数） |

### Synthetic Data
- **类名**：`CdlSyntheticData`
- **d2lcore 函数**：`synthetic_data(w, b, num_examples)`
- **功能**：生成合成的线性回归数据集。随机生成权重和特征，通过 \( y = Xw + b + \text{噪声} \) 生成标签。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_features` | `INT` | 2 | 特征维度（1~1000） |
  | `num_examples` | `INT` | 100 | 样本数量（1~1000000） |
  | `noise_std` | `FLOAT` | 0.01 | 噪声标准差（0~10.0） |
  | `seed` | `INT` | 0 | 随机种子（0~99999） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `TENSOR` | 特征矩阵 `[num_examples, num_features]` |
  | `y` | `TENSOR` | 标签向量 `[num_examples, 1]` |

### Truncate/Pad
- **类名**：`CdlTruncatePad`
- **d2lcore 函数**：`truncate_pad(line, num_steps, padding_token)`
- **功能**：将序列截断或填充到固定长度。超长的序列被截断；不足的序列用 `padding_token` 填充。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_steps` | `INT` | 64 | 目标序列长度（1~10000） |
  | `padding_token` | `INT` | 0 | 填充 token 索引（0~100000） |
  | `sequence` | `TENSOR` | — | 输入索引序列（可选；缺失时返回全填充张量） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `padded` | `TENSOR` | 截断/填充后的序列（`torch.int64`） |

### BLEU Score
- **类名**：`CdlBleu`
- **d2lcore 函数**：`bleu(pred_seq, label_seq, k)`
- **功能**：计算预测序列与参考序列之间的 BLEU 分数。支持 BLEU-1 到 BLEU-4（由 `max_n` 控制）。按空白符分词后计算 n-gram 精度和简洁惩罚。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `pred_seq` | `STRING` | `"the quick brown"` | 预测序列，空白符分隔的 token（多行） |
  | `label_seq` | `STRING` | `"the quick brown fox"` | 参考序列，空白符分隔的 token（多行） |
  | `max_n` | `INT` | 4 | 最大 n-gram 阶数（1~4） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `bleu_score` | `FLOAT` | BLEU 分数（0~1） |

### Gradient Clip
- **类名**：`CdlGradClipping`
- **d2lcore 函数**：`grad_clipping(net, theta)`
- **功能**：对模型参数执行梯度裁剪。**前提条件**：必须先通过 `loss.backward()` 计算梯度。计算所有参数梯度的 L2 范数；若超过阈值 `theta`，则按比例缩放。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `theta` | `FLOAT` | 1.0 | 梯度裁剪阈值（0.1~100.0） |
  | `model` | `cdlModel` | — | 待裁剪梯度的模型（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `norm` | `FLOAT` | 裁剪前的梯度总范数；若无模型则返回 0.0 |

### SGD Step
- **类名**：`CdlSgdStep`
- **d2lcore 函数**：`sgd(params, lr, batch_size)`
- **功能**：执行一步小批量随机梯度下降：\( \theta \leftarrow \theta - \eta \cdot g / \text{batch\_size} \)。**前提条件**：必须先通过 `loss.backward()` 计算梯度。梯度在更新后被清零。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `lr` | `FLOAT` | 0.03 | 学习率（1e-8 ~ 10.0） |
  | `batch_size` | `INT` | 32 | 批量大小（1~65536） |
  | `model` | `cdlModel` | — | 待更新的模型（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 更新后的模型；若无模型则返回 None |

---

## 9. d2l / ObjectDetection（10 个节点）

### Box Corner→Center
- **类名**：`CdlBoxCornerToCenter`
- **d2lcore 函数**：`box_corner_to_center(boxes)`
- **功能**：将边界框从角点格式 `(x1, y1, x2, y2)` 转换为中心格式 `(cx, cy, w, h)`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes` | `TENSOR` | 角点格式边界框 `[N,4]`（左上角 + 右下角） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes_ccwh` | `TENSOR` | 中心格式边界框 `[N,4]`（中心 + 宽 + 高） |

### Box Center→Corner
- **类名**：`CdlBoxCenterToCorner`
- **d2lcore 函数**：`box_center_to_corner(boxes)`
- **功能**：将边界框从中心格式 `(cx, cy, w, h)` 转回角点格式 `(x1, y1, x2, y2)`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes` | `TENSOR` | 中心格式边界框 `[N,4]`（中心 + 宽 + 高） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes_xyxy` | `TENSOR` | 角点格式边界框 `[N,4]`（左上角 + 右下角） |

### Box IoU
- **类名**：`CdlBoxIou`
- **d2lcore 函数**：`box_iou(boxes1, boxes2)`
- **功能**：计算两组边界框之间的逐对 IoU（交并比）。返回矩阵 `[N1, N2]`，其中元素 `(i,j)` 为 `boxes1[i]` 与 `boxes2[j]` 的 IoU。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes1` | `TENSOR` | 第一组框 `[N1,4]`（左上角 + 右下角） |
  | `boxes2` | `TENSOR` | 第二组框 `[N2,4]`（左上角 + 右下角） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `iou` | `TENSOR` | IoU 矩阵 `[N1, N2]` |

### NMS
- **类名**：`CdlNms`
- **d2lcore 函数**：`nms(boxes, scores, iou_threshold)`
- **功能**：对边界框执行非极大值抑制（Non-Maximum Suppression）。按分数降序排列，保留与任何更高分框的 IoU 不超过阈值的框。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `boxes` | `TENSOR` | — | 边界框 `[N,4]`（左上角 + 右下角） |
  | `scores` | `TENSOR` | — | 每个框的置信度分数 |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU 阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `keep_indices` | `TENSOR` | 被保留框的索引（`torch.int64`） |

### Multibox Prior
- **类名**：`CdlMultiboxPrior`
- **d2lcore 函数**：`multibox_prior(data, sizes, ratios)`
- **功能**：在每个像素点生成不同形状的锚框。每个像素的锚框数 = `len(sizes) + len(ratios) - 1`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `sizes` | `STRING` | `"0.75,0.5,0.25"` | 锚框尺寸列表，逗号分隔 |
  | `ratios` | `STRING` | `"1,2,0.5"` | 宽高比列表，逗号分隔 |
  | `data` | `TENSOR` | — | 输入数据（可选；用于推断空间尺寸；缺失时默认为 561×728） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `TENSOR` | 锚框 `[1, H*W*bpp, 4]`，归一化坐标（左上角 + 右下角） |

### Offset Boxes
- **类名**：`CdlOffsetBoxes`
- **d2lcore 函数**：`offset_boxes(anchors, assigned_bb, eps)`
- **功能**：计算从锚框到分配的真实框的偏移量。中心坐标差缩放 10 倍；宽高比取对数后缩放 5 倍。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `anchors` | `TENSOR` | — | 锚框 `[N,4]`（左上角 + 右下角） |
  | `assigned_bb` | `TENSOR` | — | 分配的真实框 `[N,4]`（左上角 + 右下角） |
  | `eps` | `FLOAT` | 1e-6 | 防止除零的小 epsilon（1e-12~1e-3） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `offsets` | `TENSOR` | 偏移量 `[N,4]`（dx, dy, dw, dh） |

### Offset Inverse
- **类名**：`CdlOffsetInverse`
- **d2lcore 函数**：`offset_inverse(anchors, offset_preds)`
- **功能**：通过逆变换根据锚框和预测偏移量重建角点格式的边界框坐标。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `TENSOR` | 锚框 `[N,4]`（左上角 + 右下角） |
  | `offset_preds` | `TENSOR` | 预测偏移量 `[N,4]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `predicted_bbox` | `TENSOR` | 预测框 `[N,4]`（左上角 + 右下角） |

### Assign Anchor→BBox
- **类名**：`CdlAssignAnchorToBbox`
- **d2lcore 函数**：`assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)`
- **功能**：基于 IoU 将真实边界框分配给锚框。每个锚框被分配给一个 IoU ≥ 阈值的真实框，且每个真实框至少保证有一个锚框与之匹配（取 IoU 最大者）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `ground_truth` | `TENSOR` | — | 真实框 `[M,4]`（左上角 + 右下角） |
  | `anchors` | `TENSOR` | — | 锚框 `[N,4]`（左上角 + 右下角） |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU 阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors_bbox_map` | `TENSOR` | 锚框→真实框映射 `[N,]`，-1 表示无匹配（`torch.int64`） |

### Multibox Target
- **类名**：`CdlMultiboxTarget`
- **d2lcore 函数**：`multibox_target(anchors, labels)`
- **功能**：为锚框生成多框目标训练标签。对批次中每张图像，将真实框分配给锚框并计算偏移目标、掩码和类别标签。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `TENSOR` | 锚框 `[1, N, 4]`（左上角 + 右下角） |
  | `labels` | `TENSOR` | 标签 `[B, M, 5]`，格式 `[class_id, x1, y1, x2, y2]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `bbox_offset` | `TENSOR` | 边界框偏移目标 `[B, N*4]` |
  | `bbox_mask` | `TENSOR` | 边界框偏移掩码 `[B, N*4]`（匹配锚框为 1.0） |
  | `class_labels` | `TENSOR` | 锚框类别标签 `[B, N]`（背景=0，类别从 1 开始） |

### Multibox Detection
- **类名**：`CdlMultiboxDetection`
- **d2lcore 函数**：`multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)`
- **功能**：使用 NMS 从模型输出中预测边界框。结合类别预测和偏移量，通过逆偏移变换重建框，并通过 NMS 和置信度阈值过滤。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `cls_probs` | `TENSOR` | — | 类别概率 `[B, num_classes, N]` |
  | `offset_preds` | `TENSOR` | — | 偏移预测 `[B, N*4]` |
  | `anchors` | `TENSOR` | — | 锚框 `[1, N, 4]` |
  | `nms_threshold` | `FLOAT` | 0.5 | NMS IoU 阈值（0~1） |
  | `pos_threshold` | `FLOAT` | 0.01 | 正样本置信度阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `detections` | `TENSOR` | 检测结果 `[B, N, 6]`，格式 `[class_id, confidence, x1, y1, x2, y2]`（class_id=-1 表示背景） |

---

## 10. d2l / Segmentation（4 个节点）

### VOC Classes
- **类名**：`CdlVocClasses`
- **d2lcore 函数**：`VOC_CLASSES` 常量
- **功能**：获取 PASCAL VOC 21 个类别名称。单个索引查询返回该类名；索引 `-1` 返回所有 21 个类别的逗号分隔列表。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `index` | `INT` | -1 | 类别索引（-1=全部，0=背景，...，20=tv/monitor） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `class_names` | `STRING` | 类别名称（单个或逗号分隔的完整列表） |

21 个类别：`background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor`

### VOC Colormap→Label
- **类名**：`CdlVocColormap2Label`
- **d2lcore 函数**：`voc_colormap2label()`
- **功能**：构建 VOC RGB 颜色 → 类别索引查找表。输出为 \( 256^3 \) 大小的张量；通过颜色编码（R×65536 + G×256 + B）查找类别索引。
- **输入**：无
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `colormap2label` | `TENSOR` | 颜色→类别索引查找表 `[16777216]`（`torch.int64`） |

### VOC Label Indices
- **类名**：`CdlVocLabelIndices`
- **d2lcore 函数**：`voc_label_indices(colormap, colormap2label)`
- **功能**：将 VOC 标签彩色图像映射为类别索引图。将 RGB 像素编码为单一颜色值，再通过查找表转换为类别索引。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `colormap` | `IMAGE` | VOC 标签彩色图像 `[B, H, W, C]`，使用第一张图像 |
  | `colormap2label` | `TENSOR` | 颜色→标签查找表 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `label_mask` | `MASK` | 类别索引图 `[H, W]`（浮点类型） |

### VOC Random Crop
- **类名**：`CdlVocRandCrop`
- **d2lcore 函数**：`voc_rand_crop(feature, label, height, width)`
- **功能**：对特征图像和标签图像执行同步随机裁剪。使用相同的随机裁剪参数以保持特征与标签对齐。若请求尺寸超过图像尺寸则回退为中心裁剪。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `feature` | `IMAGE` | — | 特征图像 `[B, H, W, C]` |
  | `label` | `IMAGE` | — | 标签图像 `[B, H, W, C]` |
  | `height` | `INT` | 320 | 裁剪高度（1~4096，步长 32） |
  | `width` | `INT` | 480 | 裁剪宽度（1~4096，步长 32） |
  | `seed` | `INT` | 0 | 随机种子（可选，0~999999） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `cropped_feature` | `IMAGE` | 裁剪后的特征图像 |
  | `cropped_label` | `IMAGE` | 裁剪后的标签图像 |

---

## 11. d2l / Visualization（13 个节点）

可视化节点采用"双变体"设计模式：带 `(Output)` 后缀的版本是 ComfyUI 输出节点（直接在界面中显示交互式图表），不带后缀的版本将图表渲染为 `IMAGE` 张量，供下游节点使用。

### Show Images
- **类名**：`CdlShowImages`
- **d2lcore 函数**：同上
- **功能**：与 `Show Images (Output)` 相同，但将图表渲染为 IMAGE 张量，可传递给下游节点。
- **输入**：同上
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的网格图像 `[1, H, W, C]` |

### Show Heatmaps (Output)
- **类名**：`CdlShowHeatmapsOutput`
- **d2lcore 函数**：`show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)`
- **功能**：显示单张热力图并附颜色条。先把张量抽样到元素预算以内，再保留 `dims` 指定的轴（或由 `auto_select` 自动挑选）、其余轴按 `axis_reduce` 归约；最终保留的 1/2/3 个轴决定画什么：条码状色带（1D）、常规平面（2D）或半透明立方体（3D）。输出节点变体。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `matrices` | `TENSOR` | — | 任意秩张量；先抽样到 `max_samples` 个元素，再降到 1~3 个轴后渲染 |
  | `xlabel` | `STRING` | `""` | X 轴标签 |
  | `ylabel` | `STRING` | `""` | Y 轴标签 |
  | `figsize_w` | `FLOAT` | 2.5 | 图宽（0.5~20.0） |
  | `figsize_h` | `FLOAT` | 2.5 | 图高（0.5~20.0） |
  | `cmap` | `STRING` | `"Reds"` | matplotlib 颜色映射名称（未知名称回退为 `Reds`） |
  | `titles` | `STRING` | `""` | 图标题（可选） |
  | `max_samples` | `INT` | 262144 | 元素预算；`0` 表示不抽样 |
  | `dims` | `STRING` | `"auto"` | `auto`，或 1~3 个轴索引如 `0,1` / `-2,-1` |
  | `axis_reduce` | `COMBO` | `mean` | 被丢弃轴的归约方式：`mean` / `max` / `first` / `mid` |
  | `auto_select` | `COMBO` | `last_n` | `auto` 保留哪些轴：`first_n` / `last_n` / `most_informative_n` / `least_informative_n`（信息量 ≈ 轴长度） |
  | `auto_n` | `INT` | 0 | `auto` 保留几个轴（0 = 跟随输入秩，上限 3） |
  | `on_error` | `COMBO` | `fallback_first_n` | 高级。`error` 直接抛 `HeatmapSpecError`，`fallback_first_n` 退回取前几个轴继续渲染 |
- **输出**：无（输出节点）
- **行为变更**：旧的"多子图阵列"布局已取消 —— 一张图只画一张热力图。2D 输入在 `max_samples=0` + `dims=auto` 下仍与旧版逐像素一致。

### Show Heatmaps
- **类名**：`CdlShowHeatmaps`
- **d2lcore 函数**：同上
- **功能**：与 `Show Heatmaps (Output)` 相同，但将图表渲染为 IMAGE 张量。
- **输入**：同上
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的热力图 `[1, H, W, C]` |

### Heatmaps to 3D
- **类名**：`CdlHeatmapsTo3D`
- **d2lcore 函数**：`show_heatmaps(matrices, ...)`（复用同一前端），但把结果画成几何体而非像素
- **功能**：把抽样/降维后的热力数据导出为真正的 3D 模型。1D → 有厚度的平放条码色带；2D → 同样厚度的彩色平板；3D → 半透明立方体（六个外表面 + 三个正交中截面，内部仍然可读）。颜色量化为 64 个 OBJ 材质，透明度写在 MTL 里，因此立方体保持半透明。把 `model_3d` 连到内置 **Preview3D** 节点即可查看。
- **说明**：在 `max_samples` 预算之外，网格还会再做一次步进以控制面数（1D：256 格，2D：每轴 64，3D：每轴 32）。维度标注（"1D"/"2D"/"3D"）以几何体绘制，因为 3D 文件无法携带文字。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `matrices` | `TENSOR` | — | 与 Show Heatmaps 相同的前端：抽样、选轴、降维 |
  | `cmap` | `STRING` | `"Reds"` | matplotlib 颜色映射名称 |
  | `opacity` | `FLOAT` | 0.6 | 材质透明度（MTL 中的 `d`）；小于 1.0 时立方体内部可见 |
  | `thickness` | `FLOAT` | 0.15 | 1D / 2D 色带的厚度，单位为格 |
  | `max_samples` | `INT` | 262144 | 元素预算；`0` 表示不抽样 |
  | `dims` | `STRING` | `"auto"` | `auto`，或 1~3 个轴索引 |
  | `axis_reduce` | `COMBO` | `mean` | 被丢弃轴的归约方式 |
  | `auto_select` | `COMBO` | `last_n` | `auto` 保留哪些轴 |
  | `auto_n` | `INT` | 0 | `auto` 保留几个轴 |
  | `on_error` | `COMBO` | `fallback_first_n` | 高级。`error` 抛错，`fallback_first_n` 回退 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model_3d` | `FILE_3D_OBJ` | OBJ 网格 + 同名 MTL；连到 `Preview3D` 查看 |

### Plot
- **类名**：`CdlPlot`
- **d2lcore 函数**：`plot(X, Y, xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, figsize, axes)`
- **功能**：通用 MATLAB 风格折线图。支持多条曲线、自定义轴标签、对数/线性刻度、图例和轴范围。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `xlabel` | `STRING` | `"x"` | X 轴标签 |
  | `ylabel` | `STRING` | `"y"` | Y 轴标签 |
  | `xscale` | `COMBO` | `linear` | X 轴刻度：`linear` / `log` |
  | `yscale` | `COMBO` | `linear` | Y 轴刻度：`linear` / `log` |
  | `figsize_w` | `FLOAT` | 6.0 | 图宽度（1.0~30.0） |
  | `figsize_h` | `FLOAT` | 4.0 | 图高度（1.0~30.0） |
  | `X` | `TENSOR` | — | X 轴数据（可选，一维或二维） |
  | `Y` | `TENSOR` | — | Y 轴数据（可选） |
  | `legend` | `STRING` | `""` | 图例标签，逗号分隔（可选） |
  | `xlim_min` | `FLOAT` | -1.0 | X 轴下界（仅当 xlim_min < xlim_max 时生效） |
  | `xlim_max` | `FLOAT` | -1.0 | X 轴上界 |
  | `ylim_min` | `FLOAT` | -1.0 | Y 轴下界（仅当 ylim_min < ylim_max 时生效） |
  | `ylim_max` | `FLOAT` | -1.0 | Y 轴上界 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的图表 `[1, H, W, C]` |

### Show Trace 2D
- **类名**：`CdlShowTrace2D`
- **d2lcore 函数**：`show_trace_2d(f, results)`
- **功能**：可视化二维优化轨迹。绘制一系列点，展示参数 (x1, x2) 在优化过程中的变化。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `results` | `TENSOR` | 优化轨迹点 `[N, 2]`，每行是一个 (x1, x2) 坐标 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的轨迹图 `[1, H, W, C]` |

### Show BBoxes
- **类名**：`CdlShowBboxes`
- **d2lcore 函数**：`show_bboxes(axes, bboxes, labels, colors)`
- **功能**：在图像上绘制边界框。支持自定义标签和颜色；最多渲染 200 个框。坐标必须归一化到 [0,1] 范围。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `image` | `IMAGE` | — | 背景图像 `[B, H, W, C]`（使用第一张图像） |
  | `bboxes` | `TENSOR` | — | 边界框 `[N, 4]`，归一化坐标（左上角 + 右下角） |
  | `labels` | `STRING` | `""` | 框标签，逗号分隔（可选） |
  | `colors` | `STRING` | `"b,g,r,m,c"` | matplotlib 颜色，逗号分隔（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 绘制了边界框的图像 `[1, H, W, C]` |

### Histogram
- **类名**：`CdlHistogram`
- **功能**：绘制张量值分布的直方图，支持可调的 bins 数量、密度归一化和颜色。封装 ``matplotlib.pyplot.hist``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 输入张量（内部展平） |
  | `bins` | `INT` | 30 | 直方图分桶数 |
  | `density` | `BOOLEAN` | False | 为 True 时显示密度而非计数 |
  | `color` | `STRING` | `"#4673a6"` | 条形颜色 |
  | `alpha` | `FLOAT` | 0.7 | 条形透明度 |
  | `title` | `STRING` | `""` | 图表标题 |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 直方图 `[1, H, W, C]` |

### Bar Chart
- **类名**：`CdlBarChart`
- **功能**：绘制垂直或水平柱状图，可选数值标注。封装 ``matplotlib.pyplot.bar`` / ``barh``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `values` | `TENSOR` | — | 柱状高度（1-D 张量） |
  | `labels` | `STRING` | `""` | 分类标签，逗号分隔 |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
  | `horizontal` | `BOOLEAN` | False | 使用 ``barh`` 代替 ``bar`` |
  | `color` | `STRING` | `"#4673a6"` | 条形颜色 |
  | `annotate` | `BOOLEAN` | True | 在柱状条上显示数值 |
  | `figsize_w` | `FLOAT` | 7.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 4.0 | 图高 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 柱状图 `[1, H, W, C]` |

### Scatter
- **类名**：`CdlScatter`
- **功能**：绘制二维散点图，可选点颜色和尺寸编码第三维。封装 ``matplotlib.pyplot.scatter``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `X` | `TENSOR` | — | X 坐标（展平） |
  | `Y` | `TENSOR` | — | Y 坐标（展平） |
  | `alpha` | `FLOAT` | 0.6 | 点透明度 |
  | `cmap` | `STRING` | `"viridis"` | ``color_map`` 的 colormap |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
  | `figsize_w` | `FLOAT` | 6.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 5.0 | 图高 |
  | `color_map` | `TENSOR` | — | 逐点颜色值（可选） |
  | `size_map` | `TENSOR` | — | 逐点大小值（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 散点图 `[1, H, W, C]` |

### Confusion Matrix
- **类名**：`CdlConfusionMatrix`
- **功能**：将混淆矩阵渲染为热力图，并在每个格子中标注数值。支持行归一化和自定义数字格式。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `matrix` | `TENSOR` | — | 混淆矩阵（N×N 或展平） |
  | `class_labels` | `STRING` | `""` | 类别名称，逗号分隔 |
  | `cmap` | `STRING` | `"Blues"` | colormap 名称 |
  | `normalize` | `BOOLEAN` | False | 将行归一化到 [0,1] |
  | `fmt` | `COMBO` | `.1f` | 数字格式（`.0f` / `.1f` / `.2f` / `.3f`） |
  | `figsize_w` | `FLOAT` | 6.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 5.0 | 图高 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 混淆矩阵热力图 `[1, H, W, C]` |

### Pie Chart
- **类名**：`CdlPieChart`
- **功能**：绘制饼图（普通或甜甜圈样式）并标注百分比。封装 ``matplotlib.pyplot.pie``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `values` | `TENSOR` | — | 扇区值（1-D 张量） |
  | `labels` | `STRING` | `""` | 扇区标签，逗号分隔 |
  | `donut` | `BOOLEAN` | False | 空心中心（甜甜圈图） |
  | `explode` | `STRING` | `""` | 每扇区分裂 0/1，逗号分隔 |
  | `pctdistance` | `FLOAT` | 0.6 | 百分比标签距中心的距离 |
  | `shadow` | `BOOLEAN` | False | 饼底阴影 |
  | `figsize_w` | `FLOAT` | 6.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 6.0 | 图高 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 饼图（或甜甜圈图）`[1, H, W, C]` |

### Area Chart
- **类名**：`CdlAreaChart`
- **功能**：绘制填充面积图 —— 单序列使用 ``fill_between``，多序列堆叠使用 ``stackplot``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `Y` | `TENSOR` | — | 序列数据，`[T]` 或 `[N, T]` |
  | `stacked` | `BOOLEAN` | False | 堆叠而非叠加 |
  | `alpha` | `FLOAT` | 0.5 | 填充透明度 |
  | `color_palette` | `STRING` | `"tab10"` | matplotlib 调色盘名称 |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
  | `figsize_w` | `FLOAT` | 7.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 4.0 | 图高 |
  | `X_vals` | `TENSOR` | — | 自定义 x 轴值（可选） |
  | `labels` | `STRING` | `""` | 序列图例标签，逗号分隔（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 面积图 `[1, H, W, C]` |

---

## 12. d2l / Datasets（10 个节点）

数据集节点提供端到端的数据集管理能力：下载、加载、查看、预览和统计。

### Load Array → DataLoader
- **类名**：`CdlLoadArray`
- **d2lcore 函数**：`load_array(data_arrays, batch_size, is_train)`
- **功能**：将一个或多个张量封装为 PyTorch DataLoader。将 `TENSOR` 特征和/或标签连接到可选输入槽；节点输出 `cdlDataloader`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `batch_size` | `INT` | 32 | 批大小（1~4096） |
  | `shuffle` | `BOOLEAN` | True | 每个 epoch 是否打乱数据 |
  | `features` | `TENSOR` | — | 特征张量 X（可选） |
  | `labels` | `TENSOR` | — | 标签张量 y（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `dataloader` | `cdlDataloader` | 包装了张量的 PyTorch DataLoader |

### DataLoader Info
- **类名**：`CdlDataLoaderInfo`
- **功能**：查看 `cdlDataloader` 的属性：批次数、批大小和数据集总大小。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `dataloader` | `cdlDataloader` | 待查看的 DataLoader |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `num_batches` | `INT` | 批次总数 |
  | `batch_size` | `INT` | 每批样本数 |
  | `dataset_size` | `INT` | 样本总数 |

### Download
- **类名**：`CdlDownload`
- **d2lcore 函数**：`download(url, folder, sha1_hash)`
- **功能**：从 URL 下载文件，支持基于 SHA1 的缓存检查。若本地文件存在且 SHA1 匹配，则跳过下载。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `url` | `STRING` | `""` | 下载 URL |
  | `save_dir` | `STRING` | `"../data"` | 保存目录（可选） |
  | `sha1_hash` | `STRING` | `""` | 用于缓存的 SHA1 哈希值（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `file_path` | `STRING` | 下载/缓存文件的本地路径 |

### Download + Extract
- **类名**：`CdlDownloadExtract`
- **d2lcore 函数**：`download_extract(name, folder)`
- **功能**：下载并解压 d2l DATA_HUB 中注册的数据集。从下拉菜单中选择预注册的数据集（banana-detection、voc2012、cifar10_tiny 等）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `name` | `COMBO` | 第一个键 | DATA_HUB 中的数据集名称 |
  | `subfolder` | `STRING` | `""` | 归档文件内的子文件夹（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `extract_dir` | `STRING` | 解压后的数据集目录路径 |

### Fashion-MNIST
- **类名**：`CdlFashionMNIST`
- **d2lcore 函数**：`load_data_fashion_mnist(batch_size, resize)`
- **功能**：加载 Fashion-MNIST 图像分类数据集（训练集 6 万张 / 测试集 1 万张，10 个类别）。首次使用时自动下载（约 30 MB）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `batch_size` | `INT` | 64 | 每批样本数（1~2048） |
  | `resize` | `INT` | 28 | 缩放尺寸（0=不缩放，1~512） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `train_loader` | `cdlDataloader` | 训练 DataLoader（60,000 张图像） |
  | `test_loader` | `cdlDataloader` | 测试 DataLoader（10,000 张图像） |
  | `class_names` | `STRING` | 按行分隔的类别名称 |

10 个类别：t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot

### Bananas Detection
- **类名**：`CdlBananasDetection`
- **d2lcore 函数**：`load_data_bananas(batch_size)`
- **功能**：加载用于目标检测的香蕉检测数据集。包含带边界框标注的香蕉图像。首次使用时自动下载。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `batch_size` | `INT` | 32 | 每批样本数（1~256） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `train_loader` | `cdlDataloader` | 训练 DataLoader |
  | `val_loader` | `cdlDataloader` | 验证 DataLoader |

### VOC Segmentation
- **类名**：`CdlVOCSegmentation`
- **d2lcore 函数**：`load_data_voc(batch_size, crop_size)`
- **功能**：加载 PASCAL VOC2012 语义分割数据集（21 个类别）。首次使用时自动下载和解压（约 2 GB）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `batch_size` | `INT` | 32 | 每批样本数（1~128） |
  | `crop_height` | `INT` | 320 | 随机裁剪高度（64~1024） |
  | `crop_width` | `INT` | 480 | 随机裁剪宽度（64~2048） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `train_loader` | `cdlDataloader` | 训练 DataLoader |
  | `test_loader` | `cdlDataloader` | 测试 DataLoader |

### DataLoader Preview
- **类名**：`CdlDataLoaderPreview`
- **功能**：从 `cdlDataloader` 中取一批样本，渲染为图像网格（IMAGE 输出）。自动适配不同数据格式：图像分类显示带标签的图像；目标检测显示带边界框的图像。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `dataloader` | `cdlDataloader` | — | 待采样的 DataLoader |
  | `num_rows` | `INT` | 2 | 网格行数（1~16） |
  | `num_cols` | `INT` | 4 | 网格列数（1~16） |
  | `max_samples` | `INT` | 32 | 最大显示图像数（1~256，可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的网格图像 `[1, H, W, C]` |

### DataLoader Preview (Output)
- **类名**：`CdlDataLoaderPreviewOutput`
- **功能**：与 `CdlDataLoaderPreview` 相同，但注册为 OUTPUT_NODE，渲染后的图像网格直接在 UI 中显示。
- **输入**：与 `CdlDataLoaderPreview` 相同
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的网格图像 `[1, H, W, C]` |

### Dataset Stats
- **类名**：`CdlDataLoaderStats`
- **功能**：遍历 `cdlDataloader`，统计标签分布。自动识别标签类型：整数且落在 `[0, num_classes)` 内的离散类别标签渲染为分类柱状图；连续数值标签（如回归目标，含 `(n,1)` 二维标签）自动按 `num_classes` 个区间分桶为直方图。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `dataloader` | `cdlDataloader` | — | 待分析的 DataLoader |
  | `num_classes` | `INT` | 10 | 预期类别数/分桶数（1~1000） |
  | `class_names` | `STRING` | `""` | 逗号分隔的类别名称（可选；仅分类模式生效） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `stats_text` | `STRING` | 格式化统计文本摘要（分类模式为各类别计数，直方图模式为各区间范围与计数） |
  | `stats_image` | `IMAGE` | 标签分布图（分类柱状图或数值直方图）`[1, H, W, C]` |

---

## 13. d2l / Model Utils（7 个节点）

自主开发的模型实用工具节点（非 d2l 内容）。用于在工作流中直接检查、切换、运行、克隆与持久化 PyTorch 模型。所有节点操作 `cdlModel` 类型（任意 `nn.Module` 实例）。

### Model Info（模型信息）
- **类名**：`CdlModelInfo`
- **功能**：检查模型并报告 (1) 人类可读的摘要字符串、(2) 总参数量、(3) 可训练参数量。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意 `nn.Module` 实例 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `summary` | `STRING` | 模型类型、子模块名、模块数量与参数量 |
  | `total_params` | `INT` | 总参数量 |
  | `trainable_params` | `INT` | `requires_grad=True` 的参数量 |

### Model Mode（模型模式）
- **类名**：`CdlModelMode`（⚠️ **已弃用**，软归档至 `d2l/_Legacy/Model Utils`）
- **替代**：纯 `TENSOR` 图请改用核心节点 `Training Mode`（`Network & Layers/Training`）。注意两者**载荷不同、不可互换**：本节点切换的是整个 `cdlModel`（`model.train()` / `model.eval()`）并把模块继续传递，`Training Mode` 发布的则是驱动 `TENSOR` 级节点的 STRING。本节点功能未变，`cdlModel` 流水线仍可继续使用。
- **功能**：通过 `model.train()` / `model.eval()` 在训练与评估模式间切换，返回同一实例，使下游节点感知新模式。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `model` | `cdlModel` | — | 任意 `nn.Module` 实例 |
  | `mode` | `COMBO` | `eval` | `train`（训练模式）/ `eval`（推理模式） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 已应用模式的同一实例 |

### Model Forward（模型前向）
- **类名**：`CdlModelForward`
- **功能**：在 `torch.no_grad()` 下对输入张量执行模型前向。若输入与模型设备不同则自动迁移；前向前置为 eval 模式。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意 `nn.Module` 实例 |
  | `tensor` | `TENSOR` | 模型期望形状的输入张量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `TENSOR` | `model(tensor)`——形状取决于模型 |

### Model Layers（模型层结构）
- **类名**：`CdlModelLayers`
- **功能**：遍历 `model.named_modules()` 并以缩进树形式输出每个模块（名称 + 类名），便于检查架构。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意 `nn.Module` 实例 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `layers_str` | `STRING` | 每行一个模块，按嵌套深度缩进 |

### Model Params（模型参数）
- **类名**：`CdlModelParams`
- **功能**：遍历 `model.named_parameters()` 并输出每个参数的名称、形状与 `requires_grad`，附总数。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意 `nn.Module` 实例 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `params_str` | `STRING` | 每行一个参数 + 总数 |

### Model Clone（模型克隆）
- **类名**：`CdlModelClone`
- **功能**：返回 `copy.deepcopy(model)`——架构与权重相同但参数完全独立的实例。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 任意 `nn.Module` 实例 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `clone` | `cdlModel` | 输入模型的深拷贝 |

### Model Save（模型保存）
- **类名**：`CdlModelSave`
- **功能**：将 `torch.save(model.state_dict(), path)` 写入磁盘。仅保存权重（state_dict），重新加载需要结构匹配的模型。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `model` | `cdlModel` | — | 任意 `nn.Module` 实例 |
  | `path` | `STRING` | `"model.pt"` | 目标文件路径，如 `"C:/models/my_model.pt"` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `message` | `STRING` | 包含保存路径的确认文本 |

### Model Load（模型加载）
- **类名**：`CdlModelLoad`
- **功能**：用 `torch.load` 读取 `.pt` state_dict 并通过 `model.load_state_dict()` 应用到输入模型。模型架构必须与保存的 state_dict 匹配。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `model` | `cdlModel` | — | 将接收权重的模型实例 |
  | `path` | `STRING` | `"model.pt"` | 已保存 state_dict 文件的路径 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `model` | `cdlModel` | 已加载权重的输入模型 |

---

## 14. ComfyUI / image/color（3 个节点）

自主开发的通用 CV 图像节点（非 d2l 内容），并入 ComfyUI 核心分类 `image/color`。所有节点消费并产出 ComfyUI 原生 `IMAGE` 格式——float32 `[B, H, W, C]`，值域 `[0, 1]`——用 `torch` + `torchvision.transforms.functional` 实现。例外：Image Normalize（图像归一化）故意不将输出裁剪到 `[0, 1]`（z-score 值域）。

> 几何类节点已由 ComfyUI 核心等价节点取代并删除：`Image Resize` → `ImageScale` / `ResizeImageMaskNode`，`Image Flip` → `ImageFlip`，`Image Blur` → `ImageBlur`，`Image Crop` → `ImageCrop` / `ImageCropV2`。`Image Rotate` 因核心 `ImageRotate` 仅支持 90 度步进而保留，移入 `image/transform`。

### Image Normalize（图像归一化）
- **类名**：`CdlImageNormalize`
- **功能**：`denorm` 为 False 时执行 `(x - mean) / std`，为 True 时执行逆变换 `x * std + mean`。`mean`/`std` 为逗号分隔字符串；单个值会广播到所有通道（如 `"0.5"` 或 `"0.5,0.5,0.5"`）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `image` | `IMAGE` | — | 输入图像 `[B, H, W, C]` |
  | `mean` | `STRING` | `"0.5,0.5,0.5"` | 逗号分隔的逐通道均值 |
  | `std` | `STRING` | `"0.5,0.5,0.5"` | 逗号分隔的逐通道标准差 |
  | `denorm` | `BOOLEAN` | False | True = 反归一化，False = 归一化 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 处理后的图像 `[B, H, W, C]`（值域取决于操作） |

### Image Grayscale（图像灰度化）
- **类名**：`CdlImageGrayscale`
- **功能**：以 `num_output_channels=3` 转为灰度，保持 `[B, H, W, C]`（C=3）布局，各通道携带相同的亮度值。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 输入图像 `[B, H, W, C]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 3 通道灰度图像 `[B, H, W, C]` |

### Image Adjust（图像调整）
- **类名**：`CdlImageAdjust`
- **功能**：用给定因子应用 torchvision 亮度、对比度与饱和度调整（1.0 = 不变，>1 增强，<1 减弱，0 = 无）。等于 1.0 的因子会跳过以加速。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `image` | `IMAGE` | — | 输入图像 `[B, H, W, C]` |
  | `brightness` | `FLOAT` | 1.0 | 亮度因子（0~2） |
  | `contrast` | `FLOAT` | 1.0 | 对比度因子（0~2） |
  | `saturation` | `FLOAT` | 1.0 | 饱和度因子（0~2） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 调整后的图像 `[B, H, W, C]` |

## 15. ComfyUI / image/transform（1 个节点）

保留为 ComfyUI 核心 `image/transform` 节点：核心 `ImageRotate` 仅支持 90 度步进，本节点支持任意角度并可放大画布。

### Image Rotate（图像旋转）
- **类名**：`CdlImageRotate`
- **功能**：以双线性插值和零填充边界将每张图像旋转 `angle` 度（逆时针）。`expand` 为 True 时画布会放大以容纳旋转内容；否则输出保持输入尺寸。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `image` | `IMAGE` | — | 输入图像 `[B, H, W, C]` |
  | `angle` | `FLOAT` | 90.0 | 旋转角度（度，-360~360） |
  | `expand` | `BOOLEAN` | False | True = 放大画布以容纳旋转内容 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 旋转后的图像 `[B, H, W, C]` |

---

## 16. ComfyUI / image（1 个节点）

并入 ComfyUI 核心分类 `image`（与核心 `GetImageSize` 同级）。

### Image Stats（图像统计）
- **类名**：`CdlImageStats`
- **功能**：聚合批次内所有图像，按通道报告 mean/std/min/max 以及批次布局。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 输入图像 `[B, H, W, C]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `stats` | `STRING` | 每通道一行 + 批次摘要行 |

---

## 17. ComfyUI / Network & Layers（62 个节点）

由宿主运行时提供的核心神经网络节点（不属于 ComfyDL 子模块），分布在 `comfy_extras` 的九个模块
`nodes_activation.py`、`nodes_layers.py`、`nodes_attention.py`、`nodes_normalization.py`、
`nodes_pooling.py`、`nodes_convolution.py`、`nodes_training.py`、`nodes_nlp.py` 与 `nodes_lm.py`
中，在节点库中构成 **Comfy节点 → Network & Layers** 分支，下分 `Activation`、
`Basic`、`Attention`、`Normalization`、`Regularization`、`Training`、`Pooling`、`Convolution` 与
`Text` 九组。它们全部通过共享的 `TENSOR` 插槽类型交换数据、保持输入的
dtype/device 不变，并且都是无状态的：`weight`、`bias` 等可学习参数以张量形式从输入插槽传入，
节点内部不做初始化，因此每个节点都是纯函数，可直接与上述 ComfyDL 张量节点互连。`Training`
组另外引入 `PARAMS` 与 `OPTIMIZER` 两个一等图数据类型，让同一套无状态约定也能表达**可训练**
参数与优化循环；`Training` 与 `Text` 两组合起来又为语言模型流水线新增 `VOCAB`、`MODELSPEC` 与
`NNMODEL` 三个数据类型。

### 17.1 Activation（14 个节点）

核心激活函数节点（`comfy_extras/nodes_activation.py`）。每个节点恰好 1 个名为 `tensor` 的
`TENSOR` 输入与 1 个名为 `output` 的 `TENSOR` 输出，且无可学习参数。

| 节点 | 类名 | 额外控件 | 作用 |
|------|-------|--------------|---------|
| Sigmoid | `ActivationSigmoid` | — | `1 / (1 + exp(-x))`；输出范围 (0, 1) |
| Tanh | `ActivationTanh` | — | `tanh(x)`；输出范围 (-1, 1) |
| ReLU | `ActivationReLU` | — | `max(0, x)` |
| Leaky ReLU | `ActivationLeakyReLU` | `negative_slope` FLOAT 0.01 (0~1) | 负半轴保留小斜率的 ReLU |
| ELU | `ActivationELU` | `alpha` FLOAT 1.0 (0~100) | `x > 0` 时为 `x`，否则 `alpha * (exp(x) - 1)` |
| SELU | `ActivationSELU` | — | 自归一化 ELU（标准 scale/alpha 常数） |
| GELU | `ActivationGELU` | `approximate` COMBO none/tanh | 高斯误差线性单元 |
| SiLU | `ActivationSiLU` | — | `x * sigmoid(x)`（swish） |
| Mish | `ActivationMish` | — | `x * tanh(softplus(x))` |
| Softplus | `ActivationSoftplus` | — | `log(1 + exp(x))`（beta=1, threshold=20） |
| ReLU6 | `ActivationReLU6` | — | `min(max(0, x), 6)` |
| Hard Swish | `ActivationHardSwish` | — | `x * relu6(x + 3) / 6` |
| Identity | `ActivationIdentity` | — | 原样透传输入张量（零拷贝） |
| Softmax | `ActivationSoftmax` | `dim` INT -1 (-4~4) | 沿 `dim` 归一化（越界时按张量秩钳制） |

### 17.2 Basic（8 个节点）

核心基础层 / 张量运算节点（`comfy_extras/nodes_layers.py`）。每个节点接收 1~2 个 `TENSOR` 输入，
返回恰好 1 个名为 `output` 的 `TENSOR` 输出。`weight` / `bias` 都是普通张量输入，不存在隐藏的
参数状态。控件默认值开箱可用；`Reshape` / `Broadcast` 在 `target_shape` 无法解析时回退为原样返回
输入张量，因此不会中断工作流。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Linear | `BasicLinear` | `tensor`、`weight`（`[out, in]`）、`bias`（可选） | — | 仿射变换 `x @ weight.T + bias`（`F.linear`） |
| Embedding | `BasicEmbedding` | `tensor`（整数索引）、`weight`（`[num, dim]`） | — | 按索引查表（`F.embedding`） |
| Flatten | `BasicFlatten` | `tensor` | `start_dim` INT 1 (0~4)、`end_dim` INT -1 (-4~4) | 展平连续的维度区间（`torch.flatten`） |
| Reshape | `BasicReshape` | `tensor` | `target_shape` STRING `"1,-1"` | 重塑为目标形状（`torch.reshape`） |
| Broadcast | `BasicBroadcast` | `tensor` | `target_shape` STRING `"2,3"` | 广播为目标形状（`torch.broadcast_to`） |
| Concat | `BasicConcat` | `a`、`b` | `dim` INT -1 (-4~4) | 沿 `dim` 拼接两个张量（`torch.cat`） |
| Add | `BasicAdd` | `a`、`b` | — | 逐元素相加（带广播，`a + b`） |
| Multiply | `BasicMultiply` | `a`、`b` | — | 逐元素相乘（带广播，`a * b`） |

> `Linear` 已覆盖 `Dense` 层（同为仿射变换），`Add` 也已覆盖残差 / 跳连（残差即逐元素广播相加），
> 因此不再单独提供 `Dense`、`Residual`、`Skip Connection` 节点。

### 17.3 Normalization（7 个节点）

核心归一化节点（`comfy_extras/nodes_normalization.py`）。每个节点都以名为 `output` 的
`TENSOR` 作为第 1 个输出，保持输入的 dtype/device，且不保存任何状态。`BatchNorm` 与
`InstanceNorm` 还会在其后追加 `mean` / `var` 两个 `(C,)` 输出——本次归一化**实际使用**的
逐通道统计量，可直接接入 `Training Run Stats`（见 17.4）；其余节点恰好 1 个输出。
`weight` / `bias`（γ / β）都是普通张量
输入，从插槽接入的 `running_mean` / `running_var` 也绝不会被就地改写（同一张量可能被图中其它节点
共享）。`BatchNorm` 与 `InstanceNorm` 都是**按秩自适应**的——一个节点即覆盖 1d/2d/3d 三种形态，
因为“沿哪些维度统计”完全由张量形状决定——它们自身不带训练/推理开关，而是跟随 `mode` 插槽。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| BatchNorm | `NormalizationBatchNorm` | `tensor`、`weight`（可选）、`bias`（可选）、`running_mean` / `running_var`（可选）、`mode`（可选 STRING 插槽） | `eps` FLOAT 1e-5 (0~1e-2) | 对 `(N, C, ...)` 的第 1 维做 `F.batch_norm`；秩 2/3/4/5 分别等价于 BatchNorm1d/1d/2d/3d。另输出本次使用的逐通道 `mean` / `var`（`(C,)`）：`train` 下为本批统计量，`eval` 下为所接入的 running 统计量 |
| InstanceNorm | `NormalizationInstanceNorm` | `tensor`、`weight`、`bias`、`running_mean` / `running_var`、`mode`（除 `tensor` 外均可选） | `eps` FLOAT 1e-5 (0~1e-2) | `F.instance_norm`，按样本且按通道统计；需要秩 ≥ 3。另输出 `mean` / `var`，把逐样本统计量归约为逐通道口径（组内方差均值 + 组间均值方差），因此与 BatchNorm 的输出含义一致、可存进同一组插槽 |
| LayerNorm | `NormalizationLayerNorm` | `tensor`、`weight`（可选）、`bias`（可选） | `normalized_shape` STRING `"last"`、`eps` FLOAT 1e-5 | 对尾部维度做 `F.layer_norm`（`"8,16"` 表示最后两维） |
| GroupNorm | `NormalizationGroupNorm` | `tensor`、`weight`（可选）、`bias`（可选） | `num_groups` INT 1 (1~64)、`eps` FLOAT 1e-5 | `F.group_norm`；`num_groups=1` 即在全部通道上归一化 |
| RMSNorm | `NormalizationRMSNorm` | `tensor`、`weight`（可选） | `normalized_shape` STRING `"last"`、`eps` FLOAT 1e-6 | `F.rms_norm`；LLaMA 风格（不减均值、无偏置） |
| WeightNorm | `NormalizationWeightNorm` | `weight`、`g`（可选） | `dim` INT 0 (-8~7)、`eps` FLOAT 1e-12 | 权重重参数化 `g * v / ‖v‖₂`，范数沿 `dim` 求取 |
| SpectralNorm | `NormalizationSpectralNorm` | `weight`、`u` / `v`（可选） | `n_power_iterations` INT 10 (0~20)、`dim` INT 0、`eps` FLOAT 1e-12 | 用确定性幂迭代估计最大奇异值并据此除权重；同时输出 `sigma` |

> 训练/推理开关是一条显式连线：`Training Mode`（17.4）把 `train` / `eval` 以 STRING 形式发布，
> 接入 `BatchNorm` / `InstanceNorm` 的 `mode` 插槽。这里用连线不只是图个方便，而是正确性所需：
> 节点的缓存签名包含它自己的输入**以及所有祖先节点的输入**，所以切换下拉框会让全部消费者失效重算；
> 而“后台读取整个 prompt”的隐藏握手机制既不在签名里（算签名时甚至读不到 prompt），会让下游一直
> 复用旧输出。`LayerNorm`、`GroupNorm`、`RMSNorm`、`WeightNorm`、`SpectralNorm` 刻意不带 `mode`
> 插槽：它们的数学在训练与推理下完全一致。`eval` 模式若没有可用统计量，则回退使用本次调用的统计量；
> 控件文本无法解析时（过期的 `normalized_shape`、不能整除的 `num_groups`、重复的维度等）会回退到
> 既定默认值并打印提示，因此一个控件取值永远不会弄坏工作流。

### 17.4 Training（17 个节点）

`Network & Layers/Training` 分类下有四族节点。第一族是两个小巧的“状态”节点
（`comfy_extras/nodes_normalization.py`），把训练/推理决策与可持久化的运行统计量送入归一化节点。
第二族是训练闭环（`comfy_extras/nodes_training.py`），新增 `PARAMS` 与 `OPTIMIZER` 两个图数据
类型，并提供一个在**节点内部**完成真实优化循环的训练节点。第三、四族（`comfy_extras/nodes_lm.py`，
reform step 8）是语言模型流水线——在 `MODELSPEC` 槽上以 spec 链声明 Transformer 的结构，由 Build 节点
物化为真正的 `nn.Module` 走 `NNMODEL` 槽，再加上 Train / Forward / Generate 三件套。

**训练/推理状态（2 个节点）**

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Training Mode | `TrainingMode` | — | `mode` COMBO train/eval（默认 `train`） | 把 `train` / `eval` 以 STRING 发布给 `BatchNorm` / `InstanceNorm` 的 `mode` 插槽 |
| Training Run Stats | `TrainingRunStats` | `mean` / `var`（可选 `TENSOR` 插槽） | `running_mean` STRING `"0.0"`、`running_var` STRING `"1.0"` | 把 `running_mean` / `running_var` 以两个 1 维 `TENSOR` 输出给统计量插槽；插槽一旦连线即以**连线值**为准，此时上方的控件文本被忽略 |

> 运行统计量必须能在保存的工作流里留存，而控件是唯一能做到这一点的地方，因此它们以逗号分隔的数字
> 形式输入——每通道一个值（`"0.1,0.2,0.3"`），或只给一个值由消费者广播到全部通道。两个节点都是
> 数据源：只拖到画布上不接线是无害的，因为源节点只有在被消费者需要时才会被求值。
> 接入 `mean` / `var` 插槽时以**连线为准**：连线是被测量出来的当次运行值，而控件只保存了当初
> 画图时敲进去的数字，因此把 `train` 的统计量转交 `eval` 只需两根线，不必手工抄数。

**可学习参数、优化器设定与训练循环（9 个节点）**

`PARAMS` 是有序的 `{name: nn.Parameter}` 映射；`OPTIMIZER` 是优化器**配置**
（`OptimizerConfig`）而不是活的优化器——超参留在控件上，只有设定随连线传递。ComfyUI 会把整轮
prompt 包在 `torch.inference_mode()` 里执行，因此 autograd 图无法跨越节点边界，任何节点都不能对
另一个节点产出的张量求导：训练节点只能**自己**完成前向、反向与 `optimizer.step()`，且必须在
`torch.inference_mode(False)` 块内进行。参数名沿用模块 / `safetensors` 约定
（`layer0.weight`、`layer0.bias`、`layer1.weight` …），因此训练产物可以用 `Parameters to Tensor`
拆出来，直接接进无状态的 `Basic` / `Conv` 层节点。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|---------------|---------|
| Learnable Parameters | `TrainingParameters` | `tensor`（可选） | `name` STRING `"weight"`、`shape` STRING `"2,3"`、`init` COMBO normal/zeros/ones/xavier_uniform/kaiming_uniform（默认 `normal`）、`seed` INT 0 | 创建一个具名可训练参数；**连入的张量优先于** `shape` / `init`。输出 `PARAMS` |
| Merge Parameters | `TrainingParametersMerge` | `params a`、`params b` | — | 把两份参数集合并成一份；两侧同名的键会被**改名**（`weight` → `weight_2`）并告警，而不是覆盖，因此不会静默丢权重 |
| Parameters to Tensor | `TrainingParametersExtract` | `params` | `name` STRING `"weight"` | 把其中一个条目以普通 `TENSOR` 输出（已 detach）——这是训练产物回流 `Basic` / `Conv` 层节点的桥（`layer0.weight` → `Linear.weight`）；名字不存在时抛可读错误并列出全部可用名字 |
| Optimizer | `TrainingOptimizer` | — | `optimizer` COMBO AdamW/Adam/SGD/RMSprop（默认 `AdamW`）、`lr` FLOAT 0.01 (0~1)、`momentum` 0.9 (0~0.999)、`beta1` 0.9 (0~0.999)、`beta2` 0.999 (0~0.9999)、`eps` 1e-8 (0~1e-3)、`weight_decay` 0.01 (0~1)、`amsgrad` false | 把设定以 `OPTIMIZER` 发布；`SGD` 读 `momentum`，`Adam`/`AdamW` 读两个 beta，`RMSprop` 把 `beta2` 当作 `alpha` |
| Training Loop | `TrainingLoop` | `x`、`y`（`TENSOR`）、`optimizer`（`OPTIMIZER`）、`params`（可选 `PARAMS`，热启动） | `hidden` STRING `"8"`、`activation` COMBO relu/gelu/tanh/sigmoid/none（默认 `relu`）、`loss` COMBO mse/l1/cross_entropy（默认 `mse`）、`steps` INT 200 (1~100000)、`batch_size` INT 0 (0~65536；0 = 每步全批)、`seed` INT 0 | 训练器：按 `hidden` 搭一个小 MLP（`in_features` 取自 `x`、`out_features` 取自 `y`，激活只加在隐层之间，`hidden` 留空即纯线性回归），执行 `steps` 次「前向 + 反向 + `optimizer.step()`」。输出 `params`、`loss`（标量）、`loss_history`（1 维，每步一项）与 `prediction`（已 detach） |
| Save Parameters | `TrainingSaveParameters` | `params` | `filename_prefix` STRING `"comfydl/parameters"` | 把参数集写成输出目录下的 `.safetensors` 文件并**原样透传**该集合，因此保存不会中断图；另输出绝对 `path` |
| Load Parameters | `TrainingLoadParameters` | — | `path` STRING `"comfydl/parameters_00001_.safetensors"` | 读回参数集（相对输出目录或绝对路径）；非浮点条目被丢弃、非 float32 被升位，两者都会报告；文件缺失时抛可读错误 |
| Parameters to Text | `TrainingParametersToText` | `params` | — | 把参数集编码为 `CDLPARAMS1:<base64 safetensors>` 并推进节点自身的 UI 文本框，可直接复制粘贴进控件——这条通道能让参数随 `.json` 工作流一起保存，不碰磁盘 |
| Text to Parameters | `TrainingTextToParameters` | — | `text` STRING，多行（默认值是一个合法的 2×3 `weight`） | 把文本形式解码回 `PARAMS`；忽略首尾空白与换行，内容不可解时抛出说明「该粘贴什么」的错误 |

> 确定性与缓存：`Training Loop` 用 `seed` 控件同时给初始化和批采样播种，用完还原进程 RNG，
> 因此同样的输入永远得到同样的结果，节点的缓存输出也始终有意义。所有输出都已 detach，不会把
> autograd 图带进 ComfyUI 的输出缓存；热启动（接入 `params`）只复制名字**与形状**都匹配的条目，
> 缺失与跳过的键会逐条报告，绝不静默丢弃。
>
> `PARAMS` 与 `OPTIMIZER` 在 `comfy_api/latest/_io.py` 里以 `@comfytype` 声明，与 `TENSOR`、
> `LORA_MODEL` 完全同构，因此无需前端注册：未登记的插槽类型使用前端默认配色，可以像其他类型一样
> 连线。
>
> `loss_history` 就是一条普通 1 维张量，收敛曲线可直接交给上述任意可视化节点查看；`loss` 是它的
> 最后一个元素（标量）。

**语言模型：spec 链、物化、训练、生成（6 个节点）**

语言模型流水线（`comfy_extras/nodes_lm.py`，reform step 8）。由于梯度无法跨越节点边界（见上文），
Transformer 的**结构**以冻结蓝图链的形式声明在新 `MODELSPEC` 槽上——嵌入链接点在前，每个 Transformer
块一个链接点——`Language Model Build` 把链条物化为 `NNMODEL` 槽上的真实 `nn.Module`。
`Language Model Train` 随后在**深拷贝**上自己完成前向 + 反向 + `optimizer.step()` 闭环，返回**新的**
训练后模型，缓存的输入不被污染。数据来自 `Text` 组（17.9）：`Vocab Build → Text Encode →
Sliding Window` 产出（上下文，下一 token）样本对；损失是对**每个位置**的交叉熵（每个 token 预测它的
后继，最后一个位置预测接线的目标），因此 300 步的教学级训练即可收敛。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Language Model Embedding | `LanguageModelEmbedding` | `spec`（可选 `MODELSPEC`，替换既有链的嵌入链接点）、`vocab`（可选 `VOCAB`，覆盖控件） | `vocab_size` INT 16、`d_model` INT 32、`include_position` BOOLEAN true | 首个 spec 链接点：词嵌入宽度 + 词表大小 + 可选的固定正弦位置编码（无参数）；输出单链接点的 `spec` 链与 `d_model` |
| Language Model Transformer Block | `LanguageModelTransformerBlock` | `spec`（`MODELSPEC`） | `num_heads` INT 4 (1~64)、`d_ffn` INT 128、`activation` COMBO relu/gelu、`dropout` FLOAT 0.0 (0~0.9) | 向链追加一个 pre-LN 块（`x + attn(LN(x))`，再 `x + ffn(LN(x))`）；宽度**从链上读取**，宽度错配根本接不进来；想堆多深堆多深 |
| Language Model Build | `LanguageModelBuild` | `spec`（`MODELSPEC`） | `seed` INT 0 | 把链物化为带种子的 `nn.Module`（线性层 Xavier 均匀、bias 置零、嵌入 N(0, 0.01)；RNG 用完还原）；输出 `model` 与 `params` 参数量 |
| Language Model Train | `LanguageModelTrain` | `model`（`NNMODEL`）、`x` / `y`（`TENSOR`，来自 Sliding Window）、`optimizer`（`OPTIMIZER`） | `steps` INT 300 (1~100000)、`batch_size` INT 0（0 = 全批）、`seed` INT 0 | 训练器：在 `torch.inference_mode(False)` 内对深拷贝执行 `steps` 次「前向 + 反向 + `optimizer.step()`」；输出训练后的 `model`（eval 态）、末步 `loss`（FLOAT）与 `loss_history`（1 维）；同 seed 完全复现 |
| Language Model Forward | `LanguageModelForward` | `model`（`NNMODEL`）、`ids`（`TENSOR`，1 维流或 2 维批） | — | 纯推理前向（eval 态）；输出 `logits` `(batch, seq_len, vocab_size)` —— `[..., t, :]` 是位置 `t` **之后**那个 token 的分布 |
| Language Model Generate | `LanguageModelGenerate` | `model`（`NNMODEL`）、`vocab`（可选 `VOCAB`）、`prefix_ids`（可选 `TENSOR`，覆盖文本前缀） | `prefix` STRING `"the "`、`num_tokens` INT 16、`temperature` FLOAT 1.0（0 = 贪心）、`seed` INT 0 | 自回归续写：在本地 `torch.Generator` 上贪心或按温度采样下一 token；输出 `ids`（前缀 + 生成）与解码后的 `text`（未接 `vocab` 时为空串） |

> spec 链是冻结 dataclass 组成的元组——纯值、不含张量——因此交给 ComfyUI 缓存是安全的，链上每个节点
> 都是其输入的纯函数。参数名沿用 `state_dict` 约定（`embedding.weight`、`blocks.0.attn.q_proj.weight`、
> `head.weight` …），训练产物可用 `Parameters to Tensor` 拆解，将来也便于接保存节点。
> `Language Model Generate` 在 `temperature` 为 0 时是确定性的 argmax 走位；非 0 时在带种子的本地
> 生成器上从 softmax 采样，同一 seed 与前缀必然复现同一段续写。


### 17.5 Regularization（1 个节点）

核心正则化节点（`comfy_extras/nodes_normalization.py`）。该节点恰好 1 个名为 `output` 的
`TENSOR` 输出，保持输入的 dtype/device，且不保存任何状态。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Dropout | `RegularizationDropout` | `tensor`、`mode`（可选 STRING 插槽） | `p` FLOAT 0.5 (0~1)、`seed` INT 0（带固定 / 递增 / 递减 / 随机化下拉） | `F.dropout`，逐元素：以概率 `p` 把元素置零，并把保留的元素按 `1/(1-p)` 放大；`eval` 模式原样透传输入 |

> 与归一化节点一样，`Dropout` 按秩自适应：掩码是逐元素抽取的，一个节点即覆盖 `(N, C)`、
> `(N, C, H, W)` 乃至 0 维标量。它的掩码来自一个按张量所在设备创建、由 `seed` 控件播种的
> `torch.Generator`，而不是进程级全局 RNG：同一种子可逐比特复现掩码（因此 ComfyUI 的缓存依然有意义），
> 宿主交给其它节点的随机流也不会被打乱。种子旁的下拉决定该值在两次运行之间如何变化——设为
> `randomize` 则每次运行都得到新掩码，设为 `fixed` 则冻结掩码。
> 与训练/推理数学完全一致的 `LayerNorm` / `GroupNorm` / `RMSNorm` 不同，`Dropout` 需要训练/推理开关，
> 因此它跟随与 `BatchNorm` / `InstanceNorm` 相同的 `mode` 插槽，并在 `eval` 中原样透传输入，
> 使得推理图里遗留一个 Dropout 节点无害。`p` 取 0 或 1 时会短路为原输入或全零，完全不做随机数抽取。

### 17.6 Pooling（2 个节点）

核心池化节点（`comfy_extras/nodes_pooling.py`）。一个节点覆盖滑动窗口家族
（`MaxPool{1,2,3}d` / `AvgPool{1,2,3}d`），另一个覆盖自适应家族
（`AdaptiveAvgPool{1,2,3}d` / `AdaptiveMaxPool{1,2,3}d`）：`mode` 选最大或平均，`dims` 选秩，
十二种 torch 层类型因此收敛为两个节点。下面的每个窗口控件都同时作用于 `dims` 个空间维。缺少
batch 维的张量同样接受——内部补上缺少的前导维，结果再还原回去。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Pool | `PoolingSliding` | `tensor` | `dims` COMBO 1/2/3（默认 2）、`mode` COMBO max/avg（默认 `max`）、`kernel_size` INT 2 (1~64)、`stride` INT 0 (0~64)、`padding` INT 0 (0~32)、`dilation` INT 1 (1~16)、`ceil_mode` BOOLEAN false、`count_include_pad` BOOLEAN true | `F.max_pool{1,2,3}d` / `F.avg_pool{1,2,3}d`；`dims=2, kernel_size=2` 与 `nn.MaxPool2d(2)` 完全等价 |
| Adaptive Pool | `PoolingAdaptive` | `tensor` | `dims` COMBO 1/2/3（默认 2）、`mode` COMBO avg/max（默认 `avg`）、`output_size` INT 1 (1~512) | `F.adaptive_max_pool{1,2,3}d` / `F.adaptive_avg_pool{1,2,3}d`；每个空间维都被压到恰好 `output_size` |

> 不单独提供 `GlobalAvgPool` / `GlobalMaxPool` 节点：`output_size=1` **就是**全局池化
> （`dims=2, output_size=1` ≡ `nn.AdaptiveAvgPool2d(1)` ≡ 教科书里的 **Global Average Pooling**、`GAP`），
> 自适应节点已经覆盖它，`mode` 下拉再覆盖两种归约。`stride=0` 表示「与 `kernel_size` 相同」，
> 即 torch 自己的默认行为。有两个控件是模式专属的，另一种模式下会被忽略：`dilation` 只在
> `mode=max` 下有效，`count_include_pad` 只在 `mode=avg` 下有效（`F.avg_pool*d` 没有 `dilation` 参数，
> `F.max_pool*d` 没有 `count_include_pad`）；若被忽略的控件被设成非默认值，节点会打印提示，而不是
> 静默失效。

### 17.7 Convolution（2 个节点）

核心卷积节点（`comfy_extras/nodes_convolution.py`），是上文 `Basic` 层在网络结构上的可学习对应物。
它们与 `Linear` 遵循同一约定：节点内部不做任何初始化，`weight` 与 `bias` 都是普通 `TENSOR` 输入，
因此节点是纯函数，同一组权重可以喂给多个节点。相应地也没有 `in_channels` / `out_channels` /
`bias` 开关：通道数直接从 `weight.shape` 读出，「不要偏置」用不连 `bias` 可选插槽来表达。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Conv | `ConvolutionConv` | `tensor`、`weight`（`(out, in/groups, k...)`）、`bias`（可选） | `dims` COMBO 1/2/3（默认 2）、`groups` INT 1 (1~4096)、`stride` INT 1 (1~64)、`padding` INT 1 (0~64)、`padding_mode` COMBO zeros/reflect/replicate/circular（默认 `zeros`）、`dilation` INT 1 (1~32) | `F.conv{1,2,3}d`，支持分组卷积、膨胀卷积与四种填充模式 |
| ConvTranspose | `ConvolutionConvTranspose` | `tensor`、`weight`（`(in, out/groups, k...)`）、`bias`（可选） | `dims` COMBO 1/2/3（默认 2）、`groups` INT 1 (1~4096)、`stride` INT 2 (1~64)、`padding` INT 0 (0~64)、`output_padding` INT 0 (0~64)、`dilation` INT 1 (1~32) | `F.conv_transpose{1,2,3}d`，解码器与生成器所用的上采样对应物 |

> `Conv` 的关键在于**参数共享**：*同一个*卷积核在所有空间位置上被复用，因此权重数量只与通道数和
> 卷积核尺寸有关，而与图像尺寸无关——3×3 的核无论图像是 32×32 还是 1024×1024，每个输入/输出通道
> 都只需要 9 个权重。`dims=1/2/3` 之间变的只是秩与核形状，这正是用一个节点覆盖三种秩的原因。
>
> 两个节点的权重形状是**镜像**的：`Conv` 收 `(out_channels, in_channels/groups, k...)`，而
> `ConvTranspose` 收 `(in_channels, out_channels/groups, k...)`。`groups = C_in` 即深度卷积，
> `dilation > 1` 在不增加任何权重的前提下扩大感受野。
>
> `padding_mode` 必须手工实现：`F.conv{1,2,3}d` **没有** `padding_mode` 参数，只有 `nn.Conv*d`
> 模块才有。当 `padding_mode != "zeros"` 时，先用 `F.pad(mode=reflect|replicate|circular)` 填充输入，
> 再以 `padding=0` 调用卷积，两者不会重复填充。`F.pad` 的 `circular` 需要输入秩 ≥ 3 且填充量小于
> 对应维度尺寸；条件不满足时节点会打印可读提示并回退为补零，而不是抛出异常。`ConvTranspose`
> 刻意不提供 `padding_mode` 控件，因为 `nn.ConvTranspose*d` 与 `F.conv_transpose*d` 都不支持它。
>
> 两个节点都像 `Linear` 一样做 dtype 提升：当输入与权重同为浮点但类型不同（fp16 激活 + fp32 权重）时，
> 以更宽的类型为准，而不是抛 dtype 不匹配错误。

### 17.8 Attention（7 个节点）

核心注意力节点（`comfy_extras/nodes_attention.py`），是 `Basic` 家族在序列层上的对应物。它们遵循与
`Linear` 相同的约定：四组投影权重（q / k / v / out）是普通 `TENSOR` 输入、经插槽接线传入，节点内部
不做任何初始化，同一组权重可以喂给多个节点。训练/推理开关经 `Training Mode`（17.4）的 `mode` 连线
传递；注意力权重的 dropout 由 `seed` 控件播种的本地 `torch.Generator` 生成，同一 seed 逐位复现同一
输出。`AttentionSelf` / `AttentionCross` 是 `AttentionMultihead` 在"q/k/v 从哪来"上的常用形态封装，
`TransformerEncoderBlock` 则用接线的权重把整个 post-LN 残差块组装成一个节点——搭一个块从 7 个节点
降到 1 个。三个掩码 / 位置工具（reform step 8）为上述节点与语言模型流水线供应 `mask` 插槽的输入。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Multi-Head Attention | `AttentionMultihead` | `queries` / `keys` / `values`、`q_weight` / `k_weight` / `v_weight`（各 `(E, in)`，`E` 相同）、`out_weight`（`(E_out, E)`）、`q_bias` / `k_bias` / `v_bias` / `out_bias`（可选）、`mask`（可选）、`mode`（可选 STRING 插槽） | `num_heads` INT 4 (1~64)、`dropout_p` FLOAT 0.0 (0~0.9)、`seed` INT 0 | `nn.MultiheadAttention` 的函数式实现：投影 → 分头 → `softmax(QK^T/√d + mask)` →（train 态）带种子的 dropout → 输出投影；`E` 从权重形状读出，须能被 `num_heads` 整除 |
| Self-Attention | `AttentionSelf` | `tensor`，其余权重 / bias / mask / mode 同上 | 同上 | `AttentionMultihead` 取 `q = k = v = tensor` —— 最常用形态；`out_weight` 为方阵时输出形状与输入一致 |
| Cross-Attention | `AttentionCross` | `tensor`（查询源）、`context`（键值源），其余同上 | 同上 | `AttentionMultihead` 取 q 来自 `tensor`、k/v 来自 `context` —— encoder-decoder / 多模态常用形态 |
| Transformer Encoder Block | `TransformerEncoderBlock` | `tensor`、四组注意力权重（残差要求方阵 `(E, E)`）、`ffn1_weight`（`(ffn_dim, E)`）/ `ffn2_weight`（`(E, ffn_dim)`）、注意力与 FFN 的 bias（可选）、`ln1_weight` / `ln1_bias` / `ln2_weight` / `ln2_bias`（可选 `(E,)`；不连 = 无仿射 LayerNorm）、`mask`、`mode` | `num_heads` INT 4 (1~64)、`dropout_p` FLOAT 0.0 (0~0.9)、`ffn_activation` COMBO relu/gelu（默认 `relu`）、`seed` INT 0 | 单节点的 post-LN 编码器块：`LN → MHA → Add → LN → FFN → Add`；把上一块的输出接入下一块的 `tensor` 即可堆叠 |
| Causal Mask | `AttentionCausalMask` | — | `seq_len` INT 8 (1~65536) | 下三角布尔 `(seq_len, seq_len)` 掩码（含对角线），`True` = 可注意 —— 让注意力自回归的解码器 / 语言模型掩码 |
| Padding Mask | `AttentionPaddingMask` | `lengths`（`(batch,)` 整数） | `max_len` INT 0（0 = 取 `max(lengths)`） | 逐样本有效位掩码 `(batch, 1, 1, max_len)`：位置 `< lengths[i]` 为 `True`，填充尾部为 `False`；可广播到头上与查询维 |
| Positional Encoding | `AttentionPositionalEncoding` | `tensor`（可选 `(..., length, width)`；连线时输出 `tensor + 编码`，控件改由其形状读出） | `length` INT 8、`width` INT 32 | 固定正弦位置表 `(length, width)`（"Attention Is All You Need"，无可学习参数）；输出 `encoding` 与加性注入后的 `output` |

> 布尔 `mask` 遵循 `F.scaled_dot_product_attention` 的约定 —— `True` 表示"可注意"，与
> `torch.nn.MultiheadAttention`（`True` 为屏蔽）**相反**；浮点 additive 掩码在两侧都是直接加到
> 分数上。被屏蔽的位置使用 dtype 的有限最小值而非 `-inf`，因此整行全被屏蔽的查询 softmax 后是均匀
> 分布而不是 NaN。与 `Linear` / `Conv` 一样，浮点输入与权重类型不同时提升到公共 dtype（fp16 激活 +
> fp32 权重在 fp32 中计算）；接线错误（`num_heads` 除不尽、权重形状不匹配）会抛出写明期望形状的报错，
> 绝不吞掉。
>
> 三个工具节点都是纯函数：`Causal Mask` / `Padding Mask` 直接产出注意力节点 `mask` 插槽所需的布尔表
> （任何地方都不需要再取反），`Positional Encoding` 与 `Language Model Embedding` 链接点在
> `include_position` 开启时注入的是同一张表——独立节点服务于想自己做加性注入的编码器式堆叠。

### 17.9 Text（4 个节点）

语言模型流水线的文本侧（`comfy_extras/nodes_nlp.py`，reform step 8；既有 `comfy_extras/nodes_text.py`
仍是与之无关的 Save Text 输出节点），运行在新 `VOCAB` 槽类型与普通 `torch.long` 索引张量之上。一切
都是确定性的：词表按词频（降序、再按字典序）排序，同一语料永远构建出同一词表；`<unk>` 固定为索引
0 —— 遇到没见过的 token 映射到它而不是报错。模型侧见 `Training` 组（17.4）。

| 节点 | 类名 | 输入 | 额外控件 | 作用 |
|------|-------|--------|--------------|---------|
| Vocab Build | `TextVocabBuild` | — | `corpus` STRING 多行（默认全字母句 `"the quick brown fox jumps over the lazy dog"`）、`level` COMBO char/word（默认 `char`）、`min_freq` INT 1 | 从语料构建冻结词表；输出 `vocab`（`VOCAB`）与 `vocab_size`（INT，含 `<unk>`） |
| Text Encode | `TextEncode` | `vocab`（`VOCAB`） | `text` STRING 多行（默认 `"the quick brown"`） | 把文本编码为 1 维 `torch.long` 索引张量；未见过的 token 编为 `<unk>` 索引 |
| Text Decode | `TextDecode` | `vocab`（`VOCAB`）、`ids`（`TENSOR`） | — | 把索引解码回文本；2 维 `(batch, seq_len)` 批逐行解码并以换行拼接；越界索引解码为 `<unk>` |
| Sliding Window | `TextSlidingWindow` | `ids`（`TENSOR`，1 维流） | `window` INT 4 (1~4096) | 把 token 流切成（上下文，下一 token）样本对——语言模型的下一词预测数据集；输出 `x` `(samples, window)` 与 `y` `(samples,)` |

> `Vocab Build` 的默认语料与 `Text Encode` 的默认文本互相匹配（后者的每个词都在前者中出现），因此默认
> 图无需改动任何控件即可运行。词级会折叠空白，字符级保留标点与拼写——即 d2l time-machine 的做法；
> `min_freq` 把罕见 token 从词表中剔除（它们仍会编码，只是变成 `<unk>`）。

---

## 18. ComfyUI / utilities/conversion（6 个节点）

并入 ComfyUI 核心分类 `utilities/conversion`。这些是 ComfyDL 中**唯一**基于 ComfyUI **V3** 节点 API
（`io.ComfyNode` + `io.Schema`）而非旧式 `INPUT_TYPES` / `RETURN_TYPES` 编写的节点，因为联合输入
插槽只能用 `io.MultiType.Input` 表达。

`Value → Tensor` / `Tensor → Value` 这对节点让携带 ComfyUI 语义的值（IMAGE / MASK / LATENT /
AUDIO / SIGMAS）能够进入通用张量链路，并原样返回。`Tensor → Value` 刻意提供**五个具体类型输出槽**
而不是一个动态槽：具体槽是静态类型的，前端会从物理上阻止接错类型，同时后端校验器会做真正的类型
检查——而 `MatchType` 槽在后端完全拿不到校验。无法塞进裸张量的元数据（`noise_mask`、`batch_index`、
`type`、`sampler_rate`）走平行插槽传递。

### Value → Tensor
- **类名**：`CdlValueToTensor`
- **功能**：把任意受支持的 Comfy 值拆包成通用 `TENSOR`。IMAGE / MASK / SIGMAS 原样透传；LATENT 取 `samples`；AUDIO 取 `waveform`。无法放进裸张量的元数据从平行插槽输出。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `value` | `IMAGE` \| `MASK` \| `LATENT` \| `AUDIO` \| `SIGMAS` | — | 联合插槽；接口会自适应你接入的类型 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `TENSOR` | `TENSOR` | 拆包后的张量 |
  | `origin` | `STRING` | 尽力推断的来源标签（`IMAGE` / `MASK` / `LATENT` / `AUDIO` / `SIGMAS`）。IMAGE、MASK 与 SIGMAS 在运行时都是普通张量，因此标签按形状推断——仅供诊断，不影响转换结果 |
  | `noise_mask` | `TENSOR` | LATENT 存在 `noise_mask` 时原样输出，否则为 `None` |
  | `batch_index` | `ARRAY` | LATENT 存在 `batch_index` 时原样输出，否则为 `None` |
  | `latent_type` | `STRING` | LATENT 的 `type`（`"audio"` / `"hunyuan3dv2"`）；空串表示非特殊类型 |
  | `sampler_rate` | `INT` | AUDIO 采样率；非音频值为 `0` |

### Tensor → Value
- **类名**：`CdlTensorToValue`
- **功能**：把通用 `TENSOR` 暴露到每种类型一个的具体输出槽上。连你需要的那根线即可；未使用的槽返回 `None` 且不产生开销。从 `Value → Tensor` 接回的元数据会被折回重建的 LATENT / AUDIO 字典。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 要暴露的张量 |
  | `origin` | `STRING` | `""` | 来自 `Value → Tensor` 的可选标签；仅用于组织形状不匹配的提示语 |
  | `noise_mask` | `TENSOR` | — | 可选 LATENT 噪声掩码（默认不接线） |
  | `batch_index` | `ARRAY` | — | 可选 LATENT 批次索引（默认不接线） |
  | `latent_type` | `STRING` | `""` | 可选 LATENT `type`；空串表示省略该键，与原件一致 |
  | `sampler_rate` | `INT` | `44100` | 可选 AUDIO 采样率（1~384000） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `IMAGE` | `IMAGE` | `tensor` 原样——形状约定由调用方负责 |
  | `MASK` | `MASK` | `tensor` 原样 |
  | `LATENT` | `LATENT` | `{"samples": tensor}` 加上所有已连接的元数据槽；张量非 4 维时为 `None` |
  | `AUDIO` | `AUDIO` | `{"waveform": tensor, "sampler_rate": rate}`；张量非 2 维/3 维时为 `None` |
  | `SIGMAS` | `SIGMAS` | `tensor` 原样 |

> 当张量秩不可能匹配时，LATENT / AUDIO 槽返回 `None`，并且**只在 `origin` 表明你确实想转成该类型时**
> 才打印提示——未接线的槽保持静默。

预留节点对——目前 ComfyDL 中没有任何节点产出 `LORA_MODEL` / `LOSS_MAP`，因此这四个节点属于静态/
实验性能力。每个打包节点按 key 顺序把每个张量展平并拼接成一个 1 维 `TENSOR`，同时把布局作为平行
元数据输出；每个解包节点据此切回。同一次打包中的所有张量必须 dtype 一致：混合 dtype 会抛出可读的
`ValueError`，而不是静默提升类型（那会让往返变得有损）。

### LoRA Model → Tensor
- **类名**：`CdlLoraModelToTensor`
- **功能**：把 `LORA_MODEL`（`dict[str, torch.Tensor]`）打包成一个 1 维 `TENSOR` 加 key/shape/dtype 元数据。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `lora_model` | `LORA_MODEL` | — | 参数名到张量的映射 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `TENSOR` | `TENSOR` | 所有值按 key 顺序展平并拼接 |
  | `keys` | `ARRAY` | key 顺序，`list[str]` |
  | `shapes` | `ARRAY` | 每个张量一个 `"3,4"` 风格形状字符串；`""` 表示 0 维标量 |
  | `dtypes` | `ARRAY` | 每个张量一个 `"torch.float32"` 风格 dtype 字符串 |

### Tensor → LoRA Model
- **类名**：`CdlTensorToLoraModel`
- **功能**：`LoRA Model → Tensor` 的精确逆操作；按 `shapes` 切分扁平张量，把每段转回记录的 dtype 并重建字典。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 打包节点产出的扁平张量（任意形状都会先展平） |
  | `keys` | `ARRAY` | — | key 顺序，`list[str]` |
  | `shapes` | `ARRAY` | — | 每个张量一个形状字符串 |
  | `dtypes` | `ARRAY` | — | 每个张量一个 dtype 字符串 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `LORA_MODEL` | `LORA_MODEL` | 还原后的映射 |

### Loss Map → Tensor
- **类名**：`CdlLossMapToTensor`
- **功能**：把 `LOSS_MAP`（`{"loss": [Tensor, ...]}`）打包成一个 1 维 `TENSOR` 加 shape/dtype 元数据。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `loss_map` | `LOSS_MAP` | — | 一个 `LOSS_MAP`：`{"loss": [Tensor, ...]}`；也接受裸张量或普通列表 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `TENSOR` | `TENSOR` | 所有 loss 张量按顺序展平并拼接 |
  | `shapes` | `ARRAY` | 每个张量一个 `"3,4"` 风格形状字符串 |
  | `dtypes` | `ARRAY` | 每个张量一个 `"torch.float32"` 风格 dtype 字符串 |

### Tensor → Loss Map
- **类名**：`CdlTensorToLossMap`
- **功能**：`Loss Map → Tensor` 的精确逆操作；重建 `{"loss": [Tensor, ...]}`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `TENSOR` | — | 打包节点产出的扁平张量 |
  | `shapes` | `ARRAY` | — | 每个张量一个形状字符串 |
  | `dtypes` | `ARRAY` | — | 每个张量一个 dtype 字符串 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `LOSS_MAP` | `LOSS_MAP` | `{"loss": [Tensor, ...]}` |

> 畸形元数据永远不会中断工作流：无法解析的形状字符串会降级为 0 维标量（消耗一个元素）并打印提示，
> 而被截断的张量只会还原出放得下的那些张量。

---

## 19. ComfyUI / model（22 个节点）

**model 协议层**（`comfy_extras/nodes_model_loaders.py`、`nodes_model_merging.py` 与
`nodes_model_inference.py`）。`MODEL`、`CLIP`、`VAE` 重新成为图中的一等数据，权重再次成为工作流可以
加载、搬运、混合与写回的东西。

这一层与原生实现最大的不同，是它在 **state_dict** 层面而不是模块层面工作。脱水阶段移除了 `ldm` 模型
实现，因此这里不做任何结构识别：权重文件被当作 key → 张量的扁平映射，节点唯一理解的「结构」是那套
约定俗成的**键前缀**：

| 前缀 | 归入 |
|---|---|
| `diffusion_model.`（以及所有未匹配的键） | `MODEL` |
| `first_stage_model.` | `VAE` |
| `cond_stage_model.` / `conditioner.` / `text_encoders.` | `CLIP` |

于是任何 loader 都能直接吃 `.safetensors` / `.ckpt` 文件。三组权重装进按**引用**持有张量的容器
（不复制，内存占用即文件大小），并且逐字往返键集：载入时剥掉的外层容器前缀（`model.`、
`state_dict.`、`module.`，自动探测，也可用 `prefix_strip` 控件显式指定）在保存时原样写回。这正是
「载入 → 合并 → 保存」能产出与输入键集完全一致的文件的原因——往返在构造上就是无损的。

`MODEL` 组是「只有权重」的唯一例外：它额外被包进 `ModelPatcher`，从而是一个真正的 `MODEL` 值，
与 ComfyUI 其余部分保持类型兼容。某一组没有键时输出的是合法的**空**容器而不是 `None`，
因此「只含部分权重」的 checkpoint 不会打断下游连线。

**两档行为**，以便一眼看清本构建里什么能真正跑起来：

| 档位 | 行为 | 节点 |
|---|---|---|
| L1——真实执行 | 真正读文件、拆组、合并、落盘 | 5 个 loader、7 个合并节点、4 个保存节点（共 16 个） |
| L2——已注册但不可执行 | 节点存在且 IO 契约与原生一致，工作流可以连线并通过校验；但执行时会抛出 `RuntimeError`，说明缺少哪个模块、如何恢复（绝不会是裸的 `ModuleNotFoundError`） | `Load LoRA (Model and CLIP)`、`Load LoRA`、`VAE Decode`、`VAE Encode`、`CLIP Text Encode (Prompt)`、`CLIP Set Last Layer`（共 6 个） |

L2 是刻意保留的：IO 契约让用户今天就能搭出完整文生图流程图，而每个节点都会在 docstring 与报错信息里
写明自己在等被移除引擎的哪一块。将来把对应模块回捞回来，它们原地即成为可用节点。

### 19.1 Loaders（7 个节点）

| 节点 | 类名 | 输入 | 控件 | 输出 / 作用 |
|------|-------|--------|---------|-------------------|
| Load Checkpoint | `CheckpointLoaderSimple` | `ckpt_name` COMBO（`models/checkpoints`） | `prefix_strip` STRING `"auto"` | `MODEL`、`CLIP`、`VAE`——把一个文件拆进三个组 |
| Load Diffusion Model | `UNETLoader` | `unet_name` COMBO（`models/unet`、`models/diffusion_models`） | `prefix_strip` STRING `"auto"` | `MODEL`——整个文件就是扩散模型 |
| Load VAE | `VAELoader` | `vae_name` COMBO（`models/vae`） | `prefix_strip` STRING `"auto"` | `VAE` |
| Load CLIP | `CLIPLoader` | `clip_name` COMBO（`models/text_encoders`，同时兼容旧的 `models/clip`） | `prefix_strip` STRING `"auto"` | `CLIP` |
| Load CLIP (Dual) | `DualCLIPLoader` | `clip_name1`、`clip_name2` COMBO | `prefix_strip` STRING `"auto"` | `CLIP`——两个文件的并集，两个编码器作为一个值传递 |
| Load LoRA (Model and CLIP) | `LoraLoader` | `model` MODEL、`clip` CLIP、`lora_name` COMBO（`models/loras`） | `strength_model` FLOAT 1.0、`strength_clip` FLOAT 1.0 | `MODEL`、`CLIP`——**L2** |
| Load LoRA | `LoraLoaderModelOnly` | `model` MODEL、`lora_name` COMBO（`models/loras`） | `strength_model` FLOAT 1.0 | `MODEL`——**L2** |

> 除文件选择器外，只比原生契约多了一个 `prefix_strip` 控件，且其默认值 `auto` 开箱可用，
> 因此任何 loader 不改任何参数就能运行。

### 19.2 Merging（11 个节点）

合并节点取出两个输入的键、对齐，然后**直接对张量做运算**，把结果包成新容器返回。它们刻意不构造
`ModelPatcher` 补丁列表：补丁只有在模型真正被求值时才生效，而那条路径需要被移除的引擎。把运算提前做完，
结果就是最终、确定、可检视的——因此也可保存。只有两边都有的键会被混合；只存在于一侧的键原样带过去
并给出报告。

| 节点 | 类名 | 输入 | 控件 | 结果 |
|------|-------|--------|---------|--------|
| ModelMergeSimple | `ModelMergeSimple` | `model1`、`model2` | `ratio` FLOAT 1.0 (0~1) | `model1 * ratio + model2 * (1 - ratio)` |
| ModelMergeBlocks | `ModelMergeBlocks` | `model1`、`model2` | `input` FLOAT 1.0、`middle` FLOAT 1.0、`out` FLOAT 1.0 (0~1) | 按 UNet 块分组（`input_blocks` / `middle_block` / `output_blocks`）各给一个比例；不属于任何分组的键使用 `input`，与原生节点一致 |
| ModelMergeAdd | `ModelMergeAdd` | `model1`、`model2` | — | `model1 + model2` |
| ModelMergeSubtract | `ModelMergeSubtract` | `model1`、`model2` | `multiplier` FLOAT 1.0 (-10~10) | `model1 - multiplier * model2`（乘数为负即为加回） |
| CLIPMergeSimple | `CLIPMergeSimple` | `clip1`、`clip2` | `ratio` FLOAT 1.0 (0~1) | `ModelMergeSimple` 的 CLIP 版本 |
| CLIPMergeAdd | `CLIPMergeAdd` | `clip1`、`clip2` | — | `clip1 + clip2` |
| CLIPMergeSubtract | `CLIPMergeSubtract` | `clip1`、`clip2` | `multiplier` FLOAT 1.0 (-10~10) | `clip1 - multiplier * clip2` |

> CLIP 合并会跳过 `.position_ids` 与 `.logit_scale` 并从 `clip1` 直接拷贝：它们是指标 / 标量而不是
> 权重，原生节点跳过的也正是这两个键。

四个保存节点把结果写进 `output/`。它们没有输出——文件本身就是结果——并计入输出节点，因此以保存节点
收尾的图照样可以执行。

| 节点 | 类名 | 输入 | 控件 | 写出内容 |
|------|-------|--------|--------|--------|
| ModelSave | `ModelSave` | `model` | `filename_prefix` `"comfydl/diffusion_models"` | 仅 MODEL 组，键与来源模型完全一致 |
| VAESave | `VAESave` | `vae` | `filename_prefix` `"comfydl/vae"` | VAE 组，`Load VAE` 可直接读回的布局 |
| CLIPSave | `CLIPSave` | `clip` | `filename_prefix` `"comfydl/clip"` | CLIP 容器写成单个文件（原生节点会把双编码器拆成每族一个文件；`Load CLIP` / `Load CLIP (Dual)` 都能读回这种单文件形式） |
| Save Checkpoint | `CheckpointSave` | `model`、`clip`、`vae` | `filename_prefix` `"comfydl/checkpoints"` | 三组权重写进同一个 `.safetensors`，各自回加原本的键前缀 |

> `filename_prefix` 支持 `%date%` 之类占位符，默认值即可用，因此保存节点不改参数也能落盘。

### 19.3 Latent（2 个节点）

两个都是协议占位节点（L2）：注册了与原生一致的 IO 契约，工作流可以通过校验；但解码 / 编码图像需要被
脱水的自编码器结构，因此执行时会抛出说明这一点的 `RuntimeError`。

| 节点 | 类名 | 输入 | 输出 |
|------|-------|--------|--------|
| VAE Decode | `VAEDecode` | `samples` LATENT、`vae` VAE | `IMAGE` |
| VAE Encode | `VAEEncode` | `pixels` IMAGE、`vae` VAE | `LATENT` |

### 19.4 Conditioning（2 个节点）

同理的协议占位节点（L2）：文本编码需要文本编码器结构。

| 节点 | 类名 | 输入 | 控件 | 输出 |
|------|-------|--------|---------|--------|
| CLIP Text Encode (Prompt) | `CLIPTextEncode` | `text` STRING（多行，默认 `"a photo of a cat"`）、`clip` CLIP | — | `CONDITIONING` |
| CLIP Set Last Layer | `CLIPSetLastLayer` | `clip` CLIP | `stop_at_clip_layer` INT -1 (-24~-1) | `CLIP`——为 "clip skip" 技巧截断编码器（常用 `-2`） |

> 宿主注册表的 `model/latent` 分类下还有原生节点 `LatentCompositeMasked`，不属于本次改动，此处不列。

---

## 20. ComfyUI / 3d（1 个节点）

`Preview3D`（`comfy_extras/nodes_preview_3d.py`）是单独保留的官方 3D 预览节点，而不是上游那一整套
`Load3D` / 高斯泼溅家族。前端把 3D 画布绑定在硬编码的节点 id `Preview3D` 上，因此自定义节点永远无法
自己渲染 3D，只能产出文件并交给它。本节点就是那个接收端：`CdlHeatmapsTo3D` 的产出正喂给它，而它也是
脱水版唯一需要的 3D 节点。

| 节点 | 类名 | 输入 | 输出 |
|------|-------|--------|--------|
| Preview 3D | `Preview3D` | `model_file` STRING \| `File3D`（`obj` / `glb` / `gltf` / `fbx` / `stl` / `usdz`）、`camera_info` LOAD3D_CAMERA（可选）、`bg_image` IMAGE（可选） | — |

> 它没有输出：预览通过节点的 `ui` 载荷传给前端画布。`File3D` 对象会以
> `preview3d_<uuid>.<format>` 之名写进 `output/`；同名的 `.mtl`（`CdlHeatmapsTo3D` 会产出）
> 由生产方自己写出，而不是由本节点写，否则重命名会把这个材质文件落单。

---

## 附录

### 节点注册机制

ComfyDL 在 `nodes/__init__.py` 中使用基于 importlib 的自动发现机制：扫描 `nodes/` 目录下的所有 `.py` 文件（排除 `__init__.py`），动态导入它们，并聚合每个模块的 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`。

### 节点总数

共 **109 个节点**，分属 20 个类别均由 ComfyDL 本身提供；随宿主一起发布的节点库另加 85 个 ComfyUI
核心节点，两个口径都列在下表：

| 类别 | 数量 | 说明 |
|----------|-------|------|
| d2l/Device Utils | 3 | GPU/CPU 设备查询 |
| d2l/CV Models | 5 | CNN 基础与模型构建 |
| d2l/GAN | 2 | GAN 训练更新 |
| utilities | 4 | Windows MessageBox、NoOp 空操作、计时与神秘的 "?"（ComfyUI 核心分类） |
| d2l/Model Utils | 7 | 模型信息、模式、前向、层结构、参数、克隆与存取 |
| d2l/_Legacy/Model Utils | 1 | 已弃用（软归档）：`Model Mode`，纯 `TENSOR` 图改用核心 `Training Mode` |
| d2l/NLP Models | 12 | RNN/GRU/RNNLM、注意力与 Seq2Seq 模型构件 |
| d2l/_Legacy/NLP Models | 4 | 已弃用（软归档）：`Multi-Head Attention`、`Add & Norm`、`Transformer Encoder Block`、`Transformer Encoder` |
| d2l/NLP Utils | 5 | 文本分词与词表 |
| d2l/Tensor Basic | 5 | 张量 I/O、卷积、转置、广播、重塑、激活函数 |
| d2l/_Legacy/Tensor Basic | 3 | 已弃用（软归档）：`Broadcast`、`Reshape`、`Activation`，均有核心等价节点 |
| d2l/TorchOps | 10 | 损失、优化、评估指标 |
| d2l/ObjectDetection | 10 | 锚框、IoU、NMS |
| d2l/Segmentation | 4 | VOC 语义分割工具 |
| d2l/Visualization | 13 | 图表与边界框可视化 |
| d2l/Datasets | 10 | 数据集下载、加载、预览与统计 |
| image/color | 3 | 灰度、归一化与亮度/对比度/饱和度（ComfyUI 核心分类） |
| image/transform | 1 | 任意角度旋转 + 画布扩展（ComfyUI 核心分类） |
| image | 1 | 图像批次逐通道统计（ComfyUI 核心分类） |
| utilities/conversion | 6 | Comfy 语义值与通用 `TENSOR` 的往返转换（ComfyUI 核心分类） |
| Network & Layers/Activation | 14 | `TENSOR` 类型上的核心激活函数（ComfyUI 核心分类） |
| Network & Layers/Basic | 8 | `TENSOR` 类型上的核心基础层与张量运算（ComfyUI 核心分类） |
| Network & Layers/Attention | 7 | 多头 / 自 / 交叉注意力、组装好的 post-LN Transformer 编码器块、因果 / 填充掩码与正弦位置编码，权重走插槽（ComfyUI 核心分类） |
| Network & Layers/Normalization | 7 | `TENSOR` 类型上的核心归一化（ComfyUI 核心分类） |
| Network & Layers/Regularization | 1 | 带种子掩码的核心逐元素 dropout（ComfyUI 核心分类） |
| Network & Layers/Training | 17 | 训练/推理开关、运行统计量、可学习参数、优化器设定、训练循环与语言模型流水线（spec 链 / Build / Train / Forward / Generate）（ComfyUI 核心分类） |
| Network & Layers/Pooling | 2 | `TENSOR` 类型上的最大 / 平均池化，滑动窗口与自适应（`output_size=1` 即全局池化）（ComfyUI 核心分类） |
| Network & Layers/Convolution | 2 | `TENSOR` 类型上的卷积与转置卷积，权重走连线传入（ComfyUI 核心分类） |
| Network & Layers/Text | 4 | 核心文本流水线：`VOCAB` 类型上的词表构建、文本编码 / 解码与滑窗下一词数据集（ComfyUI 核心分类） |
| model/loaders | 7 | state_dict 层面的 checkpoint / 扩散模型 / VAE / CLIP 加载（ComfyUI 核心分类） |
| model/merging | 11 | 按键对齐的模型与 CLIP 合并，以及 `.safetensors` 落盘（ComfyUI 核心分类） |
| model/latent | 2 | `VAE Decode` / `VAE Encode` 协议占位节点（ComfyUI 核心分类） |
| model/conditioning | 2 | `CLIP Text Encode (Prompt)` / `CLIP Set Last Layer` 协议占位节点（ComfyUI 核心分类） |
| 3d | 1 | `Preview 3D`，前端绑定的 3D 预览画布（ComfyUI 核心分类） |

> 前 20 行统计 **ComfyDL 提供的 109 个节点**（其中 8 个已软归档到 `d2l/_Legacy/*`：节点不删、旧工作流照常加载，但显示名带 `(DEPRECATED)` 后缀并在节点库中移入 Legacy 分类）。`utilities`、`utilities/conversion`、`image/color`、`image/transform`、`image` 是 ComfyUI 核心分类（ComfyDL 节点并入其中），这些分类下还有 ComfyUI 原生节点。
>
> 其余 14 行是纯 ComfyUI 核心分类，不含 ComfyDL 节点：九个 `Network & Layers/*` 分组（62 个节点）、
> 四个 `model/*` 分组（22 个节点）与 `3d`（1 个节点）。因此随宿主发布的节点库总计
> **194 个节点、34 个分类** = 109 个 ComfyDL + 85 个核心节点。
>
> 有两行的数量少于宿主注册表在该分类下的实际节点数，因为注册表把本次改动未触及的原生节点也算在内：
> `model/latent`（其第三个节点是 `LatentCompositeMasked`）以及 `image`、`utilities`、`image/color`、
> `image/transform`（ComfyDL 节点与相邻的原生节点混在同一分类里）。
