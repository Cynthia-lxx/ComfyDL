# ComfyDL 节点参考

> [English version / 英文版](FUNCTIONS.md)

本文档详细说明 ComfyDL 中的每一个自定义节点：功能、输入与输出。

---

## 自定义数据类型

ComfyDL 定义了 5 种 ComfyUI 自定义数据类型，用于在节点间传递结构化数据：

| 类型名 | Python 类型 | 说明 |
|-----------|-------------|------|
| `cdlTensor` | `torch.Tensor` | 任意形状的 PyTorch 张量 |
| `cdlModel` | `nn.Module` | PyTorch 模型实例 |
| `cdlVocab` | `dict` | 词表字典，包含 `idx_to_token` 和 `token_to_idx` |
| `cdlDataloader` | `torch.utils.data.DataLoader` | PyTorch 数据加载器 |
| `cdlBbox` | `torch.Tensor [N,4]` | 边界框张量，格式为 `(x1, y1, x2, y2)` |

直接使用的 ComfyUI 标准类型：
- `IMAGE` — 图像批次，`torch.Tensor [B, H, W, C]`
- `MASK` — 掩码，`torch.Tensor [H, W]` 或 `[B, C, H, W]`
- `INT`、`FLOAT`、`STRING`、`BOOLEAN` — 基本标量类型

---

## 1. ComfyDL / Device Utils（3 个节点）

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

## 2. ComfyDL / CV Models（5 个节点）

### Corr2D
- **类名**：`CdlCorr2d`
- **d2lcore 函数**：`corr2d(X, K)`
- **功能**：对输入张量执行二维互相关运算——这是卷积操作的基础。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `input_tensor` | `cdlTensor` | 输入二维张量 |
  | `kernel` | `cdlTensor` | 核二维张量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 互相关结果 |

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

## 3. ComfyDL / GAN（2 个节点）

### Update Discriminator
- **类名**：`CdlUpdateD`
- **d2lcore 函数**：`update_D(X, Z, net_D, net_G, loss, trainer_D)`
- **功能**：执行一次 GAN 判别器的训练更新。使用真实数据 `X` 和生成器 `net_G` 从噪声 `Z` 生成的假数据，计算 BCE 损失并反向传播以更新判别器参数。使用 SGD 优化器（lr=0.01）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `cdlTensor` | 真实数据批次（可选） |
  | `Z` | `cdlTensor` | 噪声输入（可选） |
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
  | `Z` | `cdlTensor` | 噪声输入（可选） |
  | `net_D` | `cdlModel` | 判别器模型（可选） |
  | `net_G` | `cdlModel` | 生成器模型（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `loss_G` | `FLOAT` | 生成器损失；若任一输入缺失则返回 0.0 |
  | `fake_X` | `cdlTensor` | 生成器产生的假数据 |

---

## 4. ComfyDL / Misc（2 个节点）

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

---

## 5. ComfyDL / NLP Utils（5 个节点）

### Tokenize
- **类名**：`CdlTokenize`
- **d2lcore 函数**：`tokenize(lines, token)`
- **功能**：将输入文本按行分词。支持词级别（按空白符分割）和字符级别两种分词模式。每行视为一个句子；行内 token 以逗号分隔输出，行间以换行符分隔。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `text` | `STRING` | `""` | 输入文本，每行一个句子（多行） |
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
  | `tokens_a` | `STRING` | `""` | A 段，逗号分隔的 token（多行） |
  | `tokens_b` | `STRING` | `""` | B 段，逗号分隔的 token（可选，多行） |
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
  | `tokens_text` | `STRING` | `""` | Token 文本，每行一个或逗号分隔（多行） |
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
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `vocab` | `cdlVocab` | 词表字典 |
  | `tokens` | `STRING` | 逗号分隔的 token 序列（多行） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `indices` | `cdlTensor` | 编码后的索引张量（`torch.int64`） |

### Vocab Decode
- **类名**：`CdlVocabDecode`
- **d2lcore 函数**：`Vocab.to_tokens(indices)`
- **功能**：使用词表将索引张量转换回 token 字符串。超出范围的索引映射为 `<unk>`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `vocab` | `cdlVocab` | 词表字典 |
  | `indices` | `cdlTensor` | 索引张量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `tokens_str` | `STRING` | 解码后的 token 字符串，逗号分隔 |

---

## 6. ComfyDL / Tensor Basic（7 个节点）

### Tensor → String
- **类名**：`CdlTensorToStr`
- **功能**：将张量格式化为人类可读的字符串。显示张量的形状、数据类型、设备信息和数值内容。超过 `max_elems` 的大张量会被截断（显示前半部分 + 后半部分）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `cdlTensor` | — | 输入张量 |
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
  | `tensor` | `cdlTensor` | 解析后的张量（`torch.float32`） |

### Conv2D
- **类名**：`CdlConv2d`
- **功能**：对输入张量执行二维卷积，使用指定的卷积核。封装 ``torch.nn.functional.conv2d``，支持 stride 和 padding 参数。自动将 2-D/3-D 输入扩展为 4-D ``(N, C, H, W)``，结果去掉 batch 维度返回。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `input_tensor` | `cdlTensor` | — | 输入张量 |
  | `kernel` | `cdlTensor` | — | 卷积核 |
  | `stride` | `INT` | 1 | 卷积步幅（1~4） |
  | `padding` | `INT` | 0 | 零填充（0~10） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 卷积结果 |

### Transpose
- **类名**：`CdlTranspose`
- **功能**：交换张量的两个维度。封装 ``torch.transpose``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `cdlTensor` | — | 输入张量 |
  | `dim0` | `INT` | 0 | 第一个要交换的维度（0~5） |
  | `dim1` | `INT` | 1 | 第二个要交换的维度（0~5） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 转置后的张量 |

### Broadcast
- **类名**：`CdlBroadcast`
- **功能**：将张量广播到目标形状。封装 ``torch.broadcast_to``。目标形状以逗号分隔字符串输入（如 ``"3,1,4"``）。解析错误时返回原张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `cdlTensor` | — | 输入张量 |
  | `target_shape` | `STRING` | `""` | 目标形状，逗号分隔（如 ``"3,1,4"``） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 广播后的张量 |

### Reshape
- **类名**：`CdlReshape`
- **功能**：将张量变形为新的形状。封装 ``torch.reshape``。目标形状以逗号分隔字符串输入（如 ``"2,8"``、``"4,-1"``）。解析错误时返回原张量。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `cdlTensor` | — | 输入张量 |
  | `target_shape` | `STRING` | `""` | 目标形状，逗号分隔（如 ``"2,8"``） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 变形后的张量 |

### Activation
- **类名**：`CdlActivation`
- **功能**：对张量逐元素应用激活函数。通过下拉菜单选择：``relu``、``sigmoid``、``tanh``、``leaky_relu``、``elu``、``gelu``、``silu``、``softmax``、``softplus``。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `tensor` | `cdlTensor` | — | 输入张量 |
  | `func` | `COMBO` | `relu` | 激活函数：relu / sigmoid / tanh / leaky_relu / elu / gelu / silu / softmax / softplus |
  | `dim` | `INT` | -1 | softmax 的维度（-4~4） |
  | `negative_slope` | `FLOAT` | 0.01 | leaky_relu 的负斜率（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 激活后的张量 |

---

## 7. ComfyDL / TorchOps（10 个节点）

### Linear Regression
- **类名**：`CdlLinReg`
- **d2lcore 函数**：`linreg(X, w, b)`
- **功能**：线性回归前向计算：\( \hat{y} = X w + b \)
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `cdlTensor` | 输入特征矩阵 |
  | `w` | `cdlTensor` | 权重向量 |
  | `b` | `cdlTensor` | 偏置向量/标量 |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `cdlTensor` | 预测值 |

### Squared Loss
- **类名**：`CdlSquaredLoss`
- **d2lcore 函数**：`squared_loss(y_hat, y)`
- **功能**：计算平方损失：\( \frac{1}{2}(\hat{y} - y)^2 \)（注意：不做平均）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `cdlTensor` | 预测值 |
  | `y` | `cdlTensor` | 真实值（自动重塑为与 `y_hat` 相同的形状） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `loss` | `cdlTensor` | 逐元素损失 |

### Masked Softmax
- **类名**：`CdlMaskedSoftmax`
- **d2lcore 函数**：`masked_softmax(X, valid_lens)`
- **功能**：在最后一个维度上执行带掩码的 softmax。使用 `valid_lens` 指定每条序列的有效长度；超出有效长度的位置在 softmax 前被设为一个极大的负值（使概率趋近于 0）。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `X` | `cdlTensor` | 输入张量 |
  | `valid_lens` | `cdlTensor` | 有效长度张量（可选；若未提供则执行普通 softmax） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `output` | `cdlTensor` | 掩码 softmax 结果 |

### Sequence Mask
- **类名**：`CdlSequenceMask`
- **d2lcore 函数**：`sequence_mask(X, valid_len, value)`
- **功能**：将序列中超出有效长度的位置替换为指定值。常用于将填充位置置零。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `X` | `cdlTensor` | — | 输入序列张量 |
  | `valid_len` | `cdlTensor` | — | 每条序列的有效长度 |
  | `mask_value` | `FLOAT` | 0.0 | 掩码填充值（-1e9~1e9） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `masked` | `cdlTensor` | 掩码后的张量 |

### Accuracy
- **类名**：`CdlAccuracy`
- **d2lcore 函数**：`accuracy(y_hat, y)`
- **功能**：计算分类任务的正确预测数量。若预测为多类别 logits，先取 argmax，再与标签比较。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `y_hat` | `cdlTensor` | 预测值（logits 或类别索引） |
  | `y` | `cdlTensor` | 真实标签 |
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
  | `X` | `cdlTensor` | 特征矩阵 `[num_examples, num_features]` |
  | `y` | `cdlTensor` | 标签向量 `[num_examples, 1]` |

### Truncate/Pad
- **类名**：`CdlTruncatePad`
- **d2lcore 函数**：`truncate_pad(line, num_steps, padding_token)`
- **功能**：将序列截断或填充到固定长度。超长的序列被截断；不足的序列用 `padding_token` 填充。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `num_steps` | `INT` | 64 | 目标序列长度（1~10000） |
  | `padding_token` | `INT` | 0 | 填充 token 索引（0~100000） |
  | `sequence` | `cdlTensor` | — | 输入索引序列（可选；缺失时返回全填充张量） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `padded` | `cdlTensor` | 截断/填充后的序列（`torch.int64`） |

### BLEU Score
- **类名**：`CdlBleu`
- **d2lcore 函数**：`bleu(pred_seq, label_seq, k)`
- **功能**：计算预测序列与参考序列之间的 BLEU 分数。支持 BLEU-1 到 BLEU-4（由 `max_n` 控制）。按空白符分词后计算 n-gram 精度和简洁惩罚。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `pred_seq` | `STRING` | `""` | 预测序列，空白符分隔的 token（多行） |
  | `label_seq` | `STRING` | `""` | 参考序列，空白符分隔的 token（多行） |
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

## 8. ComfyDL / ObjectDetection（9 个节点）

### Box Corner→Center
- **类名**：`CdlBoxCornerToCenter`
- **d2lcore 函数**：`box_corner_to_center(boxes)`
- **功能**：将边界框从角点格式 `(x1, y1, x2, y2)` 转换为中心格式 `(cx, cy, w, h)`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes` | `cdlTensor` | 角点格式边界框 `[N,4]`（左上角 + 右下角） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes_ccwh` | `cdlTensor` | 中心格式边界框 `[N,4]`（中心 + 宽 + 高） |

### Box Center→Corner
- **类名**：`CdlBoxCenterToCorner`
- **d2lcore 函数**：`box_center_to_corner(boxes)`
- **功能**：将边界框从中心格式 `(cx, cy, w, h)` 转回角点格式 `(x1, y1, x2, y2)`。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes` | `cdlTensor` | 中心格式边界框 `[N,4]`（中心 + 宽 + 高） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes_xyxy` | `cdlTensor` | 角点格式边界框 `[N,4]`（左上角 + 右下角） |

### Box IoU
- **类名**：`CdlBoxIou`
- **d2lcore 函数**：`box_iou(boxes1, boxes2)`
- **功能**：计算两组边界框之间的逐对 IoU（交并比）。返回矩阵 `[N1, N2]`，其中元素 `(i,j)` 为 `boxes1[i]` 与 `boxes2[j]` 的 IoU。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `boxes1` | `cdlTensor` | 第一组框 `[N1,4]`（左上角 + 右下角） |
  | `boxes2` | `cdlTensor` | 第二组框 `[N2,4]`（左上角 + 右下角） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `iou` | `cdlTensor` | IoU 矩阵 `[N1, N2]` |

### NMS
- **类名**：`CdlNms`
- **d2lcore 函数**：`nms(boxes, scores, iou_threshold)`
- **功能**：对边界框执行非极大值抑制（Non-Maximum Suppression）。按分数降序排列，保留与任何更高分框的 IoU 不超过阈值的框。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `boxes` | `cdlTensor` | — | 边界框 `[N,4]`（左上角 + 右下角） |
  | `scores` | `cdlTensor` | — | 每个框的置信度分数 |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU 阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `keep_indices` | `cdlTensor` | 被保留框的索引（`torch.int64`） |

### Multibox Prior
- **类名**：`CdlMultiboxPrior`
- **d2lcore 函数**：`multibox_prior(data, sizes, ratios)`
- **功能**：在每个像素点生成不同形状的锚框。每个像素的锚框数 = `len(sizes) + len(ratios) - 1`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `sizes` | `STRING` | `"0.75,0.5,0.25"` | 锚框尺寸列表，逗号分隔 |
  | `ratios` | `STRING` | `"1,2,0.5"` | 宽高比列表，逗号分隔 |
  | `data` | `cdlTensor` | — | 输入数据（可选；用于推断空间尺寸；缺失时默认为 561×728） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `cdlTensor` | 锚框 `[1, H*W*bpp, 4]`，归一化坐标（左上角 + 右下角） |

### Offset Boxes
- **类名**：`CdlOffsetBoxes`
- **d2lcore 函数**：`offset_boxes(anchors, assigned_bb, eps)`
- **功能**：计算从锚框到分配的真实框的偏移量。中心坐标差缩放 10 倍；宽高比取对数后缩放 5 倍。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `anchors` | `cdlTensor` | — | 锚框 `[N,4]`（左上角 + 右下角） |
  | `assigned_bb` | `cdlTensor` | — | 分配的真实框 `[N,4]`（左上角 + 右下角） |
  | `eps` | `FLOAT` | 1e-6 | 防止除零的小 epsilon（1e-12~1e-3） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `offsets` | `cdlTensor` | 偏移量 `[N,4]`（dx, dy, dw, dh） |

### Offset Inverse
- **类名**：`CdlOffsetInverse`
- **d2lcore 函数**：`offset_inverse(anchors, offset_preds)`
- **功能**：通过逆变换根据锚框和预测偏移量重建角点格式的边界框坐标。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `cdlTensor` | 锚框 `[N,4]`（左上角 + 右下角） |
  | `offset_preds` | `cdlTensor` | 预测偏移量 `[N,4]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `predicted_bbox` | `cdlTensor` | 预测框 `[N,4]`（左上角 + 右下角） |

### Assign Anchor→BBox
- **类名**：`CdlAssignAnchorToBbox`
- **d2lcore 函数**：`assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)`
- **功能**：基于 IoU 将真实边界框分配给锚框。每个锚框被分配给一个 IoU ≥ 阈值的真实框，且每个真实框至少保证有一个锚框与之匹配（取 IoU 最大者）。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `ground_truth` | `cdlTensor` | — | 真实框 `[M,4]`（左上角 + 右下角） |
  | `anchors` | `cdlTensor` | — | 锚框 `[N,4]`（左上角 + 右下角） |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU 阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors_bbox_map` | `cdlTensor` | 锚框→真实框映射 `[N,]`，-1 表示无匹配（`torch.int64`） |

### Multibox Target
- **类名**：`CdlMultiboxTarget`
- **d2lcore 函数**：`multibox_target(anchors, labels)`
- **功能**：为锚框生成多框目标训练标签。对批次中每张图像，将真实框分配给锚框并计算偏移目标、掩码和类别标签。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `anchors` | `cdlTensor` | 锚框 `[1, N, 4]`（左上角 + 右下角） |
  | `labels` | `cdlTensor` | 标签 `[B, M, 5]`，格式 `[class_id, x1, y1, x2, y2]` |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `bbox_offset` | `cdlTensor` | 边界框偏移目标 `[B, N*4]` |
  | `bbox_mask` | `cdlTensor` | 边界框偏移掩码 `[B, N*4]`（匹配锚框为 1.0） |
  | `class_labels` | `cdlTensor` | 锚框类别标签 `[B, N]`（背景=0，类别从 1 开始） |

### Multibox Detection
- **类名**：`CdlMultiboxDetection`
- **d2lcore 函数**：`multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)`
- **功能**：使用 NMS 从模型输出中预测边界框。结合类别预测和偏移量，通过逆偏移变换重建框，并通过 NMS 和置信度阈值过滤。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `cls_probs` | `cdlTensor` | — | 类别概率 `[B, num_classes, N]` |
  | `offset_preds` | `cdlTensor` | — | 偏移预测 `[B, N*4]` |
  | `anchors` | `cdlTensor` | — | 锚框 `[1, N, 4]` |
  | `nms_threshold` | `FLOAT` | 0.5 | NMS IoU 阈值（0~1） |
  | `pos_threshold` | `FLOAT` | 0.01 | 正样本置信度阈值（0~1） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `detections` | `cdlTensor` | 检测结果 `[B, N, 6]`，格式 `[class_id, confidence, x1, y1, x2, y2]`（class_id=-1 表示背景） |

---

## 9. ComfyDL / Segmentation（4 个节点）

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
  | `colormap2label` | `cdlTensor` | 颜色→类别索引查找表 `[16777216]`（`torch.int64`） |

### VOC Label Indices
- **类名**：`CdlVocLabelIndices`
- **d2lcore 函数**：`voc_label_indices(colormap, colormap2label)`
- **功能**：将 VOC 标签彩色图像映射为类别索引图。将 RGB 像素编码为单一颜色值，再通过查找表转换为类别索引。
- **输入**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `colormap` | `IMAGE` | VOC 标签彩色图像 `[B, H, W, C]`，使用第一张图像 |
  | `colormap2label` | `cdlTensor` | 颜色→标签查找表 |
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

## 10. ComfyDL / Visualization（13 个节点）

可视化节点采用"双变体"设计模式：带 `(Output)` 后缀的版本是 ComfyUI 输出节点（直接在界面中显示交互式图表），不带后缀的版本将图表渲染为 `IMAGE` 张量，供下游节点使用。

### Show Images (Output)
- **类名**：`CdlShowImagesOutput`
- **d2lcore 函数**：`show_images(imgs, num_rows, num_cols, titles, scale)`
- **功能**：在网格布局中显示一批图像。输出节点变体——图像直接渲染到 ComfyUI 界面中。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `images` | `IMAGE` | — | 输入图像批次 `[B, H, W, C]` |
  | `num_rows` | `INT` | 1 | 网格行数（1~100） |
  | `num_cols` | `INT` | 4 | 网格列数（1~100） |
  | `scale` | `FLOAT` | 1.5 | 图像缩放因子（0.1~10.0） |
  | `titles` | `STRING` | `""` | 图像标题，逗号分隔（可选） |
- **输出**：无（输出节点）

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
- **功能**：在网格布局中显示热力图矩阵。输出节点变体，带颜色条。输入矩阵维度自动提升为 4D。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `matrices` | `cdlTensor` | — | 矩阵张量（2D=[H,W]→1×1，3D=[N,H,W]→N×1，4D=[N,M,H,W]） |
  | `xlabel` | `STRING` | `""` | X 轴标签 |
  | `ylabel` | `STRING` | `""` | Y 轴标签 |
  | `figsize_w` | `FLOAT` | 2.5 | 每列宽度（0.5~20.0） |
  | `figsize_h` | `FLOAT` | 2.5 | 每行高度（0.5~20.0） |
  | `cmap` | `STRING` | `"Reds"` | matplotlib 颜色映射名称 |
  | `titles` | `STRING` | `""` | 子图标题，逗号分隔（可选） |
- **输出**：无（输出节点）

### Show Heatmaps
- **类名**：`CdlShowHeatmaps`
- **d2lcore 函数**：同上
- **功能**：与 `Show Heatmaps (Output)` 相同，但将图表渲染为 IMAGE 张量。
- **输入**：同上
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 渲染后的热力图 `[1, H, W, C]` |

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
  | `X` | `cdlTensor` | — | X 轴数据（可选，一维或二维） |
  | `Y` | `cdlTensor` | — | Y 轴数据（可选） |
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
  | `results` | `cdlTensor` | 优化轨迹点 `[N, 2]`，每行是一个 (x1, x2) 坐标 |
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
  | `bboxes` | `cdlTensor` | — | 边界框 `[N, 4]`，归一化坐标（左上角 + 右下角） |
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
  | `tensor` | `cdlTensor` | — | 输入张量（内部展平） |
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
  | `values` | `cdlTensor` | — | 柱状高度（1-D 张量） |
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
  | `X` | `cdlTensor` | — | X 坐标（展平） |
  | `Y` | `cdlTensor` | — | Y 坐标（展平） |
  | `alpha` | `FLOAT` | 0.6 | 点透明度 |
  | `cmap` | `STRING` | `"viridis"` | ``color_map`` 的 colormap |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
  | `figsize_w` | `FLOAT` | 6.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 5.0 | 图高 |
  | `color_map` | `cdlTensor` | — | 逐点颜色值（可选） |
  | `size_map` | `cdlTensor` | — | 逐点大小值（可选） |
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
  | `matrix` | `cdlTensor` | — | 混淆矩阵（N×N 或展平） |
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
  | `values` | `cdlTensor` | — | 扇区值（1-D 张量） |
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
  | `Y` | `cdlTensor` | — | 序列数据，`[T]` 或 `[N, T]` |
  | `stacked` | `BOOLEAN` | False | 堆叠而非叠加 |
  | `alpha` | `FLOAT` | 0.5 | 填充透明度 |
  | `color_palette` | `STRING` | `"tab10"` | matplotlib 调色盘名称 |
  | `xlabel` | `STRING` | `""` | x 轴标签 |
  | `ylabel` | `STRING` | `""` | y 轴标签 |
  | `figsize_w` | `FLOAT` | 7.0 | 图宽 |
  | `figsize_h` | `FLOAT` | 4.0 | 图高 |
  | `X_vals` | `cdlTensor` | — | 自定义 x 轴值（可选） |
  | `labels` | `STRING` | `""` | 序列图例标签，逗号分隔（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `image` | `IMAGE` | 面积图 `[1, H, W, C]` |

---

## 11. ComfyDL / Datasets（10 个节点）

数据集节点提供端到端的数据集管理能力：下载、加载、查看、预览和统计。

### Load Array → DataLoader
- **类名**：`CdlLoadArray`
- **d2lcore 函数**：`load_array(data_arrays, batch_size, is_train)`
- **功能**：将一个或多个张量封装为 PyTorch DataLoader。将 `cdlTensor` 特征和/或标签连接到可选输入槽；节点输出 `cdlDataloader`。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `batch_size` | `INT` | 32 | 批大小（1~4096） |
  | `shuffle` | `BOOLEAN` | True | 每个 epoch 是否打乱数据 |
  | `features` | `cdlTensor` | — | 特征张量 X（可选） |
  | `labels` | `cdlTensor` | — | 标签张量 y（可选） |
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
- **功能**：与 `CdlDataLoaderPreview` 相同，但作为 OUTPUT_NODE 直接渲染——无输出槽，仅在 UI 中显示渲染预览。
- **输入**：与 `CdlDataLoaderPreview` 相同
- **输出**：无（输出节点）

### Dataset Stats
- **类名**：`CdlDataLoaderStats`
- **功能**：遍历 `cdlDataloader`，统计类别分布。渲染柱状图显示各类别的样本数。
- **输入**：
  | 名称 | 类型 | 默认值 | 说明 |
  |------|------|---------|------|
  | `dataloader` | `cdlDataloader` | — | 待分析的 DataLoader |
  | `num_classes` | `INT` | 10 | 预期类别数（1~1000） |
  | `class_names` | `STRING` | `""` | 逗号分隔的类别名称（可选） |
- **输出**：
  | 名称 | 类型 | 说明 |
  |------|------|------|
  | `stats_text` | `STRING` | 格式化后的类别统计文本摘要 |
  | `stats_image` | `IMAGE` | 类别分布柱状图 `[1, H, W, C]` |

---

## 附录

### 节点注册机制

ComfyDL 在 `nodes/__init__.py` 中使用基于 importlib 的自动发现机制：扫描 `nodes/` 目录下的所有 `.py` 文件（排除 `__init__.py`），动态导入它们，并聚合每个模块的 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`。

### 节点总数

共 **68 个节点**，分属 11 个类别：

| 类别 | 数量 | 说明 |
|----------|-------|------|
| ComfyDL/Device Utils | 3 | GPU/CPU 设备查询 |
| ComfyDL/CV Models | 5 | CNN 基础与模型构建 |
| ComfyDL/GAN | 2 | GAN 训练更新 |
| ComfyDL/Misc | 2 | Windows MessageBox 和 NoOp 空操作 |
| ComfyDL/NLP Utils | 5 | 文本分词与词表 |
| ComfyDL/Tensor Basic | 7 | 张量 I/O、卷积、转置、广播、激活函数 |
| ComfyDL/TorchOps | 10 | 损失、优化、评估指标 |
| ComfyDL/ObjectDetection | 9 | 锚框、IoU、NMS |
| ComfyDL/Segmentation | 4 | VOC 语义分割工具 |
| ComfyDL/Visualization | 13 | 图表与边界框可视化 |
| ComfyDL/Datasets | 10 | 数据集下载、加载、预览与统计 |
