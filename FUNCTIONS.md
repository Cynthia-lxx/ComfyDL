# ComfyDL Node Reference

> [中文版 / 中文版本](FUNCTIONS_zh.md)

This document details every custom node in ComfyDL: what it does, its inputs, and its outputs. The node suite began as a mapping of the `d2l` textbook codebase and keeps growing with self-developed nodes beyond it.

ComfyDL is licensed under **GPL-3.0** (see [LICENSE](LICENSE)) and also ships as part of its own GUI build, [ComfyDL_UI](https://github.com/Cynthia-lxx/ComfyDL_UI), where every node below is registered as a built-in node.

---

## Data Types

ComfyDL nodes exchange structured data through the following ComfyUI slot types:

| Type Name | Python Type | Description |
|-----------|-------------|-------------|
| `TENSOR` | `torch.Tensor` | PyTorch tensor of arbitrary shape — **ComfyUI core type**, shared with the built-in `Network & Layers` nodes (formerly `cdlTensor`) |
| `BBOX` | `torch.Tensor [N,4]` | Bounding box tensor in `(x1, y1, x2, y2)` format — **ComfyUI core type** (formerly `cdlBbox`) |
| `cdlModel` | `nn.Module` | PyTorch model instance — ComfyDL-only, no core counterpart |
| `cdlVocab` | `dict` | Vocabulary dictionary containing `idx_to_token` and `token_to_idx` — ComfyDL-only |
| `cdlDataloader` | `torch.utils.data.DataLoader` | PyTorch data loader — ComfyDL-only |

`TENSOR` and `BBOX` are declared in the ComfyUI core (`comfy/comfy_types/node_typing.py`
and `comfy_api/latest/_io.py`), so ComfyDL nodes and core nodes (such as the `Network &
Layers` family) can be wired together directly on the same slots. `cdlModel` / `cdlVocab` /
`cdlDataloader` have no core equivalent and stay ComfyDL-specific; the previous names
`cdlTensor` / `cdlBbox` are still exported as legacy aliases from `nodes/__init__.py`.

ComfyUI standard types (used directly):
- `IMAGE` — Image batch, `torch.Tensor [B, H, W, C]`
- `MASK` — Mask, `torch.Tensor [H, W]` or `[B, C, H, W]`
- `LATENT` — Latent dictionary `{"samples": ..., "noise_mask"?, "batch_index"?, "type"?}`
- `AUDIO` — Audio dictionary `{"waveform": ..., "sampler_rate": ...}`
- `SIGMAS` — Noise schedule, `torch.Tensor [N]`
- `LORA_MODEL` — Tensor collection `dict[str, torch.Tensor]` (reserved, see §18)
- `LOSS_MAP` — Tensor collection `{"loss": [torch.Tensor, ...]}` (reserved, see §18)
- `ARRAY` — Plain Python `list`, used for the parallel metadata slots of §18
- `MODEL`, `CLIP`, `VAE`, `CONDITIONING` — ComfyUI core model-protocol types, restored by §19
- `INT`, `FLOAT`, `STRING`, `BOOLEAN` — Primitive scalar types

---

## 1. d2l / Device Utils (3 nodes)

### Device Info
- **Class**: `CdlDeviceInfo`
- **d2lcore function**: `num_gpus()`
- **Purpose**: Queries the number of GPUs and CUDA availability in the current environment.
- **Inputs**: None
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `num_gpus` | `INT` | Number of available GPUs |
  | `has_cuda` | `INT` | Whether CUDA is available (1=yes, 0=no) |

### Try GPU
- **Class**: `CdlTryGpu`
- **d2lcore function**: `try_gpu(i)`
- **Purpose**: Attempts to get the device name string for GPU at `gpu_index`. Falls back to CPU if the GPU is not available.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `gpu_index` | `INT` | 0 | GPU index (0~16) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `device_str` | `STRING` | Device string, e.g. `"cuda:0"` or `"cpu"` |

### Try All GPUs
- **Class**: `CdlTryAllGpus`
- **d2lcore function**: `try_all_gpus()`
- **Purpose**: Returns a comma-separated string of all available GPU device names. Returns `"cpu"` if no GPUs are available.
- **Inputs**: None
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `device_str` | `STRING` | e.g. `"cuda:0,cuda:1,cuda:2,cuda:3"` or `"cpu"` |

---

## 2. d2l / CV Models (5 nodes)

### Corr2D
- **Class**: `CdlCorr2d`
- **d2lcore function**: `corr2d(X, K)`
- **Purpose**: Performs 2D cross-correlation on input tensor — the fundamental operation underlying convolution.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `input_tensor` | `TENSOR` | Input 2D tensor |
  | `kernel` | `TENSOR` | Kernel 2D tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Cross-correlation result |

### LeNet
- **Class**: `CdlLeNet`
- **d2lcore function**: `LeNet(lr, num_classes)`
- **Purpose**: Builds a classic LeNet-5 convolutional neural network. Uses `LazyConv2d` and `LazyLinear` — input shape is inferred on first forward pass.
- **Architecture**: `LazyConv2d(6,5) → Sigmoid → AvgPool2d(2,2) → LazyConv2d(16,5) → Sigmoid → AvgPool2d(2,2) → Flatten → LazyLinear(120) → Sigmoid → LazyLinear(84) → Sigmoid → LazyLinear(num_classes)`
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_classes` | `INT` | 10 | Number of output classes (1~1000) |
  | `lr` | `FLOAT` | 0.1 | Learning rate (reserved parameter) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | LeNet-5 model instance |

### ResNet-18
- **Class**: `CdlResNet18`
- **d2lcore function**: `resnet18(num_classes, in_channels)`
- **Purpose**: Builds a modified ResNet-18 model (smaller kernel/stride/padding, no max-pooling). Contains 4 residual block groups (2 residual blocks each) with channel sizes 64, 128, 256, 512, followed by global average pooling and a fully connected layer.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_classes` | `INT` | 10 | Number of output classes (1~10000) |
  | `in_channels` | `INT` | 1 | Number of input channels (1~1024) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | ResNet-18 model instance |

### Residual Block
- **Class**: `CdlResidual`
- **d2lcore function**: `Residual(num_channels, use_1x1conv, strides)`
- **Purpose**: Creates a single ResNet residual block. Contains two convolutional layers (Conv2d + BatchNorm + ReLU) and an optional 1×1 shortcut convolution.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_channels` | `INT` | 64 | Output channels (1~2048) |
  | `use_1x1conv` | `BOOLEAN` | False | Enable 1×1 shortcut convolution |
  | `strides` | `INT` | 1 | Convolution stride (1~4) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `block` | `cdlModel` | Residual block module instance |

### ResNeXt Block
- **Class**: `CdlResNeXtBlock`
- **d2lcore function**: `ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv, strides)`
- **Purpose**: Creates a single ResNeXt block using grouped convolutions for multi-branch structure. Contains bottleneck structure (1×1 reduction → 3×3 grouped conv → 1×1 expansion) + optional shortcut.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_channels` | `INT` | 64 | Output channels (1~2048) |
  | `groups` | `INT` | 32 | Number of groups for grouped convolution (1~1024) |
  | `bot_mul` | `FLOAT` | 0.5 | Bottleneck channel multiplier (0.125~2.0) |
  | `use_1x1conv` | `BOOLEAN` | False | Enable shortcut convolution |
  | `strides` | `INT` | 1 | Convolution stride (1~4) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `block` | `cdlModel` | ResNeXt block module instance |

---

## 3. d2l / GAN (2 nodes)

### Update Discriminator
- **Class**: `CdlUpdateD`
- **d2lcore function**: `update_D(X, Z, net_D, net_G, loss, trainer_D)`
- **Purpose**: Performs one training update of the GAN discriminator. Uses real data `X` and fake data generated by `net_G` from noise `Z`, computes BCE loss, and backpropagates to update discriminator parameters. Uses SGD optimizer (lr=0.01).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `TENSOR` | Real data batch (optional) |
  | `Z` | `TENSOR` | Noise input (optional) |
  | `net_D` | `cdlModel` | Discriminator model (optional) |
  | `net_G` | `cdlModel` | Generator model (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `loss_D` | `FLOAT` | Discriminator loss; returns 0.0 if any input is missing |

### Update Generator
- **Class**: `CdlUpdateG`
- **d2lcore function**: `update_G(Z, net_D, net_G, loss, trainer_G)`
- **Purpose**: Performs one training update of the GAN generator. Generates fake data from noise `Z` and attempts to fool the discriminator, computes BCE loss, and backpropagates to update generator parameters. Uses SGD optimizer (lr=0.01).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `Z` | `TENSOR` | Noise input (optional) |
  | `net_D` | `cdlModel` | Discriminator model (optional) |
  | `net_G` | `cdlModel` | Generator model (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `loss_G` | `FLOAT` | Generator loss; returns 0.0 if any input is missing |
  | `fake_X` | `TENSOR` | Fake data produced by the generator |

---

## 4. ComfyUI / utilities (4 nodes)

Merged into the ComfyUI core category `utilities` (frontend group: 实用工具). Class names keep the `Cdl` prefix.

### MessageBox
- **Class**: `CdlMessageBox`
- **Purpose**: Pops up a native Windows message dialog via ctypes calling `MessageBoxW` in `user32.dll`. Supports both blocking and non-blocking modes.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `title` | `STRING` | `"ComfyDL"` | Dialog title |
  | `text` | `STRING` | `"Hello from ComfyDL!"` | Dialog message text (multiline) |
  | `button_type` | `COMBO` | `MB_OK` | Button type: MB_OK / MB_OKCANCEL / MB_ABORTRETRYIGNORE / MB_YESNOCANCEL / MB_YESNO / MB_RETRYCANCEL |
  | `icon_type` | `COMBO` | `MB_ICONINFORMATION` | Icon type: MB_ICONINFORMATION / MB_ICONWARNING / MB_ICONERROR / MB_ICONQUESTION |
  | `block` | `BOOLEAN` | True | True=blocking (wait for user close), False=non-blocking |
  | `any_input` | `*` | — | Wildcard input (optional), triggers node execution |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `result` | `STRING` | Name of the button clicked by user (e.g. `"2 (IDCANCEL)"`); returns a placeholder string on non-Windows systems |

### NoOp
- **Class**: `CdlNoOp`
- **Purpose**: A no-operation node — accepts any input and performs no computation. Equivalent to Python's ``pass`` or assembly's ``NOP``. Useful as a null sink for any data type, a placeholder during workflow construction, or a debug bypass.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `any_input` | `*` | — | Wildcard input (optional), any data type — discarded |
- **Outputs**: None

### Timer (Benchmark)
- **Class**: `CdlTimer`
- **Purpose**: Benchmarks a tensor operation by running it `num_iters` times (after a 3-iteration warm-up) and reports total and average time. Operations: `sum`, `mean`, `abs`, `sqrt`, `neg`. Synchronizes CUDA before/after timing when the tensor is on GPU.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor to operate on |
  | `operation` | `COMBO` | `sum` | Operation to benchmark: sum / mean / abs / sqrt / neg |
  | `num_iters` | `INT` | 10 | Number of timed iterations (1~100000) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `report` | `STRING` | Human-readable timing report |
  | `avg_seconds` | `FLOAT` | Average seconds per iteration |

### ?
- **Class**: `CdlWhat`
- **Purpose**: 试一试就知道了~
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `OMG` | `BOOLEAN` | False | 试一试就知道了~ |
  | `any_input` | `*` | — | Optional wildcard input — ignored |
- **Outputs**: None (output node)

---

## 5. d2l / NLP Utils (5 nodes)

### Tokenize
- **Class**: `CdlTokenize`
- **d2lcore function**: `tokenize(lines, token)`
- **Purpose**: Splits input text into tokens line by line. Supports word-level (by whitespace) and character-level tokenization modes. Each line is treated as one sentence; tokens within a line are comma-separated in the output, and lines are newline-separated.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `text` | `STRING` | `"the quick brown fox\njumps over the lazy dog"` | Input text, one sentence per line (multiline) |
  | `token_mode` | `COMBO` | `word` | Tokenization mode: `word` (whitespace split) / `char` (character-level) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tokens_str` | `STRING` | Tokenized result string; tokens comma-separated per line, lines newline-separated |

### Get Tokens & Segments
- **Class**: `CdlGetTokensAndSegments`
- **d2lcore function**: `get_tokens_and_segments(tokens_a, tokens_b)`
- **Purpose**: Prepares input for BERT models. Concatenates segment A and segment B tokens with `[CLS]` and `[SEP]` markers, and generates segment IDs (0 for A, 1 for B).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tokens_a` | `STRING` | `"the,quick,brown,fox"` | Segment A, comma-separated tokens (multiline) |
  | `tokens_b` | `STRING` | `"jumps,over"` | Segment B, comma-separated tokens (optional, multiline) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tokens` | `STRING` | Concatenated token sequence (with `[CLS]` and `[SEP]`), comma-separated |
  | `segments` | `STRING` | Segment ID sequence, comma-separated (0=A, 1=B) |

### Vocab Build
- **Class**: `CdlVocabBuild`
- **d2lcore function**: `Vocab(tokens, min_freq, reserved_tokens)`
- **Purpose**: Builds a vocabulary from token text. Counts token frequencies, retains tokens with count ≥ `min_freq`, and adds reserved tokens (e.g. `<pad>`, `<bos>`, `<eos>`). The `<unk>` token is always included automatically.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tokens_text` | `STRING` | `"the quick\nbrown fox\nthe lazy dog"` | Token text, one token per line or comma-separated (multiline) |
  | `min_freq` | `INT` | 1 | Minimum frequency threshold (1~100000) |
  | `reserved_tokens` | `STRING` | `"<pad>,<bos>,<eos>"` | Reserved tokens, comma-separated |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `vocab` | `cdlVocab` | Vocabulary dict `{"idx_to_token": [...], "token_to_idx": {...}}` |
  | `vocab_size` | `INT` | Vocabulary size |

### Vocab Encode
- **Class**: `CdlVocabEncode`
- **d2lcore function**: `Vocab.__getitem__(tokens)`
- **Purpose**: Converts token strings to index tensor using the vocabulary. Tokens not found in the vocabulary are mapped to the `<unk>` index.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `vocab` | `cdlVocab` | — | Vocabulary dictionary |
  | `tokens` | `STRING` | `"the,quick,brown"` | Comma-separated token sequence (multiline) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `indices` | `TENSOR` | Encoded index tensor (`torch.int64`) |

### Vocab Decode
- **Class**: `CdlVocabDecode`
- **d2lcore function**: `Vocab.to_tokens(indices)`
- **Purpose**: Converts index tensor back to token strings using the vocabulary. Out-of-range indices are mapped to `<unk>`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `vocab` | `cdlVocab` | Vocabulary dictionary |
  | `indices` | `TENSOR` | Index tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tokens_str` | `STRING` | Decoded token string, comma-separated |

---

## 6. d2l / NLP Models (13 nodes)

NLP model builder nodes wrap the d2lcore RNN/GRU/RNNLM, attention/Transformer and Seq2Seq building blocks. All builders return a `cdlModel` that can be wired into `CdlModelForward` / `CdlModelInfo` / `CdlModelSave` etc. for inspection and inference. RNN/GRU forwards expect time-major inputs `(num_steps, batch_size, num_inputs)`; attention modules and the Transformer encoder expect batch-first inputs.

### RNN (from scratch)
- **Class**: `CdlRNNScratch`
- **d2lcore function**: `RNNScratch(num_inputs, num_hiddens, sigma)`
- **Purpose**: Builds an RNN from scratch (tanh cell, manually created parameters). Forward expects `(num_steps, batch_size, num_inputs)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_inputs` | `INT` | 32 | Input feature size (vocab size for LMs) (1~100000) |
  | `num_hiddens` | `INT` | 64 | Number of hidden units (1~4096) |
  | `sigma` | `FLOAT` | 0.01 | Std of random parameter initialization (0.0001~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | RNNScratch instance |

### RNN (high-level)
- **Class**: `CdlRNN`
- **d2lcore function**: `RNN(num_inputs, num_hiddens)`
- **Purpose**: Builds an RNN using PyTorch's high-level `nn.RNN`. Forward expects `(num_steps, batch_size, num_inputs)` and returns `(output, h_n)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_inputs` | `INT` | 32 | Input feature size (1~100000) |
  | `num_hiddens` | `INT` | 64 | Number of hidden units (1~4096) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | RNN instance |

### GRU
- **Class**: `CdlGRU`
- **d2lcore function**: `GRU(num_inputs, num_hiddens, num_layers, dropout)`
- **Purpose**: Builds a multilayer GRU using PyTorch's high-level `nn.GRU`. Forward expects `(num_steps, batch_size, num_inputs)` and returns `(output, h_n)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_inputs` | `INT` | 32 | Input feature size (1~100000) |
  | `num_hiddens` | `INT` | 64 | Hidden units per layer (1~4096) |
  | `num_layers` | `INT` | 1 | Number of stacked GRU layers (1~20) |
  | `dropout` | `FLOAT` | 0.0 | Dropout between layers (0~0.9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | GRU instance |

### RNN Language Model (from scratch)
- **Class**: `CdlRNNLMScratch`
- **d2lcore function**: `RNNLMScratch(rnn, vocab_size, lr)`
- **Purpose**: Wraps an RNN/GRU `cdlModel` (with `num_inputs == vocab_size`) into a from-scratch language model with an output projection to `vocab_size` classes. Forward takes an index tensor `(batch_size, num_steps)` and returns logits `(num_steps, batch_size, vocab_size)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `rnn` | `cdlModel` | — | RNN/GRU cdlModel with `num_inputs == vocab_size` |
  | `vocab_size` | `INT` | 32 | Vocabulary size of the output projection (2~100000) |
  | `lr` | `FLOAT` | 0.01 | Learning rate used when training (0.0001~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | RNNLMScratch instance |

### RNN Language Model (high-level)
- **Class**: `CdlRNNLM`
- **d2lcore function**: `RNNLM(rnn, vocab_size, lr)`
- **Purpose**: Wraps an RNN/GRU `cdlModel` into a language model with a high-level `LazyLinear` head. Same interface as the from-scratch version.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `rnn` | `cdlModel` | — | RNN/GRU cdlModel with `num_inputs == vocab_size` |
  | `vocab_size` | `INT` | 32 | Vocabulary size of the output projection (2~100000) |
  | `lr` | `FLOAT` | 0.01 | Learning rate used when training (0.0001~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | RNNLM instance |

### RNN LM Predict
- **Class**: `CdlRNNLMScratchPredict`
- **d2lcore function**: `RNNLMScratch.predict(prefix, num_preds, vocab, device)`
- **Purpose**: Generates text with an RNN language model given a starting prefix. Accepts a `cdlVocab` dict (from `CdlVocabBuild`) and returns the prefix followed by `num_preds` predicted tokens.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `model` | `cdlModel` | — | An RNNLMScratch / RNNLM cdlModel |
  | `vocab` | `cdlVocab` | — | Vocabulary dict from `CdlVocabBuild` |
  | `prefix` | `STRING` | `"the "` | Starting token(s), e.g. `"the "` |
  | `num_preds` | `INT` | 10 | Number of tokens to predict after prefix (1~1000) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `prediction` | `STRING` | Prefix followed by predicted tokens |

### Dot-Product Attention
- **Class**: `CdlDotProductAttention`
- **d2lcore function**: `DotProductAttention(dropout)`
- **Purpose**: Builds a scaled dot-product attention layer. Forward `(queries, keys, values, valid_lens)` with batch-first tensors `(batch, seq, num_hiddens)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `dropout` | `FLOAT` | 0.0 | Dropout on attention weights (0~0.9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Attention layer |

### Additive Attention
- **Class**: `CdlAdditiveAttention`
- **d2lcore function**: `AdditiveAttention(num_hiddens, dropout)`
- **Purpose**: Builds an additive (Bahdanau) attention layer. Forward `(queries, keys, values, valid_lens)` with batch-first tensors.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_hiddens` | `INT` | 8 | Hidden units of the additive score function (1~4096) |
  | `dropout` | `FLOAT` | 0.0 | Dropout on attention weights (0~0.9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Attention layer |

### Multi-Head Attention
- **Class**: `CdlMultiHeadAttention`
- **d2lcore function**: `MultiHeadAttention(num_hiddens, num_heads, dropout, bias)`
- **Purpose**: Builds a multi-head attention layer. `num_hiddens` must be divisible by `num_heads`. Forward `(queries, keys, values, valid_lens)` with batch-first tensors.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_hiddens` | `INT` | 8 | Model width; divisible by `num_heads` (1~4096) |
  | `num_heads` | `INT` | 4 | Number of parallel attention heads (1~64) |
  | `dropout` | `FLOAT` | 0.0 | Dropout on attention weights (0~0.9) |
  | `use_bias` | `BOOLEAN` | False | Whether Q/K/V/O projections use bias |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Multi-head attention layer |

### Positional Encoding
- **Class**: `CdlPositionalEncoding`
- **d2lcore function**: `PositionalEncoding(num_hiddens, dropout, max_len)`
- **Purpose**: Builds a sinusoidal positional encoding layer. Forward `X` of shape `(batch, seq, num_hiddens)` adds positional information.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_hiddens` | `INT` | 16 | Model width (feature dimension) (1~4096) |
  | `dropout` | `FLOAT` | 0.0 | Dropout after adding encoding (0~0.9) |
  | `max_len` | `INT` | 1000 | Maximum supported sequence length (1~100000) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Positional encoding layer |

### Position-Wise FFN
- **Class**: `CdlPositionWiseFFN`
- **d2lcore function**: `PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)`
- **Purpose**: Builds a position-wise feed-forward network (two dense layers + ReLU), applied identically to each position.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `ffn_num_hiddens` | `INT` | 64 | Hidden units of the inner dense layer (1~16384) |
  | `ffn_num_outputs` | `INT` | 16 | Output units, usually == `num_hiddens` (1~16384) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | FFN module |

### Add & Norm
- **Class**: `CdlAddNorm` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/NLP Models`)
- **Replacement**: on a bare `TENSOR` graph, compose `NormalizationLayerNorm` after `BasicAdd` — `LayerNorm(dropout(Y) + X)` is exactly the same math. This node operates on a module-level `cdlModel` rather than a `TENSOR`, so it is unchanged and still usable in `cdlModel` pipelines.
- **d2lcore function**: `AddNorm(norm_shape, dropout)`
- **Purpose**: Builds a residual connection followed by layer normalization: `LayerNorm(dropout(Y) + X)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `norm_shape` | `INT` | 16 | Feature dimension for LayerNorm (1~16384) |
  | `dropout` | `FLOAT` | 0.0 | Dropout on the residual branch (0~0.9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Add & Norm block |

### Transformer Encoder Block
- **Class**: `CdlTransformerEncoderBlock` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/NLP Models`)
- **Replacement**: on a bare `TENSOR` graph, compose the core layer nodes (multi-head attention + `BasicAdd` before `NormalizationLayerNorm` + a position-wise FFN built from `BasicLinear`). This node builds a module-level `cdlModel` rather than a `TENSOR`, so it is unchanged and still usable in `cdlModel` pipelines.
- **d2lcore function**: `TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias)`
- **Purpose**: Builds a single Transformer encoder block (multi-head attention + FFN with add & norm). Forward `(X, valid_lens)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_hiddens` | `INT` | 8 | Model width; divisible by `num_heads` (1~4096) |
  | `ffn_num_hiddens` | `INT` | 64 | Hidden units of the position-wise FFN (1~16384) |
  | `num_heads` | `INT` | 4 | Number of attention heads (1~64) |
  | `dropout` | `FLOAT` | 0.0 | Dropout probability (0~0.9) |
  | `use_bias` | `BOOLEAN` | False | Whether attention projections use bias |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Transformer encoder block |

### Transformer Encoder
- **Class**: `CdlTransformerEncoder` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/NLP Models`)
- **Replacement**: on a bare `TENSOR` graph, compose the core layer nodes (embedding + positional encoding + stacked encoder blocks). This node builds a module-level `cdlModel` rather than a `TENSOR`, so it is unchanged and still usable in `cdlModel` pipelines.
- **d2lcore function**: `TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias)`
- **Purpose**: Builds a full Transformer encoder (embedding + positional encoding + `num_blks` stacked blocks). Forward `(X, valid_lens)` with `X` of token indices `(batch, seq)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `vocab_size` | `INT` | 32 | Vocabulary size of input embeddings (2~100000) |
  | `num_hiddens` | `INT` | 8 | Model width; divisible by `num_heads` (1~4096) |
  | `ffn_num_hiddens` | `INT` | 64 | Hidden units of the position-wise FFN (1~16384) |
  | `num_heads` | `INT` | 4 | Number of attention heads (1~64) |
  | `num_blks` | `INT` | 2 | Number of stacked encoder blocks (1~50) |
  | `dropout` | `FLOAT` | 0.0 | Dropout probability (0~0.9) |
  | `use_bias` | `BOOLEAN` | False | Whether attention projections use bias |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Transformer encoder |

### Seq2Seq Encoder
- **Class**: `CdlSeq2SeqEncoder`
- **d2lcore function**: `Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)`
- **Purpose**: Builds an RNN encoder for sequence-to-sequence learning (embedding + multilayer GRU). Forward `X` of token indices `(batch, num_steps)` returns `(outputs, state)`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `vocab_size` | `INT` | 32 | Vocabulary size of source embeddings (2~100000) |
  | `embed_size` | `INT` | 16 | Embedding dimension (1~4096) |
  | `num_hiddens` | `INT` | 16 | Hidden units of each GRU layer (1~4096) |
  | `num_layers` | `INT` | 2 | Number of stacked GRU layers (1~20) |
  | `dropout` | `FLOAT` | 0.0 | Dropout between GRU layers (0~0.9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Seq2Seq encoder |

### Init Seq2Seq Weights
- **Class**: `CdlInitSeq2Seq`
- **d2lcore function**: `init_seq2seq(module)`
- **Purpose**: Applies Xavier-uniform weight initialization in place. `nn.Linear` and `nn.GRU` layers get initialized weights; other layers are left untouched.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` to initialize |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | The same model after in-place initialization |

---

## 7. d2l / Tensor Basic (5 nodes)

### Tensor → String
- **Class**: `CdlTensorToStr`
- **Purpose**: Formats a tensor as a human-readable string. Displays the tensor's shape, dtype, device, and numeric content. Large tensors exceeding `max_elems` are truncated (showing first half + second half).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor |
  | `max_elems` | `INT` | 100 | Maximum elements to display (10~10000) |
  | `precision` | `INT` | 6 | Numeric display precision (1~16 digits) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `text` | `STRING` | Formatted string (shape, dtype, device, values) |

### String → Tensor
- **Class**: `CdlStrToTensor`
- **Purpose**: Parses a string into a PyTorch tensor. Supports standard Python list literal format, e.g. `"[[1,2],[3,4]]"`, `"[1,2,3,4,5]"`. Handles whitespace and trailing commas automatically.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `text` | `STRING` | `"[[1, 2, 3], [4, 5, 6]]"` | String representation of tensor (multiline), e.g. `"[[1,2],[3,4]]"` |
  | `error_strategy` | `COMBO` | `empty_tensor` | Error handling: `empty_tensor`=return empty tensor; `zero_tensor`=return `[0.]`; `raise_error`=raise exception |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tensor` | `TENSOR` | Parsed tensor (`torch.float32`) |

### Conv2D
- **Class**: `CdlConv2d`
- **Purpose**: Performs 2D convolution on an input tensor with a kernel. Wraps ``torch.nn.functional.conv2d`` with configurable stride and padding. Auto-expands 2-D/3-D inputs to 4-D ``(N, C, H, W)`` and squeezes output back.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `input_tensor` | `TENSOR` | — | Input tensor |
  | `kernel` | `TENSOR` | — | Convolution kernel |
  | `stride` | `INT` | 1 | Convolution stride (1~4) |
  | `padding` | `INT` | 0 | Zero padding (0~10) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Convolved result |

### Transpose
- **Class**: `CdlTranspose`
- **Purpose**: Swaps two dimensions of a tensor. Wraps ``torch.transpose``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor |
  | `dim0` | `INT` | 0 | First dimension to swap (0~5) |
  | `dim1` | `INT` | 1 | Second dimension to swap (0~5) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Transposed tensor |

### Broadcast
- **Class**: `CdlBroadcast` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/Tensor Basic`)
- **Replacement**: the core `BasicBroadcast` node (`Network & Layers/Basic`) — the same `torch.broadcast_to`, operating directly on `TENSOR`.
- **Purpose**: Broadcasts a tensor to a target shape. Wraps ``torch.broadcast_to``. Enter the target shape as a comma-separated string (e.g. ``"3,1,4"``). Returns original tensor unchanged on parse error.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor |
  | `target_shape` | `STRING` | `"2,3"` | Target shape, comma-separated (e.g. ``"3,1,4"``) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Broadcasted tensor |

### Reshape
- **Class**: `CdlReshape` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/Tensor Basic`)
- **Replacement**: the core `BasicReshape` node (`Network & Layers/Basic`) — the same `torch.reshape`, operating directly on `TENSOR`.
- **Purpose**: Reshapes a tensor to a new shape. Wraps ``torch.reshape``. Enter the target shape as a comma-separated string (e.g. ``"2,8"``, ``"4,-1"``). Returns original tensor unchanged on parse error.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor |
  | `target_shape` | `STRING` | `"3,2"` | Target shape, comma-separated (e.g. ``"2,8"``) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Reshaped tensor |

### Activation
- **Class**: `CdlActivation` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/Tensor Basic`)
- **Replacement**: the 14 core `Activation*` nodes (`Network & Layers/Activation`) cover every function offered here and five more (`selu`, `mish`, `relu6`, `hardswish`, `identity`), one node per operation.
- **Purpose**: Applies an element-wise activation function to a tensor. Select the function from a combo widget: ``relu``, ``sigmoid``, ``tanh``, ``leaky_relu``, ``elu``, ``gelu``, ``silu``, ``softmax``, ``softplus``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor |
  | `func` | `COMBO` | `relu` | Activation: relu / sigmoid / tanh / leaky_relu / elu / gelu / silu / softmax / softplus |
  | `dim` | `INT` | -1 | Dimension for softmax (-4~4) |
  | `negative_slope` | `FLOAT` | 0.01 | Slope for leaky_relu (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Activated tensor |

### Random Tensor
- **Class**: `CdlRandomTensor`
- **Purpose**: Generates a random tensor with a chosen distribution (`normal`, `uniform`, `randint`). `seed >= 0` fixes the RNG for reproducible results; `-1` leaves it random.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `shape` | `STRING` | `"4,4"` | Output shape, comma-separated dims (e.g. `"4,4"`) |
  | `dist` | `COMBO` | `normal` | Distribution: normal / uniform / randint |
  | `mean` | `FLOAT` | 0.0 | Mean for `normal` |
  | `std` | `FLOAT` | 1.0 | Std deviation for `normal` |
  | `low` | `FLOAT` | 0.0 | Lower bound for `uniform` / `randint` (inclusive) |
  | `high` | `FLOAT` | 1.0 | Upper bound for `uniform` / `randint` (exclusive) |
  | `seed` | `INT` | -1 | Random seed (>= 0 fixes RNG, -1 = random) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tensor` | `TENSOR` | Random tensor of the requested shape |

---

## 8. d2l / TorchOps (10 nodes)

### Linear Regression
- **Class**: `CdlLinReg`
- **d2lcore function**: `linreg(X, w, b)`
- **Purpose**: Linear regression forward computation: \( \hat{y} = X w + b \)
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `TENSOR` | Input feature matrix |
  | `w` | `TENSOR` | Weight vector |
  | `b` | `TENSOR` | Bias vector/scalar |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `TENSOR` | Predicted values |

### Squared Loss
- **Class**: `CdlSquaredLoss`
- **d2lcore function**: `squared_loss(y_hat, y)`
- **Purpose**: Computes squared loss: \( \frac{1}{2}(\hat{y} - y)^2 \) (note: no averaging).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `TENSOR` | Predicted values |
  | `y` | `TENSOR` | Ground truth values (auto-reshaped to match `y_hat` shape) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `loss` | `TENSOR` | Element-wise loss |

### Masked Softmax
- **Class**: `CdlMaskedSoftmax`
- **d2lcore function**: `masked_softmax(X, valid_lens)`
- **Purpose**: Performs softmax with masking on the last dimension. Uses `valid_lens` to specify the effective length of each sequence; positions beyond the valid length are set to -1e6 before softmax (making their probability near 0).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `TENSOR` | Input tensor |
  | `valid_lens` | `TENSOR` | Valid length tensor (optional; if not provided, normal softmax is performed) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | Masked softmax result |

### Sequence Mask
- **Class**: `CdlSequenceMask`
- **d2lcore function**: `sequence_mask(X, valid_len, value)`
- **Purpose**: Replaces positions in a sequence that exceed the valid length with a specified value. Commonly used to zero out padding positions.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `X` | `TENSOR` | — | Input sequence tensor |
  | `valid_len` | `TENSOR` | — | Effective length for each sequence |
  | `mask_value` | `FLOAT` | 0.0 | Mask fill value (-1e9~1e9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `masked` | `TENSOR` | Masked tensor |

### Accuracy
- **Class**: `CdlAccuracy`
- **d2lcore function**: `accuracy(y_hat, y)`
- **Purpose**: Computes the number of correct predictions for classification. If the prediction is multi-class logits, argmax is taken first, then compared against labels.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `TENSOR` | Predictions (logits or class indices) |
  | `y` | `TENSOR` | Ground truth labels |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `accuracy` | `FLOAT` | Number of correct predictions (float) |
  | `count` | `INT` | Number of correct predictions (int) |

### Synthetic Data
- **Class**: `CdlSyntheticData`
- **d2lcore function**: `synthetic_data(w, b, num_examples)`
- **Purpose**: Generates a synthetic linear regression dataset. Randomly generates weights and features, produces labels via \( y = Xw + b + \text{noise} \).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_features` | `INT` | 2 | Feature dimension (1~1000) |
  | `num_examples` | `INT` | 100 | Number of samples (1~1000000) |
  | `noise_std` | `FLOAT` | 0.01 | Noise standard deviation (0~10.0) |
  | `seed` | `INT` | 0 | Random seed (0~99999) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `TENSOR` | Feature matrix `[num_examples, num_features]` |
  | `y` | `TENSOR` | Label vector `[num_examples, 1]` |

### Truncate/Pad
- **Class**: `CdlTruncatePad`
- **d2lcore function**: `truncate_pad(line, num_steps, padding_token)`
- **Purpose**: Truncates or pads a sequence to a fixed length. Sequences longer than the target are truncated; shorter ones are padded with `padding_token`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_steps` | `INT` | 64 | Target sequence length (1~10000) |
  | `padding_token` | `INT` | 0 | Padding token index (0~100000) |
  | `sequence` | `TENSOR` | — | Input index sequence (optional; returns all-padding tensor when absent) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `padded` | `TENSOR` | Truncated/padded sequence (`torch.int64`) |

### BLEU Score
- **Class**: `CdlBleu`
- **d2lcore function**: `bleu(pred_seq, label_seq, k)`
- **Purpose**: Computes the BLEU score between a predicted sequence and a reference sequence. Supports BLEU-1 through BLEU-4 (controlled by `max_n`). Tokenizes by whitespace, then computes n-gram precision and brevity penalty.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `pred_seq` | `STRING` | `"the quick brown"` | Predicted sequence, whitespace-separated tokens (multiline) |
  | `label_seq` | `STRING` | `"the quick brown fox"` | Reference sequence, whitespace-separated tokens (multiline) |
  | `max_n` | `INT` | 4 | Maximum n-gram order (1~4) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `bleu_score` | `FLOAT` | BLEU score (0~1) |

### Gradient Clip
- **Class**: `CdlGradClipping`
- **d2lcore function**: `grad_clipping(net, theta)`
- **Purpose**: Performs gradient clipping on model parameters. **Prerequisite**: gradients must have been computed via `loss.backward()`. Computes the L2 norm of all parameter gradients; if it exceeds threshold `theta`, scales them proportionally.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `theta` | `FLOAT` | 1.0 | Gradient clipping threshold (0.1~100.0) |
  | `model` | `cdlModel` | — | Model whose gradients to clip (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `norm` | `FLOAT` | Total gradient norm before clipping; returns 0.0 if no model |

### SGD Step
- **Class**: `CdlSgdStep`
- **d2lcore function**: `sgd(params, lr, batch_size)`
- **Purpose**: Performs one step of mini-batch stochastic gradient descent: \( \theta \leftarrow \theta - \eta \cdot g / \text{batch\_size} \). **Prerequisite**: gradients must have been computed via `loss.backward()`. Gradients are zeroed after the update.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `lr` | `FLOAT` | 0.03 | Learning rate (1e-8 ~ 10.0) |
  | `batch_size` | `INT` | 32 | Batch size (1~65536) |
  | `model` | `cdlModel` | — | Model to update (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Updated model; None if no model provided |

---

## 9. d2l / ObjectDetection (10 nodes)

### Box Corner→Center
- **Class**: `CdlBoxCornerToCenter`
- **d2lcore function**: `box_corner_to_center(boxes)`
- **Purpose**: Converts bounding boxes from corner format `(x1, y1, x2, y2)` to center format `(cx, cy, w, h)`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes` | `TENSOR` | Corner-format boxes `[N,4]` (top-left + bottom-right) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes_ccwh` | `TENSOR` | Center-format boxes `[N,4]` (center + width + height) |

### Box Center→Corner
- **Class**: `CdlBoxCenterToCorner`
- **d2lcore function**: `box_center_to_corner(boxes)`
- **Purpose**: Converts bounding boxes from center format `(cx, cy, w, h)` back to corner format `(x1, y1, x2, y2)`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes` | `TENSOR` | Center-format boxes `[N,4]` (center + width + height) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes_xyxy` | `TENSOR` | Corner-format boxes `[N,4]` (top-left + bottom-right) |

### Box IoU
- **Class**: `CdlBoxIou`
- **d2lcore function**: `box_iou(boxes1, boxes2)`
- **Purpose**: Computes pairwise IoU (Intersection over Union) between two sets of bounding boxes. Returns matrix `[N1, N2]` where element `(i,j)` is the IoU of `boxes1[i]` and `boxes2[j]`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes1` | `TENSOR` | First set of boxes `[N1,4]` (top-left + bottom-right) |
  | `boxes2` | `TENSOR` | Second set of boxes `[N2,4]` (top-left + bottom-right) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `iou` | `TENSOR` | IoU matrix `[N1, N2]` |

### NMS
- **Class**: `CdlNms`
- **d2lcore function**: `nms(boxes, scores, iou_threshold)`
- **Purpose**: Performs Non-Maximum Suppression on bounding boxes. Sorts by score descending, keeps boxes whose IoU with any higher-scoring box does not exceed the threshold.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `boxes` | `TENSOR` | — | Boxes `[N,4]` (top-left + bottom-right) |
  | `scores` | `TENSOR` | — | Confidence scores for each box |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `keep_indices` | `TENSOR` | Indices of kept boxes (`torch.int64`) |

### Multibox Prior
- **Class**: `CdlMultiboxPrior`
- **d2lcore function**: `multibox_prior(data, sizes, ratios)`
- **Purpose**: Generates anchor boxes of different shapes centered at each pixel. Number of anchors per pixel = `len(sizes) + len(ratios) - 1`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `sizes` | `STRING` | `"0.75,0.5,0.25"` | Anchor size list, comma-separated |
  | `ratios` | `STRING` | `"1,2,0.5"` | Aspect ratio list, comma-separated |
  | `data` | `TENSOR` | — | Input data (optional; used to infer spatial dimensions; defaults to 561×728 when absent) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `TENSOR` | Anchors `[1, H*W*bpp, 4]`, normalized coordinates (top-left + bottom-right) |

### Offset Boxes
- **Class**: `CdlOffsetBoxes`
- **d2lcore function**: `offset_boxes(anchors, assigned_bb, eps)`
- **Purpose**: Computes offset from anchor boxes to assigned ground-truth boxes. Center coordinate differences are scaled by 10×; width/height ratios are log-scaled by 5×.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `anchors` | `TENSOR` | — | Anchors `[N,4]` (top-left + bottom-right) |
  | `assigned_bb` | `TENSOR` | — | Assigned ground-truth boxes `[N,4]` (top-left + bottom-right) |
  | `eps` | `FLOAT` | 1e-6 | Small epsilon to prevent division by zero (1e-12~1e-3) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `offsets` | `TENSOR` | Offsets `[N,4]` (dx, dy, dw, dh) |

### Offset Inverse
- **Class**: `CdlOffsetInverse`
- **d2lcore function**: `offset_inverse(anchors, offset_preds)`
- **Purpose**: Reconstructs bounding box coordinates (corner format) from anchors and predicted offsets via inverse transformation.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `TENSOR` | Anchors `[N,4]` (top-left + bottom-right) |
  | `offset_preds` | `TENSOR` | Predicted offsets `[N,4]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `predicted_bbox` | `TENSOR` | Predicted boxes `[N,4]` (top-left + bottom-right) |

### Assign Anchor→BBox
- **Class**: `CdlAssignAnchorToBbox`
- **d2lcore function**: `assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)`
- **Purpose**: Assigns ground-truth bounding boxes to anchor boxes based on IoU. Each anchor is assigned to a ground-truth box (IoU ≥ threshold), and each ground-truth box is guaranteed at least one anchor (the one with the highest IoU).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `ground_truth` | `TENSOR` | — | Ground-truth boxes `[M,4]` (top-left + bottom-right) |
  | `anchors` | `TENSOR` | — | Anchors `[N,4]` (top-left + bottom-right) |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors_bbox_map` | `TENSOR` | Anchor→ground-truth mapping `[N,]`, -1 means no match (`torch.int64`) |

### Multibox Target
- **Class**: `CdlMultiboxTarget`
- **d2lcore function**: `multibox_target(anchors, labels)`
- **Purpose**: Generates multi-box target training labels for anchors. For each image in the batch, assigns ground-truth boxes to anchors and computes offset targets, masks, and class labels.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `TENSOR` | Anchors `[1, N, 4]` (top-left + bottom-right) |
  | `labels` | `TENSOR` | Labels `[B, M, 5]`, format `[class_id, x1, y1, x2, y2]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `bbox_offset` | `TENSOR` | Bounding box offset targets `[B, N*4]` |
  | `bbox_mask` | `TENSOR` | Bounding box offset masks `[B, N*4]` (1.0 for matched anchors) |
  | `class_labels` | `TENSOR` | Anchor class labels `[B, N]` (background=0, classes start from 1) |

### Multibox Detection
- **Class**: `CdlMultiboxDetection`
- **d2lcore function**: `multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)`
- **Purpose**: Predicts bounding boxes from model outputs using NMS. Combines class predictions and offsets, reconstructs boxes via inverse offset transform, and filters through NMS and confidence thresholding.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `cls_probs` | `TENSOR` | — | Class probabilities `[B, num_classes, N]` |
  | `offset_preds` | `TENSOR` | — | Offset predictions `[B, N*4]` |
  | `anchors` | `TENSOR` | — | Anchors `[1, N, 4]` |
  | `nms_threshold` | `FLOAT` | 0.5 | NMS IoU threshold (0~1) |
  | `pos_threshold` | `FLOAT` | 0.01 | Positive confidence threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `detections` | `TENSOR` | Detection results `[B, N, 6]`, format `[class_id, confidence, x1, y1, x2, y2]` (class_id=-1 means background) |

---

## 10. d2l / Segmentation (4 nodes)

### VOC Classes
- **Class**: `CdlVocClasses`
- **d2lcore function**: `VOC_CLASSES` constant
- **Purpose**: Retrieves PASCAL VOC 21 class names. A single index query returns that class name; index `-1` returns all 21 classes as a comma-separated list.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `index` | `INT` | -1 | Class index (-1=all, 0=background, ..., 20=tv/monitor) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `class_names` | `STRING` | Class name(s) (single or comma-separated full list) |

The 21 classes: `background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor`

### VOC Colormap→Label
- **Class**: `CdlVocColormap2Label`
- **d2lcore function**: `voc_colormap2label()`
- **Purpose**: Builds a VOC RGB color → class index lookup table. Output is a tensor of size \( 256^3 \); the class index can be looked up by color encoding (R×65536 + G×256 + B).
- **Inputs**: None
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `colormap2label` | `TENSOR` | Color→class index lookup table `[16777216]` (`torch.int64`) |

### VOC Label Indices
- **Class**: `CdlVocLabelIndices`
- **d2lcore function**: `voc_label_indices(colormap, colormap2label)`
- **Purpose**: Maps a VOC label color image to a class index map. Encodes RGB pixels as a single color value, then converts to class indices via the lookup table.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `colormap` | `IMAGE` | VOC label color image `[B, H, W, C]`, uses the first image |
  | `colormap2label` | `TENSOR` | Color→label lookup table |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `label_mask` | `MASK` | Class index map `[H, W]` (float type) |

### VOC Random Crop
- **Class**: `CdlVocRandCrop`
- **d2lcore function**: `voc_rand_crop(feature, label, height, width)`
- **Purpose**: Performs synchronized random cropping on feature and label images. Uses the same random crop parameters to keep feature and label aligned. Falls back to center crop if the requested size exceeds the image size.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `feature` | `IMAGE` | — | Feature image `[B, H, W, C]` |
  | `label` | `IMAGE` | — | Label image `[B, H, W, C]` |
  | `height` | `INT` | 320 | Crop height (1~4096, step 32) |
  | `width` | `INT` | 480 | Crop width (1~4096, step 32) |
  | `seed` | `INT` | 0 | Random seed (optional, 0~999999) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `cropped_feature` | `IMAGE` | Cropped feature image |
  | `cropped_label` | `IMAGE` | Cropped label image |

---

## 11. d2l / Visualization (13 nodes)

Visualization nodes follow a "dual variant" design pattern: `(Output)` suffix versions are ComfyUI output nodes (showing interactive plots directly in the UI), while non-suffix versions render plots as `IMAGE` tensors for downstream nodes.

### Show Images
- **Class**: `CdlShowImages`
- **d2lcore function**: Same as above
- **Purpose**: Same as `Show Images (Output)`, but renders the plot as an IMAGE tensor, passable to downstream nodes.
- **Inputs**: Same as above
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered grid image `[1, H, W, C]` |

### Show Heatmaps (Output)
- **Class**: `CdlShowHeatmapsOutput`
- **d2lcore function**: `show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)`
- **Purpose**: Displays one heatmap with a colorbar. The tensor is first sampled down to an element budget, the axes named by `dims` (or picked by `auto_select`) are kept while every other axis is collapsed by `axis_reduce`, and the surviving 1/2/3 axes decide what is drawn: a bar-code like colour strip (1-D), a normal plane (2-D) or a translucent cube (3-D). Output node variant.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `matrices` | `TENSOR` | — | Any-rank tensor; sampled to `max_samples` elements, reduced to 1~3 axes, then rendered |
  | `xlabel` | `STRING` | `""` | X-axis label |
  | `ylabel` | `STRING` | `""` | Y-axis label |
  | `figsize_w` | `FLOAT` | 2.5 | Figure width (0.5~20.0) |
  | `figsize_h` | `FLOAT` | 2.5 | Figure height (0.5~20.0) |
  | `cmap` | `STRING` | `"Reds"` | matplotlib colormap name (unknown names fall back to `Reds`) |
  | `titles` | `STRING` | `""` | Figure title (optional) |
  | `max_samples` | `INT` | 262144 | Element budget; `0` never samples |
  | `dims` | `STRING` | `"auto"` | `auto`, or 1~3 axis indices such as `0,1` / `-2,-1` |
  | `axis_reduce` | `COMBO` | `mean` | How dropped axes are collapsed: `mean` / `max` / `first` / `mid` |
  | `auto_select` | `COMBO` | `last_n` | Which axes `auto` keeps: `first_n` / `last_n` / `most_informative_n` / `least_informative_n` (information ≈ axis length) |
  | `auto_n` | `INT` | 0 | How many axes `auto` keeps (0 = follow the input rank, capped at 3) |
  | `on_error` | `COMBO` | `fallback_first_n` | Advanced. `error` raises `HeatmapSpecError`, `fallback_first_n` keeps going with the leading axes |
- **Outputs**: None (output node)
- **Behaviour change**: the old "grid of sub-plots" layout is gone — one figure, one heatmap. A 2-D input with `max_samples=0` and `dims=auto` is still pixel-identical to the old rendering.

### Show Heatmaps
- **Class**: `CdlShowHeatmaps`
- **d2lcore function**: Same as above
- **Purpose**: Same as `Show Heatmaps (Output)`, but renders the plot as an IMAGE tensor.
- **Inputs**: Same as above
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered heatmap `[1, H, W, C]` |

### Heatmaps to 3D
- **Class**: `CdlHeatmapsTo3D`
- **d2lcore function**: `show_heatmaps(matrices, ...)` (the same front end), rendered as geometry instead of pixels
- **Purpose**: Exports the sampled / reduced heatmap as a real 3D model. 1-D becomes a flat colour ribbon with a real thickness, 2-D a coloured plate of the same thickness, 3-D a translucent cube (six outer faces plus the three orthogonal mid-planes so the interior stays readable). Colour is quantised into 64 OBJ materials and the alpha lives in the MTL, so the cube stays see-through. Connect `model_3d` to the built-in **Preview3D** node to look at it.
- **Note**: After the `max_samples` budget, the mesh is strided once more to a polygon budget (1-D: 256 cells, 2-D: 64 per axis, 3-D: 32 per axis). The caption ("1D"/"2D"/"3D") is drawn as geometry, because a 3D file cannot carry text.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `matrices` | `TENSOR` | — | Same front end as Show Heatmaps: sampled, axis-selected, reduced |
  | `cmap` | `STRING` | `"Reds"` | matplotlib colormap name |
  | `opacity` | `FLOAT` | 0.6 | Material alpha (`d` in the MTL); below 1.0 the cube interior stays visible |
  | `thickness` | `FLOAT` | 0.15 | Thickness of the 1-D / 2-D ribbon, in cell units |
  | `max_samples` | `INT` | 262144 | Element budget; `0` never samples |
  | `dims` | `STRING` | `"auto"` | `auto`, or 1~3 axis indices |
  | `axis_reduce` | `COMBO` | `mean` | How dropped axes are collapsed |
  | `auto_select` | `COMBO` | `last_n` | Which axes `auto` keeps |
  | `auto_n` | `INT` | 0 | How many axes `auto` keeps |
  | `on_error` | `COMBO` | `fallback_first_n` | Advanced. `error` raises, `fallback_first_n` recovers |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model_3d` | `FILE_3D_OBJ` | OBJ mesh + sibling MTL; connect it to `Preview3D` |

### Plot
- **Class**: `CdlPlot`
- **d2lcore function**: `plot(X, Y, xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, figsize, axes)`
- **Purpose**: General-purpose MATLAB-style line plot. Supports multiple curves, custom axis labels, log/linear scales, legend, and axis limits.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `xlabel` | `STRING` | `"x"` | X-axis label |
  | `ylabel` | `STRING` | `"y"` | Y-axis label |
  | `xscale` | `COMBO` | `linear` | X-axis scale: `linear` / `log` |
  | `yscale` | `COMBO` | `linear` | Y-axis scale: `linear` / `log` |
  | `figsize_w` | `FLOAT` | 6.0 | Figure width (1.0~30.0) |
  | `figsize_h` | `FLOAT` | 4.0 | Figure height (1.0~30.0) |
  | `X` | `TENSOR` | — | X-axis data (optional, 1D or 2D) |
  | `Y` | `TENSOR` | — | Y-axis data (optional) |
  | `legend` | `STRING` | `""` | Legend labels, comma-separated (optional) |
  | `xlim_min` | `FLOAT` | -1.0 | X-axis lower bound (only effective when xlim_min < xlim_max) |
  | `xlim_max` | `FLOAT` | -1.0 | X-axis upper bound |
  | `ylim_min` | `FLOAT` | -1.0 | Y-axis lower bound (only effective when ylim_min < ylim_max) |
  | `ylim_max` | `FLOAT` | -1.0 | Y-axis upper bound |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered plot `[1, H, W, C]` |

### Show Trace 2D
- **Class**: `CdlShowTrace2D`
- **d2lcore function**: `show_trace_2d(f, results)`
- **Purpose**: Visualizes a 2D optimization trajectory. Plots a sequence of points showing how parameters (x1, x2) change during optimization.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `results` | `TENSOR` | Optimization trajectory points `[N, 2]`, each row is a (x1, x2) coordinate |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered trajectory plot `[1, H, W, C]` |

### Show BBoxes
- **Class**: `CdlShowBboxes`
- **d2lcore function**: `show_bboxes(axes, bboxes, labels, colors)`
- **Purpose**: Draws bounding boxes on an image. Supports custom labels and colors; renders up to 200 boxes. Coordinates must be normalized to [0,1].
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `image` | `IMAGE` | — | Background image `[B, H, W, C]` (first image used) |
  | `bboxes` | `TENSOR` | — | Boxes `[N, 4]`, normalized coordinates (top-left + bottom-right) |
  | `labels` | `STRING` | `""` | Box labels, comma-separated (optional) |
  | `colors` | `STRING` | `"b,g,r,m,c"` | matplotlib colors, comma-separated (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Image with bounding boxes drawn `[1, H, W, C]` |

### Histogram
- **Class**: `CdlHistogram`
- **Purpose**: Draws a histogram of tensor value distribution with configurable bins, density normalisation and colour. Wraps ``matplotlib.pyplot.hist``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Input tensor (flattened internally) |
  | `bins` | `INT` | 30 | Number of histogram bins |
  | `density` | `BOOLEAN` | False | If True show density instead of count |
  | `color` | `STRING` | `"#4673a6"` | Bar face colour |
  | `alpha` | `FLOAT` | 0.7 | Bar transparency |
  | `title` | `STRING` | `""` | Plot title |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Histogram plot `[1, H, W, C]` |

### Bar Chart
- **Class**: `CdlBarChart`
- **Purpose**: Draws a vertical or horizontal bar chart with optional value annotations. Wraps ``matplotlib.pyplot.bar`` / ``barh``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `values` | `TENSOR` | — | Bar heights (1-D tensor) |
  | `labels` | `STRING` | `""` | Category labels, comma-separated |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
  | `horizontal` | `BOOLEAN` | False | Use ``barh`` instead of ``bar`` |
  | `color` | `STRING` | `"#4673a6"` | Bar face colour |
  | `annotate` | `BOOLEAN` | True | Show numeric values on bars |
  | `figsize_w` | `FLOAT` | 7.0 | Figure width |
  | `figsize_h` | `FLOAT` | 4.0 | Figure height |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Bar chart `[1, H, W, C]` |

### Scatter
- **Class**: `CdlScatter`
- **Purpose**: Draws a 2-D scatter plot with optional point colour and size encodings. Wraps ``matplotlib.pyplot.scatter``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `X` | `TENSOR` | — | X coordinates (flattened) |
  | `Y` | `TENSOR` | — | Y coordinates (flattened) |
  | `alpha` | `FLOAT` | 0.6 | Point transparency |
  | `cmap` | `STRING` | `"viridis"` | Colormap for ``color_map`` |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
  | `figsize_w` | `FLOAT` | 6.0 | Figure width |
  | `figsize_h` | `FLOAT` | 5.0 | Figure height |
  | `color_map` | `TENSOR` | — | Per-point colour values (optional) |
  | `size_map` | `TENSOR` | — | Per-point size values (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Scatter plot `[1, H, W, C]` |

### Confusion Matrix
- **Class**: `CdlConfusionMatrix`
- **Purpose**: Renders a confusion matrix as a heatmap with per-cell numeric annotations. Supports row-wise normalisation and configurable number format.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `matrix` | `TENSOR` | — | Confusion matrix (N×N or flat) |
  | `class_labels` | `STRING` | `""` | Class names, comma-separated |
  | `cmap` | `STRING` | `"Blues"` | Colormap name |
  | `normalize` | `BOOLEAN` | False | Normalise rows to [0,1] |
  | `fmt` | `COMBO` | `.1f` | Number format (`.0f`, `.1f`, `.2f`, `.3f`) |
  | `figsize_w` | `FLOAT` | 6.0 | Figure width |
  | `figsize_h` | `FLOAT` | 5.0 | Figure height |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Confusion matrix heatmap `[1, H, W, C]` |

### Pie Chart
- **Class**: `CdlPieChart`
- **Purpose**: Draws a pie chart (regular or donut style) with percentage labels. Wraps ``matplotlib.pyplot.pie``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `values` | `TENSOR` | — | Slice values (1-D tensor) |
  | `labels` | `STRING` | `""` | Slice labels, comma-separated |
  | `donut` | `BOOLEAN` | False | Hollow centre (donut chart) |
  | `explode` | `STRING` | `""` | Comma-separated 0/1 per slice |
  | `pctdistance` | `FLOAT` | 0.6 | Distance of percentage labels from centre |
  | `shadow` | `BOOLEAN` | False | Drop shadow beneath pie |
  | `figsize_w` | `FLOAT` | 6.0 | Figure width |
  | `figsize_h` | `FLOAT` | 6.0 | Figure height |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Pie (or donut) chart `[1, H, W, C]` |

### Area Chart
- **Class**: `CdlAreaChart`
- **Purpose**: Draws a filled area chart — single series with ``fill_between``, or stacked multi-series with ``stackplot``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `Y` | `TENSOR` | — | Series data, `[T]` or `[N, T]` |
  | `stacked` | `BOOLEAN` | False | Stack series instead of overlay |
  | `alpha` | `FLOAT` | 0.5 | Fill transparency |
  | `color_palette` | `STRING` | `"tab10"` | matplotlib palette name |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
  | `figsize_w` | `FLOAT` | 7.0 | Figure width |
  | `figsize_h` | `FLOAT` | 4.0 | Figure height |
  | `X_vals` | `TENSOR` | — | Custom x-axis values (optional) |
  | `labels` | `STRING` | `""` | Series legend labels, comma-separated (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Area chart `[1, H, W, C]` |

---

## 12. d2l / Datasets (10 nodes)

Datasets nodes provide end-to-end dataset management: download, load, inspect, preview, and compute statistics.

### Load Array → DataLoader
- **Class**: `CdlLoadArray`
- **d2lcore function**: `load_array(data_arrays, batch_size, is_train)`
- **Purpose**: Wraps one or more tensors into a PyTorch DataLoader. Connect `TENSOR` features and/or labels to the optional slots; the node outputs a `cdlDataloader`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `batch_size` | `INT` | 32 | Batch size (1~4096) |
  | `shuffle` | `BOOLEAN` | True | Shuffle data on each epoch |
  | `features` | `TENSOR` | — | Feature tensor X (optional) |
  | `labels` | `TENSOR` | — | Label tensor y (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `dataloader` | `cdlDataloader` | PyTorch DataLoader wrapping the tensors |

### DataLoader Info
- **Class**: `CdlDataLoaderInfo`
- **Purpose**: Inspects a `cdlDataloader` and reports its properties: number of batches, batch size, and total dataset size.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `dataloader` | `cdlDataloader` | DataLoader to inspect |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `num_batches` | `INT` | Total number of batches |
  | `batch_size` | `INT` | Samples per batch |
  | `dataset_size` | `INT` | Total number of samples |

### Download
- **Class**: `CdlDownload`
- **d2lcore function**: `download(url, folder, sha1_hash)`
- **Purpose**: Downloads a file from a URL with SHA1-based cache checking. If the local file exists and matches the hash, download is skipped.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `url` | `STRING` | `""` | Download URL |
  | `save_dir` | `STRING` | `"../data"` | Save directory (optional) |
  | `sha1_hash` | `STRING` | `""` | Expected SHA1 hash for caching (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `file_path` | `STRING` | Path to the downloaded/cached file |

### Download + Extract
- **Class**: `CdlDownloadExtract`
- **d2lcore function**: `download_extract(name, folder)`
- **Purpose**: Downloads and extracts a dataset registered in the d2l DATA_HUB. Select from a dropdown of pre-registered datasets (banana-detection, voc2012, cifar10_tiny, etc.).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `name` | `COMBO` | first key | Dataset name from DATA_HUB |
  | `subfolder` | `STRING` | `""` | Subfolder inside archive (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `extract_dir` | `STRING` | Path to the extracted dataset directory |

### Fashion-MNIST
- **Class**: `CdlFashionMNIST`
- **d2lcore function**: `load_data_fashion_mnist(batch_size, resize)`
- **Purpose**: Loads the Fashion-MNIST image classification dataset (60k train / 10k test, 10 classes). Downloads automatically on first use (~30 MB).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `batch_size` | `INT` | 64 | Samples per batch (1~2048) |
  | `resize` | `INT` | 28 | Resize dimensions (0 = no resize, 1~512) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `train_loader` | `cdlDataloader` | Training DataLoader (60,000 images) |
  | `test_loader` | `cdlDataloader` | Test DataLoader (10,000 images) |
  | `class_names` | `STRING` | Newline-separated class names |

10 classes: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot

### Bananas Detection
- **Class**: `CdlBananasDetection`
- **d2lcore function**: `load_data_bananas(batch_size)`
- **Purpose**: Loads the banana detection dataset for object detection. Contains banana images with bounding box annotations. Downloads automatically on first use.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `batch_size` | `INT` | 32 | Samples per batch (1~256) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `train_loader` | `cdlDataloader` | Training DataLoader |
  | `val_loader` | `cdlDataloader` | Validation DataLoader |

### VOC Segmentation
- **Class**: `CdlVOCSegmentation`
- **d2lcore function**: `load_data_voc(batch_size, crop_size)`
- **Purpose**: Loads the PASCAL VOC2012 semantic segmentation dataset (21 classes). Downloads and extracts automatically on first use (~2 GB).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `batch_size` | `INT` | 32 | Samples per batch (1~128) |
  | `crop_height` | `INT` | 320 | Random crop height (64~1024) |
  | `crop_width` | `INT` | 480 | Random crop width (64~2048) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `train_loader` | `cdlDataloader` | Training DataLoader |
  | `test_loader` | `cdlDataloader` | Test DataLoader |

### DataLoader Preview
- **Class**: `CdlDataLoaderPreview`
- **Purpose**: Samples one batch from a `cdlDataloader` and renders it as an image grid (IMAGE output). Adapts to different formats: image classification displays images with labels; object detection shows images with bounding boxes.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `dataloader` | `cdlDataloader` | — | DataLoader to sample from |
  | `num_rows` | `INT` | 2 | Grid rows (1~16) |
  | `num_cols` | `INT` | 4 | Grid columns (1~16) |
  | `max_samples` | `INT` | 32 | Max images to show (1~256, optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered grid image `[1, H, W, C]` |

### DataLoader Preview (Output)
- **Class**: `CdlDataLoaderPreviewOutput`
- **Purpose**: Same as `CdlDataLoaderPreview` but registered as an OUTPUT_NODE so the rendered grid is displayed directly in the UI.
- **Inputs**: Same as `CdlDataLoaderPreview`
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered grid image `[1, H, W, C]` |

### Dataset Stats
- **Class**: `CdlDataLoaderStats`
- **Purpose**: Iterates over a `cdlDataloader` and computes label distribution statistics. The label type is auto-detected: discrete integer class indices within `[0, num_classes)` are rendered as a per-class bar chart; continuous values (e.g. regression targets, including 2D `(n,1)` labels) are automatically bucketed into a histogram with `num_classes` bins.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `dataloader` | `cdlDataloader` | — | DataLoader to analyze |
  | `num_classes` | `INT` | 10 | Expected number of classes / histogram bins (1~1000) |
  | `class_names` | `STRING` | `""` | Comma-separated class names (optional; classification mode only) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `stats_text` | `STRING` | Formatted text summary (per-class counts, or bin ranges + counts for histograms) |
  | `stats_image` | `IMAGE` | Label distribution chart (class bar chart or value histogram) `[1, H, W, C]` |

---

## 13. d2l / Model Utils (7 nodes)

Self-developed model utility nodes (not from d2l). They help inspect, switch, run, clone and persist PyTorch models directly on the workflow graph. All nodes operate on the `cdlModel` type (any `nn.Module` instance).

### Model Info
- **Class**: `CdlModelInfo`
- **Purpose**: Inspects a model and reports (1) a human-readable summary string, (2) the total parameter count and (3) the trainable parameter count.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` instance |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `summary` | `STRING` | Model type, submodule names, module count, parameter counts |
  | `total_params` | `INT` | Total number of parameters |
  | `trainable_params` | `INT` | Number of parameters with `requires_grad=True` |

### Model Mode
- **Class**: `CdlModelMode` (⚠️ **deprecated**, soft-archived to `d2l/_Legacy/Model Utils`)
- **Replacement**: on a bare `TENSOR` graph use the core `Training Mode` node (`Network & Layers/Training`). The two payloads are **not** interchangeable: this node switches a whole `cdlModel` (`model.train()` / `model.eval()`) and passes the module on, whereas `Training Mode` publishes the STRING that drives `TENSOR`-level nodes. This node is unchanged and still usable in `cdlModel` pipelines.
- **Purpose**: Switches a model between training and evaluation mode via `model.train()` / `model.eval()`. Returns the same instance so downstream nodes observe the new mode.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `model` | `cdlModel` | — | Any `nn.Module` instance |
  | `mode` | `COMBO` | `eval` | `train` (training mode) / `eval` (inference mode) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | The same instance with the mode applied |

### Model Forward
- **Class**: `CdlModelForward`
- **Purpose**: Runs a forward pass of `model` on an input tensor under `torch.no_grad()`. The input is moved to the model's device if they differ; the model is switched to eval mode first.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` instance |
  | `tensor` | `TENSOR` | Input tensor of the shape the model expects |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `TENSOR` | `model(tensor)` — shape depends on the model |

### Model Layers
- **Class**: `CdlModelLayers`
- **Purpose**: Walks `model.named_modules()` and renders an indented tree of every module (name + class), so you can inspect the architecture.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` instance |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `layers_str` | `STRING` | One module per line, indented by nesting depth |

### Model Params
- **Class**: `CdlModelParams`
- **Purpose**: Walks `model.named_parameters()` and renders name, shape and `requires_grad` for each parameter, plus the total count.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` instance |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `params_str` | `STRING` | One parameter per line + total count |

### Model Clone
- **Class**: `CdlModelClone`
- **Purpose**: Returns `copy.deepcopy(model)` — an independent instance with the same architecture and weights but no shared parameters.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | Any `nn.Module` instance |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `clone` | `cdlModel` | A deep copy of the input model |

### Model Save
- **Class**: `CdlModelSave`
- **Purpose**: Writes `torch.save(model.state_dict(), path)` to disk. Only weights are saved (state_dict), so reloading requires a model with a matching architecture.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `model` | `cdlModel` | — | Any `nn.Module` instance |
  | `path` | `STRING` | `"model.pt"` | Target file path, e.g. `"C:/models/my_model.pt"` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `message` | `STRING` | Confirmation text including the saved path |

### Model Load
- **Class**: `CdlModelLoad`
- **Purpose**: Reads a `.pt` state_dict with `torch.load` and applies it to the input model via `model.load_state_dict()`. The model architecture must match the saved state_dict.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `model` | `cdlModel` | — | Model instance that will receive the weights |
  | `path` | `STRING` | `"model.pt"` | Path of the saved state_dict file |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `model` | `cdlModel` | The input model with loaded weights |

---

## 14. ComfyUI / image/color (3 nodes)

Self-developed general CV image nodes (not from d2l), merged into the ComfyUI core category `image/color`. All nodes consume and produce the native ComfyUI `IMAGE` format — float32 `[B, H, W, C]` with values in `[0, 1]` — and are implemented with `torch` + `torchvision.transforms.functional`. Exception: `Image Normalize` deliberately does not clip its output to `[0, 1]` (z-score range).

> Geometry nodes removed in favour of the ComfyUI core equivalents: `Image Resize` → `ImageScale` / `ResizeImageMaskNode`, `Image Flip` → `ImageFlip`, `Image Blur` → `ImageBlur`, `Image Crop` → `ImageCrop` / `ImageCropV2`. `Image Rotate` was kept (the core `ImageRotate` only supports 90-degree steps) and moved to `image/transform`.

### Image Normalize
- **Class**: `CdlImageNormalize`
- **Purpose**: Applies `(x - mean) / std` when `denorm` is False, or the inverse `x * std + mean` when `denorm` is True. `mean`/`std` are comma-separated strings; a single value broadcasts to all channels (e.g. `"0.5"` or `"0.5,0.5,0.5"`).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `image` | `IMAGE` | — | Input images `[B, H, W, C]` |
  | `mean` | `STRING` | `"0.5,0.5,0.5"` | Comma-separated per-channel means |
  | `std` | `STRING` | `"0.5,0.5,0.5"` | Comma-separated per-channel standard deviations |
  | `denorm` | `BOOLEAN` | False | True = denormalize, False = normalize |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Processed images `[B, H, W, C]` (range depends on the op) |

### Image Grayscale
- **Class**: `CdlImageGrayscale`
- **Purpose**: Converts images to grayscale with `num_output_channels=3`, preserving the `[B, H, W, C]` (C=3) layout while all channels carry the same luminance value.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Input images `[B, H, W, C]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | 3-channel grayscale images `[B, H, W, C]` |

### Image Adjust
- **Class**: `CdlImageAdjust`
- **Purpose**: Applies torchvision brightness, contrast and saturation adjustments with the given factors (1.0 = unchanged, >1 stronger, <1 weaker, 0 = none). Factors equal to 1.0 are skipped for speed.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `image` | `IMAGE` | — | Input images `[B, H, W, C]` |
  | `brightness` | `FLOAT` | 1.0 | Brightness factor (0~2) |
  | `contrast` | `FLOAT` | 1.0 | Contrast factor (0~2) |
  | `saturation` | `FLOAT` | 1.0 | Saturation factor (0~2) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Adjusted images `[B, H, W, C]` |

## 15. ComfyUI / image/transform (1 node)

Kept as a ComfyUI core `image/transform` node: the core `ImageRotate` node only supports 90-degree steps, while this one accepts an arbitrary angle plus optional canvas expansion.

### Image Rotate
- **Class**: `CdlImageRotate`
- **Purpose**: Rotates every image by `angle` degrees (counter-clockwise) with bilinear interpolation and zero-filled borders. When `expand` is True the canvas is enlarged so rotated content is not clipped; otherwise the output keeps the input size.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `image` | `IMAGE` | — | Input images `[B, H, W, C]` |
  | `angle` | `FLOAT` | 90.0 | Rotation angle in degrees (-360~360) |
  | `expand` | `BOOLEAN` | False | True = enlarge canvas to fit rotated content |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rotated images `[B, H, W, C]` |

---

## 16. ComfyUI / image (1 node)

Merged into the ComfyUI core category `image` (next to the core `GetImageSize` node).

### Image Stats
- **Class**: `CdlImageStats`
- **Purpose**: Aggregates all images in the batch and reports per channel the mean, std, min and max values, plus the batch layout.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Input images `[B, H, W, C]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `stats` | `STRING` | One line per channel + batch summary line |

---

## 17. ComfyUI / Network & Layers (45 nodes)

Core neural-network nodes shipped by the host runtime (not part of the ComfyDL submodule).
They live in six `comfy_extras` modules — `nodes_activation.py`, `nodes_layers.py`,
`nodes_normalization.py`, `nodes_pooling.py`, `nodes_convolution.py` and `nodes_training.py` —
and form the **Comfy nodes → Network & Layers** branch of the node library, split into the `Activation`,
`Basic`, `Normalization`, `Regularization`, `Training`, `Pooling` and `Convolution` groups below. All of them exchange data on the shared `TENSOR` slot type, preserve the input
dtype/device, and are stateless: learnable parameters such as `weight` and `bias` are tensors fed
through input slots instead of being initialised inside the node, so a node is a pure function and
can be wired straight to the ComfyDL tensor nodes listed above. The `Training` group additionally
introduces two first-class graph value types, `PARAMS` and `OPTIMIZER`, which let the same
stateless convention express *trainable* parameters and an optimisation loop.

### 17.1 Activation (14 nodes)

Core activation-function nodes (`comfy_extras/nodes_activation.py`). Each node takes exactly
one `TENSOR` input named `tensor` and returns exactly one `TENSOR` output named `output`,
and has no learnable parameters.

| Node | Class | Extra widget | Purpose |
|------|-------|--------------|---------|
| Sigmoid | `ActivationSigmoid` | — | `1 / (1 + exp(-x))`; output range (0, 1) |
| Tanh | `ActivationTanh` | — | `tanh(x)`; output range (-1, 1) |
| ReLU | `ActivationReLU` | — | `max(0, x)` |
| Leaky ReLU | `ActivationLeakyReLU` | `negative_slope` FLOAT 0.01 (0~1) | ReLU with a small slope for `x < 0` |
| ELU | `ActivationELU` | `alpha` FLOAT 1.0 (0~100) | `x` if `x > 0`, else `alpha * (exp(x) - 1)` |
| SELU | `ActivationSELU` | — | Self-normalizing ELU (standard scale/alpha constants) |
| GELU | `ActivationGELU` | `approximate` COMBO none/tanh | Gaussian error linear unit |
| SiLU | `ActivationSiLU` | — | `x * sigmoid(x)` (swish) |
| Mish | `ActivationMish` | — | `x * tanh(softplus(x))` |
| Softplus | `ActivationSoftplus` | — | `log(1 + exp(x))` (beta=1, threshold=20) |
| ReLU6 | `ActivationReLU6` | — | `min(max(0, x), 6)` |
| Hard Swish | `ActivationHardSwish` | — | `x * relu6(x + 3) / 6` |
| Identity | `ActivationIdentity` | — | Zero-copy pass-through of the input tensor |
| Softmax | `ActivationSoftmax` | `dim` INT -1 (-4~4) | Normalizes along `dim` (clamped to the tensor rank) |

### 17.2 Basic (8 nodes)

Core basic-layer / tensor-op nodes (`comfy_extras/nodes_layers.py`). Each node takes one or
two `TENSOR` inputs and returns exactly one `TENSOR` output named `output`. `weight` / `bias`
are ordinary tensor inputs, so there is no hidden parameter state. Widget defaults are usable
as-is; `Reshape` / `Broadcast` fall back to returning the input tensor unchanged when
`target_shape` cannot be parsed, so they never break a workflow.

| Node | Class | Inputs | Extra widget | Purpose |
|------|-------|--------|--------------|---------|
| Linear | `BasicLinear` | `tensor`, `weight` (`[out, in]`), `bias` (optional) | — | Affine transform `x @ weight.T + bias` (`F.linear`) |
| Embedding | `BasicEmbedding` | `tensor` (integer indices), `weight` (`[num, dim]`) | — | Looks up rows of `weight` (`F.embedding`) |
| Flatten | `BasicFlatten` | `tensor` | `start_dim` INT 1 (0~4), `end_dim` INT -1 (-4~4) | Flattens a contiguous dim range (`torch.flatten`) |
| Reshape | `BasicReshape` | `tensor` | `target_shape` STRING `"1,-1"` | Reshapes to the given shape (`torch.reshape`) |
| Broadcast | `BasicBroadcast` | `tensor` | `target_shape` STRING `"2,3"` | Expands to the given shape (`torch.broadcast_to`) |
| Concat | `BasicConcat` | `a`, `b` | `dim` INT -1 (-4~4) | Concatenates two tensors along `dim` (`torch.cat`) |
| Add | `BasicAdd` | `a`, `b` | — | Element-wise addition with broadcasting (`a + b`) |
| Multiply | `BasicMultiply` | `a`, `b` | — | Element-wise multiplication with broadcasting (`a * b`) |

> `Linear` supersedes a `Dense` layer (same affine transform), and `Add` also covers a
> residual / skip connection — both are a plain broadcast element-wise sum — so no separate
> `Dense`, `Residual` or `Skip Connection` node is shipped.

### 17.3 Normalization (7 nodes)

Core normalization nodes (`comfy_extras/nodes_normalization.py`). Every node returns a `TENSOR`
named `output` as its first output, preserves the input dtype/device and keeps no state.
`BatchNorm` and `InstanceNorm` append a `mean` / `var` pair after it — the per-channel statistics
this call **actually normalized with**, ready to be wired into `Training Run Stats` (see 17.4) —
while the remaining nodes return exactly one output.
`weight` / `bias` (γ / β) are ordinary tensor inputs, and a `running_mean` / `running_var`
tensor that is wired in is never modified in place, because the same tensor may be shared with
other nodes. `BatchNorm` and `InstanceNorm` are **rank adaptive** — one node each covers the
1d/2d/3d flavours, since the shape alone decides which dimensions the statistics are taken over
— and they follow a `mode` slot instead of owning a train/eval switch of their own.

| Node | Class | Inputs | Extra widget | Purpose |
|------|-------|--------|--------------|---------|
| BatchNorm | `NormalizationBatchNorm` | `tensor`, `weight` (optional), `bias` (optional), `running_mean` / `running_var` (optional), `mode` (optional STRING socket) | `eps` FLOAT 1e-5 (0~1e-2) | `F.batch_norm` over dimension 1 of `(N, C, ...)`; rank 2/3/4/5 behave like BatchNorm1d/1d/2d/3d. Also outputs the per-channel `mean` / `var` (`(C,)`) this call used: the batch statistics in `train`, the wired running statistics in `eval` |
| InstanceNorm | `NormalizationInstanceNorm` | `tensor`, `weight`, `bias`, `running_mean` / `running_var`, `mode` — all optional except `tensor` | `eps` FLOAT 1e-5 (0~1e-2) | `F.instance_norm`, statistics per sample *and* per channel; needs rank ≥ 3. Also outputs `mean` / `var`, collapsing the per-sample statistics to one value per channel (mean of within-sample variances + variance of the per-sample means), so the pair means the same thing as BatchNorm's and fits the same slots |
| LayerNorm | `NormalizationLayerNorm` | `tensor`, `weight` (optional), `bias` (optional) | `normalized_shape` STRING `"last"`, `eps` FLOAT 1e-5 | `F.layer_norm` over the trailing dimensions (`"8,16"` = the last two) |
| GroupNorm | `NormalizationGroupNorm` | `tensor`, `weight` (optional), `bias` (optional) | `num_groups` INT 1 (1~64), `eps` FLOAT 1e-5 | `F.group_norm`; `num_groups=1` normalizes over all channels |
| RMSNorm | `NormalizationRMSNorm` | `tensor`, `weight` (optional) | `normalized_shape` STRING `"last"`, `eps` FLOAT 1e-6 | `F.rms_norm`; LLaMA-style (no mean subtraction, no bias) |
| WeightNorm | `NormalizationWeightNorm` | `weight`, `g` (optional) | `dim` INT 0 (-8~7), `eps` FLOAT 1e-12 | Weight re-parameterization `g * v / ‖v‖₂` taken along `dim` |
| SpectralNorm | `NormalizationSpectralNorm` | `weight`, `u` / `v` (optional) | `n_power_iterations` INT 10 (0~20), `dim` INT 0, `eps` FLOAT 1e-12 | Divides a weight by a deterministic power-iteration estimate of its largest singular value; also outputs `sigma` |

> The train/eval switch is an explicit link: `Training Mode` (17.4) publishes `train` / `eval` as
> a STRING wired into the `mode` slot of `BatchNorm` / `InstanceNorm`. A link is required for
> correctness, not just for convenience — a node's cache signature covers its own inputs *and its
> ancestors' inputs*, so flipping the dropdown invalidates every consumer, whereas a hidden
> prompt-reading handshake is not part of the signature (and cannot even see the prompt while the
> signature is built) and would keep replaying stale outputs. `LayerNorm`, `GroupNorm`, `RMSNorm`,
> `WeightNorm` and `SpectralNorm` deliberately have no `mode` slot: their math is identical in
> training and inference. An `eval` mode without usable statistics falls back to the statistics of
> the current call, and unparsable widget text (a stale `normalized_shape`, an indivisible
> `num_groups`, a repeated dimension) falls back to a documented default with a printed warning,
> so a widget value never breaks a workflow.

### 17.4 Training (11 nodes)

Two families share the `Network & Layers/Training` category. The first is the pair of small "state"
nodes (`comfy_extras/nodes_normalization.py`) that carry the train/inference decision and the
persistent running statistics into the normalization nodes. The second is the training closure
(`comfy_extras/nodes_training.py`), which adds the two graph value types `PARAMS` and `OPTIMIZER`
and a node that runs a real optimisation loop inside itself.

**Train / eval state (2 nodes)**

| Node | Class | Inputs | Extra widget | Purpose |
|------|-------|--------|--------------|---------|
| Training Mode | `TrainingMode` | — | `mode` COMBO train/eval (default `train`) | Publishes `train` / `eval` as a STRING for the `mode` slots of `BatchNorm` / `InstanceNorm` |
| Training Run Stats | `TrainingRunStats` | `mean` / `var` (optional `TENSOR` sockets) | `running_mean` STRING `"0.0"`, `running_var` STRING `"1.0"` | Emits `running_mean` / `running_var` as two 1-D `TENSOR`s for the statistics slots; once a socket is linked the **link wins** and the widget text above it is ignored |

> Running statistics have to survive in a saved workflow, and a widget is the only place that
> does, so they are typed in as comma separated numbers — one value per channel
> (`"0.1,0.2,0.3"`) or a single value that the consumer broadcasts to every channel. Both nodes
> are sources: leaving one on the canvas without wiring it is harmless, because a source node is
> only evaluated when a consumer asks for it.
> A linked `mean` / `var` socket **wins over** the widget text: the link carries the measured value
> of the run that produced it, while the widget only holds whatever was typed when the graph was
> saved, so handing a run's statistics forward to `eval` is a matter of two wires.

**Learnable parameters, optimizer settings and the training loop (9 nodes)**

`PARAMS` is an ordered `{name: nn.Parameter}` mapping; `OPTIMIZER` is an optimizer *configuration*
(`OptimizerConfig`), not a live optimizer — the hyper-parameters live on widgets and only the
setting travels on the link. ComfyUI runs an entire prompt inside `torch.inference_mode()`, so an
autograd graph cannot cross a node boundary and no node can differentiate another node's tensor:
the trainer therefore runs forward, backward and `optimizer.step()` **itself**, inside a
`torch.inference_mode(False)` block. Parameter names follow the module / `safetensors` convention
(`layer0.weight`, `layer0.bias`, `layer1.weight`, …), so a trained set can be taken apart with
`Parameters to Tensor` and wired straight into the stateless `Basic` / `Conv` layer nodes.

| Node | Class | Inputs | Extra widgets | Purpose |
|------|-------|--------|---------------|---------|
| Learnable Parameters | `TrainingParameters` | `tensor` (optional) | `name` STRING `"weight"`, `shape` STRING `"2,3"`, `init` COMBO normal/zeros/ones/xavier_uniform/kaiming_uniform (default `normal`), `seed` INT 0 | Creates one named trainable parameter; a **wired tensor wins over** `shape` / `init`. Produces `PARAMS` |
| Merge Parameters | `TrainingParametersMerge` | `params a`, `params b` | — | Concatenates two sets into one; a name present on both sides is *renamed* (`weight` → `weight_2`) with a warning instead of being overwritten, so no weight is ever lost silently |
| Parameters to Tensor | `TrainingParametersExtract` | `params` | `name` STRING `"weight"` | Publishes one entry as a plain `TENSOR` (detached) — the bridge from the trainer back to the `Basic` / `Conv` layer nodes (`layer0.weight` → `Linear.weight`); an unknown name raises a readable error listing every available one |
| Optimizer | `TrainingOptimizer` | — | `optimizer` COMBO AdamW/Adam/SGD/RMSprop (default `AdamW`), `lr` FLOAT 0.01 (0~1), `momentum` 0.9 (0~0.999), `beta1` 0.9 (0~0.999), `beta2` 0.999 (0~0.9999), `eps` 1e-8 (0~1e-3), `weight_decay` 0.01 (0~1), `amsgrad` false | Publishes the settings as `OPTIMIZER`; `SGD` reads `momentum`, `Adam`/`AdamW` read the betas, `RMSprop` reads `beta2` as its `alpha` |
| Training Loop | `TrainingLoop` | `x`, `y` (`TENSOR`), `optimizer` (`OPTIMIZER`), `params` (optional `PARAMS`, warm start) | `hidden` STRING `"8"`, `activation` COMBO relu/gelu/tanh/sigmoid/none (default `relu`), `loss` COMBO mse/l1/cross_entropy (default `mse`), `steps` INT 200 (1~100000), `batch_size` INT 0 (0~65536; 0 = whole dataset per step), `seed` INT 0 | The trainer: builds an MLP (`in_features` from `x`, `out_features` from `y`, activation between the hidden layers only, `hidden` empty = plain linear regression) and runs `steps` × (forward + backward + `optimizer.step()`). Outputs `params`, `loss` (scalar), `loss_history` (1-D, one entry per step) and `prediction` (detached) |
| Save Parameters | `TrainingSaveParameters` | `params` | `filename_prefix` STRING `"comfydl/parameters"` | Writes a `.safetensors` file into the output folder and **passes the set through**, so saving does not end the graph; also outputs the absolute `path` |
| Load Parameters | `TrainingLoadParameters` | — | `path` STRING `"comfydl/parameters_00001_.safetensors"` | Reads a set back (relative to the output folder, or absolute); non-float entries are dropped and non-float32 ones promoted, both with a report, and a missing file raises a readable error |
| Parameters to Text | `TrainingParametersToText` | `params` | — | Encodes the set as `CDLPARAMS1:<base64 safetensors>` and pushes it into the node's own UI box, so it can be copied out and pasted into a widget — the channel that survives inside a saved `.json` workflow without touching the disk |
| Text to Parameters | `TrainingTextToParameters` | — | `text` STRING, multiline (default = a valid 2×3 `weight`) | Decodes the text form back into `PARAMS`; surrounding whitespace is ignored and an unusable string raises an error that says what to paste |

> Determinism and caching: `Training Loop` seeds both the initialisation and the batch shuffling
> from its `seed` widget and restores the process RNG afterwards, so the same inputs always produce
> the same run and the node's cached output stays meaningful. Every output is detached, so no
> autograd graph is left inside ComfyUI's cache, and a warm start (`params` linked) copies only the
> entries whose name *and* shape match the freshly built network, reporting every missing and every
> skipped key rather than dropping it silently.
>
> `PARAMS` and `OPTIMIZER` are declared in `comfy_api/latest/_io.py` (`@comfytype`) exactly like the
> `TENSOR` and `LORA_MODEL` types, so they need no frontend registration: unregistered slot types
> render with the frontend's default colour and can be wired like any other type.
>
> The `loss_history` output is an ordinary 1-D tensor, so the convergence curve can be inspected
> with any of the visualisation nodes above; `loss` is its last entry as a scalar.


### 17.5 Regularization (1 node)

Core regularization nodes (`comfy_extras/nodes_normalization.py`). The node returns exactly one
`TENSOR` output named `output`, preserves the input dtype/device and keeps no state.

| Node | Class | Inputs | Extra widget | Purpose |
|------|-------|--------|--------------|---------|
| Dropout | `RegularizationDropout` | `tensor`, `mode` (optional STRING socket) | `p` FLOAT 0.5 (0~1), `seed` INT 0 (with a fixed / increment / decrement / randomize dropdown) | `F.dropout`, element-wise: zeroes each element with probability `p` and rescales the survivors by `1/(1-p)`; `eval` passes the input through |

> Like the normalizers, `Dropout` is rank adaptive: the mask is drawn element by element, so one
> node covers `(N, C)`, `(N, C, H, W)` and a bare scalar. Its mask comes from a `torch.Generator`
> built for the tensor's own device and seeded by the `seed` widget, never from the process-wide
> RNG: the same seed reproduces the mask bit for bit (so ComfyUI's caching stays meaningful) and
> the RNG stream the host hands to other nodes is left alone. The dropdown next to the seed is what
> moves the value between runs — `randomize` for a fresh mask on every run, `fixed` to freeze it.
> Unlike `LayerNorm` / `GroupNorm` / `RMSNorm`, whose math is identical in both modes, `Dropout`
> needs the train/eval switch, so it follows the same `mode` slot as `BatchNorm` / `InstanceNorm`
> and passes the input through untouched in `eval`, which makes a leftover Dropout harmless in an
> inference graph. A `p` of 0 or 1 short-circuits to the input or to zeros and draws no random
> number at all.

### 17.6 Pooling (2 nodes)

Core pooling nodes (`comfy_extras/nodes_pooling.py`). One node covers the sliding-window family
(`MaxPool{1,2,3}d` / `AvgPool{1,2,3}d`) and one the adaptive family (`AdaptiveAvgPool{1,2,3}d` /
`AdaptiveMaxPool{1,2,3}d`), so twelve torch layer types collapse into two nodes: `mode` picks max
or average, `dims` picks the rank, and each window widget below is applied to all `dims` spatial
dimensions at once. A tensor that lacks the batch dimension is accepted too — the missing leading
dimensions are added internally and dropped from the result again.

| Node | Class | Inputs | Extra widgets | Purpose |
|------|-------|--------|---------------|---------|
| Pool | `PoolingSliding` | `tensor` | `dims` COMBO 1/2/3 (default 2), `mode` COMBO max/avg (default `max`), `kernel_size` INT 2 (1~64), `stride` INT 0 (0~64), `padding` INT 0 (0~32), `dilation` INT 1 (1~16), `ceil_mode` BOOLEAN false, `count_include_pad` BOOLEAN true | `F.max_pool{1,2,3}d` / `F.avg_pool{1,2,3}d`; `dims=2, kernel_size=2` reproduces `nn.MaxPool2d(2)` exactly |
| Adaptive Pool | `PoolingAdaptive` | `tensor` | `dims` COMBO 1/2/3 (default 2), `mode` COMBO avg/max (default `avg`), `output_size` INT 1 (1~512) | `F.adaptive_max_pool{1,2,3}d` / `F.adaptive_avg_pool{1,2,3}d`; every spatial dimension becomes exactly `output_size` |

> No `GlobalAvgPool` / `GlobalMaxPool` node is shipped: `output_size=1` **is** global pooling
> (`dims=2, output_size=1` ≡ `nn.AdaptiveAvgPool2d(1)` ≡ the textbook **Global Average Pooling**,
> `GAP`), so the adaptive node already covers it and its `mode` dropdown covers both reductions.
> `stride=0` means "same as `kernel_size`", which is torch's own default. Two widgets are
> mode-specific and ignored otherwise: `dilation` exists only for `mode=max` and `count_include_pad`
> only for `mode=avg`, because `F.avg_pool*d` has no `dilation` argument and `F.max_pool*d` has no
> `count_include_pad`; a non-default value of the ignored widget prints a note instead of silently
> doing nothing.

### 17.7 Convolution (2 nodes)

Core convolution nodes (`comfy_extras/nodes_convolution.py`), the learnable spatial counterpart of
the `Basic` layers above. They follow the same convention as `Linear`: nothing is initialised
inside the node — `weight` and `bias` are ordinary `TENSOR` inputs, so the node is a pure function
and one set of weights can be fed to several nodes. Consequently there is no `in_channels` /
`out_channels` / `bias` switch: the channel counts are read off `weight.shape`, and "no bias" is
expressed by leaving the optional `bias` slot unconnected.

| Node | Class | Inputs | Extra widgets | Purpose |
|------|-------|--------|---------------|---------|
| Conv | `ConvolutionConv` | `tensor`, `weight` (`(out, in/groups, k...)`), `bias` (optional) | `dims` COMBO 1/2/3 (default 2), `groups` INT 1 (1~4096), `stride` INT 1 (1~64), `padding` INT 1 (0~64), `padding_mode` COMBO zeros/reflect/replicate/circular (default `zeros`), `dilation` INT 1 (1~32) | `F.conv{1,2,3}d`, with grouped / dilated kernels and four padding modes |
| ConvTranspose | `ConvolutionConvTranspose` | `tensor`, `weight` (`(in, out/groups, k...)`), `bias` (optional) | `dims` COMBO 1/2/3 (default 2), `groups` INT 1 (1~4096), `stride` INT 2 (1~64), `padding` INT 0 (0~64), `output_padding` INT 0 (0~64), `dilation` INT 1 (1~32) | `F.conv_transpose{1,2,3}d` — the upsampling counterpart decoders and generators use |

> **Parameter sharing** is the point of `Conv`: the *same* kernel is reused at every spatial
> position, so the number of weights depends on the channel counts and the kernel size but never on
> the image size — a 3×3 kernel needs nine weights per input/output channel whether the image is
> 32×32 or 1024×1024. Only the rank and the kernel shape change between `dims=1/2/3`, which is why
> one node covers all three.
>
> The weight shapes of the two nodes are **mirrored**: `Conv` takes `(out_channels,
> in_channels/groups, k...)` while `ConvTranspose` takes `(in_channels, out_channels/groups, k...)`.
> `groups = C_in` gives a depthwise convolution and `dilation > 1` widens the receptive field
> without adding a single weight.
>
> `padding_mode` has to be implemented by hand: `F.conv{1,2,3}d` has **no** `padding_mode`
> argument, only the `nn.Conv*d` *modules* have one. With `padding_mode != "zeros"` the input is
> first padded with `F.pad(mode=reflect|replicate|circular)` and the convolution then runs with
> `padding=0`, so the two never double-pad. `F.pad`'s `circular` mode needs an input of rank 3 or
> more and a pad smaller than the corresponding dimension; when that does not hold the node prints
> a readable message and falls back to zero padding instead of raising. `ConvTranspose` deliberately
> has no `padding_mode` widget at all, because neither `nn.ConvTranspose*d` nor `F.conv_transpose*d`
> supports one.
>
> Both nodes promote dtypes the way `Linear` does: when the input and the weight are both floating
> point but different (fp16 activations with fp32 weights), the common type wins and the
> computation happens there instead of raising a dtype-mismatch error.

---

## 18. ComfyUI / utilities/conversion (6 nodes)

Merged into the ComfyUI core category `utilities/conversion`. These are the only ComfyDL nodes
written against the ComfyUI **V3** node API (`io.ComfyNode` + `io.Schema`) rather than the legacy
`INPUT_TYPES` / `RETURN_TYPES` style, because a union input slot can only be expressed with
`io.MultiType.Input`.

The pair `Value → Tensor` / `Tensor → Value` lets a value that carries ComfyUI semantics
(IMAGE / MASK / LATENT / AUDIO / SIGMAS) travel through generic tensor pipelines and come back
unchanged. `Tensor → Value` deliberately exposes **five concrete output sockets** instead of one
dynamic socket: a concrete socket is statically typed, so the frontend physically prevents a
mismatched link *and* the backend validator performs a real check — a `MatchType` socket gets no
backend checking at all. Metadata that cannot fit inside a bare tensor (`noise_mask`,
`batch_index`, `type`, `sampler_rate`) travels on parallel sockets.

### Value → Tensor
- **Class**: `CdlValueToTensor`
- **Purpose**: Unwraps any supported Comfy value into a generic `TENSOR`. IMAGE / MASK / SIGMAS pass through unchanged; LATENT yields its `samples`; AUDIO yields its `waveform`. Metadata that cannot live inside a bare tensor is emitted on parallel sockets.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `value` | `IMAGE` \| `MASK` \| `LATENT` \| `AUDIO` \| `SIGMAS` | — | Union slot; the socket adapts to whatever you connect |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `TENSOR` | `TENSOR` | The unwrapped tensor |
  | `origin` | `STRING` | Best-effort source label (`IMAGE` / `MASK` / `LATENT` / `AUDIO` / `SIGMAS`). IMAGE, MASK and SIGMAS are all plain tensors at runtime, so their label is inferred from the shape — diagnostic only, it never changes the conversion |
  | `noise_mask` | `TENSOR` | LATENT `noise_mask` when present, else `None` |
  | `batch_index` | `ARRAY` | LATENT `batch_index` when present, else `None` |
  | `latent_type` | `STRING` | LATENT `type` (`"audio"` / `"hunyuan3dv2"`); empty means no special type |
  | `sampler_rate` | `INT` | AUDIO sample rate, or `0` for non-audio values |

### Tensor → Value
- **Class**: `CdlTensorToValue`
- **Purpose**: Exposes a generic `TENSOR` on one concrete socket per type. Wire the socket you need; unused sockets return `None` and cost nothing. Metadata received from `Value → Tensor` is folded back into the rebuilt LATENT / AUDIO dictionaries.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | The tensor to expose |
  | `origin` | `STRING` | `""` | Optional label from `Value → Tensor`; only used to phrase shape-mismatch hints |
  | `noise_mask` | `TENSOR` | — | Optional LATENT noise mask (left unconnected by default) |
  | `batch_index` | `ARRAY` | — | Optional LATENT batch index (left unconnected by default) |
  | `latent_type` | `STRING` | `""` | Optional LATENT `type`; empty means the key is omitted, matching the original |
  | `sampler_rate` | `INT` | `44100` | Optional AUDIO sample rate (1~384000) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `IMAGE` | `IMAGE` | `tensor` unchanged — shape conventions are the caller's responsibility |
  | `MASK` | `MASK` | `tensor` unchanged |
  | `LATENT` | `LATENT` | `{"samples": tensor}` plus whichever metadata sockets are connected; `None` when the tensor is not 4-D |
  | `AUDIO` | `AUDIO` | `{"waveform": tensor, "sampler_rate": rate}`; `None` when the tensor is not 2-D/3-D |
  | `SIGMAS` | `SIGMAS` | `tensor` unchanged |

> The LATENT / AUDIO sockets return `None` when the tensor rank cannot possibly fit, and print a
> hint **only when `origin` says that is what you intended** — an unconnected socket stays silent.

Reserved pairs — nothing in ComfyDL produces `LORA_MODEL` / `LOSS_MAP` yet, so these four nodes are
static/experimental. Each pack node flattens and concatenates every tensor (in key order) into one
1-D `TENSOR` and emits the layout as parallel metadata; each unpack node slices it back. All
tensors in one pack must share a dtype: a mixed batch raises a readable `ValueError` instead of
silently promoting, which would make the round trip lossy.

### LoRA Model → Tensor
- **Class**: `CdlLoraModelToTensor`
- **Purpose**: Packs a `LORA_MODEL` (`dict[str, torch.Tensor]`) into one 1-D `TENSOR` plus key/shape/dtype metadata.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `lora_model` | `LORA_MODEL` | — | Mapping of parameter name to tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `TENSOR` | `TENSOR` | Every value flattened and concatenated, in key order |
  | `keys` | `ARRAY` | Key order, as `list[str]` |
  | `shapes` | `ARRAY` | One `"3,4"` style shape string per tensor; `""` denotes a 0-D scalar |
  | `dtypes` | `ARRAY` | One `"torch.float32"` style dtype string per tensor |

### Tensor → LoRA Model
- **Class**: `CdlTensorToLoraModel`
- **Purpose**: Exact inverse of `LoRA Model → Tensor`; slices the flat tensor by `shapes`, casts each chunk back to its recorded dtype and rebuilds the dictionary.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Flat tensor produced by the pack node (any shape is flattened first) |
  | `keys` | `ARRAY` | — | Key order, as `list[str]` |
  | `shapes` | `ARRAY` | — | One shape string per tensor |
  | `dtypes` | `ARRAY` | — | One dtype string per tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `LORA_MODEL` | `LORA_MODEL` | The restored mapping |

### Loss Map → Tensor
- **Class**: `CdlLossMapToTensor`
- **Purpose**: Packs a `LOSS_MAP` (`{"loss": [Tensor, ...]}`) into one 1-D `TENSOR` plus shape/dtype metadata.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `loss_map` | `LOSS_MAP` | — | A `LOSS_MAP`: `{"loss": [Tensor, ...]}`; a bare tensor or a plain list is also accepted |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `TENSOR` | `TENSOR` | Every loss tensor flattened and concatenated, in order |
  | `shapes` | `ARRAY` | One `"3,4"` style shape string per tensor |
  | `dtypes` | `ARRAY` | One `"torch.float32"` style dtype string per tensor |

### Tensor → Loss Map
- **Class**: `CdlTensorToLossMap`
- **Purpose**: Exact inverse of `Loss Map → Tensor`; rebuilds `{"loss": [Tensor, ...]}`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `TENSOR` | — | Flat tensor produced by the pack node |
  | `shapes` | `ARRAY` | — | One shape string per tensor |
  | `dtypes` | `ARRAY` | — | One dtype string per tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `LOSS_MAP` | `LOSS_MAP` | `{"loss": [Tensor, ...]}` |

> Malformed metadata never aborts a workflow: a shape string that cannot be parsed degrades to a
> 0-D scalar (consuming one element) with a printed hint, and a truncated tensor yields only the
> tensors that fit.

---

## 19. ComfyUI / model (22 nodes)

The **model protocol layer** (`comfy_extras/nodes_model_loaders.py`, `nodes_model_merging.py` and
`nodes_model_inference.py`). `MODEL`, `CLIP` and `VAE` are back in the graph as first-class values,
so weights are again something a workflow can load, move around, blend and write back.

What makes this layer different from the native implementation is that it works at the
**state_dict** level instead of the module level. The dehydration pass removed the `ldm` model
implementations, so nothing here recognises an architecture: a weight file is treated as a flat
mapping of key → tensor, and the only structure the nodes understand is the well-known *key prefix*:

| Prefix | Bucket |
|---|---|
| `diffusion_model.` (plus every unmatched key) | `MODEL` |
| `first_stage_model.` | `VAE` |
| `cond_stage_model.` / `conditioner.` / `text_encoders.` | `CLIP` |

Every loader can therefore work with a plain `.safetensors` / `.ckpt` file. The buckets are held by
containers that keep their tensors **by reference** (no copy, so the memory cost equals the file
size) and round-trip their keys verbatim: whatever outer container prefix was stripped on the way
in (`model.`, `state_dict.`, `module.` — auto-detected, or set explicitly with the `prefix_strip`
widget) is written back on the way out. That is what makes `load → merge → save` produce a file
whose keys match the input — the round trip is lossless by construction.

The `MODEL` bucket is the one exception to "just weights": it is additionally wrapped in a
`ModelPatcher` so it is a real `MODEL` value and stays type-compatible with the rest of ComfyUI. A
bucket with no keys still yields a valid **empty** container rather than `None`, so a
partially-populated checkpoint never breaks the links below it.

**Two tiers of behaviour**, so it is clear what actually runs in this build:

| Tier | Behaviour | Nodes |
|---|---|---|
| L1 — really executes | reads, splits, merges and writes real weights | the 5 loaders, the 7 merge nodes and the 4 save nodes (16) |
| L2 — registered, not executable | the node exists with the native IO contract, so a workflow can be wired and validated, but running it raises a `RuntimeError` that names the missing module and how to restore it (never a bare `ModuleNotFoundError`) | `Load LoRA (Model and CLIP)`, `Load LoRA`, `VAE Decode`, `VAE Encode`, `CLIP Text Encode (Prompt)`, `CLIP Set Last Layer` (6) |

L2 is deliberate: the IO contract is what lets a user lay out a full txt2img graph today, and each
node states in its docstring and in its error message which piece of the removed engine it is
waiting for. Restoring the dehydrated module turns each of them into a working node in place.

### 19.1 Loaders (7 nodes)

| Node | Class | Inputs | Widgets | Outputs / Purpose |
|------|-------|--------|---------|-------------------|
| Load Checkpoint | `CheckpointLoaderSimple` | `ckpt_name` COMBO (`models/checkpoints`) | `prefix_strip` STRING `"auto"` | `MODEL`, `CLIP`, `VAE` — splits one file into the three buckets |
| Load Diffusion Model | `UNETLoader` | `unet_name` COMBO (`models/unet`, `models/diffusion_models`) | `prefix_strip` STRING `"auto"` | `MODEL` — the whole file is the diffusion model |
| Load VAE | `VAELoader` | `vae_name` COMBO (`models/vae`) | `prefix_strip` STRING `"auto"` | `VAE` |
| Load CLIP | `CLIPLoader` | `clip_name` COMBO (`models/text_encoders`, legacy `models/clip` is searched too) | `prefix_strip` STRING `"auto"` | `CLIP` |
| Load CLIP (Dual) | `DualCLIPLoader` | `clip_name1`, `clip_name2` COMBO | `prefix_strip` STRING `"auto"` | `CLIP` — the union of both files, so two encoders arrive as one value |
| Load LoRA (Model and CLIP) | `LoraLoader` | `model` MODEL, `clip` CLIP, `lora_name` COMBO (`models/loras`) | `strength_model` FLOAT 1.0, `strength_clip` FLOAT 1.0 | `MODEL`, `CLIP` — **L2** |
| Load LoRA | `LoraLoaderModelOnly` | `model` MODEL, `lora_name` COMBO (`models/loras`) | `strength_model` FLOAT 1.0 | `MODEL` — **L2** |

> `prefix_strip` is the only widget added on top of the native contract, and its `auto` default is
> already usable, so no loader has to be edited before it runs.

### 19.2 Merging (11 nodes)

The merge nodes take the two inputs' keys, align them and do the arithmetic **directly on the
tensors**, returning the result as a new container. They deliberately do not build a `ModelPatcher`
patch list: a patch is only applied when the model is evaluated, which needs the removed engine.
Doing the maths up front means the result is already final, deterministic and inspectable — and
therefore saveable. Only keys both inputs share are blended; a key that exists in one input only is
carried over unchanged and reported.

| Node | Class | Inputs | Widgets | Result |
|------|-------|--------|---------|--------|
| ModelMergeSimple | `ModelMergeSimple` | `model1`, `model2` | `ratio` FLOAT 1.0 (0~1) | `model1 * ratio + model2 * (1 - ratio)` |
| ModelMergeBlocks | `ModelMergeBlocks` | `model1`, `model2` | `input` FLOAT 1.0, `middle` FLOAT 1.0, `out` FLOAT 1.0 (0~1) | one ratio per UNet block group (`input_blocks` / `middle_block` / `output_blocks`); keys in no group use `input`, matching the native node |
| ModelMergeAdd | `ModelMergeAdd` | `model1`, `model2` | — | `model1 + model2` |
| ModelMergeSubtract | `ModelMergeSubtract` | `model1`, `model2` | `multiplier` FLOAT 1.0 (-10~10) | `model1 - multiplier * model2` (a negative multiplier adds it back) |
| CLIPMergeSimple | `CLIPMergeSimple` | `clip1`, `clip2` | `ratio` FLOAT 1.0 (0~1) | the CLIP counterpart of `ModelMergeSimple` |
| CLIPMergeAdd | `CLIPMergeAdd` | `clip1`, `clip2` | — | `clip1 + clip2` |
| CLIPMergeSubtract | `CLIPMergeSubtract` | `clip1`, `clip2` | `multiplier` FLOAT 1.0 (-10~10) | `clip1 - multiplier * clip2` |

> The `CLIP` merges skip `.position_ids` and `.logit_scale` and copy them from `clip1`: they are
> indices / scalars rather than weights, which is exactly why the native nodes skip them too.

The four save nodes write into `output/`. They have no outputs — the file is the result — and they
count as output nodes, so a graph that ends in a save node still runs.

| Node | Class | Inputs | Widget | Writes |
|------|-------|--------|--------|--------|
| ModelSave | `ModelSave` | `model` | `filename_prefix` `"comfydl/diffusion_models"` | the MODEL bucket only, keyed exactly like the model it came from |
| VAESave | `VAESave` | `vae` | `filename_prefix` `"comfydl/vae"` | the VAE bucket, in the layout `Load VAE` expects back |
| CLIPSave | `CLIPSave` | `clip` | `filename_prefix` `"comfydl/clip"` | the CLIP container as a single file (the native node would split a dual encoder into one file per family; `Load CLIP` / `Load CLIP (Dual)` read this single-file form back) |
| Save Checkpoint | `CheckpointSave` | `model`, `clip`, `vae` | `filename_prefix` `"comfydl/checkpoints"` | all three buckets in one `.safetensors`, each with its original key prefix replayed |

> `filename_prefix` accepts `%date%` style placeholders and defaults to a usable path, so a save
> node works without being edited.

### 19.3 Latent (2 nodes)

Both are protocol placeholders (L2): the native IO contract is registered so a workflow validates,
but decoding / encoding an image needs the autoencoder architecture that was dehydrated, so
executing them raises a `RuntimeError` saying so.

| Node | Class | Inputs | Output |
|------|-------|--------|--------|
| VAE Decode | `VAEDecode` | `samples` LATENT, `vae` VAE | `IMAGE` |
| VAE Encode | `VAEEncode` | `pixels` IMAGE, `vae` VAE | `LATENT` |

### 19.4 Conditioning (2 nodes)

Protocol placeholders (L2) for the same reason: text encoding needs a text-encoder architecture.

| Node | Class | Inputs | Widgets | Output |
|------|-------|--------|---------|--------|
| CLIP Text Encode (Prompt) | `CLIPTextEncode` | `text` STRING (multiline, default `"a photo of a cat"`), `clip` CLIP | — | `CONDITIONING` |
| CLIP Set Last Layer | `CLIPSetLastLayer` | `clip` CLIP | `stop_at_clip_layer` INT -1 (-24~-1) | `CLIP` — truncates the encoder for the "clip skip" trick (`-2` is the usual choice) |

> The host registry's `model/latent` category also holds the native `LatentCompositeMasked`, which
> is not part of this refactor and is not listed here.

---

## 20. ComfyUI / 3d (1 node)

`Preview3D` (`comfy_extras/nodes_preview_3d.py`) is the stock 3D preview node kept on its own,
instead of as part of upstream's `Load3D` / Gaussian-splat family. The frontend binds its 3D canvas
to the hard-coded node id `Preview3D`, so a custom node can never render 3D itself — it can only
produce a file and hand it over. This node is that receiving end: it is what `CdlHeatmapsTo3D`
feeds, and it is the only 3D node the dehydrated build needs.

| Node | Class | Inputs | Output |
|------|-------|--------|--------|
| Preview 3D | `Preview3D` | `model_file` STRING \| `File3D` (`obj` / `glb` / `gltf` / `fbx` / `stl` / `usdz`), `camera_info` LOAD3D_CAMERA (optional), `bg_image` IMAGE (optional) | — |

> It has no outputs: the preview travels through the node's `ui` payload, which the frontend routes
> to the canvas. A `File3D` object is written to `output/` under a generated
> `preview3d_<uuid>.<format>` name; a sibling `.mtl` (as produced by `CdlHeatmapsTo3D`) is written
> by the producer rather than by this node, because the rename would otherwise leave the material
> file behind.

---

## Appendix

### Node Registration Mechanism

ComfyDL uses an importlib-based auto-discovery mechanism in `nodes/__init__.py`: it scans all `.py` files under the `nodes/` directory (excluding `__init__.py`), dynamically imports them, and aggregates each module's `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

### Total Node Count

**109 nodes** across 20 categories come from ComfyDL itself; the shipped node library adds 68
ComfyUI core nodes on top. Both registers are listed below:

| Category | Count | Description |
|----------|-------|-------------|
| d2l/Device Utils | 3 | GPU/CPU device queries |
| d2l/CV Models | 5 | CNN fundamentals & model construction |
| d2l/GAN | 2 | GAN training updates |
| utilities | 4 | Windows MessageBox, NoOp pass-through, timing & a mysterious "?" (ComfyUI core category) |
| d2l/Model Utils | 7 | Model info, mode, forward, layers, params, clone & persistence |
| d2l/_Legacy/Model Utils | 1 | Deprecated (soft-archived): `Model Mode`; use the core `Training Mode` on tensor graphs |
| d2l/NLP Models | 13 | RNN/GRU/RNNLM, attention & Seq2Seq model building blocks |
| d2l/_Legacy/NLP Models | 3 | Deprecated (soft-archived): `Add & Norm`, `Transformer Encoder Block`, `Transformer Encoder` |
| d2l/NLP Utils | 5 | Text tokenization & vocabularies |
| d2l/Tensor Basic | 5 | Tensor I/O, conv, transpose, broadcast, reshape, activation |
| d2l/_Legacy/Tensor Basic | 3 | Deprecated (soft-archived): `Broadcast`, `Reshape`, `Activation`, all with core equivalents |
| d2l/TorchOps | 10 | Loss, optimization, metrics |
| d2l/ObjectDetection | 10 | Anchor boxes, IoU, NMS |
| d2l/Segmentation | 4 | VOC semantic segmentation tools |
| d2l/Visualization | 13 | Plots, charts & bounding box visualization |
| d2l/Datasets | 10 | Dataset download, loading, preview, and statistics |
| image/color | 3 | Grayscale, normalize & brightness/contrast/saturation (ComfyUI core category) |
| image/transform | 1 | Arbitrary-angle rotation + expand (ComfyUI core category) |
| image | 1 | Per-channel image batch statistics (ComfyUI core category) |
| utilities/conversion | 6 | Comfy value ↔ generic `TENSOR` round-trip (ComfyUI core category) |
| Network & Layers/Activation | 14 | Core activation functions on the `TENSOR` type (ComfyUI core category) |
| Network & Layers/Basic | 8 | Core basic layers & tensor ops on the `TENSOR` type (ComfyUI core category) |
| Network & Layers/Normalization | 7 | Core normalizations on the `TENSOR` type (ComfyUI core category) |
| Network & Layers/Regularization | 1 | Core element-wise dropout with a seeded mask (ComfyUI core category) |
| Network & Layers/Training | 11 | Train/eval switch, running statistics, learnable parameters, optimizer settings and the training loop (ComfyUI core category) |
| Network & Layers/Pooling | 2 | Max / average pooling, sliding-window and adaptive (`output_size=1` is global pooling) (ComfyUI core category) |
| Network & Layers/Convolution | 2 | Convolution & transposed convolution on the `TENSOR` type, weights wired in (ComfyUI core category) |
| model/loaders | 7 | Checkpoint / diffusion-model / VAE / CLIP loaders at state_dict level (ComfyUI core category) |
| model/merging | 11 | Key-aligned model & CLIP merging plus `.safetensors` saving (ComfyUI core category) |
| model/latent | 2 | `VAE Decode` / `VAE Encode` protocol placeholders (ComfyUI core category) |
| model/conditioning | 2 | `CLIP Text Encode (Prompt)` / `CLIP Set Last Layer` protocol placeholders (ComfyUI core category) |
| 3d | 1 | `Preview 3D`, the frontend-bound 3D preview canvas (ComfyUI core category) |

> The first 20 rows list the **109 ComfyDL-provided nodes** (7 of them soft-archived into
> `d2l/_Legacy/*`: nothing was removed, old workflows still load, but their display names carry a
> `(DEPRECATED)` suffix and the node library moves them into the Legacy categories). `utilities`, `utilities/conversion`, `image/color`, `image/transform` and `image` are ComfyUI core categories that ComfyDL nodes were merged into, so those categories also contain native ComfyUI nodes.
>
> The other 12 rows are pure ComfyUI core categories with no ComfyDL nodes: the seven
> `Network & Layers/*` groups (45 nodes), the four `model/*` groups (22 nodes) and `3d` (1 node).
> The shipped library therefore totals **177 nodes across 32 categories** = 109 ComfyDL + 68 core.
>
> Two rows list fewer nodes than the host registry holds in that category, because the registry
> also counts native nodes that this refactor did not touch: `model/latent` (whose third node is
> `LatentCompositeMasked`), and `image`, `utilities`, `image/color`, `image/transform`, which mix
> ComfyDL nodes with the native ones they were merged next to.
