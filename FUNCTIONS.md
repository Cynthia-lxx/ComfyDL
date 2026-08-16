# ComfyDL Node Reference

> [中文版 / 中文版本](FUNCTIONS_zh.md)

This document details every custom node in ComfyDL: what it does, its inputs, and its outputs. The node suite began as a mapping of the `d2l` textbook codebase and keeps growing with self-developed nodes beyond it.

---

## Custom Data Types

ComfyDL defines 5 custom ComfyUI data types for passing structured data between nodes:

| Type Name | Python Type | Description |
|-----------|-------------|-------------|
| `cdlTensor` | `torch.Tensor` | PyTorch tensor of arbitrary shape |
| `cdlModel` | `nn.Module` | PyTorch model instance |
| `cdlVocab` | `dict` | Vocabulary dictionary containing `idx_to_token` and `token_to_idx` |
| `cdlDataloader` | `torch.utils.data.DataLoader` | PyTorch data loader |
| `cdlBbox` | `torch.Tensor [N,4]` | Bounding box tensor in `(x1, y1, x2, y2)` format |

ComfyUI standard types (used directly):
- `IMAGE` — Image batch, `torch.Tensor [B, H, W, C]`
- `MASK` — Mask, `torch.Tensor [H, W]` or `[B, C, H, W]`
- `INT`, `FLOAT`, `STRING`, `BOOLEAN` — Primitive scalar types

---

## 1. ComfyDL / Device Utils (3 nodes)

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

## 2. ComfyDL / CV Models (5 nodes)

### Corr2D
- **Class**: `CdlCorr2d`
- **d2lcore function**: `corr2d(X, K)`
- **Purpose**: Performs 2D cross-correlation on input tensor — the fundamental operation underlying convolution.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `input_tensor` | `cdlTensor` | Input 2D tensor |
  | `kernel` | `cdlTensor` | Kernel 2D tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Cross-correlation result |

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

## 3. ComfyDL / GAN (2 nodes)

### Update Discriminator
- **Class**: `CdlUpdateD`
- **d2lcore function**: `update_D(X, Z, net_D, net_G, loss, trainer_D)`
- **Purpose**: Performs one training update of the GAN discriminator. Uses real data `X` and fake data generated by `net_G` from noise `Z`, computes BCE loss, and backpropagates to update discriminator parameters. Uses SGD optimizer (lr=0.01).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `cdlTensor` | Real data batch (optional) |
  | `Z` | `cdlTensor` | Noise input (optional) |
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
  | `Z` | `cdlTensor` | Noise input (optional) |
  | `net_D` | `cdlModel` | Discriminator model (optional) |
  | `net_G` | `cdlModel` | Generator model (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `loss_G` | `FLOAT` | Generator loss; returns 0.0 if any input is missing |
  | `fake_X` | `cdlTensor` | Fake data produced by the generator |

---

## 4. ComfyDL / Misc (2 nodes)

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

---

## 5. ComfyDL / NLP Utils (5 nodes)

### Tokenize
- **Class**: `CdlTokenize`
- **d2lcore function**: `tokenize(lines, token)`
- **Purpose**: Splits input text into tokens line by line. Supports word-level (by whitespace) and character-level tokenization modes. Each line is treated as one sentence; tokens within a line are comma-separated in the output, and lines are newline-separated.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `text` | `STRING` | `""` | Input text, one sentence per line (multiline) |
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
  | `tokens_a` | `STRING` | `""` | Segment A, comma-separated tokens (multiline) |
  | `tokens_b` | `STRING` | `""` | Segment B, comma-separated tokens (optional, multiline) |
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
  | `tokens_text` | `STRING` | `""` | Token text, one token per line or comma-separated (multiline) |
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
  | Name | Type | Description |
  |------|------|-------------|
  | `vocab` | `cdlVocab` | Vocabulary dictionary |
  | `tokens` | `STRING` | Comma-separated token sequence (multiline) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `indices` | `cdlTensor` | Encoded index tensor (`torch.int64`) |

### Vocab Decode
- **Class**: `CdlVocabDecode`
- **d2lcore function**: `Vocab.to_tokens(indices)`
- **Purpose**: Converts index tensor back to token strings using the vocabulary. Out-of-range indices are mapped to `<unk>`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `vocab` | `cdlVocab` | Vocabulary dictionary |
  | `indices` | `cdlTensor` | Index tensor |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tokens_str` | `STRING` | Decoded token string, comma-separated |

---

## 6. ComfyDL / Tensor Basic (7 nodes)

### Tensor → String
- **Class**: `CdlTensorToStr`
- **Purpose**: Formats a tensor as a human-readable string. Displays the tensor's shape, dtype, device, and numeric content. Large tensors exceeding `max_elems` are truncated (showing first half + second half).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `cdlTensor` | — | Input tensor |
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
  | `text` | `STRING` | `""` | String representation of tensor (multiline), e.g. `"[[1,2],[3,4]]"` |
  | `error_strategy` | `COMBO` | `empty_tensor` | Error handling: `empty_tensor`=return empty tensor; `zero_tensor`=return `[0.]`; `raise_error`=raise exception |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `tensor` | `cdlTensor` | Parsed tensor (`torch.float32`) |

### Conv2D
- **Class**: `CdlConv2d`
- **Purpose**: Performs 2D convolution on an input tensor with a kernel. Wraps ``torch.nn.functional.conv2d`` with configurable stride and padding. Auto-expands 2-D/3-D inputs to 4-D ``(N, C, H, W)`` and squeezes output back.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `input_tensor` | `cdlTensor` | — | Input tensor |
  | `kernel` | `cdlTensor` | — | Convolution kernel |
  | `stride` | `INT` | 1 | Convolution stride (1~4) |
  | `padding` | `INT` | 0 | Zero padding (0~10) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Convolved result |

### Transpose
- **Class**: `CdlTranspose`
- **Purpose**: Swaps two dimensions of a tensor. Wraps ``torch.transpose``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `cdlTensor` | — | Input tensor |
  | `dim0` | `INT` | 0 | First dimension to swap (0~5) |
  | `dim1` | `INT` | 1 | Second dimension to swap (0~5) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Transposed tensor |

### Broadcast
- **Class**: `CdlBroadcast`
- **Purpose**: Broadcasts a tensor to a target shape. Wraps ``torch.broadcast_to``. Enter the target shape as a comma-separated string (e.g. ``"3,1,4"``). Returns original tensor unchanged on parse error.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `cdlTensor` | — | Input tensor |
  | `target_shape` | `STRING` | `""` | Target shape, comma-separated (e.g. ``"3,1,4"``) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Broadcasted tensor |

### Reshape
- **Class**: `CdlReshape`
- **Purpose**: Reshapes a tensor to a new shape. Wraps ``torch.reshape``. Enter the target shape as a comma-separated string (e.g. ``"2,8"``, ``"4,-1"``). Returns original tensor unchanged on parse error.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `cdlTensor` | — | Input tensor |
  | `target_shape` | `STRING` | `""` | Target shape, comma-separated (e.g. ``"2,8"``) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Reshaped tensor |

### Activation
- **Class**: `CdlActivation`
- **Purpose**: Applies an element-wise activation function to a tensor. Select the function from a combo widget: ``relu``, ``sigmoid``, ``tanh``, ``leaky_relu``, ``elu``, ``gelu``, ``silu``, ``softmax``, ``softplus``.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `tensor` | `cdlTensor` | — | Input tensor |
  | `func` | `COMBO` | `relu` | Activation: relu / sigmoid / tanh / leaky_relu / elu / gelu / silu / softmax / softplus |
  | `dim` | `INT` | -1 | Dimension for softmax (-4~4) |
  | `negative_slope` | `FLOAT` | 0.01 | Slope for leaky_relu (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Activated tensor |

---

## 7. ComfyDL / TorchOps (10 nodes)

### Linear Regression
- **Class**: `CdlLinReg`
- **d2lcore function**: `linreg(X, w, b)`
- **Purpose**: Linear regression forward computation: \( \hat{y} = X w + b \)
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `cdlTensor` | Input feature matrix |
  | `w` | `cdlTensor` | Weight vector |
  | `b` | `cdlTensor` | Bias vector/scalar |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `cdlTensor` | Predicted values |

### Squared Loss
- **Class**: `CdlSquaredLoss`
- **d2lcore function**: `squared_loss(y_hat, y)`
- **Purpose**: Computes squared loss: \( \frac{1}{2}(\hat{y} - y)^2 \) (note: no averaging).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `cdlTensor` | Predicted values |
  | `y` | `cdlTensor` | Ground truth values (auto-reshaped to match `y_hat` shape) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `loss` | `cdlTensor` | Element-wise loss |

### Masked Softmax
- **Class**: `CdlMaskedSoftmax`
- **d2lcore function**: `masked_softmax(X, valid_lens)`
- **Purpose**: Performs softmax with masking on the last dimension. Uses `valid_lens` to specify the effective length of each sequence; positions beyond the valid length are set to -1e6 before softmax (making their probability near 0).
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `X` | `cdlTensor` | Input tensor |
  | `valid_lens` | `cdlTensor` | Valid length tensor (optional; if not provided, normal softmax is performed) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `output` | `cdlTensor` | Masked softmax result |

### Sequence Mask
- **Class**: `CdlSequenceMask`
- **d2lcore function**: `sequence_mask(X, valid_len, value)`
- **Purpose**: Replaces positions in a sequence that exceed the valid length with a specified value. Commonly used to zero out padding positions.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `X` | `cdlTensor` | — | Input sequence tensor |
  | `valid_len` | `cdlTensor` | — | Effective length for each sequence |
  | `mask_value` | `FLOAT` | 0.0 | Mask fill value (-1e9~1e9) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `masked` | `cdlTensor` | Masked tensor |

### Accuracy
- **Class**: `CdlAccuracy`
- **d2lcore function**: `accuracy(y_hat, y)`
- **Purpose**: Computes the number of correct predictions for classification. If the prediction is multi-class logits, argmax is taken first, then compared against labels.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `y_hat` | `cdlTensor` | Predictions (logits or class indices) |
  | `y` | `cdlTensor` | Ground truth labels |
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
  | `X` | `cdlTensor` | Feature matrix `[num_examples, num_features]` |
  | `y` | `cdlTensor` | Label vector `[num_examples, 1]` |

### Truncate/Pad
- **Class**: `CdlTruncatePad`
- **d2lcore function**: `truncate_pad(line, num_steps, padding_token)`
- **Purpose**: Truncates or pads a sequence to a fixed length. Sequences longer than the target are truncated; shorter ones are padded with `padding_token`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `num_steps` | `INT` | 64 | Target sequence length (1~10000) |
  | `padding_token` | `INT` | 0 | Padding token index (0~100000) |
  | `sequence` | `cdlTensor` | — | Input index sequence (optional; returns all-padding tensor when absent) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `padded` | `cdlTensor` | Truncated/padded sequence (`torch.int64`) |

### BLEU Score
- **Class**: `CdlBleu`
- **d2lcore function**: `bleu(pred_seq, label_seq, k)`
- **Purpose**: Computes the BLEU score between a predicted sequence and a reference sequence. Supports BLEU-1 through BLEU-4 (controlled by `max_n`). Tokenizes by whitespace, then computes n-gram precision and brevity penalty.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `pred_seq` | `STRING` | `""` | Predicted sequence, whitespace-separated tokens (multiline) |
  | `label_seq` | `STRING` | `""` | Reference sequence, whitespace-separated tokens (multiline) |
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

## 8. ComfyDL / ObjectDetection (10 nodes)

### Box Corner→Center
- **Class**: `CdlBoxCornerToCenter`
- **d2lcore function**: `box_corner_to_center(boxes)`
- **Purpose**: Converts bounding boxes from corner format `(x1, y1, x2, y2)` to center format `(cx, cy, w, h)`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes` | `cdlTensor` | Corner-format boxes `[N,4]` (top-left + bottom-right) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes_ccwh` | `cdlTensor` | Center-format boxes `[N,4]` (center + width + height) |

### Box Center→Corner
- **Class**: `CdlBoxCenterToCorner`
- **d2lcore function**: `box_center_to_corner(boxes)`
- **Purpose**: Converts bounding boxes from center format `(cx, cy, w, h)` back to corner format `(x1, y1, x2, y2)`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes` | `cdlTensor` | Center-format boxes `[N,4]` (center + width + height) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes_xyxy` | `cdlTensor` | Corner-format boxes `[N,4]` (top-left + bottom-right) |

### Box IoU
- **Class**: `CdlBoxIou`
- **d2lcore function**: `box_iou(boxes1, boxes2)`
- **Purpose**: Computes pairwise IoU (Intersection over Union) between two sets of bounding boxes. Returns matrix `[N1, N2]` where element `(i,j)` is the IoU of `boxes1[i]` and `boxes2[j]`.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `boxes1` | `cdlTensor` | First set of boxes `[N1,4]` (top-left + bottom-right) |
  | `boxes2` | `cdlTensor` | Second set of boxes `[N2,4]` (top-left + bottom-right) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `iou` | `cdlTensor` | IoU matrix `[N1, N2]` |

### NMS
- **Class**: `CdlNms`
- **d2lcore function**: `nms(boxes, scores, iou_threshold)`
- **Purpose**: Performs Non-Maximum Suppression on bounding boxes. Sorts by score descending, keeps boxes whose IoU with any higher-scoring box does not exceed the threshold.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `boxes` | `cdlTensor` | — | Boxes `[N,4]` (top-left + bottom-right) |
  | `scores` | `cdlTensor` | — | Confidence scores for each box |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `keep_indices` | `cdlTensor` | Indices of kept boxes (`torch.int64`) |

### Multibox Prior
- **Class**: `CdlMultiboxPrior`
- **d2lcore function**: `multibox_prior(data, sizes, ratios)`
- **Purpose**: Generates anchor boxes of different shapes centered at each pixel. Number of anchors per pixel = `len(sizes) + len(ratios) - 1`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `sizes` | `STRING` | `"0.75,0.5,0.25"` | Anchor size list, comma-separated |
  | `ratios` | `STRING` | `"1,2,0.5"` | Aspect ratio list, comma-separated |
  | `data` | `cdlTensor` | — | Input data (optional; used to infer spatial dimensions; defaults to 561×728 when absent) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `cdlTensor` | Anchors `[1, H*W*bpp, 4]`, normalized coordinates (top-left + bottom-right) |

### Offset Boxes
- **Class**: `CdlOffsetBoxes`
- **d2lcore function**: `offset_boxes(anchors, assigned_bb, eps)`
- **Purpose**: Computes offset from anchor boxes to assigned ground-truth boxes. Center coordinate differences are scaled by 10×; width/height ratios are log-scaled by 5×.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `anchors` | `cdlTensor` | — | Anchors `[N,4]` (top-left + bottom-right) |
  | `assigned_bb` | `cdlTensor` | — | Assigned ground-truth boxes `[N,4]` (top-left + bottom-right) |
  | `eps` | `FLOAT` | 1e-6 | Small epsilon to prevent division by zero (1e-12~1e-3) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `offsets` | `cdlTensor` | Offsets `[N,4]` (dx, dy, dw, dh) |

### Offset Inverse
- **Class**: `CdlOffsetInverse`
- **d2lcore function**: `offset_inverse(anchors, offset_preds)`
- **Purpose**: Reconstructs bounding box coordinates (corner format) from anchors and predicted offsets via inverse transformation.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `cdlTensor` | Anchors `[N,4]` (top-left + bottom-right) |
  | `offset_preds` | `cdlTensor` | Predicted offsets `[N,4]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `predicted_bbox` | `cdlTensor` | Predicted boxes `[N,4]` (top-left + bottom-right) |

### Assign Anchor→BBox
- **Class**: `CdlAssignAnchorToBbox`
- **d2lcore function**: `assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)`
- **Purpose**: Assigns ground-truth bounding boxes to anchor boxes based on IoU. Each anchor is assigned to a ground-truth box (IoU ≥ threshold), and each ground-truth box is guaranteed at least one anchor (the one with the highest IoU).
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `ground_truth` | `cdlTensor` | — | Ground-truth boxes `[M,4]` (top-left + bottom-right) |
  | `anchors` | `cdlTensor` | — | Anchors `[N,4]` (top-left + bottom-right) |
  | `iou_threshold` | `FLOAT` | 0.5 | IoU threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors_bbox_map` | `cdlTensor` | Anchor→ground-truth mapping `[N,]`, -1 means no match (`torch.int64`) |

### Multibox Target
- **Class**: `CdlMultiboxTarget`
- **d2lcore function**: `multibox_target(anchors, labels)`
- **Purpose**: Generates multi-box target training labels for anchors. For each image in the batch, assigns ground-truth boxes to anchors and computes offset targets, masks, and class labels.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `anchors` | `cdlTensor` | Anchors `[1, N, 4]` (top-left + bottom-right) |
  | `labels` | `cdlTensor` | Labels `[B, M, 5]`, format `[class_id, x1, y1, x2, y2]` |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `bbox_offset` | `cdlTensor` | Bounding box offset targets `[B, N*4]` |
  | `bbox_mask` | `cdlTensor` | Bounding box offset masks `[B, N*4]` (1.0 for matched anchors) |
  | `class_labels` | `cdlTensor` | Anchor class labels `[B, N]` (background=0, classes start from 1) |

### Multibox Detection
- **Class**: `CdlMultiboxDetection`
- **d2lcore function**: `multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)`
- **Purpose**: Predicts bounding boxes from model outputs using NMS. Combines class predictions and offsets, reconstructs boxes via inverse offset transform, and filters through NMS and confidence thresholding.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `cls_probs` | `cdlTensor` | — | Class probabilities `[B, num_classes, N]` |
  | `offset_preds` | `cdlTensor` | — | Offset predictions `[B, N*4]` |
  | `anchors` | `cdlTensor` | — | Anchors `[1, N, 4]` |
  | `nms_threshold` | `FLOAT` | 0.5 | NMS IoU threshold (0~1) |
  | `pos_threshold` | `FLOAT` | 0.01 | Positive confidence threshold (0~1) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `detections` | `cdlTensor` | Detection results `[B, N, 6]`, format `[class_id, confidence, x1, y1, x2, y2]` (class_id=-1 means background) |

---

## 9. ComfyDL / Segmentation (4 nodes)

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
  | `colormap2label` | `cdlTensor` | Color→class index lookup table `[16777216]` (`torch.int64`) |

### VOC Label Indices
- **Class**: `CdlVocLabelIndices`
- **d2lcore function**: `voc_label_indices(colormap, colormap2label)`
- **Purpose**: Maps a VOC label color image to a class index map. Encodes RGB pixels as a single color value, then converts to class indices via the lookup table.
- **Inputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `colormap` | `IMAGE` | VOC label color image `[B, H, W, C]`, uses the first image |
  | `colormap2label` | `cdlTensor` | Color→label lookup table |
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

## 10. ComfyDL / Visualization (13 nodes)

Visualization nodes follow a "dual variant" design pattern: `(Output)` suffix versions are ComfyUI output nodes (showing interactive plots directly in the UI), while non-suffix versions render plots as `IMAGE` tensors for downstream nodes.

### Show Images (Output)
- **Class**: `CdlShowImagesOutput`
- **d2lcore function**: `show_images(imgs, num_rows, num_cols, titles, scale)`
- **Purpose**: Displays a batch of images in a grid layout. Output node variant — images render directly in the ComfyUI interface.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `images` | `IMAGE` | — | Input image batch `[B, H, W, C]` |
  | `num_rows` | `INT` | 1 | Grid rows (1~100) |
  | `num_cols` | `INT` | 4 | Grid columns (1~100) |
  | `scale` | `FLOAT` | 1.5 | Image scaling factor (0.1~10.0) |
  | `titles` | `STRING` | `""` | Image titles, comma-separated (optional) |
- **Outputs**: None (output node)

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
- **Purpose**: Displays heatmap matrices in a grid layout. Output node variant, with colorbar. Input matrix dimensions are auto-promoted to 4D.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `matrices` | `cdlTensor` | — | Matrix tensor (2D=[H,W]→1×1, 3D=[N,H,W]→N×1, 4D=[N,M,H,W]) |
  | `xlabel` | `STRING` | `""` | X-axis label |
  | `ylabel` | `STRING` | `""` | Y-axis label |
  | `figsize_w` | `FLOAT` | 2.5 | Per-column width (0.5~20.0) |
  | `figsize_h` | `FLOAT` | 2.5 | Per-row height (0.5~20.0) |
  | `cmap` | `STRING` | `"Reds"` | matplotlib colormap name |
  | `titles` | `STRING` | `""` | Subplot titles, comma-separated (optional) |
- **Outputs**: None (output node)

### Show Heatmaps
- **Class**: `CdlShowHeatmaps`
- **d2lcore function**: Same as above
- **Purpose**: Same as `Show Heatmaps (Output)`, but renders the plot as an IMAGE tensor.
- **Inputs**: Same as above
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Rendered heatmap `[1, H, W, C]` |

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
  | `X` | `cdlTensor` | — | X-axis data (optional, 1D or 2D) |
  | `Y` | `cdlTensor` | — | Y-axis data (optional) |
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
  | `results` | `cdlTensor` | Optimization trajectory points `[N, 2]`, each row is a (x1, x2) coordinate |
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
  | `bboxes` | `cdlTensor` | — | Boxes `[N, 4]`, normalized coordinates (top-left + bottom-right) |
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
  | `tensor` | `cdlTensor` | — | Input tensor (flattened internally) |
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
  | `values` | `cdlTensor` | — | Bar heights (1-D tensor) |
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
  | `X` | `cdlTensor` | — | X coordinates (flattened) |
  | `Y` | `cdlTensor` | — | Y coordinates (flattened) |
  | `alpha` | `FLOAT` | 0.6 | Point transparency |
  | `cmap` | `STRING` | `"viridis"` | Colormap for ``color_map`` |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
  | `figsize_w` | `FLOAT` | 6.0 | Figure width |
  | `figsize_h` | `FLOAT` | 5.0 | Figure height |
  | `color_map` | `cdlTensor` | — | Per-point colour values (optional) |
  | `size_map` | `cdlTensor` | — | Per-point size values (optional) |
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
  | `matrix` | `cdlTensor` | — | Confusion matrix (N×N or flat) |
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
  | `values` | `cdlTensor` | — | Slice values (1-D tensor) |
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
  | `Y` | `cdlTensor` | — | Series data, `[T]` or `[N, T]` |
  | `stacked` | `BOOLEAN` | False | Stack series instead of overlay |
  | `alpha` | `FLOAT` | 0.5 | Fill transparency |
  | `color_palette` | `STRING` | `"tab10"` | matplotlib palette name |
  | `xlabel` | `STRING` | `""` | x-axis label |
  | `ylabel` | `STRING` | `""` | y-axis label |
  | `figsize_w` | `FLOAT` | 7.0 | Figure width |
  | `figsize_h` | `FLOAT` | 4.0 | Figure height |
  | `X_vals` | `cdlTensor` | — | Custom x-axis values (optional) |
  | `labels` | `STRING` | `""` | Series legend labels, comma-separated (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `image` | `IMAGE` | Area chart `[1, H, W, C]` |

---

## 11. ComfyDL / Datasets (10 nodes)

Datasets nodes provide end-to-end dataset management: download, load, inspect, preview, and compute statistics.

### Load Array → DataLoader
- **Class**: `CdlLoadArray`
- **d2lcore function**: `load_array(data_arrays, batch_size, is_train)`
- **Purpose**: Wraps one or more tensors into a PyTorch DataLoader. Connect `cdlTensor` features and/or labels to the optional slots; the node outputs a `cdlDataloader`.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `batch_size` | `INT` | 32 | Batch size (1~4096) |
  | `shuffle` | `BOOLEAN` | True | Shuffle data on each epoch |
  | `features` | `cdlTensor` | — | Feature tensor X (optional) |
  | `labels` | `cdlTensor` | — | Label tensor y (optional) |
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
- **Purpose**: Same as `CdlDataLoaderPreview` but renders directly as an OUTPUT_NODE — no output slots, just the rendered preview in the UI.
- **Inputs**: Same as `CdlDataLoaderPreview`
- **Outputs**: None (output node)

### Dataset Stats
- **Class**: `CdlDataLoaderStats`
- **Purpose**: Iterates over a `cdlDataloader` and computes class distribution statistics. Renders a bar chart showing per-class counts.
- **Inputs**:
  | Name | Type | Default | Description |
  |------|------|---------|-------------|
  | `dataloader` | `cdlDataloader` | — | DataLoader to analyze |
  | `num_classes` | `INT` | 10 | Expected number of classes (1~1000) |
  | `class_names` | `STRING` | `""` | Comma-separated class names (optional) |
- **Outputs**:
  | Name | Type | Description |
  |------|------|-------------|
  | `stats_text` | `STRING` | Formatted text summary of class counts |
  | `stats_image` | `IMAGE` | Bar chart of class distribution `[1, H, W, C]` |

---

## Appendix

### Node Registration Mechanism

ComfyDL uses an importlib-based auto-discovery mechanism in `nodes/__init__.py`: it scans all `.py` files under the `nodes/` directory (excluding `__init__.py`), dynamically imports them, and aggregates each module's `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

### Total Node Count

**71 nodes** across 11 categories:

| Category | Count | Description |
|----------|-------|-------------|
| ComfyDL/Device Utils | 3 | GPU/CPU device queries |
| ComfyDL/CV Models | 5 | CNN fundamentals & model construction |
| ComfyDL/GAN | 2 | GAN training updates |
| ComfyDL/Misc | 2 | Windows MessageBox & NoOp pass-through |
| ComfyDL/NLP Utils | 5 | Text tokenization & vocabularies |
| ComfyDL/Tensor Basic | 7 | Tensor I/O, conv, transpose, broadcast, activation |
| ComfyDL/TorchOps | 10 | Loss, optimization, metrics |
| ComfyDL/ObjectDetection | 10 | Anchor boxes, IoU, NMS |
| ComfyDL/Segmentation | 4 | VOC semantic segmentation tools |
| ComfyDL/Visualization | 13 | Plots, charts & bounding box visualization |
| ComfyDL/Datasets | 10 | Dataset download, loading, preview, and statistics |
