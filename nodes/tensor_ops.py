"""
d2lcore/TensorOps - Tensor operations, metrics, and loss functions.

d2lcore functions:
  - linreg(X, w, b)            : Linear regression forward pass
  - squared_loss(y_hat, y)     : Squared loss
  - sgd(params, lr, batch_size): Minibatch SGD
  - masked_softmax(X, valid_lens): Masked softmax
  - sequence_mask(X, valid_len, value): Mask irrelevant entries
  - accuracy(y_hat, y)         : Compute number of correct predictions
  - synthetic_data(w, b, num_examples): Generate synthetic linear data
  - evaluate_loss(net, data_iter, loss): Evaluate model loss
  - evaluate_accuracy_gpu(net, data_iter, device): GPU accuracy eval
  - grad_clipping(net, theta)  : Clip gradients
  - check_len(a, n)            : Check list length
  - check_shape(a, shape)      : Check tensor shape
  - truncate_pad(line, num_steps, padding_token): Truncate/pad sequence
  - bleu(pred_seq, label_seq, k): Compute BLEU score
"""

import torch
import math
import collections

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlLinReg:
    """Linear regression model: y = X @ w + b.

    d2lcore: linreg(X, w, b)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("cdlTensor",),
                "w": ("cdlTensor",),
                "b": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("y_hat",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, X, w, b):
        result = torch.matmul(X, w) + b
        return (result,)


NODE_CLASS_MAPPINGS["CdlLinReg"] = CdlLinReg
NODE_DISPLAY_NAME_MAPPINGS["CdlLinReg"] = "Linear Regression"


class CdlSquaredLoss:
    """Squared loss: (y_hat - y)^2 / 2.

    d2lcore: squared_loss(y_hat, y)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "y_hat": ("cdlTensor",),
                "y": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("loss",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, y_hat, y):
        y = y.reshape(y_hat.shape)
        result = (y_hat - y) ** 2 / 2
        return (result,)


NODE_CLASS_MAPPINGS["CdlSquaredLoss"] = CdlSquaredLoss
NODE_DISPLAY_NAME_MAPPINGS["CdlSquaredLoss"] = "Squared Loss"


class CdlMaskedSoftmax:
    """Masked softmax operation on the last axis.

    d2lcore: masked_softmax(X, valid_lens)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("cdlTensor",),
            },
            "optional": {
                "valid_lens": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, X, valid_lens=None):
        if valid_lens is None:
            return (torch.nn.functional.softmax(X, dim=-1),)

        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = X.clone()  # 避免就地修改上游输入张量
        X_flat = X.reshape(-1, shape[-1])
        maxlen = X_flat.size(1)
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
        X_flat[~mask] = -1e6
        result = torch.nn.functional.softmax(X_flat.reshape(shape), dim=-1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlMaskedSoftmax"] = CdlMaskedSoftmax
NODE_DISPLAY_NAME_MAPPINGS["CdlMaskedSoftmax"] = "Masked Softmax"


class CdlSequenceMask:
    """Mask irrelevant entries in sequences.

    d2lcore: sequence_mask(X, valid_len, value)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("cdlTensor",),
                "valid_len": ("cdlTensor",),
                "mask_value": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("masked",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, X, valid_len, mask_value):
        maxlen = X.size(1)
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X_out = X.clone()
        X_out[~mask] = mask_value
        return (X_out,)


NODE_CLASS_MAPPINGS["CdlSequenceMask"] = CdlSequenceMask
NODE_DISPLAY_NAME_MAPPINGS["CdlSequenceMask"] = "Sequence Mask"


class CdlAccuracy:
    """Compute the number of correct predictions.

    d2lcore: accuracy(y_hat, y)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "y_hat": ("cdlTensor",),
                "y": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("accuracy", "count")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(dim=1)
        cmp = y_hat.type(y.dtype) == y
        count = float(cmp.type(y.dtype).sum())
        return (count, int(count))


NODE_CLASS_MAPPINGS["CdlAccuracy"] = CdlAccuracy
NODE_DISPLAY_NAME_MAPPINGS["CdlAccuracy"] = "Accuracy"


class CdlSyntheticData:
    """Generate synthetic linear data: y = Xw + b + noise.

    d2lcore: synthetic_data(w, b, num_examples)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_features": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1}),
                "num_examples": ("INT", {"default": 100, "min": 1, "max": 1000000, "step": 100}),
                "noise_std": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlTensor", "cdlTensor")
    RETURN_NAMES = ("X", "y")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, num_features, num_examples, noise_std, seed):
        torch.manual_seed(seed)
        w = torch.normal(0, 1, (num_features, 1))
        b = torch.tensor([0.0])
        X = torch.normal(0, 1, (num_examples, num_features))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, noise_std, y.shape)
        return (X, y.reshape((-1, 1)))


NODE_CLASS_MAPPINGS["CdlSyntheticData"] = CdlSyntheticData
NODE_DISPLAY_NAME_MAPPINGS["CdlSyntheticData"] = "Synthetic Data"


class CdlTruncatePad:
    """Truncate or pad a sequence to a fixed length.

    d2lcore: truncate_pad(line, num_steps, padding_token)
    Takes a list of token indices and pads/truncates to num_steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_steps": ("INT", {"default": 64, "min": 1, "max": 10000, "step": 1}),
                "padding_token": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                "sequence": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("padded",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, num_steps, padding_token, sequence=None):
        if sequence is None:
            # Return a dummy padded tensor
            result = torch.full((num_steps,), padding_token, dtype=torch.long)
            return (result,)

        seq_list = sequence.tolist()
        if len(seq_list) > num_steps:
            seq_list = seq_list[:num_steps]
        else:
            seq_list = seq_list + [padding_token] * (num_steps - len(seq_list))
        return (torch.tensor(seq_list, dtype=torch.long),)


NODE_CLASS_MAPPINGS["CdlTruncatePad"] = CdlTruncatePad
NODE_DISPLAY_NAME_MAPPINGS["CdlTruncatePad"] = "Truncate/Pad"


class CdlBleu:
    """Compute BLEU score between prediction and reference.

    d2lcore: bleu(pred_seq, label_seq, k)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pred_seq": ("STRING", {"default": "the quick brown", "multiline": True}),
                "label_seq": ("STRING", {"default": "the quick brown fox", "multiline": True}),
                "max_n": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("bleu_score",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, pred_seq, label_seq, max_n):
        pred_tokens = pred_seq.split(' ')
        label_tokens = label_seq.split(' ')
        len_pred = len(pred_tokens)
        len_label = len(label_tokens)
        score = math.exp(min(0, 1 - len_label / max(len_pred, 1)))
        for n in range(1, min(max_n, len_pred) + 1):
            num_matches = 0
            label_subs = collections.defaultdict(int)
            for i in range(len_label - n + 1):
                label_subs[' '.join(label_tokens[i: i + n])] += 1
            for i in range(len_pred - n + 1):
                key = ' '.join(pred_tokens[i: i + n])
                if label_subs[key] > 0:
                    num_matches += 1
                    label_subs[key] -= 1
            denom = max(len_pred - n + 1, 1)
            score *= math.pow(num_matches / denom, math.pow(0.5, n))
        return (score,)


NODE_CLASS_MAPPINGS["CdlBleu"] = CdlBleu
NODE_DISPLAY_NAME_MAPPINGS["CdlBleu"] = "BLEU Score"


class CdlGradClipping:
    """Clip gradients of a model.

    d2lcore: grad_clipping(net, theta)
    Note: Gradients must already be computed. This node clips existing grads.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "theta": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("norm",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, theta, model=None):
        if model is None:
            return (0.0,)

        if isinstance(model, torch.nn.Module):
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        else:
            params = getattr(model, 'params', [])

        if not params:
            return (0.0,)

        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
        return (float(norm.item()),)


NODE_CLASS_MAPPINGS["CdlGradClipping"] = CdlGradClipping
NODE_DISPLAY_NAME_MAPPINGS["CdlGradClipping"] = "Gradient Clip"


class CdlSgdStep:
    """Minibatch stochastic gradient descent step.

    d2lcore: sgd(params, lr, batch_size)
    Note: Gradients must already be computed. This applies the SGD update.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lr": ("FLOAT", {"default": 0.03, "min": 1e-8, "max": 10.0, "step": 0.001}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 65536, "step": 1}),
            },
            "optional": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/TorchOps"

    def execute(self, lr, batch_size, model=None):
        if model is None:
            return (None,)

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad / batch_size
                    param.grad.zero_()
        return (model,)


NODE_CLASS_MAPPINGS["CdlSgdStep"] = CdlSgdStep
NODE_DISPLAY_NAME_MAPPINGS["CdlSgdStep"] = "SGD Step"
