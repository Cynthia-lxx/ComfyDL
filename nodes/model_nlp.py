"""
d2lcore/NLP Models - RNN/GRU/RNNLM model builders and inference.

d2lcore classes:
  - RNNScratch(num_inputs, num_hiddens, sigma) : RNN from scratch (tanh cell)
  - RNN(num_inputs, num_hiddens)               : RNN via nn.RNN (high-level)
  - GRU(num_inputs, num_hiddens, num_layers, dropout): GRU via nn.GRU
  - RNNLMScratch(rnn, vocab_size, lr)          : RNN language model from scratch
  - RNNLM(rnn, vocab_size, lr)                 : RNN language model via nn.LazyLinear

All model builders return a cdlModel (d2l Module) that can be wired into
CdlModelForward / CdlModelInfo / CdlModelSave etc. for inspection and
inference. The RNNLM* builders take an existing RNN/GRU cdlModel as input
and wrap it with an output projection to vocab_size classes.
"""

import torch

try:
    from ..src.d2lcore.torch import RNNScratch, RNN, GRU, RNNLMScratch, RNNLM
except ImportError:  # 直接以 ComfyDL 为顶层包运行（如冒烟测试）
    from src.d2lcore.torch import RNNScratch, RNN, GRU, RNNLMScratch, RNNLM

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class _SafeTokenList(list):
    """idx_to_token 列表：越界索引返回 '<unk>'，防止预测超出词表。"""

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list.__getitem__(self, i)
        try:
            return list.__getitem__(self, i)
        except IndexError:
            return '<unk>'


class _VocabProxy:
    """把 cdlVocab（dict 形式）适配为 d2l Vocab 接口，供 predict() 使用。"""

    def __init__(self, vocab_dict):
        self.idx_to_token = _SafeTokenList(vocab_dict.get('idx_to_token') or [])
        self.token_to_idx = dict(vocab_dict.get('token_to_idx') or {})
        self._unk = self.token_to_idx.get('<unk>', 0)

    def __getitem__(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, self._unk)
        return [self.token_to_idx.get(t, self._unk) for t in tokens]


def _device_of(model):
    """取模型第一个参数的 device（无参数时回退 cpu）。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


class CdlRNNScratch:
    """Build an RNN from scratch (tanh cell, manual parameters).

    d2lcore: RNNScratch(num_inputs, num_hiddens, sigma)
    Inputs:
        num_inputs (INT): input feature size (vocab size for LMs)
        num_hiddens (INT): number of hidden units
        sigma (FLOAT): std of the random parameter initialization
    Outputs:
        model (cdlModel): RNNScratch instance (forward expects
            (num_steps, batch_size, num_inputs))
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inputs": ("INT", {"default": 32, "min": 1, "max": 100000, "step": 1}),
                "num_hiddens": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "sigma": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_inputs, num_hiddens, sigma):
        model = RNNScratch(num_inputs, num_hiddens, sigma=sigma)
        return (model,)


NODE_CLASS_MAPPINGS["CdlRNNScratch"] = CdlRNNScratch
NODE_DISPLAY_NAME_MAPPINGS["CdlRNNScratch"] = "RNN (from scratch)"


class CdlRNN:
    """Build an RNN using PyTorch's high-level nn.RNN.

    d2lcore: RNN(num_inputs, num_hiddens)
    Inputs:
        num_inputs (INT): input feature size
        num_hiddens (INT): number of hidden units
    Outputs:
        model (cdlModel): RNN instance (forward expects
            (num_steps, batch_size, num_inputs), returns (output, h_n))
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inputs": ("INT", {"default": 32, "min": 1, "max": 100000, "step": 1}),
                "num_hiddens": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_inputs, num_hiddens):
        model = RNN(num_inputs, num_hiddens)
        return (model,)


NODE_CLASS_MAPPINGS["CdlRNN"] = CdlRNN
NODE_DISPLAY_NAME_MAPPINGS["CdlRNN"] = "RNN (high-level)"


class CdlGRU:
    """Build a multilayer GRU using PyTorch's high-level nn.GRU.

    d2lcore: GRU(num_inputs, num_hiddens, num_layers, dropout)
    Inputs:
        num_inputs (INT): input feature size
        num_hiddens (INT): number of hidden units per layer
        num_layers (INT): number of stacked GRU layers
        dropout (FLOAT): dropout probability between layers (0 = disabled)
    Outputs:
        model (cdlModel): GRU instance (forward expects
            (num_steps, batch_size, num_inputs), returns (output, h_n))
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inputs": ("INT", {"default": 32, "min": 1, "max": 100000, "step": 1}),
                "num_hiddens": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_inputs, num_hiddens, num_layers, dropout):
        model = GRU(num_inputs, num_hiddens, num_layers, dropout=dropout)
        return (model,)


NODE_CLASS_MAPPINGS["CdlGRU"] = CdlGRU
NODE_DISPLAY_NAME_MAPPINGS["CdlGRU"] = "GRU"


class CdlRNNLMScratch:
    """Wrap an RNN model into a language model (from-scratch output layer).

    d2lcore: RNNLMScratch(rnn, vocab_size, lr)
    Inputs:
        rnn (cdlModel): an RNN/GRU cdlModel with num_inputs == vocab_size
        vocab_size (INT): vocabulary size of the output projection
        lr (FLOAT): learning rate used when training the model
    Outputs:
        model (cdlModel): RNNLMScratch instance (forward expects an index
            tensor of shape (batch_size, num_steps); returns logits of
            shape (num_steps, batch_size, vocab_size))
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rnn": ("cdlModel",),
                "vocab_size": ("INT", {"default": 32, "min": 2, "max": 100000, "step": 1}),
                "lr": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, rnn, vocab_size, lr):
        model = RNNLMScratch(rnn, vocab_size, lr=lr)
        return (model,)


NODE_CLASS_MAPPINGS["CdlRNNLMScratch"] = CdlRNNLMScratch
NODE_DISPLAY_NAME_MAPPINGS["CdlRNNLMScratch"] = "RNN Language Model (from scratch)"


class CdlRNNLM:
    """Wrap an RNN model into a language model (high-level LazyLinear head).

    d2lcore: RNNLM(rnn, vocab_size, lr)
    Inputs:
        rnn (cdlModel): an RNN/GRU cdlModel with num_inputs == vocab_size
        vocab_size (INT): vocabulary size of the output projection
        lr (FLOAT): learning rate used when training the model
    Outputs:
        model (cdlModel): RNNLM instance (forward expects an index tensor
            of shape (batch_size, num_steps); returns logits of shape
            (num_steps, batch_size, vocab_size))
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rnn": ("cdlModel",),
                "vocab_size": ("INT", {"default": 32, "min": 2, "max": 100000, "step": 1}),
                "lr": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, rnn, vocab_size, lr):
        model = RNNLM(rnn, vocab_size, lr=lr)
        return (model,)


NODE_CLASS_MAPPINGS["CdlRNNLM"] = CdlRNNLM
NODE_DISPLAY_NAME_MAPPINGS["CdlRNNLM"] = "RNN Language Model (high-level)"


class CdlRNNLMScratchPredict:
    """Generate text with an RNN language model given a prefix.

    d2lcore: RNNLMScratch.predict(prefix, num_preds, vocab, device)
    Inputs:
        model (cdlModel): an RNNLMScratch / RNNLM cdlModel
        vocab (cdlVocab): vocabulary dict from CdlVocabBuild
        prefix (STRING): starting token(s), e.g. "the "
        num_preds (INT): number of characters/words to predict after prefix
    Outputs:
        prediction (STRING): the prefix followed by num_preds predicted tokens
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
                "vocab": ("cdlVocab",),
                "prefix": ("STRING", {"default": "the ", "placeholder": "prefix tokens, e.g. 'the '" }),
                "num_preds": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prediction",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, model, vocab, prefix, num_preds):
        if not hasattr(model, 'predict'):
            raise TypeError(
                f'模型类型 {type(model).__name__} 不支持 predict()，'
                '请使用 RNNLMScratch/RNNLM 模型节点构造的语言模型。')
        device = _device_of(model)
        pred = model.predict(prefix, num_preds, _VocabProxy(vocab), device=device)
        return (pred,)


NODE_CLASS_MAPPINGS["CdlRNNLMScratchPredict"] = CdlRNNLMScratchPredict
NODE_DISPLAY_NAME_MAPPINGS["CdlRNNLMScratchPredict"] = "RNN LM Predict"
