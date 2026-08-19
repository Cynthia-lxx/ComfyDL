"""
d2lcore/Seq2Seq - Sequence-to-sequence model builders.

d2lcore classes:
  - Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
  - init_seq2seq(module)                          : Xavier weight initializer

Notes:
  d2lcore 提供 Encoder/Decoder/AttentionDecoder 抽象基类但没有具体的
  decoder 实现，因此完整 Seq2Seq(encoder, decoder, tgt_pad, lr) 无法开箱
  构建，这里仅提供可开箱使用的 encoder 构造器与权重初始化工具。编码器
  输出 (outputs, state)，可直接接入 CdlModelForward 等进行前向验证。
"""

import torch

try:
    from ..src.d2lcore.torch import Seq2SeqEncoder, init_seq2seq
except ImportError:  # 直接以 ComfyDL 为顶层包运行（如冒烟测试）
    from src.d2lcore.torch import Seq2SeqEncoder, init_seq2seq

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlSeq2SeqEncoder:
    """Build an RNN encoder for sequence-to-sequence learning.

    d2lcore: Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
    Inputs:
        vocab_size (INT): vocabulary size of the source token embeddings
        embed_size (INT): embedding dimension of the input tokens
        num_hiddens (INT): hidden units of each GRU layer
        num_layers (INT): number of stacked GRU layers
        dropout (FLOAT): dropout probability between GRU layers
    Outputs:
        model (cdlModel): encoder; forward(X) with X of shape
            (batch_size, num_steps) of token indices, returns
            (outputs, state) with shapes
            (num_steps, batch_size, num_hiddens) and
            (num_layers, batch_size, num_hiddens).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocab_size": ("INT", {"default": 32, "min": 2, "max": 100000, "step": 1}),
                "embed_size": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                "num_hiddens": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                "num_layers": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, vocab_size, embed_size, num_hiddens, num_layers, dropout):
        model = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens,
                               num_layers, dropout=dropout)
        return (model,)


NODE_CLASS_MAPPINGS["CdlSeq2SeqEncoder"] = CdlSeq2SeqEncoder
NODE_DISPLAY_NAME_MAPPINGS["CdlSeq2SeqEncoder"] = "Seq2Seq Encoder"


class CdlInitSeq2Seq:
    """Apply Xavier weight initialization to a sequence-to-sequence model.

    d2lcore: init_seq2seq(module)
    Inputs:
        model (cdlModel): any nn.Module; nn.Linear and nn.GRU layers get
            Xavier-uniform initialized weights (other layers untouched)
    Outputs:
        model (cdlModel): the same model after in-place weight initialization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, model):
        model.apply(init_seq2seq)
        return (model,)


NODE_CLASS_MAPPINGS["CdlInitSeq2Seq"] = CdlInitSeq2Seq
NODE_DISPLAY_NAME_MAPPINGS["CdlInitSeq2Seq"] = "Init Seq2Seq Weights"
