"""
d2lcore/Attention - Attention mechanisms and Transformer components.

d2lcore classes:
  - DotProductAttention(dropout)                    : scaled dot-product attention
  - AdditiveAttention(num_hiddens, dropout)         : additive (Bahdanau) attention
  - MultiHeadAttention(num_hiddens, num_heads, dropout, bias)
  - PositionalEncoding(num_hiddens, dropout, max_len)
  - PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
  - AddNorm(norm_shape, dropout)
  - TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias)
  - TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                       num_blks, dropout, use_bias)

All builder nodes return a cdlModel (nn.Module) that can be wired into
CdlModelForward / CdlModelInfo / CdlModelSave etc. for inspection and
inference. Attention modules expect batch-first inputs
(batch_size, seq_len, num_hiddens); positional encoding and the transformer
encoder accept token index tensors of shape (batch_size, seq_len).
"""

import torch

try:
    from ..src.d2lcore.torch import (
        DotProductAttention, AdditiveAttention, MultiHeadAttention,
        PositionalEncoding, PositionWiseFFN, AddNorm,
        TransformerEncoderBlock, TransformerEncoder,
    )
except ImportError:  # 直接以 ComfyDL 为顶层包运行（如冒烟测试）
    from src.d2lcore.torch import (
        DotProductAttention, AdditiveAttention, MultiHeadAttention,
        PositionalEncoding, PositionWiseFFN, AddNorm,
        TransformerEncoderBlock, TransformerEncoder,
    )

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlDotProductAttention:
    """Build a scaled dot-product attention layer.

    d2lcore: DotProductAttention(dropout)
    Inputs:
        dropout (FLOAT): dropout probability applied to attention weights
    Outputs:
        model (cdlModel): attention layer; forward(queries, keys, values,
            valid_lens) with all tensors of shape
            (batch_size, seq_len, num_hiddens). Returns attention-weighted
            values of shape (batch_size, no. of queries, value dim).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, dropout):
        model = DotProductAttention(dropout)
        return (model,)


NODE_CLASS_MAPPINGS["CdlDotProductAttention"] = CdlDotProductAttention
NODE_DISPLAY_NAME_MAPPINGS["CdlDotProductAttention"] = "Dot-Product Attention"


class CdlAdditiveAttention:
    """Build an additive (Bahdanau) attention layer.

    d2lcore: AdditiveAttention(num_hiddens, dropout)
    Inputs:
        num_hiddens (INT): hidden units of the additive score function
        dropout (FLOAT): dropout probability applied to attention weights
    Outputs:
        model (cdlModel): attention layer; forward(queries, keys, values,
            valid_lens) with queries/keys/values of shape
            (batch_size, seq_len, num_hiddens).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_hiddens": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_hiddens, dropout):
        model = AdditiveAttention(num_hiddens, dropout)
        return (model,)


NODE_CLASS_MAPPINGS["CdlAdditiveAttention"] = CdlAdditiveAttention
NODE_DISPLAY_NAME_MAPPINGS["CdlAdditiveAttention"] = "Additive Attention"


class CdlMultiHeadAttention:
    """Build a multi-head attention layer.

    d2lcore: MultiHeadAttention(num_hiddens, num_heads, dropout, bias)
    Inputs:
        num_hiddens (INT): model width; must be divisible by num_heads
        num_heads (INT): number of parallel attention heads
        dropout (FLOAT): dropout probability applied to attention weights
        use_bias (BOOLEAN): whether the Q/K/V/O projections use bias
    Outputs:
        model (cdlModel): attention layer; forward(queries, keys, values,
            valid_lens) with tensors of shape
            (batch_size, seq_len, num_hiddens). Returns
            (batch_size, no. of queries, num_hiddens).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_hiddens": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
                "num_heads": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "use_bias": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_hiddens, num_heads, dropout, use_bias):
        if num_hiddens % num_heads != 0:
            raise ValueError(
                f'num_hiddens ({num_hiddens}) 必须能被 num_heads ({num_heads}) 整除。')
        model = MultiHeadAttention(num_hiddens, num_heads, dropout, bias=use_bias)
        return (model,)


NODE_CLASS_MAPPINGS["CdlMultiHeadAttention"] = CdlMultiHeadAttention
NODE_DISPLAY_NAME_MAPPINGS["CdlMultiHeadAttention"] = "Multi-Head Attention"


class CdlPositionalEncoding:
    """Build a sinusoidal positional encoding layer.

    d2lcore: PositionalEncoding(num_hiddens, dropout, max_len)
    Inputs:
        num_hiddens (INT): model width (feature dimension)
        dropout (FLOAT): dropout probability applied after adding encoding
        max_len (INT): maximum supported sequence length
    Outputs:
        model (cdlModel): encoding layer; forward(X) with X of shape
            (batch_size, seq_len, num_hiddens) adds positional information.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_hiddens": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "max_len": ("INT", {"default": 1000, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_hiddens, dropout, max_len):
        model = PositionalEncoding(num_hiddens, dropout, max_len)
        return (model,)


NODE_CLASS_MAPPINGS["CdlPositionalEncoding"] = CdlPositionalEncoding
NODE_DISPLAY_NAME_MAPPINGS["CdlPositionalEncoding"] = "Positional Encoding"


class CdlPositionWiseFFN:
    """Build a position-wise feed-forward network (two dense layers + ReLU).

    d2lcore: PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
    Inputs:
        ffn_num_hiddens (INT): hidden units of the inner dense layer
        ffn_num_outputs (INT): output units (usually == num_hiddens)
    Outputs:
        model (cdlModel): FFN applied identically to each position of X of
            shape (batch_size, seq_len, ffn_num_outputs).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ffn_num_hiddens": ("INT", {"default": 64, "min": 1, "max": 16384, "step": 1}),
                "ffn_num_outputs": ("INT", {"default": 16, "min": 1, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, ffn_num_hiddens, ffn_num_outputs):
        model = PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs)
        return (model,)


NODE_CLASS_MAPPINGS["CdlPositionWiseFFN"] = CdlPositionWiseFFN
NODE_DISPLAY_NAME_MAPPINGS["CdlPositionWiseFFN"] = "Position-Wise FFN"


class CdlAddNorm:
    """Build a residual connection followed by layer normalization.

    d2lcore: AddNorm(norm_shape, dropout)
    Inputs:
        norm_shape (INT): feature dimension for LayerNorm
        dropout (FLOAT): dropout probability applied to the residual branch
    Outputs:
        model (cdlModel): block; forward(X, Y) returns
            LayerNorm(dropout(Y) + X) of the same shape as X.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "norm_shape": ("INT", {"default": 16, "min": 1, "max": 16384, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, norm_shape, dropout):
        model = AddNorm(norm_shape, dropout)
        return (model,)


NODE_CLASS_MAPPINGS["CdlAddNorm"] = CdlAddNorm
NODE_DISPLAY_NAME_MAPPINGS["CdlAddNorm"] = "Add & Norm"


class CdlTransformerEncoderBlock:
    """Build a single Transformer encoder block.

    d2lcore: TransformerEncoderBlock(num_hiddens, ffn_num_hiddens,
                                     num_heads, dropout, use_bias)
    Inputs:
        num_hiddens (INT): model width; must be divisible by num_heads
        ffn_num_hiddens (INT): hidden units of the position-wise FFN
        num_heads (INT): number of attention heads
        dropout (FLOAT): dropout probability
        use_bias (BOOLEAN): whether attention projections use bias
    Outputs:
        model (cdlModel): block; forward(X, valid_lens) with X of shape
            (batch_size, seq_len, num_hiddens), returns same shape.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_hiddens": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
                "ffn_num_hiddens": ("INT", {"default": 64, "min": 1, "max": 16384, "step": 1}),
                "num_heads": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "use_bias": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias):
        if num_hiddens % num_heads != 0:
            raise ValueError(
                f'num_hiddens ({num_hiddens}) 必须能被 num_heads ({num_heads}) 整除。')
        model = TransformerEncoderBlock(num_hiddens, ffn_num_hiddens,
                                        num_heads, dropout, use_bias=use_bias)
        return (model,)


NODE_CLASS_MAPPINGS["CdlTransformerEncoderBlock"] = CdlTransformerEncoderBlock
NODE_DISPLAY_NAME_MAPPINGS["CdlTransformerEncoderBlock"] = "Transformer Encoder Block"


class CdlTransformerEncoder:
    """Build a Transformer encoder (embedding + positional encoding + blocks).

    d2lcore: TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                num_heads, num_blks, dropout, use_bias)
    Inputs:
        vocab_size (INT): vocabulary size of the input token embeddings
        num_hiddens (INT): model width; must be divisible by num_heads
        ffn_num_hiddens (INT): hidden units of the position-wise FFN
        num_heads (INT): number of attention heads
        num_blks (INT): number of stacked encoder blocks
        dropout (FLOAT): dropout probability
        use_bias (BOOLEAN): whether attention projections use bias
    Outputs:
        model (cdlModel): encoder; forward(X, valid_lens) with X of shape
            (batch_size, seq_len) of token indices, returns
            (batch_size, seq_len, num_hiddens).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocab_size": ("INT", {"default": 32, "min": 2, "max": 100000, "step": 1}),
                "num_hiddens": ("INT", {"default": 8, "min": 1, "max": 4096, "step": 1}),
                "ffn_num_hiddens": ("INT", {"default": 64, "min": 1, "max": 16384, "step": 1}),
                "num_heads": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "num_blks": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "use_bias": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Models"

    def execute(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                num_blks, dropout, use_bias):
        if num_hiddens % num_heads != 0:
            raise ValueError(
                f'num_hiddens ({num_hiddens}) 必须能被 num_heads ({num_heads}) 整除。')
        model = TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   use_bias=use_bias)
        return (model,)


NODE_CLASS_MAPPINGS["CdlTransformerEncoder"] = CdlTransformerEncoder
NODE_DISPLAY_NAME_MAPPINGS["CdlTransformerEncoder"] = "Transformer Encoder"
