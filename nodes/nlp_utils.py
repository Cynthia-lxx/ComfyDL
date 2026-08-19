"""
d2lcore/NLP Utils - Natural language processing utility functions.

d2lcore functions:
  - tokenize(lines, token)       : Split text into word/character tokens
  - get_tokens_and_segments(tokens_a, tokens_b): BERT input construction
  - Vocab(tokens, min_freq, reserved_tokens): Build vocabulary
"""

import collections
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlTokenize:
    """Split text lines into word or character tokens.

    d2lcore: tokenize(lines, token)
    token='word': split by whitespace
    token='char': split into individual characters
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "the quick brown fox\njumps over the lazy dog", "multiline": True, "placeholder": "Input text, one sentence per line"}),
                "token_mode": (["word", "char"], {"default": "word"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tokens_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Utils"

    def execute(self, text, token_mode):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if token_mode == 'word':
            tokens = [line.split() for line in lines]
        else:
            tokens = [list(line) for line in lines]

        # Serialize as JSON-like string for pass-through
        token_strs = [','.join(t) for t in tokens]
        return ('\n'.join(token_strs),)


NODE_CLASS_MAPPINGS["CdlTokenize"] = CdlTokenize
NODE_DISPLAY_NAME_MAPPINGS["CdlTokenize"] = "Tokenize"


class CdlGetTokensAndSegments:
    """Get tokens and segment IDs for BERT input.

    d2lcore: get_tokens_and_segments(tokens_a, tokens_b)
    Returns tokens with [CLS] and [SEP] markers, and segment IDs (0 for A, 1 for B).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokens_a": ("STRING", {"default": "the,quick,brown,fox", "multiline": True, "placeholder": "comma-separated tokens for segment A"}),
            },
            "optional": {
                "tokens_b": ("STRING", {"default": "jumps,over", "multiline": True, "placeholder": "comma-separated tokens for segment B (optional)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tokens", "segments")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Utils"

    def execute(self, tokens_a, tokens_b=None):
        tokens_a_list = [t.strip() for t in tokens_a.split(',') if t.strip()]
        tokens = ['<cls>'] + tokens_a_list + ['<sep>']
        segments = [0] * (len(tokens_a_list) + 2)

        if tokens_b and tokens_b.strip():
            tokens_b_list = [t.strip() for t in tokens_b.split(',') if t.strip()]
            tokens += tokens_b_list + ['<sep>']
            segments += [1] * (len(tokens_b_list) + 1)

        return (','.join(tokens), ','.join(str(s) for s in segments))


NODE_CLASS_MAPPINGS["CdlGetTokensAndSegments"] = CdlGetTokensAndSegments
NODE_DISPLAY_NAME_MAPPINGS["CdlGetTokensAndSegments"] = "Get Tokens & Segments"


class CdlVocabBuild:
    """Build a vocabulary from token lists.

    d2lcore: Vocab(tokens, min_freq, reserved_tokens)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokens_text": ("STRING", {"default": "the quick\nbrown fox\nthe lazy dog", "multiline": True,
                                            "placeholder": "One token per line, or comma-separated per line"}),
                "min_freq": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "reserved_tokens": ("STRING", {"default": "<pad>,<bos>,<eos>", "placeholder": "comma-separated reserved tokens"}),
            }
        }

    RETURN_TYPES = ("cdlVocab", "INT")
    RETURN_NAMES = ("vocab", "vocab_size")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Utils"

    def execute(self, tokens_text, min_freq, reserved_tokens):
        # Parse input
        reserved_list = [t.strip() for t in reserved_tokens.split(',') if t.strip()]
        lines = [line.strip() for line in tokens_text.split('\n') if line.strip()]
        token_lists = []
        for line in lines:
            if ',' in line:
                token_lists.append([t.strip() for t in line.split(',') if t.strip()])
            else:
                token_lists.append(line.split())

        if token_lists and isinstance(token_lists[0], list):
            tokens_flat = [token for line in token_lists for token in line]
        else:
            tokens_flat = token_lists

        # Count frequencies
        counter = collections.Counter(tokens_flat)
        token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # Build index
        idx_to_token = list(sorted(set(['<unk>'] + reserved_list + [
            token for token, freq in token_freqs if freq >= min_freq])))
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

        vocab_dict = {
            'idx_to_token': idx_to_token,
            'token_to_idx': token_to_idx,
        }
        return (vocab_dict, len(idx_to_token))


NODE_CLASS_MAPPINGS["CdlVocabBuild"] = CdlVocabBuild
NODE_DISPLAY_NAME_MAPPINGS["CdlVocabBuild"] = "Vocab Build"


class CdlVocabEncode:
    """Encode tokens to indices using a vocabulary.

    d2lcore: Vocab.__getitem__(tokens)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocab": ("cdlVocab",),
                "tokens": ("STRING", {"default": "the,quick,brown", "multiline": True, "placeholder": "comma-separated tokens"}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("indices",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Utils"

    def execute(self, vocab, tokens):
        token_list = [t.strip() for t in tokens.split(',') if t.strip()]
        token_to_idx = vocab.get('token_to_idx', {})
        unk_idx = token_to_idx.get('<unk>', 0)
        indices = [token_to_idx.get(t, unk_idx) for t in token_list]
        return (torch.tensor(indices, dtype=torch.long),)


NODE_CLASS_MAPPINGS["CdlVocabEncode"] = CdlVocabEncode
NODE_DISPLAY_NAME_MAPPINGS["CdlVocabEncode"] = "Vocab Encode"


class CdlVocabDecode:
    """Decode indices to tokens using a vocabulary.

    d2lcore: Vocab.to_tokens(indices)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocab": ("cdlVocab",),
                "indices": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tokens_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/NLP Utils"

    def execute(self, vocab, indices):
        idx_to_token = vocab.get('idx_to_token', [])
        if hasattr(indices, '__len__') and (isinstance(indices, list) or indices.numel() > 1):
            idx_list = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
            tokens = [idx_to_token[int(i)] if int(i) < len(idx_to_token) else '<unk>' for i in idx_list]
        else:
            i = int(indices.item()) if hasattr(indices, 'item') else int(indices)
            tokens = [idx_to_token[i] if i < len(idx_to_token) else '<unk>']
        return (','.join(tokens),)


NODE_CLASS_MAPPINGS["CdlVocabDecode"] = CdlVocabDecode
NODE_DISPLAY_NAME_MAPPINGS["CdlVocabDecode"] = "Vocab Decode"
