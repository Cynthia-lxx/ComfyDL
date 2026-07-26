"""
ComfyDL/Tensor Basic - Tensor utility nodes for pretty printing and parsing.

Nodes:
  - Tensor → String : Pretty-print tensor to a formatted string
  - String → Tensor : Parse a string representation into a tensor
"""

import re
import ast
import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlTensorToStr:
    """Convert a tensor to a pretty-printed string.

    Shows shape, dtype, device info, and formatted values.
    Large tensors are truncated according to max_elems.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "max_elems": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "precision": ("INT", {"default": 6, "min": 1, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, tensor, max_elems, precision):
        t = tensor
        device_str = str(t.device)
        total_elems = t.numel()

        lines = [f"shape: {list(t.shape)}  dtype: {t.dtype}  device: {device_str}"]

        if total_elems <= max_elems:
            torch.set_printoptions(precision=precision, threshold=max_elems + 1, linewidth=120)
            lines.append(str(t))
        else:
            flat = t.flatten()
            front = flat[:max_elems // 2].tolist()
            back = flat[-max_elems // 2:].tolist()

            def _fmt_list(lst):
                return "[" + ", ".join(f"{v:.{precision}g}" for v in lst) + "]"

            lines.append(f"tensor({_fmt_list(front)}, ...")
            lines.append(f"      ...({total_elems - max_elems} elements omitted)...")
            lines.append(f"      {_fmt_list(back)})")
            lines.append(f"(truncated; total elements: {total_elems})")

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS["CdlTensorToStr"] = CdlTensorToStr
NODE_DISPLAY_NAME_MAPPINGS["CdlTensorToStr"] = "Tensor \u2192 String"


class CdlStrToTensor:
    """Convert a string representation to a tensor.

    Parses strings like:
        "[[1, 2], [3, 4]]"
        "[[1, 2],\\n[3, 4], \\n]"
        "[1, 2, 3, 4, 5]"

    Error handling strategies (widget):
        - empty_tensor : return torch.tensor([])
        - zero_tensor  : return torch.tensor([0.])
        - raise_error  : propagate the exception
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True,
                                     "placeholder": "e.g. [[1, 2], [3, 4]]"}),
                "error_strategy": (["empty_tensor", "zero_tensor", "raise_error"],
                                   {"default": "empty_tensor"}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, text, error_strategy):
        if not text or not text.strip():
            return (self._handle_error(error_strategy, "Empty input string"),)

        try:
            # --- Normalize the input string ---
            s = text.strip()

            # 1. Collapse all whitespace (newlines, tabs, spaces) into nothing
            s = re.sub(r'\s+', '', s)

            # 2. Remove trailing commas before closing brackets: e.g. [1,2,] -> [1,2]
            s = re.sub(r',(?=\s*])', '', s)

            # --- Parse as a Python literal ---
            data = ast.literal_eval(s)

            # --- Convert to tensor ---
            if isinstance(data, (int, float)):
                data = [data]  # scalar -> 1-element list

            result = torch.tensor(data, dtype=torch.float32)
            return (result,)

        except Exception as e:
            return (self._handle_error(error_strategy, str(e)),)

    @staticmethod
    def _handle_error(strategy, msg):
        if strategy == "raise_error":
            raise ValueError(f"String \u2192 Tensor conversion failed: {msg}")
        elif strategy == "zero_tensor":
            return torch.tensor([0.], dtype=torch.float32)
        else:  # empty_tensor
            return torch.tensor([], dtype=torch.float32)


NODE_CLASS_MAPPINGS["CdlStrToTensor"] = CdlStrToTensor
NODE_DISPLAY_NAME_MAPPINGS["CdlStrToTensor"] = "String \u2192 Tensor"
