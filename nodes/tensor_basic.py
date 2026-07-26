"""
ComfyDL/Tensor Basic - Tensor utility nodes for pretty printing, parsing,
and fundamental tensor operations.

Nodes:
  - Tensor → String : Pretty-print tensor to a formatted string
  - String → Tensor : Parse a string representation into a tensor
  - Conv2D         : 2D convolution (wraps torch.nn.functional.conv2d)
  - Transpose      : Swap two tensor dimensions
  - Broadcast      : Broadcast tensor to a target shape
  - Reshape        : Reshape tensor to a new shape
  - Activation     : Element-wise activation (relu/sigmoid/tanh/...)
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


class CdlConv2d:
    """Perform 2D convolution on an input tensor using a kernel.

    Wraps ``torch.nn.functional.conv2d`` with configurable stride and padding.
    Accepts input and kernel tensors via slots; stride/padding via widgets.

    Input tensors are auto-expanded to 4-D ``(N, C, H, W)`` as needed.
    The result is squeezed back to remove the batch dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("cdlTensor",),
                "kernel": ("cdlTensor",),
                "stride": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, input_tensor, kernel, stride, padding):
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
        elif kernel.dim() == 3:
            kernel = kernel.unsqueeze(0)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        result = torch.nn.functional.conv2d(
            input_tensor, kernel, stride=stride, padding=padding
        )
        return (result.squeeze(0),)


NODE_CLASS_MAPPINGS["CdlConv2d"] = CdlConv2d
NODE_DISPLAY_NAME_MAPPINGS["CdlConv2d"] = "Conv2D"


class CdlTranspose:
    """Transpose a tensor by swapping two dimensions.

    Wraps ``torch.transpose``. Specify dim0 and dim1 to swap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "dim0": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "dim1": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, tensor, dim0, dim1):
        result = torch.transpose(tensor, dim0, dim1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlTranspose"] = CdlTranspose
NODE_DISPLAY_NAME_MAPPINGS["CdlTranspose"] = "Transpose"


class CdlBroadcast:
    """Broadcast a tensor to a target shape.

    Wraps ``torch.broadcast_to``. Enter the target shape as a
    comma-separated string (e.g. ``\"3,1,4\"``), respecting broadcasting rules.
    Returns the original tensor unchanged on parse error.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "target_shape": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "e.g. 3,1,4",
                }),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, tensor, target_shape):
        if not target_shape or not target_shape.strip():
            return (tensor,)
        try:
            shape = tuple(
                int(x.strip()) for x in target_shape.split(",") if x.strip()
            )
            result = torch.broadcast_to(tensor, shape)
            return (result,)
        except (ValueError, RuntimeError) as e:
            print(f"[CdlBroadcast] Error: {e}. Returning original tensor.")
            return (tensor,)


NODE_CLASS_MAPPINGS["CdlBroadcast"] = CdlBroadcast
NODE_DISPLAY_NAME_MAPPINGS["CdlBroadcast"] = "Broadcast"


class CdlReshape:
    """Reshape a tensor to a new shape.

    Wraps ``torch.reshape``. Enter the target shape as a comma-separated
    string (e.g. ``\"2,8\"`` or ``\"4,-1\"``). Returns the original tensor
    unchanged on parse error.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "target_shape": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "e.g. 2,8",
                }),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, tensor, target_shape):
        if not target_shape or not target_shape.strip():
            return (tensor,)
        try:
            shape = tuple(
                int(x.strip()) for x in target_shape.split(",") if x.strip()
            )
            result = torch.reshape(tensor, shape)
            return (result,)
        except (ValueError, RuntimeError) as e:
            print(f"[CdlReshape] Error: {e}. Returning original tensor.")
            return (tensor,)


NODE_CLASS_MAPPINGS["CdlReshape"] = CdlReshape
NODE_DISPLAY_NAME_MAPPINGS["CdlReshape"] = "Reshape"


class CdlActivation:
    """Apply an element-wise activation function to a tensor.

    Supported functions (Combo widget):
        - relu / sigmoid / tanh / leaky_relu / elu / gelu / silu
        - softmax / softplus

    ``dim`` only affects softmax; ``negative_slope`` only affects leaky_relu.
    """

    ACTIVATION_LIST = [
        "relu", "sigmoid", "tanh", "leaky_relu",
        "elu", "gelu", "silu", "softmax", "softplus",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "func": (cls.ACTIVATION_LIST, {"default": "relu"}),
                "dim": ("INT", {"default": -1, "min": -4, "max": 4, "step": 1}),
                "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Tensor Basic"

    def execute(self, tensor, func, dim, negative_slope):
        if func == "relu":
            result = torch.nn.functional.relu(tensor)
        elif func == "sigmoid":
            result = torch.sigmoid(tensor)
        elif func == "tanh":
            result = torch.tanh(tensor)
        elif func == "leaky_relu":
            result = torch.nn.functional.leaky_relu(tensor, negative_slope=negative_slope)
        elif func == "elu":
            result = torch.nn.functional.elu(tensor)
        elif func == "gelu":
            result = torch.nn.functional.gelu(tensor)
        elif func == "silu":
            result = torch.nn.functional.silu(tensor)
        elif func == "softmax":
            result = torch.nn.functional.softmax(tensor, dim=dim)
        elif func == "softplus":
            result = torch.nn.functional.softplus(tensor)
        else:
            result = tensor
        return (result,)


NODE_CLASS_MAPPINGS["CdlActivation"] = CdlActivation
NODE_DISPLAY_NAME_MAPPINGS["CdlActivation"] = "Activation"
