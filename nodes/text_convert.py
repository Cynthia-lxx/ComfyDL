"""CdlToText — cast ANY pipeline value into a STRING.

The one-liner that makes wildcard sockets practical: Notebook ``result``,
tensor scalars, ints, dicts — anything can flow into text-only consumers
(``SaveText`` / ``Show Text`` / ``Display Any`` style nodes) which would
otherwise fail at runtime with ``TypeError: write() argument must be str,
not int`` (STRING targets cannot accept a wildcard's arbitrary payload).

Conversion rules mirror Python's ``str()`` for everyday values, with two
quality-of-life upgrades for teaching workflows:
  - ``None``          -> ``""`` (empty output instead of the literal "None")
  - tensors           -> compact summary ``Tensor(shape=..., dtype=..., ...)``
    instead of a multi-line wall of numbers (use the Visualization family to
    inspect real tensor data).
"""

from __future__ import annotations

import torch


def _to_text(value) -> str:
    """Cast one value to text with teaching-friendly formatting."""
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(value.item())
        summary = (f"Tensor(shape={tuple(value.shape)}, "
                   f"dtype={value.dtype})")
        if value.numel() <= 16:
            return f"{summary} values={value.flatten().tolist()}"
        return summary
    if isinstance(value, (list, tuple)) and value:
        # Short homogeneous numeric lists render like Python; anything long
        # gets the same tensor-style summary treatment.
        if len(value) > 16:
            return f"{type(value).__name__}(len={len(value)})"
        return str(value)
    return str(value)


class CdlToText:
    """Cast any input to a STRING output.

    What: the wildcard-to-text adapter. A ``Notebook → Cell`` result (or any
          ``*`` socket payload) is only valid downstream where the declared
          type matches at runtime; text consumers declare STRING, so a raw
          int/dict/tensor result crashes them mid-run. This node converts
          anything to text (Python ``str()`` semantics, plus ``None -> ""``
          and compact tensor/list summaries) and forwards the STRING.
    In:   ``value`` — anything (wildcard socket, forceInput).
    Out:  ``text`` — the value rendered as text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "utilities/conversion"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Wildcard * input bypasses backend type validation (see CdlNoOp).
        return True

    def execute(self, value=None, unique_id=None, extra_pnginfo=None):
        return (_to_text(value),)


NODE_CLASS_MAPPINGS = {"CdlToText": CdlToText}
NODE_DISPLAY_NAME_MAPPINGS = {"CdlToText": "To Text"}
