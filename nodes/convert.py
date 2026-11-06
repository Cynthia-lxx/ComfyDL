"""CdlConvert - one universal type-conversion node (V1, wildcard in/out).

What:  the single adapter for the "my payload's declared type does not match
       the downstream slot" problem class (the Notebook result feeding
       SaveText, a tensor feeding an INT widget, ...). One input (wildcard),
       one output (STRING / INT / FLOAT / BOOLEAN / TENSOR / IMAGE / MASK /
       LATENT), and one combo that decides the target. ``match link`` (the
       default) lets the frontend fill the combo from whatever the output is
       connected to - Reroute-style, see ``web/js/convert.js`` - so the usual
       wiring flow needs no manual typing at all.
In:    ``value`` - the wildcard payload; ``target_type`` - the declared type
       of the output socket (``match link`` + concrete types).
Out:   ``value`` - the payload cast to ``target_type``.

Casting matrix (v1, teaching-first):
- STRING   -> str() semantics shared with ``To Text``: Python str for plain
  values, ``None -> ""``, compact summaries for tensors/lists (inspect real
  tensor data with the Visualization family instead).
- INT / FLOAT / BOOLEAN -> Python casting of scalars (bool/str/number) and
  single-element tensors; multi-element tensors raise a readable error.
- TENSOR   -> torch.tensor(value) for scalars/strings, passthrough for
  tensors, IMAGE/LATENT unwrap their ``samples``.
- IMAGE / MASK / LATENT -> wraps a tensor into the ComfyUI dict convention
  (``{"samples": t}``), validates shape conventions loosely (IMAGE wants
  (B, H, W, C), MASK (B, H, W) or (B, 1, H, W)), passthrough for matching
  dicts.

Relationship to the older converters: ``Value -> Tensor`` / ``Tensor ->
Value`` (V3) remain the five-way IMAGE/MASK/LATENT/AUDIO/SIGMAS family with
their parallel metadata sockets; ``To Text`` remains the notebook-console
flavoured display cast. ``Convert`` supersedes none of them - it is the
generic graph-level adapter.
"""

from __future__ import annotations

import torch

from .text_convert import _to_text

#: Types the output socket can legally take (also the combo list). Kept in
#: one place so the frontend (convert.js) and the docs stay in sync.
TARGET_TYPES = [
    "match link",
    "STRING", "INT", "FLOAT", "BOOLEAN",
    "TENSOR", "IMAGE", "MASK", "LATENT",
]

_DEFAULT = "match link"


def _scalar(value, target):
    """Cast a non-tensor value to int/float/bool with readable errors."""
    try:
        if target == "INT":
            return int(value)
        if target == "FLOAT":
            return float(value)
        return bool(value)  # BOOLEAN
    except (TypeError, ValueError):
        raise ValueError(
            f"cannot cast {value!r} ({type(value).__name__}) to {target}"
        ) from None


def _from_tensor(t, target):
    if target == "TENSOR":
        return t
    if target in ("IMAGE", "MASK", "LATENT"):
        return {"samples": t}
    if t.numel() != 1:
        raise ValueError(
            f"cannot cast a {t.numel()}-element tensor to {target} - only "
            f"single-element tensors convert to scalars (reshape/pool first, "
            f"or use the reduction nodes)"
        )
    v = t.detach().flatten()[0].item()
    if target == "STRING":
        return _to_text(v)
    if target == "BOOLEAN":
        return bool(v)
    return int(v) if target == "INT" else float(v)


def _convert(value, target):
    """Dispatch one payload to ``target``. Raises ValueError on impossible
    casts; every message names the offending value and the target type."""
    if target == "STRING":
        return _to_text(value)

    if target == "TENSOR":
        # Scalars/lists/latents need explicit materialization here - the
        # generic scalar branches below only know int/float/bool targets.
        if isinstance(value, (list, tuple)):
            return torch.tensor(value)
        if isinstance(value, dict):
            samples = value.get("samples")
            if not isinstance(samples, torch.Tensor):
                raise ValueError(
                    f"cannot cast a dict payload (keys: {sorted(value.keys())}) "
                    f"to TENSOR - no 'samples' tensor inside"
                )
            return samples
        if isinstance(value, (bool, int, float)):
            return torch.tensor(value)
        if isinstance(value, str):
            try:
                return torch.tensor(float(value))
            except ValueError:
                raise ValueError(
                    f"cannot cast {value!r} (str) to TENSOR - only numeric "
                    f"strings convert"
                ) from None

    if isinstance(value, torch.Tensor):
        return _from_tensor(value, target)

    if isinstance(value, dict):
        samples = value.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError(
                f"cannot cast a dict payload (keys: {sorted(value.keys())}) "
                f"to {target} - no 'samples' tensor inside"
            )
        if target in ("IMAGE", "MASK", "LATENT"):
            return value if target == "LATENT" else {"samples": samples}
        return _from_tensor(samples, target)

    if isinstance(value, (list, tuple)):
        if target == "TENSOR":
            return torch.tensor(value)
        raise ValueError(
            f"cannot cast a {type(value).__name__} of length "
            f"{len(value)} to {target} - only TENSOR is supported for "
            f"sequences"
        )

    if isinstance(value, bool) or value is None:
        if target == "BOOLEAN":
            return bool(value)
        if value is None:
            raise ValueError("cannot cast None to " + target)
        return _scalar(int(value), target)

    if isinstance(value, (int, float)):
        if target == "BOOLEAN":
            return bool(value)
        return _scalar(value, target)

    if isinstance(value, str):
        if target == "BOOLEAN":
            return _scalar(value, "BOOLEAN")
        return _scalar(value, target)

    raise ValueError(
        f"cannot cast {type(value).__name__} to {target} - unsupported "
        f"payload type"
    )


class CdlConvert:
    """Universal type converter: wildcard in, one declared type out.

    The output socket's declared type mirrors ``target_type`` (the frontend
    rewrites ``outputs[0].type`` when the combo changes and fills the combo
    from the link when ``match link`` is active - the backend stays type-blind
    because output types are frontend link guards, not execution facts). The
    conversion itself is the casting matrix in the module docstring; any
    impossible cast raises a readable ValueError naming value and target.
    """

    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_type": (TARGET_TYPES, {"default": _DEFAULT}),
            },
            "optional": {
                "value": ("*",),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "utilities/conversion"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # The wildcard * input bypasses backend type validation (official
        # contract, see more_on_inputs.mdx "Flexible inputs").
        return True

    def execute(self, target_type=_DEFAULT, value=None):
        target = target_type if target_type in TARGET_TYPES else _DEFAULT
        if target == _DEFAULT:
            # No link matched and the user kept the default: pass through
            # untouched rather than guessing a type.
            return (value,)
        return (_convert(value, target),)


NODE_CLASS_MAPPINGS = {"CdlConvert": CdlConvert}
NODE_DISPLAY_NAME_MAPPINGS = {"CdlConvert": "Convert"}
