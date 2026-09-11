"""
utilities/conversion - Comfy semantic value <-> generic TENSOR conversion.

This module lets values that carry a ComfyUI semantic meaning (IMAGE, MASK,
LATENT, AUDIO, SIGMAS) travel through generic tensor pipelines and come back
unchanged.  It is the first ComfyDL module written against the ComfyUI V3 node
API (``io.ComfyNode`` + ``io.Schema``) instead of the legacy
``INPUT_TYPES`` / ``RETURN_TYPES`` style -- the V3 schema is the only way to
declare a *union* input slot (``io.MultiType.Input``).

Design notes
------------
* One generic "collect" node accepts any supported Comfy value and emits a
  plain ``TENSOR``; one generic "dispatch" node exposes one concrete output
  slot per supported type, so the destination type is chosen by *which socket
  the user wires*.
* **No ``io.MatchType`` is involved.**  The backend validator
  (``comfy_execution/validation.py``) short-circuits any MatchType comparison
  and defers the check to the frontend, so a concrete output slot is the only
  form that also gets *backend* type checking.  It is also the pattern used by
  the official ``comfy_extras/nodes_number_convert.py`` node.
* Values that already *are* ``torch.Tensor`` (IMAGE / MASK / SIGMAS) are passed
  through untouched.  LATENT and AUDIO carry extra metadata (``noise_mask``,
  ``batch_index``, ``type``, ``sampler_rate``); that metadata travels on
  parallel sockets so a collect -> dispatch round trip rebuilds the original
  dictionary key for key.
* LORA_MODEL and LOSS_MAP are *tensor collections* rather than single tensors.
  They get dedicated pack/unpack node pairs, documented as experimental /
  reserved: nothing in ComfyDL currently produces those types yet.

Nodes:
  - Value -> Tensor      : Comfy value (IMAGE/MASK/LATENT/AUDIO/SIGMAS) -> TENSOR
  - Tensor -> Value      : TENSOR -> five concrete Comfy output slots
  - LoRA Model -> Tensor : LORA_MODEL (dict[str, Tensor]) -> TENSOR + metadata
  - Tensor -> LoRA Model : TENSOR + metadata -> LORA_MODEL
  - Loss Map -> Tensor   : LOSS_MAP ({"loss": [Tensor, ...]}) -> TENSOR + metadata
  - Tensor -> Loss Map   : TENSOR + metadata -> LOSS_MAP
"""

import torch

from comfy_api.latest import io

# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

#: Core-style category.  These nodes live in the ComfyDL submodule but are
#: merged into the host's global registry by ``nodes.py:init_builtin_dl_nodes()``,
#: so they appear next to the built-in nodes rather than under the ``d2l/*`` tree.
CATEGORY = "utilities/conversion"

#: Fallback sample rate used when an AUDIO value carries no usable
#: ``sampler_rate`` (44.1 kHz is the most common rate in practice).
DEFAULT_SAMPLER_RATE = 44100

#: Comfy value types the generic collect / dispatch pair understands.
SUPPORTED_ORIGINS = ("IMAGE", "MASK", "LATENT", "AUDIO", "SIGMAS")

_TO_TENSOR_ALIASES = [
    "to tensor", "value to tensor", "as tensor", "convert to tensor",
    "tensor cast", "cast to tensor", "image to tensor", "mask to tensor",
    "latent to tensor", "audio to tensor", "sigmas to tensor", "unwrap",
]

_FROM_TENSOR_ALIASES = [
    "from tensor", "tensor to image", "tensor to mask", "tensor to latent",
    "tensor to audio", "tensor to sigmas", "convert tensor", "cast tensor",
    "dispatch tensor", "wrap",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _encode_shape(tensor: torch.Tensor) -> str:
    """Encode a tensor shape as a comma separated string.

    A 0-D (scalar) tensor encodes as the empty string, which
    :func:`_decode_shape` turns back into ``()``.
    """
    return ",".join(str(int(dim)) for dim in tensor.shape)


def _decode_shape(value):
    """Decode a shape string produced by :func:`_encode_shape`.

    Accepts ``None``, a ``"3,4"`` style string (full-width separators are
    tolerated) or an already-iterable of dimensions.

    Returns
    -------
    tuple
        The decoded shape; ``()`` means a 0-D scalar.
    None
        The value is not a parseable shape.  Callers decide whether to skip the
        entry or fall back to a scalar, so a malformed metadata socket can never
        abort a workflow.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raw = str(value).strip()
        if not raw:
            return ()
        items = raw.replace("\uff0c", ",").replace("\uff1b", ",").replace(";", ",").split(",")

    dims = []
    for item in items:
        try:
            dim = int(float(str(item).strip()))
        except (TypeError, ValueError):
            return None
        if dim < 0:
            return None
        dims.append(dim)
    return tuple(dims)


def _dtype_from_string(value) -> torch.dtype:
    """Resolve ``"torch.float32"`` / ``"float32"`` back to a ``torch.dtype``."""
    name = str(value).strip()
    if name.startswith("torch."):
        name = name[len("torch."):]
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def _has_batch_index(value) -> bool:
    """True when ``value`` looks like a usable LATENT ``batch_index``."""
    return isinstance(value, (list, tuple)) and len(value) > 0


def _guess_origin(tensor: torch.Tensor) -> str:
    """Best-effort label for a bare tensor coming through the union input.

    IMAGE, MASK and SIGMAS are all plain ``torch.Tensor`` at runtime, so the
    frontend's choice of socket cannot be recovered here.  This only feeds the
    diagnostic ``origin`` socket; it never changes how the value is converted.
    """
    rank = tensor.dim()
    if rank == 0:
        return "TENSOR"
    if rank == 1:
        # [N] -- the SIGMAS layout.
        return "SIGMAS"
    if rank == 2:
        # [H, W] -- a single mask.
        return "MASK"
    if rank == 3:
        # [B, H, W] -- a batched mask.
        return "MASK"
    if rank == 4:
        # [B, H, W, C] is IMAGE; a channel-first 4-D tensor is treated as MASK.
        return "IMAGE" if int(tensor.shape[-1]) in (1, 3, 4) else "MASK"
    return "TENSOR"


def _cat_tensors(tensors: list, what: str) -> torch.Tensor:
    """Flatten and concatenate a list of tensors into one 1-D tensor.

    All tensors must share a dtype: concatenation would otherwise fail with an
    opaque error, and promoting silently would make the round trip lossy.
    """
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) > 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise ValueError(
            f"{what}: all tensors must share one dtype, got [{names}]. "
            "Cast them to a common dtype before packing."
        )
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _split_tensors(flat: torch.Tensor, shapes, dtypes, what: str) -> list:
    """Inverse of :func:`_cat_tensors`: slice ``flat`` back into tensors.

    ``shapes`` and ``dtypes`` are the parallel metadata lists emitted by the
    matching pack node.  A shape that cannot be parsed degrades to a 0-D scalar
    (consuming one element) instead of dropping the entry, and a truncated
    ``flat`` yields only the tensors that fit -- neither case raises.

    The incoming tensor is flattened first: the pack node emits 1-D, but a
    tensor that travelled through generic ops may arrive with any shape, and
    slicing a multi-dimensional tensor along axis 0 would mis-index it.
    """
    if not isinstance(flat, torch.Tensor):
        raise TypeError(f"{what}: expected a TENSOR input, got {type(flat).__name__!r}.")
    flat = flat.reshape(-1)

    shapes = list(shapes) if isinstance(shapes, (list, tuple)) else []
    dtypes = list(dtypes) if isinstance(dtypes, (list, tuple)) else []

    results = []
    offset = 0
    for index, shape_spec in enumerate(shapes):
        shape = _decode_shape(shape_spec)
        if shape is None:
            print(
                f"[{what}] cannot parse shape {shape_spec!r}; "
                "falling back to a 0-D scalar for this entry."
            )
            shape = ()
        count = 1
        for dim in shape:
            count *= dim
        if offset + count > flat.numel():
            print(
                f"[{what}] metadata describes more elements than the tensor holds; "
                f"stopping after {len(results)} tensor(s)."
            )
            break
        dtype_spec = dtypes[index] if index < len(dtypes) else flat.dtype
        chunk = flat[offset:offset + count].reshape(shape)
        try:
            chunk = chunk.to(_dtype_from_string(dtype_spec))
        except (RuntimeError, TypeError):
            pass
        results.append(chunk)
        offset += count
    return results


# --------------------------------------------------------------------------- #
# Generic collect: Comfy value -> TENSOR
# --------------------------------------------------------------------------- #

class CdlValueToTensor(io.ComfyNode):
    """Convert any supported Comfy semantic value into a generic ``TENSOR``.

    What it does:
        Accepts **one** value on a union input slot and unwraps it to a plain
        ``torch.Tensor`` that any generic tensor node can consume.  Metadata
        that has no place inside a bare tensor is emitted on parallel sockets so
        that :class:`CdlTensorToValue` can rebuild the original object.

    Inputs:
        value (IMAGE | MASK | LATENT | AUDIO | SIGMAS): the value to unwrap.
            IMAGE / MASK / SIGMAS pass through unchanged; LATENT yields its
            ``samples``; AUDIO yields its ``waveform``.

    Outputs:
        tensor (TENSOR): the unwrapped tensor.
        origin (STRING): best-effort source label, for diagnostics and for the
            dispatch node's mismatch hints.  IMAGE / MASK / SIGMAS are all plain
            tensors at runtime, so their label is inferred from the shape.
        noise_mask (TENSOR): LATENT ``noise_mask`` when present, else ``None``.
        batch_index (ARRAY): LATENT ``batch_index`` when present, else ``None``.
        latent_type (STRING): LATENT ``type`` (``"audio"`` / ``"hunyuan3dv2"``),
            or ``""`` when the latent carries no special type.
        sampler_rate (INT): AUDIO sample rate, or ``0`` for non-audio values.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlValueToTensor",
            display_name="Value \u2192 Tensor",
            category=CATEGORY,
            description=(
                "Unwrap a Comfy value (IMAGE/MASK/LATENT/AUDIO/SIGMAS) into a "
                "generic TENSOR. Metadata travels on parallel sockets."
            ),
            search_aliases=list(_TO_TENSOR_ALIASES),
            inputs=[
                io.MultiType.Input(
                    "value",
                    [io.Image, io.Mask, io.Latent, io.Audio, io.Sigmas],
                    display_name="value",
                    tooltip="Any supported Comfy value; the socket adapts to what you connect.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="TENSOR"),
                io.String.Output(display_name="origin"),
                io.Tensor.Output(display_name="noise_mask"),
                io.Array.Output(display_name="batch_index"),
                io.String.Output(display_name="latent_type"),
                io.Int.Output(display_name="sampler_rate"),
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        origin, tensor, noise_mask, batch_index, latent_type, sampler_rate = _split_value(value)
        return io.NodeOutput(tensor, origin, noise_mask, batch_index, latent_type, sampler_rate)


def _split_value(value) -> tuple:
    """Split a Comfy value into ``(origin, tensor, noise_mask, batch_index,
    latent_type, sampler_rate)``.

    Raises ``TypeError`` for unsupported types so the failure is readable
    instead of surfacing as a mysterious downstream shape error.
    """
    if isinstance(value, dict):
        if "samples" in value:
            return (
                "LATENT",
                value["samples"],
                value.get("noise_mask"),
                value.get("batch_index"),
                str(value.get("type") or ""),
                0,
            )
        if "waveform" in value:
            try:
                rate = int(value.get("sampler_rate", 0) or 0)
            except (TypeError, ValueError):
                rate = 0
            return ("AUDIO", value["waveform"], None, None, "", rate)
        raise TypeError(
            "CdlValueToTensor: unsupported mapping value; expected a LATENT "
            "('samples') or AUDIO ('waveform') dictionary, "
            f"got keys {sorted(value.keys())!r}."
        )

    if isinstance(value, torch.Tensor):
        return (_guess_origin(value), value, None, None, "", 0)

    raise TypeError(
        "CdlValueToTensor: unsupported value type "
        f"{type(value).__name__!r}; connect an IMAGE, MASK, LATENT, AUDIO or SIGMAS socket."
    )


NODE_CLASS_MAPPINGS["CdlValueToTensor"] = CdlValueToTensor
NODE_DISPLAY_NAME_MAPPINGS["CdlValueToTensor"] = "Value \u2192 Tensor"


# --------------------------------------------------------------------------- #
# Generic dispatch: TENSOR -> Comfy values (one concrete slot per type)
# --------------------------------------------------------------------------- #

class CdlTensorToValue(io.ComfyNode):
    """Expose a generic ``TENSOR`` on one concrete Comfy socket per type.

    What it does:
        Takes a ``TENSOR`` and offers it simultaneously as IMAGE, MASK, LATENT,
        AUDIO and SIGMAS.  The user picks the destination type by wiring the
        matching output socket; unused sockets return ``None`` and cost nothing.
        Metadata received from :class:`CdlValueToTensor` is folded back into the
        reconstructed LATENT / AUDIO dictionaries.

    Why five concrete sockets instead of one dynamic socket:
        A concrete socket is statically typed, so the frontend physically
        prevents mismatched links *and* the backend ``validate_node_input``
        performs a real type check.  A ``MatchType`` socket would get no backend
        checking at all.

    Inputs:
        tensor (TENSOR): the tensor to expose.
        origin (STRING): optional label emitted by the collect node; used only
            to phrase shape-mismatch hints.  An empty value means "unknown".
        noise_mask (TENSOR): optional LATENT noise mask.
        batch_index (ARRAY): optional LATENT batch index (``list[int]``).
        latent_type (STRING): optional LATENT ``type``; an empty string means
            "no special type" and the key is omitted, matching the original.
        sampler_rate (INT): optional AUDIO sample rate; defaults to 44100.

    Outputs:
        IMAGE (IMAGE): ``tensor`` unchanged.
        MASK (MASK): ``tensor`` unchanged.
        LATENT (LATENT): ``{"samples": tensor}`` plus whichever metadata sockets
            are connected; ``None`` when the tensor is not 4-D.
        AUDIO (AUDIO): ``{"waveform": tensor, "sampler_rate": rate}``; ``None``
            when the tensor is not 2-D/3-D.
        SIGMAS (SIGMAS): ``tensor`` unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlTensorToValue",
            display_name="Tensor \u2192 Value",
            category=CATEGORY,
            description=(
                "Expose a generic TENSOR on concrete IMAGE / MASK / LATENT / "
                "AUDIO / SIGMAS sockets. Wire the one you need."
            ),
            search_aliases=list(_FROM_TENSOR_ALIASES),
            inputs=[
                io.Tensor.Input("tensor", tooltip="The tensor to expose."),
                io.String.Input(
                    "origin",
                    optional=True,
                    default="",
                    tooltip="Optional label from a Value \u2192 Tensor node; improves mismatch hints.",
                ),
                io.Tensor.Input(
                    "noise_mask",
                    optional=True,
                    tooltip="Optional LATENT noise mask, forwarded from a Value \u2192 Tensor node.",
                ),
                io.Array.Input(
                    "batch_index",
                    optional=True,
                    tooltip="Optional LATENT batch index (list[int]), forwarded from a Value \u2192 Tensor node.",
                ),
                io.String.Input(
                    "latent_type",
                    optional=True,
                    default="",
                    tooltip="Optional LATENT type ('audio' / 'hunyuan3dv2'). Empty means none.",
                ),
                io.Int.Input(
                    "sampler_rate",
                    optional=True,
                    default=DEFAULT_SAMPLER_RATE,
                    min=1,
                    max=384000,
                    step=1,
                    tooltip="AUDIO sample rate in Hz, forwarded from a Value \u2192 Tensor node.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Mask.Output(display_name="MASK"),
                io.Latent.Output(display_name="LATENT"),
                io.Audio.Output(display_name="AUDIO"),
                io.Sigmas.Output(display_name="SIGMAS"),
            ],
        )

    @classmethod
    def execute(cls, tensor, origin=None, noise_mask=None, batch_index=None,
                latent_type=None, sampler_rate=None) -> io.NodeOutput:
        label = str(origin or "")

        # IMAGE / MASK / SIGMAS are plain tensors: pass them through untouched.
        # Their shape conventions are the caller's responsibility -- a 4-D image
        # may legitimately have any channel count, so guessing here would only
        # produce false rejections.
        image = tensor
        mask = tensor
        sigmas = tensor

        latent = _build_latent(tensor, label, noise_mask, batch_index, latent_type)
        audio = _build_audio(tensor, label, sampler_rate)

        return io.NodeOutput(image, mask, latent, audio, sigmas)


def _build_latent(tensor, label, noise_mask, batch_index, latent_type):
    """Rebuild a LATENT dictionary, or ``None`` when the tensor cannot be one."""
    if tensor.dim() != 4:
        if label == "LATENT":
            print(
                "[CdlTensorToValue] LATENT expects a 4-D tensor [B, C, H, W]; "
                f"got shape {tuple(tensor.shape)}. The LATENT socket returns None."
            )
        return None

    latent = {"samples": tensor}
    # Only copy metadata that was actually supplied, so the rebuilt dictionary
    # matches the original key for key.
    if noise_mask is not None:
        latent["noise_mask"] = noise_mask
    if _has_batch_index(batch_index):
        latent["batch_index"] = list(batch_index)
    if latent_type:
        latent["type"] = str(latent_type)
    return latent


def _build_audio(tensor, label, sampler_rate):
    """Rebuild an AUDIO dictionary, or ``None`` when the tensor cannot be one."""
    if tensor.dim() not in (2, 3):
        if label == "AUDIO":
            print(
                "[CdlTensorToValue] AUDIO expects a 2-D/3-D tensor [..., C, T]; "
                f"got shape {tuple(tensor.shape)}. The AUDIO socket returns None."
            )
        return None

    try:
        rate = int(sampler_rate)
    except (TypeError, ValueError):
        rate = 0
    if rate <= 0:
        rate = DEFAULT_SAMPLER_RATE
    return {"waveform": tensor, "sampler_rate": rate}


NODE_CLASS_MAPPINGS["CdlTensorToValue"] = CdlTensorToValue
NODE_DISPLAY_NAME_MAPPINGS["CdlTensorToValue"] = "Tensor \u2192 Value"


# --------------------------------------------------------------------------- #
# LORA_MODEL pack / unpack (reserved)
#
# LORA_MODEL is ``dict[str, torch.Tensor]`` -- a *collection*, so it cannot be
# expressed as a single tensor without remembering how the parts were laid out.
# The pack node flattens and concatenates every tensor (in insertion order) into
# one 1-D TENSOR and emits three parallel metadata sockets; the unpack node
# slices it back.  Nothing in ComfyDL produces LORA_MODEL yet, which is why
# these nodes are documented as reserved / experimental.
# --------------------------------------------------------------------------- #

class CdlLoraModelToTensor(io.ComfyNode):
    """Pack a ``LORA_MODEL`` dictionary into one ``TENSOR`` plus metadata.

    What it does:
        Concatenates every tensor of ``dict[str, torch.Tensor]`` into a single
        1-D tensor, flattening each value first.  The key order, per-tensor
        shape and per-tensor dtype are emitted on parallel sockets so
        :class:`CdlTensorToLoraModel` can restore the dictionary exactly.

    Inputs:
        lora_model (LORA_MODEL): mapping of parameter name to tensor.

    Outputs:
        tensor (TENSOR): every value flattened and concatenated, in key order.
        keys (ARRAY): the key order, as ``list[str]``.
        shapes (ARRAY): one ``"3,4"`` style shape string per tensor; ``""``
            denotes a 0-D scalar.
        dtypes (ARRAY): one ``"torch.float32"`` style dtype string per tensor.

    Notes:
        All tensors must share one dtype; a mixed batch raises ``ValueError``
        rather than silently promoting (which would make the round trip lossy).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlLoraModelToTensor",
            display_name="LoRA Model \u2192 Tensor",
            category=CATEGORY,
            description=(
                "Reserved: pack a LORA_MODEL (dict[str, Tensor]) into one TENSOR "
                "plus key/shape/dtype metadata. No ComfyDL node produces "
                "LORA_MODEL yet."
            ),
            search_aliases=["lora to tensor", "pack lora", "flatten lora model"],
            is_experimental=True,
            inputs=[
                io.LoraModel.Input(
                    "lora_model",
                    tooltip="Mapping of parameter name to tensor.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="TENSOR"),
                io.Array.Output(display_name="keys"),
                io.Array.Output(display_name="shapes"),
                io.Array.Output(display_name="dtypes"),
            ],
        )

    @classmethod
    def execute(cls, lora_model) -> io.NodeOutput:
        if not isinstance(lora_model, dict):
            raise TypeError(
                "CdlLoraModelToTensor: expected a LORA_MODEL dictionary, "
                f"got {type(lora_model).__name__!r}."
            )

        keys = [str(key) for key in lora_model.keys()]
        tensors = [lora_model[key] for key in lora_model.keys()]
        for key, tensor in zip(keys, tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"CdlLoraModelToTensor: entry {key!r} is "
                    f"{type(tensor).__name__!r}, not a torch.Tensor."
                )

        flat = _cat_tensors(tensors, "CdlLoraModelToTensor")
        shapes = [_encode_shape(tensor) for tensor in tensors]
        dtypes = [str(tensor.dtype) for tensor in tensors]
        return io.NodeOutput(flat, keys, shapes, dtypes)


NODE_CLASS_MAPPINGS["CdlLoraModelToTensor"] = CdlLoraModelToTensor
NODE_DISPLAY_NAME_MAPPINGS["CdlLoraModelToTensor"] = "LoRA Model \u2192 Tensor"


class CdlTensorToLoraModel(io.ComfyNode):
    """Unpack a ``TENSOR`` plus metadata back into a ``LORA_MODEL`` dictionary.

    What it does:
        Exact inverse of :class:`CdlLoraModelToTensor`.  Slices the flat tensor
        according to ``shapes``, casts each chunk back to its recorded dtype and
        rebuilds the ``dict[str, torch.Tensor]`` under the recorded key order.

    Inputs:
        tensor (TENSOR): the flat tensor produced by the pack node.
        keys (ARRAY): key order, as ``list[str]``.
        shapes (ARRAY): one ``"3,4"`` style shape string per tensor.
        dtypes (ARRAY): one ``"torch.float32"`` style dtype string per tensor.

    Outputs:
        lora_model (LORA_MODEL): the restored mapping.

    Notes:
        Inconsistent metadata never aborts a workflow: unparseable entries are
        skipped and a truncated tensor yields only the tensors that fit.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlTensorToLoraModel",
            display_name="Tensor \u2192 LoRA Model",
            category=CATEGORY,
            description=(
                "Reserved: rebuild a LORA_MODEL (dict[str, Tensor]) from a flat "
                "TENSOR plus the metadata emitted by LoRA Model \u2192 Tensor."
            ),
            search_aliases=["tensor to lora", "unpack lora", "rebuild lora model"],
            is_experimental=True,
            inputs=[
                io.Tensor.Input("tensor", tooltip="Flat tensor produced by the pack node."),
                io.Array.Input("keys", tooltip="Key order, as list[str]."),
                io.Array.Input("shapes", tooltip="One shape string per tensor."),
                io.Array.Input("dtypes", tooltip="One dtype string per tensor."),
            ],
            outputs=[
                io.LoraModel.Output(display_name="LORA_MODEL"),
            ],
        )

    @classmethod
    def execute(cls, tensor, keys, shapes, dtypes) -> io.NodeOutput:
        chunks = _split_tensors(tensor, shapes, dtypes, "CdlTensorToLoraModel")
        key_list = list(keys) if isinstance(keys, (list, tuple)) else []

        if len(key_list) != len(chunks):
            print(
                "[CdlTensorToLoraModel] key/shape metadata disagree "
                f"({len(key_list)} key(s) vs {len(chunks)} tensor(s)); "
                "keeping the ones that line up."
            )

        lora_model = {}
        for key, chunk in zip(key_list, chunks):
            lora_model[str(key)] = chunk
        return io.NodeOutput(lora_model)


NODE_CLASS_MAPPINGS["CdlTensorToLoraModel"] = CdlTensorToLoraModel
NODE_DISPLAY_NAME_MAPPINGS["CdlTensorToLoraModel"] = "Tensor \u2192 LoRA Model"


# --------------------------------------------------------------------------- #
# LOSS_MAP pack / unpack (reserved)
#
# LOSS_MAP is ``{"loss": [Tensor, ...]}`` -- an ordered list with no keys, so
# only the shape and dtype layout has to travel on the side.
# --------------------------------------------------------------------------- #

class CdlLossMapToTensor(io.ComfyNode):
    """Pack a ``LOSS_MAP`` into one ``TENSOR`` plus shape/dtype metadata.

    What it does:
        Flattens and concatenates ``loss_map["loss"]`` into a single 1-D tensor;
        the per-tensor shape and dtype are emitted as parallel metadata.

    Inputs:
        loss_map (LOSS_MAP): ``{"loss": [Tensor, ...]}``.

    Outputs:
        tensor (TENSOR): every loss tensor flattened and concatenated, in order.
        shapes (ARRAY): one ``"3,4"`` style shape string per tensor.
        dtypes (ARRAY): one ``"torch.float32"`` style dtype string per tensor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlLossMapToTensor",
            display_name="Loss Map \u2192 Tensor",
            category=CATEGORY,
            description=(
                "Reserved: pack a LOSS_MAP ({\"loss\": [Tensor, ...]}) into one "
                "TENSOR plus shape/dtype metadata."
            ),
            search_aliases=["loss to tensor", "pack loss map", "flatten losses"],
            is_experimental=True,
            inputs=[
                io.LossMap.Input("loss_map", tooltip="A LOSS_MAP: {'loss': [Tensor, ...]}."),
            ],
            outputs=[
                io.Tensor.Output(display_name="TENSOR"),
                io.Array.Output(display_name="shapes"),
                io.Array.Output(display_name="dtypes"),
            ],
        )

    @classmethod
    def execute(cls, loss_map) -> io.NodeOutput:
        losses = _extract_losses(loss_map)

        flat = _cat_tensors(losses, "CdlLossMapToTensor")
        shapes = [_encode_shape(tensor) for tensor in losses]
        dtypes = [str(tensor.dtype) for tensor in losses]
        return io.NodeOutput(flat, shapes, dtypes)


def _extract_losses(loss_map) -> list:
    """Return the tensor list held by a LOSS_MAP, validating its contents."""
    if isinstance(loss_map, dict):
        losses = loss_map.get("loss")
    else:
        losses = loss_map
    if losses is None:
        return []
    if isinstance(losses, torch.Tensor):
        losses = [losses]
    if not isinstance(losses, (list, tuple)):
        raise TypeError(
            "CdlLossMapToTensor: expected a LOSS_MAP dictionary with a 'loss' "
            f"list, got {type(loss_map).__name__!r}."
        )
    result = []
    for index, tensor in enumerate(losses):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"CdlLossMapToTensor: loss[{index}] is "
                f"{type(tensor).__name__!r}, not a torch.Tensor."
            )
        result.append(tensor)
    return result


NODE_CLASS_MAPPINGS["CdlLossMapToTensor"] = CdlLossMapToTensor
NODE_DISPLAY_NAME_MAPPINGS["CdlLossMapToTensor"] = "Loss Map \u2192 Tensor"


class CdlTensorToLossMap(io.ComfyNode):
    """Unpack a ``TENSOR`` plus metadata back into a ``LOSS_MAP``.

    What it does:
        Exact inverse of :class:`CdlLossMapToTensor`.  Slices the flat tensor by
        ``shapes``, casts each chunk back to its recorded dtype and returns
        ``{"loss": [Tensor, ...]}``.

    Inputs:
        tensor (TENSOR): the flat tensor produced by the pack node.
        shapes (ARRAY): one ``"3,4"`` style shape string per tensor.
        dtypes (ARRAY): one ``"torch.float32"`` style dtype string per tensor.

    Outputs:
        loss_map (LOSS_MAP): ``{"loss": [Tensor, ...]}``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlTensorToLossMap",
            display_name="Tensor \u2192 Loss Map",
            category=CATEGORY,
            description=(
                "Reserved: rebuild a LOSS_MAP ({\"loss\": [Tensor, ...]}) from a "
                "flat TENSOR plus the metadata emitted by Loss Map \u2192 Tensor."
            ),
            search_aliases=["tensor to loss", "unpack loss map", "rebuild losses"],
            is_experimental=True,
            inputs=[
                io.Tensor.Input("tensor", tooltip="Flat tensor produced by the pack node."),
                io.Array.Input("shapes", tooltip="One shape string per tensor."),
                io.Array.Input("dtypes", tooltip="One dtype string per tensor."),
            ],
            outputs=[
                io.LossMap.Output(display_name="LOSS_MAP"),
            ],
        )

    @classmethod
    def execute(cls, tensor, shapes, dtypes) -> io.NodeOutput:
        losses = _split_tensors(tensor, shapes, dtypes, "CdlTensorToLossMap")
        return io.NodeOutput({"loss": losses})


NODE_CLASS_MAPPINGS["CdlTensorToLossMap"] = CdlTensorToLossMap
NODE_DISPLAY_NAME_MAPPINGS["CdlTensorToLossMap"] = "Tensor \u2192 Loss Map"
