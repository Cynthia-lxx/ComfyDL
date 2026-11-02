"""Dataset Pack/Unpack — column-to-port dynamic binding for the ``DATASET`` type.

    DATASET --unpack--> [col_0, col_1, ...]     (one TENSOR port per column)
    [col_0, ...] --pack--> DATASET              (columns merge back into a table)

Unlike :mod:`.data_list` (native ComfyUI list semantics, one port), this family
binds **columns to ports**: the number of output/input slots changes with the
table schema, so a training node can consume ``x0``/``x1``/... as individual
tensors and a feature-engineering branch can re-pack them.

Design (ported from ComfyUI-List-Utils, godmt Pack/Unpack, and
ComfyUI-basic_data_handling ``ContainsDynamicDict`` — see
``.codebuddy/memory/refs_survey/survey_D_utils.md``):

* **Truth lives in the serialized workflow.**  ``Pack`` reads its own input
  port names back through the hidden ``unique_id``/``extra_pnginfo`` channel
  (the frontend writes the real column name into each port when a link is
  made, and the port metadata is serialized with the workflow).  This keeps
  the Python side free of per-port declarations while preserving user-chosen
  column names — including inside subgraphs (execution IDs are colon chains
  like ``"19:84"``, resolved against ``workflow.definitions.subgraphs``).
* **Dynamic inputs** use the ``ContainsDynamicDict`` trick: ``INPUT_TYPES``
  declares only ``col_0`` but validation accepts any ``col_<N>``; the
  frontend grows/shrinks the slots (``app/dataops_assets/dataops.js``).
* **Dynamic outputs** (Unpack, same module) use the ``ByPassTypeTuple`` hack:
  ``RETURN_TYPES`` is a tuple whose ``__getitem__`` always yields an
  ``AnyType`` wildcard, so the backend allows an arbitrary number of output
  links while the frontend mirrors the real ports.

Efficiency notes:
  * Unpack emits zero-copy column views (``features[:, i]``);
  * Pack stacks the columns with a single ``torch.stack`` (one O(N*C) copy —
    the packed table then lives independently of the source tensors).

Both nodes live in the ``d2l/DataOps`` category.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Prefix used for Pack's dynamic input ports.  A fixed prefix (instead of raw
# column names) keeps the frontend's add/remove/renumber logic unambiguous —
# a column may legally end in a digit, but the *port* name always starts with
# this prefix (the real column name rides in the port's serialized name/label
# and in the workflow reflection below).
COL_PREFIX = "col_"


# --------------------------------------------------------------------- #
# Type-system helpers (wildcard machinery, List-Utils style)
# --------------------------------------------------------------------- #


class AnyType(str):
    """Wildcard type: compares unequal to nothing, so it links with anything.

    ComfyUI validates links with ``input_type != output_type``; returning
    ``False`` from ``__ne__`` lets a port declared as ``AnyType`` accept (and
    be accepted by) every concrete type.
    """

    def __ne__(self, __value: object) -> bool:
        return False


ANY = AnyType("*")


class ByPassTypeTuple(tuple):
    """A tuple that yields an ``AnyType`` wildcard for every index.

    ``RETURN_TYPES = ByPassTypeTuple(("*",))`` keeps ``len() == 1`` (so
    serialization and the "at least one output" contract stay valid) while
    ``RETURN_TYPES[i]`` for any ``i >= 0`` returns ``ANY`` — the backend then
    permits an arbitrary number of output links without a type mismatch.
    The real port count/names/types are maintained by the frontend.
    """

    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        return AnyType(item) if isinstance(item, str) else item


class ContainsDynamicDict(dict):
    """INPUT_TYPES dict that "contains" any ``<prefix><digits>`` key.

    Declare one seed entry (e.g. ``"col_0": ("TENSOR", {"_dynamic": "number"})``);
    validation then accepts ``col_1`` … ``col_N`` as if they were declared,
    which is what lets the frontend grow an unbounded column list without a
    server round-trip.  The ``_dynamic`` marker inside the entry's options is
    consumed by the frontend (it is preserved through JSON serialization).
    """

    def __init__(self, seed: dict):
        super().__init__(seed)
        self._prefixes: List[str] = []
        for key, spec in seed.items():
            opts = spec[1] if isinstance(spec, tuple) and len(spec) > 1 else {}
            if isinstance(opts, dict) and opts.get("_dynamic") == "number":
                self._prefixes.append(key.rstrip("0123456789"))

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            return any(
                key.startswith(p) and key[len(p):].isdigit()
                for p in self._prefixes
            )
        return False

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            for p in self._prefixes:
                if key.startswith(p) and key[len(p):].isdigit():
                    return super().__getitem__(p + "0")
        raise KeyError(key)


# --------------------------------------------------------------------- #
# Workflow reflection (hidden inputs -> serialized graph)
# --------------------------------------------------------------------- #


def _unwrap_single(value: Any) -> Any:
    """Unwrap the one-element list the engine puts around hidden inputs."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _workflow_from_extra_pnginfo(extra_pnginfo: Any) -> dict:
    """Extract the serialized workflow dict from the EXTRA_PNGINFO payload."""
    extra_pnginfo = _unwrap_single(extra_pnginfo)
    if not isinstance(extra_pnginfo, dict):
        return {}
    workflow = extra_pnginfo.get("workflow", extra_pnginfo)
    return workflow if isinstance(workflow, dict) else {}


def _find_node(graph: Any, node_id: Any) -> Optional[dict]:
    """Find a node in a serialized graph (matched as string on ``id``)."""
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    return next(
        (n for n in nodes if str(n.get("id")) == str(node_id)), None
    )


def _find_subgraph_definition(workflow: dict, definition_id: Any) -> Optional[dict]:
    """Look up a subgraph definition by id (``workflow.definitions.subgraphs``)."""
    definitions = workflow.get("definitions", {})
    subgraphs = (
        definitions.get("subgraphs", [])
        if isinstance(definitions, dict)
        else []
    )
    return next(
        (sg for sg in subgraphs if str(sg.get("id")) == str(definition_id)),
        None,
    )


def resolve_workflow_node(extra_pnginfo: Any, unique_id: Any) -> Tuple[Optional[dict], Optional[dict]]:
    """Resolve a node (and its owning graph) from hidden execution metadata.

    ``unique_id`` is the engine's node identifier: a plain id at graph root
    (``"7"``) or a colon chain for nested subgraphs (``"19:84:7"`` — each
    intermediate segment is a SubgraphNode whose serialized ``type`` field is
    a subgraph definition id).  Returns ``(node, graph)`` or ``(None, None)``
    when the workflow payload is missing or the chain cannot be followed —
    callers must degrade gracefully (fallback port naming), never raise.
    """
    workflow = _workflow_from_extra_pnginfo(extra_pnginfo)
    graph: Optional[dict] = workflow
    parts = str(_unwrap_single(unique_id)).split(":")
    for index, node_id in enumerate(parts):
        node = _find_node(graph, node_id)
        if node is None:
            return None, None
        if index == len(parts) - 1:
            return node, graph
        graph = _find_subgraph_definition(workflow, node.get("type"))
        if graph is None:
            return None, None
    return None, None


def reflect_input_names(
    extra_pnginfo: Any, unique_id: Any, fallback_names: List[str]
) -> List[str]:
    """Return the serialized port names for this node's dynamic inputs.

    Falls back to ``fallback_names`` (the raw ``col_<N>`` kwargs keys) when
    the workflow payload is unavailable (headless runs, unit tests, exports
    without metadata) — reflection is an enhancement, never a requirement.
    """
    node, _ = resolve_workflow_node(extra_pnginfo, unique_id)
    if not node:
        return list(fallback_names)
    serialized = [str(i.get("name", "")) for i in node.get("inputs", [])]
    names: List[str] = []
    for raw in fallback_names:
        match = next((s for s in serialized if s == raw), None)
        # A renamed port keeps the col_ prefix as its serialized name but the
        # frontend stores the display name in ``label``; prefer label, then
        # name, then the raw fallback.
        if match is not None:
            entry = next(
                (i for i in node.get("inputs", []) if str(i.get("name", "")) == raw),
                {},
            )
            names.append(str(entry.get("label") or entry.get("name") or raw))
        else:
            names.append(raw)
    return names


# --------------------------------------------------------------------- #
# CdlDatasetPack
# --------------------------------------------------------------------- #


def _as_column(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Coerce a connected value into a single-column ``(n,)``/``(n,1)`` tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"CdlDatasetPack: input {name!r} must be a TENSOR, got "
            f"{type(tensor).__name__}"
        )
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(
        f"CdlDatasetPack: input {name!r} has shape {tuple(tensor.shape)} — "
        "each port carries exactly ONE column ((n,) or (n,1)); reshape or "
        "split multi-column tensors upstream."
    )


class CdlDatasetPack:
    """Pack per-column tensor ports into one ``DATASET`` table.

    Inputs (dynamic):
        col_0, col_1, ... : one TENSOR port per column — ``(n,)`` or ``(n,1)``.
        All ports must agree on ``n`` (row count).  The frontend adds a port
        per connection (``ContainsDynamicDict`` + dataops.js); unconnected
        ports are simply absent from the execution, so nothing to strip.

    Widgets:
        target_name: when a connected column's port name equals this string
        (default ``"target"``) that column becomes the dataset's ``labels``
        and is excluded from ``feature_names``; when nothing matches, the
        dataset is built unlabelled.  Empty string disables the target
        mechanism entirely.

    Port naming:
        The real column names come from the serialized workflow via the
        hidden ``unique_id``/``extra_pnginfo`` reflection (the frontend writes
        the upstream column name into the port when you connect a link, and
        that name travels with the saved workflow).  Without workflow
        metadata the raw port names (``col_0``…) are used as-is.

    Output:
        one ``DATASET`` — ``features`` is a single ``torch.stack`` of the
        columns (one O(n*c) copy), ``labels`` the target column, metadata
        gains a ``pack`` provenance note (columns / target / rows).  The
        output is a first-class ``DATASET`` (not a private container type),
        so it plugs straight into the readers/writers/trainers/preview.

    Raises a ``ValueError`` naming the offending ports when columns disagree
    on row count or when duplicate column names would collide.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_name": ("STRING", {"default": "target"}),
            },
            "optional": ContainsDynamicDict({
                "col_0": (
                    "TENSOR",
                    {
                        "forceInput": True,
                        "_dynamic": "number",
                        "tooltip": "One column of the table ((n,) or (n,1)). "
                                   "Connect more values to add more columns.",
                    },
                ),
            }),
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/DataOps"

    def execute(
        self,
        target_name: str = "target",
        unique_id: Any = None,
        extra_pnginfo: Any = None,
        **columns: torch.Tensor,
    ) -> Tuple[CdlDataset]:
        if not columns:
            raise ValueError(
                "CdlDatasetPack: no columns connected — connect at least one "
                "TENSOR port (col_0, col_1, ...)."
            )

        # Reflection: port order in kwargs follows the prompt's input order,
        # which matches the serialized workflow's inputs order.
        raw_names = list(columns.keys())
        names = reflect_input_names(extra_pnginfo, unique_id, raw_names)

        target_name = str(target_name or "")
        cols: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        feat_names: List[str] = []
        labels: Optional[torch.Tensor] = None

        for name, value in zip(names, columns.values()):
            col = _as_column(value, name)
            if cols and col.shape[0] != cols[0].shape[0]:
                raise ValueError(
                    f"CdlDatasetPack: column {name!r} has {col.shape[0]} rows "
                    f"but column {names[0]!r} has {cols[0].shape[0]} — all "
                    "ports must agree on the row count."
                )
            if target_name and name == target_name:
                if labels is not None:
                    raise ValueError(
                        f"CdlDatasetPack: two columns are named "
                        f"{target_name!r} — duplicate port names cannot both "
                        "be the target."
                    )
                labels = value  # keep the target exactly as connected
                # (shape-faithful: a (n,) target stays (n,), so an
                # Unpack -> Pack round-trip reproduces the source labels)
                continue
            if name in feat_names:
                raise ValueError(
                    f"CdlDatasetPack: duplicate column name {name!r} — "
                    "rename one of the ports (column names must be unique)."
                )
            cols.append(col)
            feat_names.append(name)

        if not cols:
            raise ValueError(
                f"CdlDatasetPack: the only connected column is named "
                f"{target_name!r} (the target) — a DATASET needs at least "
                "one feature column."
            )

        features = torch.cat(cols, dim=1) if any(
            c.shape[1] != 1 for c in cols
        ) else torch.stack([c.squeeze(1) for c in cols], dim=1)

        n_rows = int(features.shape[0])
        meta = {
            "pack": {
                "columns": list(feat_names),
                "target": target_name if labels is not None else "",
                "rows": n_rows,
            }
        }
        return (CdlDataset(
            features=features,
            labels=labels,
            feature_names=feat_names,
            target_name=target_name if labels is not None else "",
            meta=meta,
        ),)


NODE_CLASS_MAPPINGS.update({
    "CdlDatasetPack": CdlDatasetPack,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "CdlDatasetPack": "Dataset Pack (columns → ports)",
})


# --------------------------------------------------------------------- #
# CdlDatasetUnpack
# --------------------------------------------------------------------- #


class CdlDatasetUnpack:
    """Unpack a ``DATASET`` into one output port per column.

    Input:
        dataset: the source table (``DATASET``, by reference).

    Outputs (dynamic):
        one port per feature column, then one port for the target when the
        dataset is labelled.  ``RETURN_TYPES`` is a :class:`ByPassTypeTuple`
        so the backend accepts an arbitrary number of output links; the real
        port names/types are maintained by the frontend (``dataops.js``):
        ports appear either by mirroring a connected upstream
        ``CdlDatasetPack`` (static channel) or after the first execution
        (runtime channel — this node returns its column schema through the
        ``ui`` payload, the same channel the App Panel uses, and the ports
        then persist with the saved workflow).

    Efficiency:
        every output is a zero-copy column view (``features[:, i]``); nothing
        is copied, so unpacking a 1 GB table is free.

    Raises a ``ValueError`` when the dataset has no columns at all (nothing
    to unpack — an empty ``feature_names`` list with no labels is invalid).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ByPassTypeTuple((ANY,))
    RETURN_NAMES = ByPassTypeTuple(("col_0",))
    OUTPUT_TOOLTIPS = ByPassTypeTuple(("One column of the table as a tensor.",))
    FUNCTION = "execute"
    CATEGORY = "d2l/DataOps"

    def execute(self, dataset, unique_id=None, extra_pnginfo=None):
        names = list(dataset.feature_names)
        columns: List[torch.Tensor] = [
            dataset.features[:, i] for i in range(dataset.features.shape[1])
        ]
        if dataset.labels is not None:
            if not dataset.target_name:
                # keep the port name scheme stable even for nameless targets
                names.append("target")
            else:
                names.append(dataset.target_name)
            columns.append(dataset.labels)
        if not names:
            raise ValueError(
                "CdlDatasetUnpack: the dataset has zero feature columns and "
                "no labels — nothing to unpack (readers should always emit "
                "at least one feature column)."
            )

        schema = [
            {"name": n, "type": "TENSOR"} for n in names
        ]
        # ui payload drives the frontend's runtime channel: dataops.js reads
        # message.columns onExecuted and (re)builds the output ports; the
        # ports then persist inside the serialized workflow.
        return {
            "ui": {"columns": schema},
            "result": tuple(columns),
        }


NODE_CLASS_MAPPINGS.update({
    "CdlDatasetUnpack": CdlDatasetUnpack,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "CdlDatasetUnpack": "Dataset Unpack (ports ← columns)",
})
