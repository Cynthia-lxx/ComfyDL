"""
Dataset list semantics — native ComfyUI list flow for the ``DATASET`` type.

    DATASET --split--> [DATASET, DATASET, ...]   (a native ComfyUI list)
    [DATASET, ...] --merge--> DATASET
    DATASET --getitem--> DATASET (single row)

This is the first node family in the repo that uses the framework's native
list channel: ``OUTPUT_IS_LIST`` makes one execution emit *N* values on a
single output port (the engine fans them out to N downstream links), and
``INPUT_IS_LIST`` collects all incoming links into one Python list (the
harness and the engine both wrap *every* input, see execution.py).  Prior to
this family the repo's nodes were strictly one-value-per-port.

Efficiency notes:
  * unshuffled batches are **zero-copy tensor views** (``features[a:b]``) —
    the whole split of a 1 GB table costs nothing but the Python wrappers;
  * shuffled batches are gathered with ``index_select`` (one copy, same as
    every DataLoader);
  * merging is a single ``torch.cat`` (O(N) copy — concatenating a large
    dataset doubles its memory for the lifetime of the merged tensor).

All three nodes live in the ``d2l/Datasets`` category.
"""

from __future__ import annotations

from typing import List

import torch

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _row_view(ds: CdlDataset, start: int, end: int, meta_extra: dict) -> CdlDataset:
    """A zero-copy ``(start:end)`` row view of a dataset, with split metadata."""
    meta = dict(ds.meta)
    meta.update(meta_extra)
    return CdlDataset(
        features=ds.features[start:end],
        labels=None if ds.labels is None else ds.labels[start:end],
        feature_names=list(ds.feature_names),
        target_name=ds.target_name,
        meta=meta,
    )


class CdlDatasetSplitBatches:
    """Split a ``DATASET`` into a native list of fixed-size row batches.

    Inputs:
        dataset   : the ``DATASET`` to split.
        batch_size: rows per batch (1..1000000, default 32).
        drop_last : drop the final short batch instead of emitting it
                    (default False — the remainder is emitted as a short
                    batch; True gives uniformly sized batches, like a
                    training DataLoader with ``drop_last=True``).
        shuffle   : permute the rows before splitting (default False).
        perm_seed : permutation seed — the split is deterministic for a
                    given (seed, n) pair and uses a local ``torch.Generator``,
                    so the global RNG state is never touched.  (Named
                    ``perm_seed`` rather than ``seed`` on purpose: the
                    frontend attaches a ``control_after_generate`` companion
                    widget to inputs named ``seed``/``noise_seed``, and that
                    companion mis-aligns ``widgets_values`` on load when the
                    node's first input is a forceInput slot — live-reproduced
                    2026-11-01.)

    Output:
        ``OUTPUT_IS_LIST=(True,)`` — one execution yields a list of
        ``CdlDataset`` row views; the engine fans the list out to every
        downstream link (one batch per link).  Unshuffled batches share
        storage with the input (zero-copy views).

    An empty input dataset yields an empty list (no batches).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 1000000}),
                "drop_last": ("BOOLEAN", {"default": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "perm_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("batches",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset, batch_size, drop_last=False, shuffle=False,
                perm_seed=0):
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        n = dataset.n_samples
        if shuffle:
            gen = torch.Generator()
            gen.manual_seed(int(perm_seed))
            order = torch.randperm(n, generator=gen)
        else:
            order = None  # keep the zero-copy contiguous slicing path

        split_meta = {
            "split": {
                "batch_size": batch_size,
                "drop_last": bool(drop_last),
                "shuffle": bool(shuffle),
                "perm_seed": int(perm_seed),
            }
        }
        batches: List[CdlDataset] = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n and drop_last:
                break
            info = {**split_meta, "split": {**split_meta["split"], "rows": end - start}}
            if order is None:
                batches.append(_row_view(dataset, start, end, info))
            else:
                idx = order[start:end]
                meta = dict(dataset.meta)
                meta.update(info)
                batches.append(CdlDataset(
                    features=dataset.features.index_select(0, idx),
                    labels=None if dataset.labels is None
                    else dataset.labels.index_select(0, idx),
                    feature_names=list(dataset.feature_names),
                    target_name=dataset.target_name,
                    meta=meta,
                ))
        return (batches,)


class CdlDatasetFromList:
    """Merge a native list of ``DATASET``s back into a single ``DATASET``.

    Input:
        dataset: with ``INPUT_IS_LIST`` the engine delivers every incoming
        link as one Python list — the complementary consumer of
        :class:`CdlDatasetSplitBatches` (roundtrip split -> merge must
        reproduce the original rows, minus rows dropped by ``drop_last``).

    Output:
        one ``DATASET`` whose features/labels are ``torch.cat`` of the
        inputs (one O(N) copy; feature names and target must agree).
        Metadata is carried from the first item plus a ``merged`` note.

    Raises a ``ValueError`` listing the concrete mismatch when the items
    disagree on feature names, target name or label presence — the message
    always names the differing columns so the offending link is obvious.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset):
        items: List[CdlDataset] = (
            list(dataset) if isinstance(dataset, (list, tuple)) else [dataset]
        )
        if not items:
            raise ValueError(
                "CdlDatasetFromList received an empty dataset list — connect at "
                "least one DATASET link (a split with drop_last=True on an "
                "empty source also produces zero batches)."
            )
        first = items[0]
        for pos, other in enumerate(items[1:], start=1):
            if other.feature_names != first.feature_names:
                extra = [c for c in other.feature_names if c not in first.feature_names]
                missing = [c for c in first.feature_names if c not in other.feature_names]
                raise ValueError(
                    "CdlDatasetFromList: item 0 and item "
                    f"{pos} disagree on feature columns "
                    f"(item 0: {first.feature_names}; item {pos}: "
                    f"{other.feature_names}; missing: {missing}, extra: {extra})"
                )
            if other.target_name != first.target_name:
                raise ValueError(
                    "CdlDatasetFromList: item 0 and item "
                    f"{pos} disagree on target name "
                    f"({first.target_name!r} vs {other.target_name!r})"
                )
            if (other.labels is None) != (first.labels is None):
                raise ValueError(
                    "CdlDatasetFromList: mixing labelled and unlabelled "
                    f"datasets (item 0 labelled={first.labels is not None}, "
                    f"item {pos} labelled={other.labels is not None})"
                )
        features = torch.cat([it.features for it in items], dim=0)
        labels = (
            torch.cat([it.labels for it in items], dim=0)
            if first.labels is not None else None
        )
        meta = dict(first.meta)
        meta["merged"] = {
            "items": len(items),
            "rows": int(features.shape[0]),
        }
        return (CdlDataset(
            features=features,
            labels=labels,
            feature_names=list(first.feature_names),
            target_name=first.target_name,
            meta=meta,
        ),)


class CdlDatasetGetItem:
    """Pick one row of a ``DATASET`` as a single-sample ``DATASET`` view.

    Inputs:
        dataset: the source table.
        index  : zero-based row index (default 0).  Negative indices are
                 rejected on purpose — with the engine's list semantics an
                 accidental ``-1`` would silently pick the last row of the
                 wrong batch.

    Output:
        a ``CdlDataset`` with ``n_samples == 1`` (a zero-copy view of the
        source row); feature names / target / metadata are carried over and
        the metadata gains the picked ``row`` index.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("row",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset, index):
        index = int(index)
        n = dataset.n_samples
        if not 0 <= index < n:
            raise ValueError(
                f"CdlDatasetGetItem: index {index} is out of range for a "
                f"dataset with n_samples={n} (indices run 0..{n - 1})"
            )
        meta = dict(dataset.meta)
        meta["row"] = index
        return (CdlDataset(
            features=dataset.features[index:index + 1],
            labels=None if dataset.labels is None else dataset.labels[index:index + 1],
            feature_names=list(dataset.feature_names),
            target_name=dataset.target_name,
            meta=meta,
        ),)


NODE_CLASS_MAPPINGS = {
    "CdlDatasetSplitBatches": CdlDatasetSplitBatches,
    "CdlDatasetFromList": CdlDatasetFromList,
    "CdlDatasetGetItem": CdlDatasetGetItem,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CdlDatasetSplitBatches": "Dataset Split Batches",
    "CdlDatasetFromList": "Dataset From List",
    "CdlDatasetGetItem": "Dataset Get Item",
}
