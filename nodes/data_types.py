"""
CdlDataset — the universal ``DATASET`` type for ComfyDL training & data-IO nodes.

A :class:`CdlDataset` bundles the data that flows between the dataset readers,
the regression trainers and the adapters:

    features      : torch.Tensor  (n_samples, n_features)  -- the model inputs
    labels        : torch.Tensor | None  (n_samples,) or (n_samples, 1)
    feature_names : list[str]      -- column names for the features
    target_name   : str            -- column name of the label (if any)
    meta          : dict           -- free-form metadata (source, dtypes, ...)

Instances are passed **by reference** between nodes (never copied), so a big
tabular dataset costs nothing extra to route through the graph. Build one with
:meth:`CdlDataset.from_tensors` (from torch tensors) or
:meth:`CdlDataset.from_dataframe` (from a pandas DataFrame, zero-copy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is an optional heavy dep
    pd = None


@dataclass
class CdlDataset:
    features: torch.Tensor
    labels: Optional[torch.Tensor] = None
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_tensors(cls, X, y=None, feature_names=None, target_name="", meta=None):
        """Build a dataset from torch tensors (or array-likes).

        ``X`` is coerced to a 2-D ``(n_samples, n_features)`` float tensor; a
        1-D input is treated as a single feature column. ``y`` is optional and
        kept as-is (typically ``(n_samples,)`` for regression).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
        else:
            feature_names = list(feature_names)
            if len(feature_names) != X.shape[1]:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries but X has "
                    f"{X.shape[1]} feature columns"
                )
        return cls(
            features=X,
            labels=y,
            feature_names=feature_names,
            target_name=target_name,
            meta=dict(meta) if meta else {},
        )

    @classmethod
    def from_dataframe(cls, df, target=None, feature_names=None, dtype="float32"):
        """Build a dataset from a pandas DataFrame with a zero-copy numpy view.

        ``target`` (a column name) becomes the labels tensor and is removed from
        the features. When ``target`` is ``None`` the dataset is unlabelled.
        ``feature_names`` restricts/orders the feature columns; defaults to all
        remaining columns.
        """
        if pd is None:
            raise ImportError(
                "pandas is required to build a DATASET from a DataFrame; "
                "install it with `pip install pandas`."
            )
        df = df.copy(deep=False)  # avoid mutating the caller's frame
        y = None
        target_name = ""
        if target is not None:
            if target not in df.columns:
                raise KeyError(f"target column {target!r} not found in DataFrame")
            y = torch.from_numpy(df.pop(target).to_numpy().astype(dtype))
            target_name = target

        feats = list(feature_names) if feature_names else list(df.columns)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(f"feature columns not found: {missing}")
        X = torch.from_numpy(df[feats].to_numpy(dtype=dtype))
        return cls(
            features=X,
            labels=y,
            feature_names=feats,
            target_name=target_name,
            meta={"n_samples": int(X.shape[0])},
        )

    # ------------------------------------------------------------------ #
    # Views / helpers
    # ------------------------------------------------------------------ #
    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def with_labels(self, y) -> "CdlDataset":
        """Return a copy that points at a new label tensor (shared features)."""
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        return CdlDataset(
            features=self.features,
            labels=y,
            feature_names=list(self.feature_names),
            target_name=self.target_name,
            meta=dict(self.meta),
        )

    def to_dataframe(self):
        """Materialise the dataset back into a pandas DataFrame (copies)."""
        if pd is None:
            raise ImportError(
                "pandas is required to export a DATASET to a DataFrame; "
                "install it with `pip install pandas`."
            )
        cols = {
            name: self.features[:, i].numpy()
            for i, name in enumerate(self.feature_names)
        }
        df = pd.DataFrame(cols)
        if self.labels is not None:
            df[self.target_name or "label"] = self.labels.reshape(-1).numpy()
        return df

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CdlDataset(n={self.n_samples}, features={self.n_features}, "
            f"labels={'yes' if self.labels is not None else 'no'}, "
            f"target={self.target_name or '-'})"
        )
