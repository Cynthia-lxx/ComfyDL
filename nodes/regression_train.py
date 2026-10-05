"""
regression_train — a coarse-grained, production-style regression trainer.

Where :class:`CdlLinRegTrain` (in ``linreg_train.py``) is the *transparent,
teaching* recipe (four ports you can verify by hand), this node is the
"ship it" counterpart: feed it a labelled table and it does the whole
production dance in one box:

    DATASET  ──►  train/validation split  ──►  standardise features
                  (optional)                     (z-score on the train split)
                              │
                              ▼
        nn_model  +  metrics(mae / rmse)  +  predictions  +  loss_history
                    (optionally written to disk for later inference)

Design notes
------------
* The mechanism deliberately mirrors the rehydration ``TrainingLoop``
  (``comfy_extras.nodes_training``): an Adam optimiser, mini-batch gradient
  descent, and **early stopping on a smoothed held-out loss with a rollback to
  the best weights** (patience / min-delta widgets). It is implemented with
  plain ``torch`` so ``comfydl`` stays decoupled from the rehydration
  ``comfy`` core and the node can be unit-tested on its own.
* The feature standardisation is folded **into** the returned ``nn_model``: a
  small :class:`_Regressor` module keeps the train mean / std as buffers and
  applies them in ``forward`` *before* the (linear or MLP) core. The saved or
  reused model is therefore self-contained — pushing raw features through
  ``CdlNNForward`` reproduces the in-training predictions with no external
  pre-processing.
* ``hidden`` lets you graduate from plain linear regression (empty string) to a
  small feed-forward network (e.g. ``"16,8"``) without leaving the node,
  exactly like ``TrainingLoop``'s hidden-layer widget.
* Training reports through ComfyUI's official progress channel: exactly one
  ``ProgressBar`` call per optimiser step, with a live loss-curve preview
  frame attached (``comfy.loss_preview.LossCurvePreviewer``, rate-limited
  inside the previewer, final frame force-rendered). Both helpers are
  imported lazily inside ``execute`` — outside the ComfyUI host (unit tests,
  plain scripts) the node simply trains without any preview, in line with
  "a preview is a courtesy, not a contract".

Inputs
------
* ``dataset`` (DATASET, optional, forceInput): the preferred conduit. Wins over
  ``X`` / ``y`` when present. Must be labelled.
* ``X`` (TENSOR, required): feature matrix ``[n, f]`` (fallback when no
  ``DATASET`` is wired).
* ``y`` (TENSOR, required): target ``[n]`` or ``[n, k]`` (fallback).
* Widgets: ``test_size`` (val fraction; 0 = no split, evaluate on all),
  ``standardize`` (yes/no), ``hidden`` (comma widths or empty),
  ``activation`` (between hidden layers only), ``loss`` (mse/mae),
  ``steps``, ``batch_size`` (0 = full batch), ``lr``, ``seed``,
  ``early_stop_patience`` (0 = off), ``early_stop_min_delta``, ``save_path``
  (empty = do not persist).

Outputs
-------
* ``nn_model`` : the trained :class:`torch.nn.Module` (standardisation included).
* ``mae`` / ``rmse`` : scalar TENSORs, in the original target space, measured on
  the validation split (or the whole set when ``test_size == 0``).
* ``predictions`` : model output on the full input, ``[n, k]``.
* ``loss_history`` : 1-D TENSOR, one entry per executed optimiser step (truncated
  to the best step when early stopping fires).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": None,
}

_LOSS = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
}


def _parse_hidden(hidden: str) -> Tuple[int, ...]:
    """Parse a ``"16,8"`` style width list into a tuple of positive ints."""
    text = (hidden or "").strip()
    if not text:
        return ()
    sizes: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            # Unreadable text falls back to a single hidden layer of width 8,
            # matching TrainingLoop's defensive behaviour.
            return (8,)
        if value > 0:
            sizes.append(value)
    return tuple(sizes)


def _make_progress(steps: int):
    """Create the ``(ProgressBar, LossCurvePreviewer)`` pair for live reporting.

    What: wires the node into ComfyUI's official progress side channel —
          ``comfy.utils.ProgressBar.update_absolute(value, total, preview)`` —
          and a curve previewer whose frames ride that channel's third
          argument.
    In:   ``steps`` — the total optimiser steps (the progress bar's total).
    Out:  the pair, or ``(None, None)`` when the host ``comfy`` core is not
          importable (unit tests, headless scripts): the node then trains
          without any reporting. Any failure degrades to no preview instead of
          breaking the run — a preview is a courtesy, not a contract.
    """
    try:
        import comfy.utils
        from comfy import loss_preview

        return (
            comfy.utils.ProgressBar(int(steps)),
            loss_preview.LossCurvePreviewer(title="loss"),
        )
    except Exception:  # noqa: BLE001 - no host, no preview
        return None, None


def _report_step(pbar, previewer, loss: float, done: int, total: int, force: bool) -> None:
    """Push exactly one progress call for one optimiser step.

    What: the per-step reporting contract shared with ``TrainingLoop`` — one
          call per step either way, so external step counters stay exact.
    In:   ``pbar`` / ``previewer`` — the pair from :func:`_make_progress`
          (either may be ``None``); ``loss`` — the detached scalar training
          loss of this step; ``done`` / ``total`` — progress counters;
          ``force`` — ``True`` on a run's final step (natural end or early
          stop) so the last pushed frame is the complete curve.
    """
    frame = None
    if pbar is not None and previewer is not None:
        frame = previewer.record(loss, force=force)
    if pbar is None:
        return
    if frame is not None:
        pbar.update_absolute(done, total, frame)
    else:
        pbar.update(1)


class _Regressor(nn.Module):
    """Linear / MLP regressor with standardisation folded into the module.

    ``forward`` computes ``z = (x - mean) / std`` (skipped when ``mean`` /
    ``std`` are ``None``) and then runs ``core(z)``. The statistics are
    registered as buffers so the whole model is self-contained and survives a
    ``state_dict`` save / load.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Tuple[int, ...] = (),
        activation: str = "relu",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if mean is not None:
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        layers: List[nn.Module] = []
        prev = in_features
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if act_cls is not None:
                layers.append(act_cls())
            prev = width
        layers.append(nn.Linear(prev, out_features))
        self.core = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "mean"):
            x = (x - self.mean) / self.std
        return self.core(x)


class _EarlyStop:
    """Moving-average early stop, mirroring ``TrainingLoop``'s tracker.

    Watches a smoothed loss (trailing window of 8) and, once ``patience`` steps
    pass without an improvement of at least ``min_delta``, signals a stop. Tracks
    the best snapshot externally so the caller can roll back.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.counter = 0
        self.improved = False
        self._window: List[float] = []

    def update(self, step: int, loss: float) -> bool:
        self._window.append(loss)
        if len(self._window) > 8:
            self._window.pop(0)
        smoothed = sum(self._window) / len(self._window)
        self.improved = False
        if self.best is None or smoothed < self.best - self.min_delta:
            self.best = smoothed
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
        if self.patience > 0 and self.counter >= self.patience:
            return True
        return False


class CdlRegressionTrain:
    """Train a regression model end-to-end and return a reusable ``nn_model``.

    One node covers the production regression pipeline: an optional
    train/validation split, feature standardisation, mini-batch Adam training
    with early stopping (rolled back to the best weights), and the metrics a
    production user actually wants (MAE / RMSE on the held-out set). The
    standardisation is baked into the returned module so it can be dropped
    straight into ``CdlNNForward`` or saved for later inference.

    See the module docstring for the full input / output contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("TENSOR",),
                "y": ("TENSOR",),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.95, "step": 0.05}),
                "standardize": (["yes", "no"], {"default": "yes"}),
                "hidden": ("STRING", {"default": "", "placeholder": "e.g. 16 or 16,8 (empty = linear)"}),
                "activation": (["relu", "tanh", "sigmoid", "none"], {"default": "relu"}),
                "loss": (["mse", "mae"], {"default": "mse"}),
                "steps": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1}),
                "lr": ("FLOAT", {"default": 1e-1, "min": 1e-5, "max": 1.0, "step": 1e-3}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "early_stop_patience": ("INT", {"default": 30, "min": 0, "max": 100000, "step": 1}),
                "early_stop_min_delta": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "save_path": ("STRING", {"default": "", "placeholder": "e.g. C:/models/reg.pt (empty = do not save)"}),
            },
            "optional": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("nn_model", "TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("nn_model", "mae", "rmse", "predictions", "loss_history")
    FUNCTION = "execute"
    CATEGORY = "d2l/Training"

    def execute(
        self,
        X,
        y,
        dataset=None,
        test_size=0.2,
        standardize="yes",
        hidden="",
        activation="relu",
        loss="mse",
        steps=500,
        batch_size=0,
        lr=1e-1,
        seed=0,
        early_stop_patience=30,
        early_stop_min_delta=1e-4,
        save_path="",
    ):
        # ------------------------------------------------------------------ #
        # 1. Resolve the data source (DATASET wins, mirroring the adapters).
        # ------------------------------------------------------------------ #
        if dataset is not None:
            if not isinstance(dataset, CdlDataset):
                raise TypeError(
                    "CdlRegressionTrain: 'dataset' must be a DATASET object, "
                    f"got {type(dataset).__name__}"
                )
            if dataset.labels is None:
                raise ValueError(
                    "CdlRegressionTrain: the DATASET has no labels; provide a "
                    "labelled dataset or wire the X / y TENSOR inputs instead."
                )
            X, y = dataset.features, dataset.labels

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if y.dim() == 0:
            y = y.reshape(1, 1)
        elif y.dim() == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"CdlRegressionTrain: X has {X.shape[0]} rows but y has "
                f"{y.shape[0]}; the sample counts must match."
            )

        n, f = X.shape
        out = y.shape[1] if y.dim() > 1 else 1

        # ------------------------------------------------------------------ #
        # 2. Train / validation split.
        # ------------------------------------------------------------------ #
        do_split = float(test_size) > 0.0 and n >= 2
        gen = torch.Generator().manual_seed(int(seed))
        if do_split:
            perm = torch.randperm(n, generator=gen)
            n_val = int(round(n * float(test_size)))
            n_val = min(max(n_val, 1), n - 1)  # keep both splits non-empty
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
        else:
            train_idx = torch.arange(n)
            val_idx = torch.arange(n)

        # ------------------------------------------------------------------ #
        # 3. Standardise features (z-score) using TRAIN statistics only.
        # ------------------------------------------------------------------ #
        if standardize == "yes":
            mean = X[train_idx].mean(dim=0)
            std = X[train_idx].std(dim=0, unbiased=False)
            # Guard against zero-variance columns (e.g. a constant feature).
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
        else:
            mean = std = None

        # ------------------------------------------------------------------ #
        # 4. Build the model (standardisation folded in as buffers).
        # ------------------------------------------------------------------ #
        hidden_sizes = _parse_hidden(hidden)
        model = _Regressor(
            f, out, hidden_sizes, activation,
            mean.clone() if mean is not None else None,
            std.clone() if std is not None else None,
        )

        train_X, train_y = X[train_idx], y[train_idx]
        val_X, val_y = X[val_idx], y[val_idx]

        loss_fn = _LOSS[loss]
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        bs = int(batch_size)
        bs = n if bs <= 0 else min(bs, n)
        shuffle_gen = torch.Generator().manual_seed(int(seed))

        history: List[torch.Tensor] = []
        stopper = _EarlyStop(int(early_stop_patience), float(early_stop_min_delta))
        best_state: Optional[dict] = None
        best_len = 0
        stopped_at: Optional[int] = None

        pbar, previewer = _make_progress(int(steps))
        total_steps = int(steps)
        last_step = total_steps - 1

        # Everything runs outside ComfyUI's inference_mode: a gradient cannot
        # cross a node boundary, so the whole fit happens inside one scope.
        with torch.inference_mode(False):
            for step in range(total_steps):
                if bs < train_X.shape[0]:
                    order = torch.randperm(train_X.shape[0], generator=shuffle_gen)[:bs]
                    xb, yb = train_X[order], train_y[order]
                else:
                    xb, yb = train_X, train_y
                pred = model(xb)
                step_loss = loss_fn(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                step_loss.backward()
                optimizer.step()
                history.append(step_loss.detach().reshape(()))

                with torch.no_grad():
                    vloss = float(loss_fn(model(val_X), val_y).detach())
                if stopper.update(step, vloss):
                    stopped_at = step
                    # Early stop ends the run on this step: force-render so the
                    # last pushed frame is the complete curve.
                    _report_step(
                        pbar, previewer, float(step_loss.detach()),
                        step + 1, total_steps, force=True,
                    )
                    break
                if stopper.improved:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_len = len(history)
                _report_step(
                    pbar, previewer, float(step_loss.detach()),
                    step + 1, total_steps, force=(step == last_step),
                )

            if stopped_at is not None and best_state is not None:
                model.load_state_dict(best_state)
                del history[best_len:]

            with torch.no_grad():
                pred_all = model(X).detach()
                pred_val = model(val_X).detach()
                diff = pred_val - val_y
                mae = diff.abs().mean()
                rmse = diff.pow(2).mean().sqrt()

        loss_history = torch.stack(history).detach()

        # ------------------------------------------------------------------ #
        # 5. Optionally persist (state_dict: weights + standardization buffers).
        # ------------------------------------------------------------------ #
        if save_path:
            torch.save(model.state_dict(), save_path)

        return (
            model,
            mae.detach(),
            rmse.detach(),
            pred_all.detach(),
            loss_history.detach(),
        )


NODE_CLASS_MAPPINGS["CdlRegressionTrain"] = CdlRegressionTrain
NODE_DISPLAY_NAME_MAPPINGS["CdlRegressionTrain"] = "Regression Train"
