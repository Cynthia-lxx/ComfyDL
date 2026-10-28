"""
classification_train — the two-form classification trainers.

Following the project's two-form rule (every operator ships as a
*max-productivity one-box* AND a *fine-grained textbook recipe*), this module
mirrors the regression pair:

* :class:`CdlSoftmaxTrain` — the **textbook** softmax-regression recipe
  (``d2l`` ch. 3): zero-initialised ``W`` / ``b``, hand-written
  softmax cross-entropy, full-batch gradient descent, five transparent ports
  (``W`` / ``b`` / ``logits`` / ``probabilities`` / ``loss_history``) you can
  verify by hand. The numerical-stability trick (shift by the row max before
  exponentiating) is spelled out in the loss expression itself.

* :class:`CdlClassificationTrain` — the **production** one-box trainer:
  optional stratified train/validation split, feature standardisation folded
  into the model, mini-batch Adam with early stopping (smoothed tracked
  metric + rollback to the best weights), breakpoint continuation (crash site
  M3) and live loss-curve preview — the exact mechanism of
  ``CdlRegressionTrain`` with a cross-entropy head and class-index labels.

* :class:`CdlClassificationModelLoad` — skeleton-free persistence: rebuilds
  the classifier architecture straight from a saved ``state_dict``.

Label contract (ADR-1)
----------------------
Labels are **class indices**. Whatever arrives (long tensor, float tensor,
``[n]`` or ``[n, 1]``, from a DATASET or a raw slot) is coerced by
:func:`_resolve_labels` into a 1-D ``long`` tensor via ``round``; the number
of classes is inferred as ``max(y) + 1``. Labels must therefore be the
integers ``0 .. k-1`` — string-to-index mapping is out of scope for now.

Host inference mode
--------------------
Same discipline as the regression trainers: ComfyUI evaluates the whole
prompt inside ``torch.inference_mode()``, so every incoming tensor is an
inference tensor. All materialisation (``.clone()``), model creation and the
training loop run inside one ``torch.inference_mode(False)`` scope.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .data_types import CdlDataset
from .regression_train import (
    _EarlyStop,
    _make_progress,
    _parse_hidden,
    _Regressor,
    _report_step,
)
from comfy.train_checkpoint import (
    current_node_id as _ckpt_node_id,
    discard as _ckpt_discard,
    load as _ckpt_load,
    run_key as _ckpt_run_key,
    save as _ckpt_save,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _resolve_data(X, y, dataset, node_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve the data conduits into materialised ``(X [n,f] float32, y [n] long)``.

    A wired labelled ``DATASET`` wins over the raw ``X`` / ``y`` slots
    (the ComfyDL adapter convention). Everything is cloned inside the caller's
    normal-mode scope so downstream in-place training is legal (see the module
    docstring on host inference mode).
    """
    if dataset is not None:
        if not isinstance(dataset, CdlDataset):
            raise TypeError(
                f"{node_name}: 'dataset' must be a DATASET object, "
                f"got {type(dataset).__name__}"
            )
        if dataset.labels is None:
            raise ValueError(
                f"{node_name}: the DATASET has no labels; provide a labelled "
                "dataset or wire the X / y TENSOR inputs instead."
            )
        X, y = dataset.features, dataset.labels

    X = torch.as_tensor(X, dtype=torch.float32).clone()
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if X.dim() != 2:
        raise ValueError(
            f"{node_name}: X must be a 2-D [n, f] feature matrix, "
            f"got {tuple(X.shape)}."
        )

    y = torch.as_tensor(y).clone()
    if y.dim() == 0:
        y = y.unsqueeze(0)
    y = y.reshape(-1)
    if y.is_floating_point():
        y = torch.round(y)
    y = y.to(torch.long)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{node_name}: X has {X.shape[0]} rows but y has {y.shape[0]}; "
            "the sample counts must match."
        )
    return X, y


def _infer_num_classes(y: torch.Tensor, node_name: str) -> int:
    """Class count = ``max(y) + 1``; labels must be ``0 .. k-1`` (ADR-1)."""
    if y.numel() == 0:
        raise ValueError(f"{node_name}: the label vector is empty.")
    k = int(y.max().item()) + 1
    if int(y.min().item()) < 0:
        raise ValueError(
            f"{node_name}: labels must be class indices 0..k-1; found a "
            f"negative value ({int(y.min().item())})."
        )
    return k


class CdlSoftmaxTrain:
    """Train a softmax-regression classifier from scratch — the textbook recipe.

    What: the classification counterpart of ``CdlLinRegTrain``. Zero-initialised
          ``W [k, f]`` / ``b [k]``, the hand-written softmax cross-entropy
          (shift by the row max for numerical stability, exponentiate, normalise,
          pick the true class probability, take the negative log) and plain
          full-batch gradient descent. Because the problem is convex and the
          initialisation/updates are deterministic, the whole run is
          reproducible without a seed — there is deliberately no seed widget.

    In:
      ``X`` (TENSOR, required): features ``[n, f]``.
      ``y`` (TENSOR, required): class indices ``0..k-1`` (float labels are
      rounded; ``[n]`` or ``[n, 1]`` both accepted).
      ``dataset`` (DATASET, optional, forceInput): labelled dataset; wins over
      ``X`` / ``y`` when present.
      Widgets: ``num_steps`` (full-batch GD steps), ``lr``.

    Out:
      ``W`` ``(k, f)`` / ``b`` ``(k,)`` learned parameters;
      ``logits`` ``(n, k)`` training-set scores ``X @ W.T + b``;
      ``probabilities`` ``(n, k)`` softmax of the logits (rows sum to 1);
      ``loss_history`` ``(num_steps,)`` per-step cross-entropy curve.

    Errors: raises ``ValueError`` for mismatched sample counts, empty labels
    or negative class indices.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("TENSOR",),
                "y": ("TENSOR",),
                "num_steps": ("INT", {"default": 300, "min": 1, "max": 100000, "step": 1}),
                "lr": ("FLOAT", {"default": 0.1, "min": 1e-5, "max": 1.0, "step": 1e-3}),
            },
            "optional": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("W", "b", "logits", "probabilities", "loss_history")
    FUNCTION = "execute"
    CATEGORY = "d2l/Training"

    def execute(self, X, y, num_steps, lr, dataset=None):
        with torch.inference_mode(False):
            X, y = _resolve_data(X, y, dataset, "CdlSoftmaxTrain")
            n, f = X.shape
            k = _infer_num_classes(y, "CdlSoftmaxTrain")

            # Textbook zero initialisation; full-batch GD is deterministic,
            # so no RNG (and therefore no seed widget) is involved at all.
            W = torch.zeros((k, f), dtype=torch.float32, requires_grad=True)
            b = torch.zeros((k,), dtype=torch.float32, requires_grad=True)

            history: List[torch.Tensor] = []
            rows = torch.arange(n)
            for _ in range(int(num_steps)):
                logits = torch.matmul(X, W.T) + b
                # Softmax cross-entropy, spelled out: shift by the row max so
                # the exponential never overflows, normalise, then take the
                # negative log of the true-class probability.
                shifted = logits - logits.max(dim=1, keepdim=True).values
                exp = shifted.exp()
                p = exp / exp.sum(dim=1, keepdim=True)
                loss = -p[rows, y].log().mean()
                loss.backward()
                with torch.no_grad():
                    W -= float(lr) * W.grad
                    b -= float(lr) * b.grad
                    W.grad.zero_()
                    b.grad.zero_()
                history.append(loss.detach())

            with torch.no_grad():
                logits = torch.matmul(X, W.T) + b
                shifted = logits - logits.max(dim=1, keepdim=True).values
                exp = shifted.exp()
                probabilities = exp / exp.sum(dim=1, keepdim=True)

        return (
            W.detach(),
            b.detach(),
            logits.detach(),
            probabilities.detach(),
            torch.stack(history).detach(),
        )


NODE_CLASS_MAPPINGS["CdlSoftmaxTrain"] = CdlSoftmaxTrain
NODE_DISPLAY_NAME_MAPPINGS["CdlSoftmaxTrain"] = "Softmax Regression Train"


class CdlClassificationTrain:
    """Train a classification model end-to-end and return a reusable ``nn_model``.

    One node covers the production classification pipeline — the exact
    mechanism of ``CdlRegressionTrain`` with two classification-specific
    changes (stratified split, cross-entropy head) and one extra widget:

    * an **optional stratified train/validation split** — each class is split
      proportionally (a random permutation per class), so a small validation
      set still contains every class;
    * the model head outputs ``k`` **logits** trained with cross-entropy
      (no softmax inside the model — ``F.cross_entropy`` fuses it, and the
      probabilities stay available through the ``probabilities`` output);
    * early stopping tracks either the smoothed validation **loss** (default)
      or validation **accuracy** (``early_stop_metric`` widget; accuracy is
      monitored as ``1 - acc`` so the "smaller is better" tracker applies).

    The feature standardisation is folded into the returned module (train
    mean/std as buffers), so the ``nn_model`` is self-contained — push raw
    features through ``CdlNNForward`` to reproduce logits, or read the
    ``probabilities`` output directly.

    Inputs / widgets mirror ``CdlRegressionTrain`` (``test_size``,
    ``standardize``, ``hidden``, ``activation``, ``steps``, ``batch_size``,
    ``lr``, ``seed``, ``early_stop_patience``, ``early_stop_min_delta``,
    ``save_path``, optional ``dataset`` slot) plus ``early_stop_metric``.
    ``hidden`` defaults to ``"32"`` — classification needs a non-linear core.

    Outputs: ``nn_model`` / ``accuracy`` (held-out overall accuracy) /
    ``predictions`` (class indices on the full input, ``[n]`` long) /
    ``probabilities`` (softmax of the logits, ``[n, k]``) / ``loss_history``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("TENSOR",),
                "y": ("TENSOR",),
                "test_size": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.95, "step": 0.05}),
                "standardize": (["yes", "no"], {"default": "yes"}),
                "hidden": ("STRING", {"default": "32", "placeholder": "e.g. 16 or 16,8 (empty = linear)"}),
                "activation": (["relu", "tanh", "sigmoid", "none"], {"default": "relu"}),
                "steps": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1}),
                "lr": ("FLOAT", {"default": 3e-2, "min": 1e-5, "max": 1.0, "step": 1e-3}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "early_stop_metric": (["loss", "accuracy"], {"default": "loss"}),
                "early_stop_patience": ("INT", {"default": 30, "min": 0, "max": 100000, "step": 1}),
                "early_stop_min_delta": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "save_path": ("STRING", {"default": "", "placeholder": "e.g. C:/models/classifier.pt (empty = do not save)"}),
            },
            "optional": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("nn_model", "TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("nn_model", "accuracy", "predictions", "probabilities", "loss_history")
    FUNCTION = "execute"
    CATEGORY = "d2l/Training"

    def execute(
        self,
        X,
        y,
        dataset=None,
        test_size=0.2,
        standardize="yes",
        hidden="32",
        activation="relu",
        steps=500,
        batch_size=0,
        lr=3e-2,
        seed=0,
        early_stop_metric="loss",
        early_stop_patience=30,
        early_stop_min_delta=1e-4,
        save_path="",
    ):
        with torch.inference_mode(False):
            X, y = _resolve_data(X, y, dataset, "CdlClassificationTrain")
            n, f = X.shape
            k = _infer_num_classes(y, "CdlClassificationTrain")

            # ---------------------------------------------------------- #
            # Stratified train/validation split (per-class proportional). #
            # ---------------------------------------------------------- #
            do_split = float(test_size) > 0.0 and n >= 2
            gen = torch.Generator().manual_seed(int(seed))
            if do_split:
                n_val = int(round(n * float(test_size)))
                # At least one class must keep at least one training sample;
                # cap the validation take per class at size-1.
                n_val = min(max(n_val, 1), n - k)
                if n_val < 1:
                    do_split = False
            if do_split:
                val_parts, train_parts = [], []
                for c in range(k):
                    idx = torch.nonzero(y == c, as_tuple=False).reshape(-1)
                    idx = idx[torch.randperm(idx.shape[0], generator=gen)]
                    take = min(int(round(idx.shape[0] * float(test_size))), idx.shape[0] - 1)
                    take = max(take, 0)
                    val_parts.append(idx[:take])
                    train_parts.append(idx[take:])
                val_idx = torch.cat(val_parts)
                train_idx = torch.cat(train_parts)
                if val_idx.shape[0] == 0 or train_idx.shape[0] == 0:
                    # Degenerate split (e.g. a class with a single sample) —
                    # fall back to evaluating on the whole set, like the
                    # regression trainer does for test_size=0.
                    train_idx = torch.arange(n)
                    val_idx = torch.arange(n)
            else:
                train_idx = torch.arange(n)
                val_idx = torch.arange(n)

            # ---------------------------------------------------------- #
            # Standardise features (z-score) using TRAIN stats only; the #
            # statistics are folded into the model as buffers.          #
            # ---------------------------------------------------------- #
            if standardize == "yes":
                mean = X[train_idx].mean(dim=0)
                std = X[train_idx].std(dim=0, unbiased=False)
                std = torch.where(std < 1e-8, torch.ones_like(std), std)
            else:
                mean = std = None

            model = _Regressor(
                f, k, _parse_hidden(hidden), activation,
                mean.clone() if mean is not None else None,
                std.clone() if std is not None else None,
            )

            train_X, train_y = X[train_idx], y[train_idx]
            val_X, val_y = X[val_idx], y[val_idx]

            optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
            bs = int(batch_size)
            bs = train_X.shape[0] if bs <= 0 else min(bs, train_X.shape[0])
            shuffle_gen = torch.Generator().manual_seed(int(seed))

            history: List[torch.Tensor] = []
            stopper = _EarlyStop(int(early_stop_patience), float(early_stop_min_delta))
            best_state: Optional[dict] = None
            best_len = 0
            stopped_at: Optional[int] = None

            # Breakpoint continuation (crash site M3) — same contract as the
            # regression trainer: resume on the same prompt_id, discard on a
            # fresh Run, disabled outside the host.
            ckpt_run = _ckpt_run_key({
                "node": "CdlClassificationTrain",
                "node_id": _ckpt_node_id(),
                "widgets": [test_size, standardize, hidden, activation, steps,
                            batch_size, lr, seed, early_stop_metric,
                            early_stop_patience, early_stop_min_delta],
                "x": X, "y": y,
            })
            start_step = 0
            saved = _ckpt_load(ckpt_run)
            if saved is not None:
                try:
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    history = list(saved["history"])
                    stopper.best = saved["stopper_best"]
                    stopper.counter = int(saved["stopper_counter"])
                    stopper._window = list(saved["stopper_window"])
                    best_state = saved["best_state"]
                    best_len = int(saved["best_len"])
                    stopped_at = saved["stopped_at"]
                    shuffle_gen.set_state(saved["shuffle_gen"])
                    torch.set_rng_state(saved["torch_rng"])
                    start_step = int(saved["step"]) + 1
                except Exception:  # noqa: BLE001 - corrupt: train fresh
                    start_step = 0

            pbar, previewer = _make_progress(int(steps))
            total_steps = int(steps)
            last_step = total_steps - 1

            def _val_signal() -> float:
                """The tracked metric on the validation split.

                ``loss``  → held-out cross-entropy;
                ``accuracy`` → ``1 - acc`` so the smaller-is-better
                early-stop tracker applies unchanged.
                """
                with torch.no_grad():
                    vlogits = model(val_X)
                    if early_stop_metric == "accuracy":
                        vpred = vlogits.argmax(dim=1)
                        return 1.0 - float((vpred == val_y).to(torch.float32).mean())
                    return float(
                        torch.nn.functional.cross_entropy(vlogits, val_y).detach()
                    )

            from comfy.model_management import InterruptProcessingException

            step = start_step - 1
            try:
                for step in range(start_step, total_steps):
                    if bs < train_X.shape[0]:
                        order = torch.randperm(train_X.shape[0], generator=shuffle_gen)[:bs]
                        xb, yb = train_X[order], train_y[order]
                    else:
                        xb, yb = train_X, train_y
                    pred = model(xb)
                    step_loss = torch.nn.functional.cross_entropy(pred, yb)
                    optimizer.zero_grad(set_to_none=True)
                    step_loss.backward()
                    optimizer.step()
                    history.append(step_loss.detach().reshape(()))

                    if stopper.update(step, _val_signal()):
                        stopped_at = step
                        _report_step(
                            pbar, previewer, float(step_loss.detach()),
                            step + 1, total_steps, force=True,
                        )
                        break
                    if stopper.improved:
                        best_state = {key: v.detach().clone() for key, v in model.state_dict().items()}
                        best_len = len(history)
                    _report_step(
                        pbar, previewer, float(step_loss.detach()),
                        step + 1, total_steps, force=(step == last_step),
                    )
            except InterruptProcessingException:
                _ckpt_save(ckpt_run, {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "stopper_best": stopper.best,
                    "stopper_counter": stopper.counter,
                    "stopper_window": list(stopper._window),
                    "best_state": best_state,
                    "best_len": best_len,
                    "stopped_at": stopped_at,
                    "shuffle_gen": shuffle_gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                })
                raise

            if stopped_at is not None and best_state is not None:
                model.load_state_dict(best_state)
                del history[best_len:]

            with torch.no_grad():
                logits_all = model(X).detach()
                probabilities = torch.softmax(logits_all, dim=1)
                predictions = logits_all.argmax(dim=1)
                accuracy = (predictions[val_idx] == val_y).to(torch.float32).mean()

        loss_history = torch.stack(history).detach()
        _ckpt_discard(ckpt_run)

        if save_path:
            torch.save(model.state_dict(), save_path)

        return (
            model,
            accuracy.detach(),
            predictions.detach(),
            probabilities.detach(),
            loss_history.detach(),
        )


NODE_CLASS_MAPPINGS["CdlClassificationTrain"] = CdlClassificationTrain
NODE_DISPLAY_NAME_MAPPINGS["CdlClassificationTrain"] = "Classification Train"


class CdlClassificationModelLoad:
    """Rebuild a classification model from a saved state_dict — no skeleton needed.

    What: the persistence counterpart of :class:`CdlClassificationTrain`.
          The generic ``Model Load`` node needs a *live* model to load weights
          into; this node instead reconstructs the ``_Regressor`` architecture
          (the classifier IS a standardised MLP with a ``k``-wide logit head)
          straight from the saved ``state_dict``: input width from the first
          linear weight, hidden widths from the alternating linear weights,
          class count from the last one, standardisation buffers restored when
          present. The result is a self-contained ``nn_model`` producing
          **logits** — apply ``argmax`` (or a softmax) for class predictions.

    In:   ``path`` — the ``.pt`` file written by ``Model Save`` (or the
          trainer's ``save_path`` widget); ``activation`` — the activation
          used between hidden layers (not recoverable from a state_dict; keep
          it consistent with training).
    Out:  ``model`` — the restored ``nn_model`` (eval mode);
          ``info`` — a one-line summary of the restored architecture.
    Errors: raises a readable ``ValueError`` for a missing file or a
          state_dict that does not look like a ComfyDL MLP (no
          ``core.0.weight``).

    Scheduling note: identical to ``CdlRegressionModelLoad`` — the loader has
    **no graph dependency on ``Model Save``** and the scheduler may run it
    first; never put save and load in one graph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "output/classification_model.pt"}),
                "activation": (["relu", "tanh", "sigmoid"], {"default": "relu"}),
            },
        }

    RETURN_TYPES = ("nn_model", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "execute"
    CATEGORY = "d2l/Training"

    def execute(self, path, activation="relu"):
        try:
            sd = torch.load(path, map_location="cpu")
        except FileNotFoundError as exc:
            raise ValueError(
                f"CdlClassificationModelLoad: file not found: {path!r}"
            ) from exc
        if not isinstance(sd, dict) or "core.0.weight" not in sd:
            raise ValueError(
                "CdlClassificationModelLoad: the file does not look like a "
                "saved ComfyDL classifier (no 'core.0.weight'); use "
                "'Model Load' with a matching skeleton for other "
                "architectures."
            )

        with torch.inference_mode(False):
            sd = {
                key: v.clone() if isinstance(v, torch.Tensor) else v
                for key, v in sd.items()
            }
            in_features = int(sd["core.0.weight"].shape[1])
            hidden: List[int] = []
            idx = 0
            while f"core.{idx + 2}.weight" in sd:
                hidden.append(int(sd[f"core.{idx}.weight"].shape[0]))
                idx += 2
            out_features = int(sd[f"core.{idx}.weight"].shape[0])
            mean = sd.get("mean")
            std = sd.get("std")

            model = _Regressor(
                in_features, out_features, tuple(hidden), activation, mean, std
            )
            model.load_state_dict(sd)
            model.eval()

        info = (
            f"restored classifier: in={in_features}, "
            f"hidden={list(hidden) if hidden else 'linear'}, "
            f"classes={out_features}, standardized={mean is not None}, "
            f"activation={activation}"
        )
        return (model, info)


NODE_CLASS_MAPPINGS["CdlClassificationModelLoad"] = CdlClassificationModelLoad
NODE_DISPLAY_NAME_MAPPINGS["CdlClassificationModelLoad"] = "Classification Model Load"
