"""
linreg_train — a fine-grained, from-scratch linear regression trainer.

Unlike the coarse ``TrainingLoop`` (which builds a full MLP and runs an
``OPTIMIZER``/``SCHEDULER`` stack), this node implements the textbook
``d2l`` linear regression recipe directly:

    y_hat = X @ w + b
    loss  = mean( (y_hat - y) ** 2 / 2 )      # squared loss
    w, b  <-  SGD on that loss (mini-batch)

It is deliberately transparent so learners can wire the four outputs straight
into the existing stateless nodes and *verify* the training:

    CdlLinRegTrain --w,b--> CdlLinReg --y_hat--> CdlSquaredLoss --loss--> ...
                      |
                      `--y_hat-------------------------------------------^

The four ports are:

    w            : (n_features, 1) weight vector (TENSOR)
    b            : (1,)            bias scalar    (TENSOR)
    loss_history : (num_steps,)   per-step squared-loss curve (TENSOR, 1-D)
    y_hat        : (n_samples, 1) predictions on the training inputs (TENSOR)

The node accepts the data either as a labelled ``DATASET`` (features + labels)
or as two raw ``TENSOR``s ``X`` / ``y``.  When both are given the ``DATASET``
wins, mirroring the adapter convention used elsewhere in ComfyDL.
"""

from __future__ import annotations

import torch

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlLinRegTrain:
    """Train a linear regression model ``y = X @ w + b`` from scratch.

    Runs mini-batch SGD on the mean squared loss for ``num_steps`` steps.
    Outputs the learned ``w`` / ``b``, the per-step ``loss_history`` and the
    training ``y_hat`` so the result can be checked against ``CdlLinReg`` +
    ``CdlSquaredLoss``.

    Inputs (one of the two conduits must carry labelled data):
      - ``X`` (TENSOR, required) : feature matrix ``[n, f]``
      - ``y`` (TENSOR, required) : label vector ``[n]`` or ``[n, 1]``
      - ``dataset`` (DATASET, optional, forceInput) : labelled dataset; wins
        over ``X`` / ``y`` when present

    Widgets:
      - ``num_steps``  INT    : optimisation steps (default 100)
      - ``batch_size`` INT    : rows sampled per step; 0 = whole dataset
      - ``lr``         FLOAT  : SGD learning rate (default 0.03)
      - ``seed``       INT    : RNG seed for the batch shuffle (default 0)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("TENSOR",),
                "y": ("TENSOR",),
                "num_steps": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
                "batch_size": ("INT", {"default": 32, "min": 0, "max": 65536, "step": 1}),
                "lr": ("FLOAT", {"default": 0.03, "min": 1e-5, "max": 1.0, "step": 1e-3}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
            "optional": {
                "dataset": ("DATASET", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "TENSOR")
    RETURN_NAMES = ("w", "b", "loss_history", "y_hat")
    FUNCTION = "execute"
    CATEGORY = "d2l/Training"

    def execute(self, X, y, num_steps, batch_size, lr, seed, dataset=None):
        # --- resolve the data source -------------------------------------- #
        if dataset is not None:
            if not isinstance(dataset, CdlDataset):
                raise TypeError(
                    "CdlLinRegTrain: 'dataset' must be a DATASET object, "
                    f"got {type(dataset).__name__}"
                )
            if dataset.labels is None:
                raise ValueError(
                    "CdlLinRegTrain: the DATASET has no labels; provide a "
                    "labelled dataset or wire the X / y TENSOR inputs instead."
                )
            X, y = dataset.features, dataset.labels

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"CdlLinRegTrain: X has {X.shape[0]} rows but y has "
                f"{y.shape[0]}; the sample counts must match."
            )

        n, f = X.shape
        # batch_size == 0 means "use the whole dataset every step"
        bs = n if batch_size == 0 else min(batch_size, n)

        w = torch.zeros((f, 1), dtype=torch.float32, requires_grad=True)
        b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)

        generator = torch.Generator().manual_seed(int(seed))
        loss_history: list[torch.Tensor] = []

        # ComfyUI evaluates inside torch.inference_mode(); autograd cannot cross
        # a node boundary, so we open our own gradient scope for the whole run.
        with torch.inference_mode(False):
            for _ in range(int(num_steps)):
                idx = torch.randint(0, n, (bs,), generator=generator)
                Xb, yb = X[idx], y[idx]
                y_hat_b = torch.matmul(Xb, w) + b
                loss = ((y_hat_b - yb) ** 2 / 2).mean()
                loss.backward()
                with torch.no_grad():
                    w -= lr * w.grad
                    b -= lr * b.grad
                    w.grad.zero_()
                    b.grad.zero_()
                loss_history.append(loss.detach())

        with torch.no_grad():
            y_hat = torch.matmul(X, w) + b

        return (
            w.detach(),
            b.detach(),
            torch.stack(loss_history).detach(),
            y_hat.detach(),
        )


NODE_CLASS_MAPPINGS["CdlLinRegTrain"] = CdlLinRegTrain
NODE_DISPLAY_NAME_MAPPINGS["CdlLinRegTrain"] = "Linear Regression Train"
