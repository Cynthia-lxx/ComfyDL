"""
data_gen — formula / model-driven synthetic tabular data generator.

Where :class:`CdlSyntheticData` (``tensor_ops.py``) is the *textbook* generator
(fixed ``y = Xw + b + noise`` recipe used by the d2l chapters), this module is
the *general-purpose* one: it turns a math formula — or an already-trained
regression model — into a labelled :class:`CdlDataset`, with optional and
configurable noise. It is the out-of-the-box data source for the regression
example workflows: no files, no downloads, just widgets.

Two modes
---------
* **formula mode** (default): the ``formula`` widget is a mathematical
  expression over the variables ``x0, x1, ...`` (the number of features is
  derived from the highest variable index referenced). Each row is one sample:
  ``y = formula(x0, x1, ...) + noise``. Example::

      2 + 3*x0 - 1.5*x1 + 0.5*sin(6*x0)

* **model mode**: wire an ``nn_model`` into the optional ``model`` slot and
  the formula is ignored — features are sampled from the chosen distribution
  and labels come from ``model(X) + noise``. This is the "digital twin"
  round-trip: train a regressor (e.g. ``CdlRegressionTrain`` with
  ``save_path``), reload it, and re-simulate unlimited fresh data from it.

The expression is validated against a strict AST whitelist before evaluation,
so a formula can only do arithmetic, call the whitelisted math functions and
reference the sample variables — it cannot touch attributes, subscripts,
lambdas, comprehensions or any Python builtin. All evaluation is vectorised
(the formula is evaluated once on the whole feature matrix, not per row).

Inputs (widgets)
----------------
* ``formula`` (STRING, multiline): the expression; only used in formula mode.
* ``num_examples`` (INT): rows to generate (1 ~ 1,000,000).
* ``x_min`` / ``x_max`` (FLOAT): the sampling window per feature.
* ``sampling`` (combo): ``uniform`` → ``U[x_min, x_max]`` per feature;
  ``normal`` → ``N(mean=(x_min+x_max)/2, std=(x_max-x_min)/6)`` so ~99.7% of
  the samples fall inside the window.
* ``noise`` (combo): ``none`` / ``gaussian`` (``N(0, noise_std)``) /
  ``uniform`` (``U[-noise_std, +noise_std]``).
* ``noise_std`` (FLOAT): the noise scale (ignored when ``noise=none``).
* ``seed`` (INT): RNG seed; generation is fully deterministic per seed
  (a local ``torch.Generator`` is used — never the global RNG state).

Inputs (slots)
--------------
* ``model`` (nn_model, optional, forceInput): when connected, switches to
  model mode (see above).

Outputs
-------
* ``X`` : TENSOR ``(num_examples, n_features)`` float32 feature matrix.
* ``y`` : TENSOR ``(num_examples, 1)`` float32 labels (clean signal + noise).
* ``dataset`` : ``DATASET`` bundling both (feature names ``x0..x{k-1}``,
  target ``y``, provenance metadata) — plugs straight into
  ``CdlRegressionTrain`` / ``CdlDatasetPreview`` / the writers.

Failure policy
--------------
A bad formula, an inverted ``x_min > x_max`` window or a non-finite signal
raise a readable ``ValueError`` — data generation problems are user input
problems and should stop the graph loudly, not degrade silently.
"""

from __future__ import annotations

import ast
import math
import re
from typing import List, Tuple

import torch

from .data_types import CdlDataset

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ---------------------------------------------------------------------- #
# Formula evaluation (AST-whitelisted, vectorised)
# ---------------------------------------------------------------------- #

def _fold_min(*args):
    result = args[0]
    for value in args[1:]:
        result = torch.minimum(result, value)
    return result


def _fold_max(*args):
    result = args[0]
    for value in args[1:]:
        result = torch.maximum(result, value)
    return result


#: Functions / constants a formula may reference. Only these names (plus the
#: sample variables ``x0, x1, ...``) resolve during evaluation.
_FORMULA_FUNCS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "log2": torch.log2,
    "log10": torch.log10,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "sign": torch.sign,
    "sigmoid": torch.sigmoid,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "round": torch.round,
    "min": _fold_min,
    "max": _fold_max,
    "pow": torch.pow,
}

_FORMULA_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_VAR_RE = re.compile(r"^x(\d+)$")

#: AST node types a formula may contain (anything else is rejected).
_ALLOWED_OPS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
)


def _validate_node(node: ast.AST) -> None:
    """Recursively enforce the formula grammar; raise on anything else."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_OPS):
            raise ValueError(
                "CdlFormulaDataGen: operator "
                f"'{type(node.op).__name__}' is not allowed in a formula."
            )
        _validate_node(node.left)
        _validate_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(
                "CdlFormulaDataGen: only unary +/- are allowed in a formula."
            )
        _validate_node(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FORMULA_FUNCS:
            raise ValueError(
                "CdlFormulaDataGen: only the math functions "
                f"{sorted(_FORMULA_FUNCS)} may be called."
            )
        if node.keywords:
            raise ValueError(
                "CdlFormulaDataGen: keyword arguments are not allowed in a "
                "formula call."
            )
        for arg in node.args:
            _validate_node(arg)
    elif isinstance(node, ast.Name):
        if node.id in _FORMULA_FUNCS or node.id in _FORMULA_CONSTS:
            return
        if _VAR_RE.match(node.id):
            return
        raise ValueError(
            f"CdlFormulaDataGen: unknown name '{node.id}' in formula; "
            "variables must be x0, x1, ... and functions must come from the "
            "whitelist."
        )
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(
                "CdlFormulaDataGen: only numeric literals are allowed in a "
                "formula."
            )
    else:
        raise ValueError(
            "CdlFormulaDataGen: formula may only contain arithmetic, "
            "whitelisted function calls and x-variables "
            f"(got '{type(node).__name__}')."
        )


def _compile_formula(formula: str) -> Tuple[int, List[int]]:
    """Validate a formula and return ``(n_features, sorted_var_indices)``.

    The feature count is one more than the highest variable index referenced,
    so a formula that only uses ``x0`` / ``x2`` still generates three columns
    (with ``x1`` simply unused by the signal).
    """
    text = (formula or "").strip()
    if not text:
        raise ValueError(
            "CdlFormulaDataGen: the formula is empty. Write an expression "
            "over the variables x0, x1, ... e.g. '2 + 3*x0 - 1.5*x1'."
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"CdlFormulaDataGen: the formula is not valid Python-style "
            f"math: {exc.msg} (position {exc.offset})."
        ) from exc
    _validate_node(tree)

    indices = sorted(
        {
            int(match.group(1))
            for match in (_VAR_RE.match(node.id) for node in ast.walk(tree) if isinstance(node, ast.Name))
            if match
        }
    )
    if not indices:
        raise ValueError(
            "CdlFormulaDataGen: the formula must reference at least one "
            "variable (x0, x1, ...) — a constant formula has no features to "
            "learn from."
        )
    return indices[-1] + 1, indices


def _eval_formula(formula: str, X: torch.Tensor) -> torch.Tensor:
    """Evaluate the (already validated) formula on the feature matrix."""
    namespace: dict = {"__builtins__": {}}
    namespace.update(_FORMULA_FUNCS)
    namespace.update(_FORMULA_CONSTS)
    for index in range(X.shape[1]):
        namespace[f"x{index}"] = X[:, index]
    result = eval(compile(formula.strip(), "<cdl-formula>", "eval"), namespace)  # noqa: S307
    result = torch.as_tensor(result, dtype=torch.float32)
    return result.reshape(-1, 1)


def _model_in_features(model) -> int:
    """Infer the expected feature count from a model's first 2-D weight."""
    for param in model.parameters():
        if param.dim() == 2 and param.shape[1] > 0:
            return int(param.shape[1])
    raise ValueError(
        "CdlFormulaDataGen: could not infer the input width of the given "
        "model (no 2-D weight found); wire a regression model built by "
        "CdlRegressionTrain / LanguageModelBuild / Network & Layers nodes."
    )


class CdlFormulaDataGen:
    """Generate a labelled tabular dataset from a formula or a trained model.

    What it does: samples a feature matrix from the chosen distribution
    (uniform / normal over the ``[x_min, x_max]`` window), computes the label
    either by evaluating the ``formula`` expression on every row (formula
    mode) or by pushing the rows through the wired ``model`` (model mode), and
    optionally perturbs the labels with gaussian or uniform noise. Returns the
    raw tensors plus a ready-to-use ``DATASET``.

    In:
      Widgets: ``formula`` (expression over x0, x1, ...), ``num_examples``,
      ``x_min`` / ``x_max`` (sampling window), ``sampling`` (uniform/normal),
      ``noise`` (none/gaussian/uniform), ``noise_std``, ``seed``.
      Slot: ``model`` (nn_model, optional) — when present, formula mode is
      bypassed and labels come from ``model(X)``.

    Out:
      ``X`` ``(n, k)`` TENSOR, ``y`` ``(n, 1)`` TENSOR, ``dataset`` DATASET
      (feature names ``x0..x{k-1}``, target ``y``).

    Failure: raises ``ValueError`` with a readable message for an empty or
    non-whitelisted formula, an inverted sampling window, or non-finite
    labels; the feature count in model mode is inferred from the model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formula": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "2 + 3*x0 - 1.5*x1 + 0.5*sin(6*x0)",
                    },
                ),
                "num_examples": ("INT", {"default": 1000, "min": 1, "max": 1000000, "step": 100}),
                "x_min": ("FLOAT", {"default": -3.0, "min": -1e6, "max": 1e6, "step": 0.1}),
                "x_max": ("FLOAT", {"default": 3.0, "min": -1e6, "max": 1e6, "step": 0.1}),
                "sampling": (["uniform", "normal"], {"default": "uniform"}),
                "noise": (["none", "gaussian", "uniform"], {"default": "gaussian"}),
                "noise_std": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
            "optional": {
                "model": ("nn_model", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "DATASET")
    RETURN_NAMES = ("X", "y", "dataset")
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(
        self,
        formula,
        num_examples,
        x_min,
        x_max,
        sampling="uniform",
        noise="gaussian",
        noise_std=0.1,
        seed=0,
        model=None,
    ):
        n = int(num_examples)
        lo, hi = float(x_min), float(x_max)
        if hi < lo:
            raise ValueError(
                f"CdlFormulaDataGen: x_max ({hi}) is below x_min ({lo}); "
                "the sampling window must satisfy x_min <= x_max."
            )

        if model is not None:
            n_features = _model_in_features(model)
            source = "model"
        else:
            n_features, _ = _compile_formula(formula)
            source = "formula"

        # Local generator: deterministic per seed without touching the global
        # RNG state (project convention, see the random-number pitfall note).
        generator = torch.Generator().manual_seed(int(seed))
        if sampling == "normal":
            mean, std = (lo + hi) / 2.0, max((hi - lo) / 6.0, 1e-12)
            X = torch.randn((n, n_features), generator=generator) * std + mean
        else:
            X = lo + (hi - lo) * torch.rand((n, n_features), generator=generator)
        X = X.to(torch.float32)

        if model is not None:
            with torch.no_grad():
                y = model(X).detach()
            y = y.reshape(-1, 1).to(torch.float32)
        else:
            y = _eval_formula(formula, X)

        if noise == "gaussian":
            y = y + torch.randn(y.shape, generator=generator) * float(noise_std)
        elif noise == "uniform":
            y = y + (2.0 * torch.rand(y.shape, generator=generator) - 1.0) * float(noise_std)

        if not torch.isfinite(y).all():
            raise ValueError(
                "CdlFormulaDataGen: the generated labels contain non-finite "
                "values (inf/nan). Check the formula domain (e.g. log of "
                "non-positive values, division by zero) or reduce the noise."
            )

        dataset = CdlDataset.from_tensors(
            X,
            y,
            feature_names=[f"x{i}" for i in range(n_features)],
            target_name="y",
            meta={
                "source": source,
                "formula": "" if model is not None else str(formula).strip(),
                "noise": noise,
                "noise_std": float(noise_std),
                "seed": int(seed),
            },
        )
        return (X, y, dataset)


NODE_CLASS_MAPPINGS["CdlFormulaDataGen"] = CdlFormulaDataGen
NODE_DISPLAY_NAME_MAPPINGS["CdlFormulaDataGen"] = "Formula Data Generator"
