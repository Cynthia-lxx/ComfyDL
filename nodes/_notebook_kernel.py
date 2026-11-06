"""Notebook kernel: per-workflow exec namespaces for CdlNotebookCell.

Mechanism ported from the ComfyUI-Notebook reference (liusida), trimmed to
the four mechanisms the SP-7 craft-6 plan calls for:

1. **Per-workflow kernel** — one ``types.ModuleType`` per workflow id; cells
   executed into ``kernel.__dict__`` share variables across cells and runs,
   exactly like Jupyter keeps a namespace per notebook.
2. **Interruptible execution** — cells run in a daemon thread; the host
   thread polls ``comfy.model_management.processing_interrupted()`` and sets
   a flag that injected interrupt checks (wrappers around ``range``/``iter``/
   ``next``/``map``/...) raise as ``InterruptProcessingException``.
3. **Streaming stdout** — handled by the node (TeeOutput + ``send_sync``),
   this module only owns the kernel store.
4. **Variable inspection** — :func:`describe_variables` summarises the
   kernel namespace (name / type / truncated repr) for the
   ``/comfydl/notebook/variables`` route.

Gating: the ``comfydl.notebook`` switch (default **off**) must be enabled in
the switch registry before any cell executes — user code runs without a
sandbox, so this feature is Tier-2 opt-in by design. The switch registration
mirrors ``app/spine/bus.py`` (module-level, ``hot_toggle=True``) but lives in
the submodule, so it is wrapped in try/except for bare-repo imports.
"""

from __future__ import annotations

import inspect
import sys
import threading
import types
from typing import Any, Dict, List, Optional

SWITCH_ID = "comfydl.notebook"

_LOCK = threading.RLock()
_KERNELS: Dict[str, types.ModuleType] = {}

#: Modules injected into every fresh kernel so simple teaching cells work
#: out of the box (``torch``/``np``); users can import anything else.
_PRELOAD: Dict[str, Any] = {}

_PRELOAD_ATTEMPTED = False

_VAR_REPR_LIMIT = 120

#: Names the executor injects into the kernel for interruptible loops; they
#: are plumbing, not user state, so inspection skips them.
_INJECTED_NAMES = frozenset({
    "range", "enumerate", "iter", "next", "zip", "map", "filter",
    "check_interrupt",
})


def _ensure_preload() -> None:
    """Fill the preload dict once (best effort; never raises)."""
    global _PRELOAD_ATTEMPTED
    if _PRELOAD_ATTEMPTED:
        return
    _PRELOAD_ATTEMPTED = True
    import importlib

    for name, alias in (("torch", "torch"), ("numpy", "np")):
        try:
            _PRELOAD[alias] = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - optional at kernel-creation time
            pass


# --------------------------------------------------------------------------- #
# Switch (comfydl.notebook) — default OFF
# --------------------------------------------------------------------------- #

def _registry():
    try:
        from comfy.profiling import switch_registry

        return switch_registry
    except Exception:  # noqa: BLE001 - bare-submodule / sandbox import
        return None


try:
    _reg = _registry()
    if _reg is not None:
        _reg.register_switch(
            SWITCH_ID,
            default=False,      # Tier-2: opt-in, no sandbox
            hot_toggle=True,
            self_meter=False,
        )
except Exception:  # noqa: BLE001 - never break node import over telemetry
    pass


def enabled() -> bool:
    """Whether notebook execution is permitted (never raises)."""
    reg = _registry()
    if reg is None:
        return False
    try:
        return bool(reg.is_enabled(SWITCH_ID))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Kernel store
# --------------------------------------------------------------------------- #

def kernel_for(workflow_id: str) -> types.ModuleType:
    """Return (creating if needed) the kernel module for one workflow."""
    _ensure_preload()
    with _LOCK:
        kernel = _KERNELS.get(workflow_id)
        if kernel is None:
            kernel = types.ModuleType(f"cdl_notebook_kernel_{workflow_id}")
            kernel.__dict__.update(_PRELOAD)
            _KERNELS[workflow_id] = kernel
        return kernel


def clear(workflow_id: Optional[str] = None) -> int:
    """Drop one workflow's kernel, or all of them; returns kernels removed."""
    with _LOCK:
        if workflow_id is None:
            count = len(_KERNELS)
            _KERNELS.clear()
            return count
        if _KERNELS.pop(workflow_id, None) is not None:
            return 1
        return 0


def active_workflows() -> List[str]:
    """Ids of workflows that currently hold a kernel."""
    with _LOCK:
        return list(_KERNELS.keys())


def describe_variables(workflow_id: str, limit: int = 64) -> List[Dict[str, str]]:
    """Summarise a kernel's namespace for the inspection route.

    Skips module/dunder entries (the preload and plumbing); each row carries
    the variable name, its type name and a truncated ``repr`` so the payload
    stays small and never ships raw tensor contents.
    """
    with _LOCK:
        kernel = _KERNELS.get(workflow_id)
        if kernel is None:
            return []
        items = list(kernel.__dict__.items())

    rows: List[Dict[str, str]] = []
    for name, value in items:
        if name.startswith("__") or isinstance(value, types.ModuleType):
            continue
        if name in _INJECTED_NAMES and callable(value):
            continue
        try:
            repr_text = repr(value)
        except Exception:  # noqa: BLE001 - repr of arbitrary user objects
            repr_text = "<repr failed>"
        if len(repr_text) > _VAR_REPR_LIMIT:
            repr_text = repr_text[: _VAR_REPR_LIMIT - 3] + "..."
        rows.append({
            "name": name,
            "type": type(value).__name__,
            "repr": repr_text,
        })
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def interrupt_check(original: Any, flag: threading.Event) -> Any:
    """Wrap one builtin so cell loops observe the interrupt flag.

    Cell code runs in a worker thread; ``flag`` is set by the host thread
    when the user interrupts the run. The wrapper raises as soon as the flag
    is observed so loops built on these builtins stop promptly instead of
    spinning forever (a daemon thread cannot be killed).
    """

    def wrapper(*args, **kwargs):
        if flag.is_set():
            raise KeyboardInterrupt("notebook cell interrupted")
        return original(*args, **kwargs)

    return wrapper
