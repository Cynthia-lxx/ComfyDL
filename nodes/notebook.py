"""CdlNotebookCell — a Jupyter-style cell node (Tier-2, default OFF).

What:  executes a multiline Python cell inside a per-workflow kernel
       namespace. Variables persist across cells and runs of the same
       workflow (like a notebook session); ``print`` output streams to the
       node UI while the cell runs; a ``Result`` variable (or the value of
       the last expression statement) becomes the ``result`` output; any
       matplotlib figure can be captured via the injected ``Notebook.plot()``
       helper. Four mechanisms ported from the ComfyUI-Notebook reference:
       per-workflow kernel, interruptible execution (host Stop button breaks
       injected builtins within one poll tick), streaming stdout
       (``TeeOutput`` + ``send_sync("executed")``), and variable inspection
       (``/comfydl/notebook/variables`` route, see ``_notebook_kernel``).
In:    ``code`` — the cell source (multiline); ``input`` — optional wildcard
       socket exposed to the cell as the ``input`` variable.
Out:   ``result`` — the cell's ``Result`` variable (wildcard ``*``);
       ``plot``  — captured matplotlib figures stacked as one IMAGE tensor
       (a 1x1 placeholder when nothing was plotted);
       ``stdout`` — everything the cell printed.
Safety: user code runs WITHOUT a sandbox — this node is gated behind the
       ``comfydl.notebook`` switch (default OFF; enable it in the Profiling
       panel switch list or via ``/comfydl/switches``). Execution errors
       propagate as node failures (the crash site captures them like any
       other node error).
"""

from __future__ import annotations

import io as io_module
import sys
import threading

import torch

from ._notebook_kernel import (
    SWITCH_ID,
    describe_variables as _describe_variables,  # re-exported for routes
    enabled as _notebook_enabled,
    interrupt_check,
    kernel_for,
)

_NODE_LOG_KEY = "cdl_notebook_log"

_PLOT_DPI = 100


class TeeOutput:
    """Write to both the original stdout and a capture buffer."""

    def __init__(self, original, capture):
        self._original = original
        self._capture = capture

    def write(self, text):
        self._original.write(text)
        self._capture.write(text)

    def flush(self):
        self._original.flush()
        self._capture.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class _PlotCollector:
    """Collects matplotlib figures rendered by the cell as IMAGE tensors."""

    def __init__(self):
        self.frames = []

    def plot(self):
        """Capture the current matplotlib figure (no-op when it is empty)."""
        try:
            import matplotlib.pyplot as plt

            fig = plt.gcf()
            if not fig.get_axes():
                return
            has_data = any(
                ax.lines or ax.patches or ax.collections or ax.images
                for ax in fig.get_axes()
            )
            if not has_data:
                return
            from io import BytesIO

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=_PLOT_DPI)
            buf.seek(0)
            try:
                from PIL import Image as PILImage

                pil_image = PILImage.open(buf)
                frame = pil_image.convert("RGB")
            except Exception:  # noqa: BLE001 - PIL unavailable
                buf.seek(0)
                return
            import numpy as np

            array = np.asarray(frame).astype("float32") / 255.0
            self.frames.append(torch.from_numpy(array)[None,])
            plt.close(fig)
            plt.close("all")
        except Exception:  # noqa: BLE001 - plotting must never kill the cell
            pass

    def tensor(self):
        if not self.frames:
            return torch.ones((1, 1, 1, 3), dtype=torch.float32)
        return torch.cat(self.frames, dim=0)


class CdlNotebookCell:
    """One Jupyter-style Python cell bound to the workflow's kernel.

    The node registers unconditionally (old workflows keep loading) but
    ``execute`` refuses to run unless the ``comfydl.notebook`` switch is
    enabled — the feature is Tier-2 opt-in because cell code executes
    without a sandbox. See the module docstring for the mechanism tour.
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "code": ("STRING", {
                    "multiline": True,
                    "default": 'x = [i ** 2 for i in range(10)]\n'
                               'print("squares:", x)\n'
                               'Result = sum(x)',
                    "tooltip": "Python source for this cell. Variables persist "
                               "across cells of the same workflow; assign "
                               "'Result' to emit it on the result socket; "
                               "call Notebook.plot() to capture figures.",
                }),
            },
            "optional": {
                "input": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("*", "IMAGE", "STRING")
    RETURN_NAMES = ("result", "plot", "stdout")
    FUNCTION = "execute"
    CATEGORY = "d2l/Notebook"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # The wildcard * input bypasses backend type validation (see CdlNoOp).
        return True

    # ------------------------------------------------------------------ #

    @staticmethod
    def _workflow_id(extra_pnginfo, unique_id) -> str:
        try:
            wid = extra_pnginfo["workflow"].get("id")
            if wid:
                return str(wid)
        except Exception:  # noqa: BLE001 - workflow metadata is optional
            pass
        return f"node-{unique_id}"

    def execute(self, code, input=None, unique_id=None, extra_pnginfo=None):
        if not _notebook_enabled():
            raise RuntimeError(
                "CdlNotebookCell is disabled: user code runs without a "
                f"sandbox, so it is gated behind the '{SWITCH_ID}' switch "
                "(default off). Enable it in the Profiling panel switch "
                "list, then re-queue."
            )
        code = str(code or "")
        if not code.strip():
            raise ValueError("CdlNotebookCell: the cell source is empty")

        from comfy.model_management import (
            InterruptProcessingException,
            processing_interrupted,
        )

        try:
            server_mod = __import__("server")
            prompt_server = server_mod.PromptServer.instance
        except Exception:  # noqa: BLE001 - headless/unit-test context
            prompt_server = None

        workflow_id = self._workflow_id(extra_pnginfo, unique_id)
        kernel = kernel_for(workflow_id)
        notebook_globals = kernel.__dict__

        plots = _PlotCollector()
        notebook_globals.update({
            "input": input,
            "Result": None,
            "Notebook": plots,
        })

        stdout_capture = io_module.StringIO()
        old_stdout = sys.stdout
        push_state = {"last_len": 0}

        def push_stdout(force=False):
            current = stdout_capture.getvalue()
            if not force and len(current) <= push_state["last_len"]:
                return
            if prompt_server is not None:
                try:
                    prompt_server.send_sync("executed", {
                        "node": unique_id,
                        "display_node": unique_id,
                        "output": {_NODE_LOG_KEY: (current,)},
                    })
                except Exception:  # noqa: BLE001 - streaming is best-effort
                    pass
            push_state["last_len"] = len(current)

        compiled = compile(code, f"<cdl_notebook:{workflow_id}>", "exec")
        execution_result = {"exception": None}
        execution_done = threading.Event()
        interrupt_flag = threading.Event()

        def execute_in_thread():
            try:
                # Inject interrupt-checking builtins so plain for/while loops
                # over range/zip/map/... observe the Stop button.
                builtins_ns = (__builtins__ if isinstance(__builtins__, dict)
                               else vars(__builtins__))
                for name in ("range", "enumerate", "iter", "next", "zip",
                             "map", "filter"):
                    notebook_globals[name] = interrupt_check(
                        builtins_ns[name], interrupt_flag)

                def check_interrupt():
                    if interrupt_flag.is_set():
                        raise KeyboardInterrupt("notebook cell interrupted")

                notebook_globals["check_interrupt"] = check_interrupt
                with torch.inference_mode(False):
                    exec(compiled, notebook_globals)  # noqa: S102 - by design
            except BaseException as exc:  # noqa: BLE001 - forwarded below
                execution_result["exception"] = exc
            finally:
                execution_done.set()

        thread = threading.Thread(target=execute_in_thread, daemon=True)
        try:
            sys.stdout = TeeOutput(old_stdout, stdout_capture)
            thread.start()
            while not execution_done.wait(timeout=0.1):
                push_stdout()
                if processing_interrupted():
                    interrupt_flag.set()
                    stdout_capture.write("\n[interrupted by user]")
                    push_stdout(force=True)
                    raise InterruptProcessingException()
        finally:
            execution_done.wait(timeout=5.0)
            push_stdout(force=True)
            sys.stdout = old_stdout

        if execution_result["exception"] is not None:
            exc = execution_result["exception"]
            stdout_capture.write(f"\n[error] {type(exc).__name__}: {exc}")
            push_stdout(force=True)
            raise exc

        plots.plot()  # auto-capture a pending figure, Jupyter-style
        result = notebook_globals.get("Result")
        stdout_text = stdout_capture.getvalue() or "[no output]"

        # Persist the final log for save/reload (front-end onConfigure reads
        # node.properties.cdl_notebook_log).
        try:
            from .data_ops import resolve_workflow_node

            node, _ = resolve_workflow_node(extra_pnginfo, unique_id)
            if node is not None:
                properties = node.get("properties")
                if isinstance(properties, dict):
                    properties[_NODE_LOG_KEY] = stdout_text
        except Exception:  # noqa: BLE001 - write-back is best-effort
            pass

        return {
            "ui": {
                _NODE_LOG_KEY: [stdout_text],
                "cdl_notebook_result": [repr(result)[:400]],
            },
            "result": (result, plots.tensor(), stdout_text),
        }


NODE_CLASS_MAPPINGS = {"CdlNotebookCell": CdlNotebookCell}
NODE_DISPLAY_NAME_MAPPINGS = {"CdlNotebookCell": "Notebook → Cell"}
