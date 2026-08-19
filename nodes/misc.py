"""
ComfyDL/Misc - Miscellaneous utility nodes.

Nodes:
  - MessageBox : Display a Windows message box via ctypes.MessageBoxW
  - NoOp       : Accept any input and do nothing (like Python's pass / asm NOP)
  - What       : Meaningless node; toggling OMG opens a browser tab (try it and see)
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Win32 MessageBox button & icon constants
_MB_BUTTONS = {
    "MB_OK":               0x00000000,
    "MB_OKCANCEL":         0x00000001,
    "MB_ABORTRETRYIGNORE": 0x00000002,
    "MB_YESNOCANCEL":      0x00000003,
    "MB_YESNO":            0x00000004,
    "MB_RETRYCANCEL":      0x00000005,
}

_MB_ICONS = {
    "MB_ICONINFORMATION": 0x00000040,
    "MB_ICONWARNING":     0x00000030,
    "MB_ICONERROR":       0x00000010,
    "MB_ICONQUESTION":    0x00000020,
}

_MB_RETURN_TEXT = {
    1: "IDOK",
    2: "IDCANCEL",
    3: "IDABORT",
    4: "IDRETRY",
    5: "IDIGNORE",
    6: "IDYES",
    7: "IDNO",
}


class CdlMessageBox:
    """Display a Windows MessageBox dialog.

    Uses ctypes to call MessageBoxW from user32.dll.
    Configurable title, text, button type, icon type, and blocking behaviour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {"default": "ComfyDL",
                                     "placeholder": "Dialog title"}),
                "text": ("STRING", {"default": "Hello from ComfyDL!",
                                    "multiline": True,
                                    "placeholder": "Dialog message"}),
                "button_type": (list(_MB_BUTTONS.keys()),
                                {"default": "MB_OK"}),
                "icon_type": (list(_MB_ICONS.keys()),
                              {"default": "MB_ICONINFORMATION"}),
                "block": ("BOOLEAN", {"default": True,
                                       "label_on": "block (wait)",
                                       "label_off": "non-block (continue)"}),
            },
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Misc"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Wildcard * input bypasses backend type validation
        return True

    def execute(self, title, text, button_type, icon_type, block, any_input=None):
        if not self._is_windows():
            return ("[MessageBox unavailable: not running on Windows]",)

        u_type = _MB_BUTTONS.get(button_type, 0) | _MB_ICONS.get(icon_type, 0)

        if block:
            # Call directly — workflow pauses until user dismisses the dialog
            ret = ctypes.windll.user32.MessageBoxW(0, text, title, u_type)
            name = _MB_RETURN_TEXT.get(ret, "UNKNOWN")
            return (f"{ret} ({name})",)
        else:
            # Spawn a daemon thread — workflow continues immediately
            def _show():
                ctypes.windll.user32.MessageBoxW(0, text, title, u_type)

            threading.Thread(target=_show, daemon=True).start()
            return ("(non-blocking: dialog shown, continuing)",)

    @staticmethod
    def _is_windows():
        try:
            return hasattr(ctypes, 'windll')
        except Exception:
            return False


class CdlNoOp:
    """A no-operation node: accepts any input and performs no computation.

    Equivalent to Python's ``pass`` statement or assembly's ``NOP``.
    Serves as:
      - A clean connection terminator / null sink for any data type
      - A placeholder node during workflow construction
      - A debug bypass for temporarily disabling downstream paths

    Inputs a single wildcard slot that accepts any data type; no returns.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Misc"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Wildcard * input bypasses backend type validation
        return True

    def execute(self, any_input=None):
        # Intentionally does nothing
        return ()


NODE_CLASS_MAPPINGS["CdlMessageBox"] = CdlMessageBox
NODE_DISPLAY_NAME_MAPPINGS["CdlMessageBox"] = "MessageBox"
NODE_CLASS_MAPPINGS["CdlNoOp"] = CdlNoOp
NODE_DISPLAY_NAME_MAPPINGS["CdlNoOp"] = "NoOp"


_OP_FUNCS = {
    "sum": lambda t: t.sum(),
    "mean": lambda t: t.mean(),
    "abs": lambda t: t.abs(),
    "sqrt": lambda t: t.sqrt(),
    "neg": lambda t: t.neg(),
}


class CdlTimer:
    """Benchmark a tensor operation: run it num_iters times and report timing.

    A short warm-up run happens first to stabilize timing. Returns a human
    readable report (STRING) and the average seconds per iteration (FLOAT).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "operation": (list(_OP_FUNCS.keys()), {"default": "sum"}),
                "num_iters": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "avg_seconds")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Misc"

    def execute(self, tensor, operation, num_iters):
        fn = _OP_FUNCS[operation]
        for _ in range(3):  # warm-up
            fn(tensor)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, num_iters)):
            fn(tensor)
        if tensor.is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        avg = dt / max(1, num_iters)
        report = (f'{operation}: {num_iters} iters, '
                  f'total {dt:.4f} s, avg {avg:.6f} s/it')
        return (report, avg)


NODE_CLASS_MAPPINGS["CdlTimer"] = CdlTimer
NODE_DISPLAY_NAME_MAPPINGS["CdlTimer"] = "Timer (Benchmark)"


class CdlWhat:
    """A meaningless node. It does nothing... unless you flip OMG.

    Accepts an optional wildcard (ANY) input which is ignored, and exposes
    a single "OMG" toggle. When OMG is True, a browser tab opens.
    What it opens? Try it and see.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "OMG": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ComfyDL/Misc"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # Wildcard * input bypasses backend type validation
        return True

    def execute(self, OMG=False, any_input=None):
        if OMG:
            self._open_url("https://www.bilibili.com/video/BV1GJ411x7h7")
        # Intentionally does nothing meaningful
        return ()

    @staticmethod
    def _open_url(url):
        """Open the given URL in the system default browser (non-blocking)."""
        try:
            if sys.platform == "win32":
                os.startfile(url)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception:
            pass


NODE_CLASS_MAPPINGS["CdlWhat"] = CdlWhat
NODE_DISPLAY_NAME_MAPPINGS["CdlWhat"] = "?"
