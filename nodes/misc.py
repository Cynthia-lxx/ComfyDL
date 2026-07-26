"""
ComfyDL/Misc - Miscellaneous utility nodes.

Nodes:
  - MessageBox : Display a Windows message box via ctypes.MessageBoxW
"""

import ctypes
import threading

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


NODE_CLASS_MAPPINGS["CdlMessageBox"] = CdlMessageBox
NODE_DISPLAY_NAME_MAPPINGS["CdlMessageBox"] = "MessageBox"
