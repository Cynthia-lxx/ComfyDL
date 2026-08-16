"""
ComfyDL/Model Utils - Self-developed model utility nodes.

Not from d2l — these nodes help inspect, switch, run, clone and persist
PyTorch models directly on the workflow graph. Everything is implemented
with torch.nn / torch primitives (no d2lcore dependency).

All nodes operate on the custom cdlModel type (any nn.Module instance).
"""

import copy

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _num_params(model, trainable_only=False):
    """Count parameters of a module (optionally only trainable ones)."""
    return sum(p.numel() for p in model.parameters()
               if not trainable_only or p.requires_grad)


class CdlModelInfo:
    """Report model parameter counts and a compact structure summary.

    What it does: inspects a model and returns (1) a human-readable
    summary string, (2) the total parameter count and (3) the trainable
    parameter count.
    Inputs:
        model (cdlModel): any nn.Module instance
    Outputs:
        summary (STRING): model type, module count and parameter counts
        total_params (INT): total number of parameters
        trainable_params (INT): number of parameters with requires_grad=True
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("summary", "total_params", "trainable_params")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model):
        total = _num_params(model)
        trainable = _num_params(model, trainable_only=True)
        n_modules = sum(1 for _ in model.modules())
        top = [type(m).__name__ for m in model.children()]
        summary = (
            f"Model: {type(model).__name__}\n"
            f"Submodules: {', '.join(top) or 'none'}\n"
            f"Modules (incl. root): {n_modules}\n"
            f"Total params: {total:,}\n"
            f"Trainable params: {trainable:,}"
        )
        return (summary, total, trainable)


NODE_CLASS_MAPPINGS["CdlModelInfo"] = CdlModelInfo
NODE_DISPLAY_NAME_MAPPINGS["CdlModelInfo"] = "Model Info"


class CdlModelMode:
    """Switch a model between training and evaluation mode.

    What it does: calls model.train() or model.eval() on the input
    model and returns the same instance, so downstream nodes observe the
    new mode.
    Inputs:
        model (cdlModel): any nn.Module instance
        mode (COMBO): "train" (training mode) or "eval" (inference mode)
    Outputs:
        model (cdlModel): the same instance with the mode applied
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
                "mode": (["train", "eval"], {"default": "eval"}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model, mode):
        if mode == "train":
            model.train()
        else:
            model.eval()
        return (model,)


NODE_CLASS_MAPPINGS["CdlModelMode"] = CdlModelMode
NODE_DISPLAY_NAME_MAPPINGS["CdlModelMode"] = "Model Mode"


class CdlModelForward:
    """Run a forward pass of the model on an input tensor.

    What it does: feeds `tensor` through `model` under torch.no_grad()
    and returns the output tensor. The input is moved to the model's
    device if they differ; the model is switched to eval mode first.
    Inputs:
        model (cdlModel): any nn.Module instance
        tensor (cdlTensor): input tensor of the shape the model expects
    Outputs:
        output (cdlTensor): model(tensor) — shape depends on the model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
                "tensor": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model, tensor):
        dev = next(model.parameters()).device
        x = tensor.to(dev)
        model.eval()
        with torch.no_grad():
            out = model(x)
        return (out,)


NODE_CLASS_MAPPINGS["CdlModelForward"] = CdlModelForward
NODE_DISPLAY_NAME_MAPPINGS["CdlModelForward"] = "Model Forward"


class CdlModelLayers:
    """List the module hierarchy (named_modules) as an indented tree.

    What it does: walks model.named_modules() and renders an indented
    tree of every module with its name and class, so you can inspect the
    architecture.
    Inputs:
        model (cdlModel): any nn.Module instance
    Outputs:
        layers_str (STRING): one module per line, indented by depth
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layers_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model):
        lines = []
        for name, mod in model.named_modules():
            depth = 0 if not name else name.count(".") + 1
            lines.append("  " * depth + (name or "(root)") + ": " + type(mod).__name__)
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS["CdlModelLayers"] = CdlModelLayers
NODE_DISPLAY_NAME_MAPPINGS["CdlModelLayers"] = "Model Layers"


class CdlModelParams:
    """List every parameter (name / shape / trainable) as text.

    What it does: walks model.named_parameters() and renders name, shape
    and requires_grad for each parameter, plus the total count.
    Inputs:
        model (cdlModel): any nn.Module instance
    Outputs:
        params_str (STRING): one parameter per line + total count
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("params_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model):
        lines = []
        total = 0
        for name, p in model.named_parameters():
            lines.append(f"{name}: shape={tuple(p.shape)} trainable={p.requires_grad}")
            total += p.numel()
        lines.append(f"Total parameters: {total:,}")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS["CdlModelParams"] = CdlModelParams
NODE_DISPLAY_NAME_MAPPINGS["CdlModelParams"] = "Model Params"


class CdlModelClone:
    """Create a deep copy of a model.

    What it does: returns copy.deepcopy(model) — an independent instance
    with the same architecture and weights but no shared parameters.
    Inputs:
        model (cdlModel): any nn.Module instance
    Outputs:
        clone (cdlModel): a deep copy of the input model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("clone",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model):
        return (copy.deepcopy(model),)


NODE_CLASS_MAPPINGS["CdlModelClone"] = CdlModelClone
NODE_DISPLAY_NAME_MAPPINGS["CdlModelClone"] = "Model Clone"


class CdlModelSave:
    """Save a model's state_dict to a .pt file.

    What it does: writes torch.save(model.state_dict(), path) to disk.
    Only weights are saved (state_dict), so reloading requires a model
    with a matching architecture.
    Inputs:
        model (cdlModel): any nn.Module instance
        path (STRING): target file path, e.g. "C:/models/my_model.pt"
    Outputs:
        message (STRING): confirmation text including the saved path
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
                "path": ("STRING", {"default": "model.pt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model, path):
        torch.save(model.state_dict(), path)
        return (f"Saved state_dict to: {path}",)


NODE_CLASS_MAPPINGS["CdlModelSave"] = CdlModelSave
NODE_DISPLAY_NAME_MAPPINGS["CdlModelSave"] = "Model Save"


class CdlModelLoad:
    """Load a state_dict from a file into a model.

    What it does: reads a .pt state_dict with torch.load and applies it
    to the input model via model.load_state_dict(). The model
    architecture must match the saved state_dict.
    Inputs:
        model (cdlModel): model instance that will receive the weights
        path (STRING): path of the saved state_dict file
    Outputs:
        model (cdlModel): the input model with loaded weights
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("cdlModel",),
                "path": ("STRING", {"default": "model.pt"}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Model Utils"

    def execute(self, model, path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        return (model,)


NODE_CLASS_MAPPINGS["CdlModelLoad"] = CdlModelLoad
NODE_DISPLAY_NAME_MAPPINGS["CdlModelLoad"] = "Model Load"
