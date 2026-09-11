"""
ComfyDL Nodes - ComfyUI custom nodes wrapping d2lcore functions.

Node Categories (matching d2l sections):
  - d2l/Tensor Basic         : tensor utilities & primitives (tensor_basic.py)
  - d2l/TorchOps             : tensor operations & metrics (tensor_ops.py)
  - d2l/CV Models            : CNN models (model_cv.py)
  - d2l/NLP Models           : RNN/GRU, attention & seq2seq models
                               (model_nlp.py, model_attention.py, model_seq2seq.py)
  - d2l/ObjectDetection      : object detection ops (object_detection.py)
  - d2l/Segmentation         : semantic segmentation (semantic_segmentation.py)
  - d2l/Visualization        : plot & display functions (visualization.py)
  - d2l/Datasets             : dataset helpers (datasets.py)
  - d2l/NLP Utils            : NLP helper functions (nlp_utils.py)
  - d2l/Model Utils          : nn.Module inspection & persistence (model_utils.py)
  - d2l/Device Utils         : GPU/CPU device utilities (device_utils.py)
  - d2l/GAN                  : GAN training functions (gan.py)

Legacy categories:
  - d2l/_Legacy/* : teaching nodes whose functionality is now covered by the core
    Comfy nodes. They stay registered (so old workflows keep loading) but are
    flagged with the ``DEPRECATED = True`` class attribute and their display name
    carries a "(DEPRECATED)" suffix, which is why the node library groups them
    under ``d2l/_Legacy/<original category>`` instead of the active categories.
    The per-node replacement is documented in the module docstrings of
    tensor_basic.py, model_attention.py and model_utils.py.

This package also declares the non-d2l categories ``image/color``,
``image/transform``, ``image`` (image_tools.py) and ``utilities``,
``utilities/conversion`` (misc.py, conversion.py).
"""

# Slot types used by the ComfyDL nodes.
#
# TENSOR / BBOX are the canonical Comfy core types (declared in
# comfy/comfy_types/node_typing.py and registered in comfy_api/latest/_io.py).
# The ComfyDL nodes use those names directly so they interoperate with the
# core Activation nodes on the same slots.

# TENSOR: Generic torch tensor of any shape (was cdlTensor).
TENSOR = "TENSOR"
# BBOX: Bounding box tensor [N, 4] (was cdlBbox; same as the core IO.BBOX).
BBOX = "BBOX"

# Legacy aliases -- prefer the canonical names above in new code.
cdlTensor = TENSOR
cdlBbox = BBOX

# No Comfy core counterpart exists for the types below, so they stay
# ComfyDL-only and keep their original names.
# cdlModel: d2l model instance (nn.Module subclass)
cdlModel = "cdlModel"
# cdlVocab: Vocabulary object
cdlVocab = "cdlVocab"
# cdlDataloader: PyTorch DataLoader
cdlDataloader = "cdlDataloader"

# Node registration - imported by __init__.py at module load
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Collect from all submodules
import importlib
import os

_module_files = [
    f[:-3] for f in os.listdir(os.path.dirname(__file__))
    if f.endswith('.py') and f != '__init__.py'
]

for mod_name in _module_files:
    try:
        mod = importlib.import_module(f'.{mod_name}', package=__package__)
        if hasattr(mod, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        if hasattr(mod, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"[ComfyDL] Warning: Failed to load {mod_name}: {e}")

_dl_loaded_msg = f"[ComfyDL] Loaded {len(NODE_CLASS_MAPPINGS)} deep learning nodes."
try:
    print("\033[92m" + _dl_loaded_msg + "\033[0m")
except UnicodeEncodeError:
    # Fall back to plain ASCII when the console encoding cannot handle ANSI text
    print(_dl_loaded_msg)
