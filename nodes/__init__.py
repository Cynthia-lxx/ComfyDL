"""
ComfyDL Nodes - ComfyUI custom nodes wrapping d2lcore functions.

Node Categories (matching d2l sections):
  - d2lcore/CV Models      : CNN models (model_cv.py)
  - d2lcore/NLP Models      : RNN/GRU models (model_nlp.py)
  - d2lcore/Attention       : Attention mechanisms (model_attention.py)
  - d2lcore/Seq2Seq         : Sequence-to-sequence models (model_seq2seq.py)
  - d2lcore/ObjectDetection : Object detection ops (object_detection.py)
  - d2lcore/Segmentation    : Semantic segmentation (semantic_segmentation.py)
  - d2lcore/Visualization   : Plot & display functions (visualization.py)
  - d2lcore/TorchOps       : Tensor operations & metrics (torch_ops.py)
  - d2lcore/NLP Utils       : NLP helper functions (nlp_utils.py)
  - d2lcore/Device Utils    : GPU/CPU device utilities (device_utils.py)
  - d2lcore/GAN             : GAN training functions (gan.py)
"""

# Custom ComfyUI types with cdl prefix
# These are used when native ComfyUI types don't match the format

# cdlTensor: Generic torch tensor of any shape
cdlTensor = "cdlTensor"
# cdlModel: d2l model instance (nn.Module subclass)
cdlModel = "cdlModel"
# cdlVocab: Vocabulary object
cdlVocab = "cdlVocab"
# cdlDataloader: PyTorch DataLoader
cdlDataloader = "cdlDataloader"
# cdlBbox: Bounding box tensor [N, 4]
cdlBbox = "cdlBbox"

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
