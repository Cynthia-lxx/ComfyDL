"""
d2lcore/Device Utils - GPU/CPU device selection utilities.

d2lcore functions:
  - cpu()          : Get CPU device
  - gpu(i)         : Get GPU device by index
  - num_gpus()     : Get number of available GPUs
  - try_gpu(i)     : Return gpu(i) if exists, else cpu()
  - try_all_gpus() : Return all available GPUs
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlDeviceInfo:
    """Get device information including available GPUs and current device.

    d2lcore: num_gpus()
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("num_gpus", "has_cuda")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Device Utils"

    def execute(self):
        num = torch.cuda.device_count()
        has_cuda = 1 if torch.cuda.is_available() else 0
        return (num, has_cuda)


NODE_CLASS_MAPPINGS["CdlDeviceInfo"] = CdlDeviceInfo
NODE_DISPLAY_NAME_MAPPINGS["CdlDeviceInfo"] = "Device Info"


class CdlTryGpu:
    """Try to get GPU device, fall back to CPU.

    d2lcore: try_gpu(i)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gpu_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("device_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Device Utils"

    def execute(self, gpu_index):
        if torch.cuda.device_count() >= gpu_index + 1:
            dev = f"cuda:{gpu_index}"
        else:
            dev = "cpu"
        return (dev,)


NODE_CLASS_MAPPINGS["CdlTryGpu"] = CdlTryGpu
NODE_DISPLAY_NAME_MAPPINGS["CdlTryGpu"] = "Try GPU"


class CdlTryAllGpus:
    """Return all available GPU device strings.

    d2lcore: try_all_gpus()
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("device_str",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Device Utils"

    def execute(self):
        n = torch.cuda.device_count()
        if n == 0:
            devices = "cpu"
        else:
            devices = ",".join(f"cuda:{i}" for i in range(n))
        return (devices,)


NODE_CLASS_MAPPINGS["CdlTryAllGpus"] = CdlTryAllGpus
NODE_DISPLAY_NAME_MAPPINGS["CdlTryAllGpus"] = "Try All GPUs"
