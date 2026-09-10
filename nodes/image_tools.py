"""
ComfyUI core image categories - Self-developed general CV image nodes.

Not from d2l. After the d2l / ComfyUI-core split, the geometry nodes that the
core already provides were dropped (CdlImageResize -> ImageScale /
ResizeImageMaskNode, CdlImageFlip -> ImageFlip, CdlImageBlur -> ImageBlur), and
the remaining nodes were merged into the ComfyUI core image categories so the
frontend shows them next to the native image nodes:

  - image/color     : Image Normalize, Image Grayscale, Image Adjust
  - image/transform : Image Rotate (core ImageRotate only supports 90-degree
                      steps, so the arbitrary-angle rotation is kept)
  - image           : Image Stats

Node classes keep the ``Cdl`` prefix. All nodes consume and produce the native
ComfyUI IMAGE format: float32 tensor [B, H, W, C] with values in [0, 1].
Internally tensors are permuted to [B, C, H, W] for processing and back again.

Implemented with torch + torchvision.transforms.functional (no d2lcore
dependency). Exceptions: the Normalize node deliberately does NOT clip
its output to [0, 1] (z-score range), which is documented below.
"""

import torch
import torchvision.transforms.functional as TF

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _to_chw(images):
    """IMAGE [B,H,W,C] -> tensor [B,C,H,W] (float32)."""
    return images.permute(0, 3, 1, 2).float()


def _to_hwc(t):
    """tensor [B,C,H,W] -> IMAGE [B,H,W,C], clipped to [0, 1]."""
    return t.permute(0, 2, 3, 1).clamp(0.0, 1.0)


class CdlImageNormalize:
    """Normalize or denormalize images with per-channel mean/std.

    What it does: applies (x - mean) / std when `denorm` is False, or the
    inverse x * std + mean when `denorm` is True. `mean`/`std` are
    comma-separated strings; a single value broadcasts to all channels,
    e.g. "0.5" or "0.5,0.5,0.5". NOTE: the output is NOT clipped to
    [0, 1] — the normalized output follows the z-score range.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        mean (STRING): comma-separated per-channel means
        std (STRING): comma-separated per-channel standard deviations
        denorm (BOOLEAN): True = denormalize (x*std+mean), False = normalize
    Outputs:
        image (IMAGE): processed images [B, H, W, C] (range depends on op)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mean": ("STRING", {"default": "0.5,0.5,0.5"}),
                "std": ("STRING", {"default": "0.5,0.5,0.5"}),
                "denorm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"

    @staticmethod
    def _parse(s, num_channels, device):
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
        if len(vals) == 1:
            vals = vals * num_channels
        if len(vals) != num_channels:
            raise ValueError(
                f"mean/std 长度 {len(vals)} 与通道数 {num_channels} 不匹配")
        return (torch.tensor(vals, dtype=torch.float32, device=device)
                .view(1, -1, 1, 1))

    def execute(self, image, mean, std, denorm):
        img = _to_chw(image)
        mean_t = self._parse(mean, img.shape[1], img.device)
        std_t = self._parse(std, img.shape[1], img.device)
        if denorm:
            out = img * std_t + mean_t
        else:
            out = (img - mean_t) / std_t
        return (out.permute(0, 2, 3, 1),)


NODE_CLASS_MAPPINGS["CdlImageNormalize"] = CdlImageNormalize
NODE_DISPLAY_NAME_MAPPINGS["CdlImageNormalize"] = "Image Normalize"


class CdlImageGrayscale:
    """Convert images to grayscale (keeps 3 output channels).

    What it does: applies rgb_to_grayscale with num_output_channels=3 so
    the [B, H, W, C] layout (C=3) is preserved while all channels carry
    the same luminance value.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
    Outputs:
        image (IMAGE): 3-channel grayscale images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"

    def execute(self, image):
        img = _to_chw(image)
        out = TF.rgb_to_grayscale(img, num_output_channels=3)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageGrayscale"] = CdlImageGrayscale
NODE_DISPLAY_NAME_MAPPINGS["CdlImageGrayscale"] = "Image Grayscale"


class CdlImageRotate:
    """Rotate images by an angle (degrees, counter-clockwise).

    What it does: rotates every image by `angle` degrees using bilinear
    interpolation and zero-filled borders. When `expand` is True the
    output canvas is enlarged so the rotated content is not clipped;
    otherwise the output keeps the input size. Kept next to the core
    ImageRotate node because that one only supports 90-degree steps.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        angle (FLOAT): rotation angle in degrees (-360~360)
        expand (BOOLEAN): True = enlarge canvas to fit rotated content
    Outputs:
        image (IMAGE): rotated images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0,
                                    "step": 1.0}),
                "expand": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/transform"

    def execute(self, image, angle, expand):
        img = _to_chw(image)
        out = TF.rotate(img, angle,
                        interpolation=TF.InterpolationMode.BILINEAR,
                        expand=bool(expand), fill=0)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageRotate"] = CdlImageRotate
NODE_DISPLAY_NAME_MAPPINGS["CdlImageRotate"] = "Image Rotate"


class CdlImageAdjust:
    """Adjust brightness / contrast / saturation in one pass.

    What it does: applies torchvision brightness, contrast and saturation
    adjustments with the given factors (1.0 = unchanged, >1 stronger,
    <1 weaker, 0 = none). Factors equal to 1.0 are skipped for speed.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        brightness (FLOAT): brightness factor (0~2)
        contrast (FLOAT): contrast factor (0~2)
        saturation (FLOAT): saturation factor (0~2)
    Outputs:
        image (IMAGE): adjusted images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                         "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                       "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,
                                         "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"

    def execute(self, image, brightness, contrast, saturation):
        img = _to_chw(image)
        if abs(brightness - 1.0) > 1e-6:
            img = TF.adjust_brightness(img, brightness)
        if abs(contrast - 1.0) > 1e-6:
            img = TF.adjust_contrast(img, contrast)
        if abs(saturation - 1.0) > 1e-6:
            img = TF.adjust_saturation(img, saturation)
        return (_to_hwc(img),)


NODE_CLASS_MAPPINGS["CdlImageAdjust"] = CdlImageAdjust
NODE_DISPLAY_NAME_MAPPINGS["CdlImageAdjust"] = "Image Adjust"


class CdlImageStats:
    """Compute per-channel statistics (mean/std/min/max) of an image batch.

    What it does: aggregates all images in the batch and reports per
    channel the mean, std, min and max values, plus the batch layout.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
    Outputs:
        stats (STRING): one line per channel + batch summary line
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "execute"
    CATEGORY = "image"

    def execute(self, image):
        img = _to_chw(image)  # [B, C, H, W]
        lines = []
        for c in range(img.shape[1]):
            ch = img[:, c]
            lines.append(
                f"channel {c}: mean={ch.mean().item():.6f} "
                f"std={ch.std().item():.6f} "
                f"min={ch.min().item():.6f} max={ch.max().item():.6f}")
        h, w = image.shape[1], image.shape[2]
        lines.append(f"batch: {img.shape[0]} images, {h}x{w} px, {img.shape[1]} channels")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS["CdlImageStats"] = CdlImageStats
NODE_DISPLAY_NAME_MAPPINGS["CdlImageStats"] = "Image Stats"
