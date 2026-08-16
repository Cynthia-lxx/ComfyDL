"""
ComfyDL/Image Tools - Self-developed general CV image nodes.

Not from d2l — a suite of practical image operations for ComfyUI users.
All nodes consume and produce the native ComfyUI IMAGE format:
float32 tensor [B, H, W, C] with values in [0, 1]. Internally tensors
are permuted to [B, C, H, W] for processing and back again.

Implemented with torch + torchvision.transforms.functional (no d2lcore
dependency). Exceptions: the Normalize node deliberately does NOT clip
its output to [0, 1] (z-score range), which is documented below.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _to_chw(images):
    """IMAGE [B,H,W,C] -> tensor [B,C,H,W] (float32)."""
    return images.permute(0, 3, 1, 2).float()


def _to_hwc(t):
    """tensor [B,C,H,W] -> IMAGE [B,H,W,C], clipped to [0, 1]."""
    return t.permute(0, 2, 3, 1).clamp(0.0, 1.0)


class CdlImageResize:
    """Resize images to a target width/height with a chosen interpolation.

    What it does: resizes each image to (height, width) using the
    selected interpolation mode. Setting a dimension to 0 keeps the
    input size on that axis.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        width (INT): target width (1~8192; 0 = keep input width)
        height (INT): target height (1~8192; 0 = keep input height)
        mode (COMBO): bilinear / nearest / bicubic / area
    Outputs:
        image (IMAGE): resized images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                "mode": (["bilinear", "nearest", "bicubic", "area"],
                         {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image, width, height, mode):
        img = _to_chw(image)
        _, _, h, w = img.shape
        tw = width if width > 0 else w
        th = height if height > 0 else h
        if mode == "area":
            # F.interpolate 支持 'area'（TF.InterpolationMode 无 AREA）
            out = F.interpolate(img, size=(th, tw), mode="area")
        else:
            interp = {
                "bilinear": TF.InterpolationMode.BILINEAR,
                "nearest": TF.InterpolationMode.NEAREST,
                "bicubic": TF.InterpolationMode.BICUBIC,
            }[mode]
            out = TF.resize(img, (th, tw), interpolation=interp)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageResize"] = CdlImageResize
NODE_DISPLAY_NAME_MAPPINGS["CdlImageResize"] = "Image Resize"


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
    CATEGORY = "ComfyDL/Image Tools"

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
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image):
        img = _to_chw(image)
        out = TF.rgb_to_grayscale(img, num_output_channels=3)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageGrayscale"] = CdlImageGrayscale
NODE_DISPLAY_NAME_MAPPINGS["CdlImageGrayscale"] = "Image Grayscale"


class CdlImageFlip:
    """Flip images horizontally or vertically.

    What it does: mirrors every image along the width axis
    (horizontal) or the height axis (vertical).
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        direction (COMBO): horizontal / vertical
    Outputs:
        image (IMAGE): flipped images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image, direction):
        img = _to_chw(image)
        if direction == "vertical":
            out = torch.flip(img, dims=[2])
        else:
            out = torch.flip(img, dims=[3])
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageFlip"] = CdlImageFlip
NODE_DISPLAY_NAME_MAPPINGS["CdlImageFlip"] = "Image Flip"


class CdlImageRotate:
    """Rotate images by an angle (degrees, counter-clockwise).

    What it does: rotates every image by `angle` degrees using bilinear
    interpolation and zero-filled borders. When `expand` is True the
    output canvas is enlarged so the rotated content is not clipped;
    otherwise the output keeps the input size.
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
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image, angle, expand):
        img = _to_chw(image)
        out = TF.rotate(img, angle,
                        interpolation=TF.InterpolationMode.BILINEAR,
                        expand=bool(expand), fill=0)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageRotate"] = CdlImageRotate
NODE_DISPLAY_NAME_MAPPINGS["CdlImageRotate"] = "Image Rotate"


class CdlImageCrop:
    """Center-crop images to height x width.

    What it does: crops each image around its center to the requested
    size. A requested dimension of 0 (or larger than the input) is
    clamped to the input size, so the output never exceeds the input.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        height (INT): crop height (1~8192; 0 = keep input height)
        width (INT): crop width (1~8192; 0 = keep input width)
    Outputs:
        image (IMAGE): cropped images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image, height, width):
        img = _to_chw(image)
        _, _, h, w = img.shape
        ch = min(height, h) if height > 0 else h
        cw = min(width, w) if width > 0 else w
        out = TF.center_crop(img, (ch, cw))
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageCrop"] = CdlImageCrop
NODE_DISPLAY_NAME_MAPPINGS["CdlImageCrop"] = "Image Crop"


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
    CATEGORY = "ComfyDL/Image Tools"

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


class CdlImageBlur:
    """Blur images with a Gaussian or mean filter.

    What it does: applies a Gaussian blur (torchvision gaussian_blur) or
    a mean (box) blur (avg_pool2d). `kernel_size` is auto-rounded up to
    the next odd number and clamped to at least 1.
    Inputs:
        image (IMAGE): input images [B, H, W, C]
        blur_type (COMBO): gaussian / mean
        kernel_size (INT): odd kernel size (1~99)
    Outputs:
        image (IMAGE): blurred images [B, H, W, C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (["gaussian", "mean"], {"default": "gaussian"}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 99, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Image Tools"

    def execute(self, image, blur_type, kernel_size):
        img = _to_chw(image)
        ks = max(1, kernel_size)
        if ks % 2 == 0:
            ks += 1
        if blur_type == "gaussian":
            out = TF.gaussian_blur(img, kernel_size=ks)
        else:
            pad = ks // 2
            out = F.avg_pool2d(img, kernel_size=ks, stride=1, padding=pad,
                               count_include_pad=False)
        return (_to_hwc(out),)


NODE_CLASS_MAPPINGS["CdlImageBlur"] = CdlImageBlur
NODE_DISPLAY_NAME_MAPPINGS["CdlImageBlur"] = "Image Blur"


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
    CATEGORY = "ComfyDL/Image Tools"

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
