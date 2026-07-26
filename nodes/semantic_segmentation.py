"""
d2lcore/Segmentation - Semantic segmentation utilities.

d2lcore functions/constants:
  - VOC_COLORMAP: VOC color mapping table
  - VOC_CLASSES: VOC 21-class names
  - voc_colormap2label(): Build RGB→class index mapping
  - voc_label_indices(colormap, colormap2label): Map VOC labels to class indices
  - voc_rand_crop(feature, label, height, width): Random crop both
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


class CdlVocClasses:
    """Get the VOC 21-class name list.

    d2lcore: VOC_CLASSES
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("class_names",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Segmentation"

    def execute(self, index):
        if index >= 0 and index < len(VOC_CLASSES):
            return (VOC_CLASSES[index],)
        return (','.join(VOC_CLASSES),)


NODE_CLASS_MAPPINGS["CdlVocClasses"] = CdlVocClasses
NODE_DISPLAY_NAME_MAPPINGS["CdlVocClasses"] = "VOC Classes"


class CdlVocColormap2Label:
    """Build VOC RGB→class index mapping table.

    d2lcore: voc_colormap2label()
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("colormap2label",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Segmentation"

    def execute(self):
        colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return (colormap2label,)


NODE_CLASS_MAPPINGS["CdlVocColormap2Label"] = CdlVocColormap2Label
NODE_DISPLAY_NAME_MAPPINGS["CdlVocColormap2Label"] = "VOC Colormap→Label"


class CdlVocLabelIndices:
    """Map VOC label colors to class indices.

    d2lcore: voc_label_indices(colormap, colormap2label)
    Input: colormap [C,H,W] image tensor, colormap2label [256^3] lookup table
    Output: [H,W] class index tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "colormap": ("IMAGE",),
                "colormap2label": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("label_mask",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Segmentation"

    def execute(self, colormap, colormap2label):
        # colormap arrives as [B,H,W,C] in ComfyUI format
        # Take first batch item, convert to [C,H,W]
        if colormap.dim() == 4:
            cm = colormap[0].permute(2, 0, 1)  # [H,W,C] → [C,H,W]
        else:
            cm = colormap

        cm = cm.permute(1, 2, 0).to(torch.int32)  # [H,W,C]
        idx = ((cm[:, :, 0] * 256 + cm[:, :, 1]) * 256 + cm[:, :, 2])
        result = colormap2label[idx.long()]
        return (result.float(),)


NODE_CLASS_MAPPINGS["CdlVocLabelIndices"] = CdlVocLabelIndices
NODE_DISPLAY_NAME_MAPPINGS["CdlVocLabelIndices"] = "VOC Label Indices"


class CdlVocRandCrop:
    """Randomly crop feature and label images together.

    d2lcore: voc_rand_crop(feature, label, height, width)
    Input: feature [B,H,W,C] (ComfyUI IMAGE), label [B,H,W,C] (ComfyUI IMAGE)
    Output: cropped feature [B,H',W',C], cropped label [B,H',W',C]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("IMAGE",),
                "label": ("IMAGE",),
                "height": ("INT", {"default": 320, "min": 1, "max": 4096, "step": 32}),
                "width": ("INT", {"default": 480, "min": 1, "max": 4096, "step": 32}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("cropped_feature", "cropped_label")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Segmentation"

    def execute(self, feature, label, height, width, seed=0):
        import torchvision.transforms.functional as TF

        torch.manual_seed(seed)

        # ComfyUI IMAGE format: [B,H,W,C] → [B,C,H,W] for torchvision
        f = feature[0:1].permute(0, 3, 1, 2)  # [1,C,H,W]
        l = label[0:1].permute(0, 3, 1, 2)     # [1,C,H,W]

        # Get params from feature
        from torchvision.transforms import RandomCrop
        try:
            i, j, h, w = RandomCrop.get_params(f[0], (height, width))
            f_cropped = TF.crop(f, i, j, h, w)
            l_cropped = TF.crop(l, i, j, h, w)
        except (ValueError, RuntimeError):
            # If crop size is larger than input, use center crop
            f_cropped = TF.center_crop(f, (height, width))
            l_cropped = TF.center_crop(l, (height, width))

        # Convert back to [B,H,W,C]
        f_out = f_cropped.permute(0, 2, 3, 1)
        l_out = l_cropped.permute(0, 2, 3, 1)

        return (f_out, l_out)


NODE_CLASS_MAPPINGS["CdlVocRandCrop"] = CdlVocRandCrop
NODE_DISPLAY_NAME_MAPPINGS["CdlVocRandCrop"] = "VOC Random Crop"
