"""
d2lcore/CV Models - CNN model building blocks and models.
"""

import torch
from torch import nn

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlCorr2d:
    """Compute 2D cross-correlation operation.

    d2lcore: corr2d(X, K)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tensor": ("cdlTensor",),
                "kernel": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/CV Models"

    def execute(self, input_tensor, kernel):
        h, w = kernel.shape
        Y = torch.zeros((input_tensor.shape[0] - h + 1, input_tensor.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (input_tensor[i: i + h, j: j + w] * kernel).sum()
        return (Y,)


NODE_CLASS_MAPPINGS["CdlCorr2d"] = CdlCorr2d
NODE_DISPLAY_NAME_MAPPINGS["CdlCorr2d"] = "Corr2D"


class CdlLeNet:
    """LeNet-5 convolutional neural network.

    d2lcore: LeNet(lr, num_classes)
    Uses LazyConv2d so input shape is inferred on first forward pass.
    Network structure:
      LazyConv2d(6, 5) → Sigmoid → AvgPool2d(2, 2) →
      LazyConv2d(16, 5) → Sigmoid → AvgPool2d(2, 2) →
      Flatten → LazyLinear(120) → Sigmoid →
      LazyLinear(84) → Sigmoid → LazyLinear(num_classes)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_classes": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
                "lr": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/CV Models"

    def execute(self, num_classes, lr):
        net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes),
        )
        return (net,)


NODE_CLASS_MAPPINGS["CdlLeNet"] = CdlLeNet
NODE_DISPLAY_NAME_MAPPINGS["CdlLeNet"] = "LeNet"


class CdlResNet18:
    """A slightly modified ResNet-18 model.

    d2lcore: resnet18(num_classes, in_channels)
    Uses smaller kernel/stride/padding and removes max-pooling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_classes": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "in_channels": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/CV Models"

    def execute(self, num_classes, in_channels):
        import sys

        def resnet_block(in_ch, out_ch, num_residuals, first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(CdlResidual._make_block(out_ch, use_1x1conv=True, strides=2))
                else:
                    blk.append(CdlResidual._make_block(out_ch))
            return nn.Sequential(*blk)

        net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        net.add_module("resnet_block2", resnet_block(64, 128, 2))
        net.add_module("resnet_block3", resnet_block(128, 256, 2))
        net.add_module("resnet_block4", resnet_block(256, 512, 2))
        net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
        return (net,)


NODE_CLASS_MAPPINGS["CdlResNet18"] = CdlResNet18
NODE_DISPLAY_NAME_MAPPINGS["CdlResNet18"] = "ResNet-18"


class CdlResidual:
    """The Residual block of ResNet.

    d2lcore: Residual(num_channels, use_1x1conv, strides)
    """

    @staticmethod
    def _make_block(num_channels, use_1x1conv=False, strides=1):
        return _ResidualModule(num_channels, use_1x1conv, strides)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_channels": ("INT", {"default": 64, "min": 1, "max": 2048, "step": 1}),
                "use_1x1conv": ("BOOLEAN", {"default": False}),
                "strides": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("block",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/CV Models"

    def execute(self, num_channels, use_1x1conv, strides):
        block = CdlResidual._make_block(num_channels, use_1x1conv, strides)
        return (block,)


class _ResidualModule(nn.Module):
    """Internal Residual wrapper for ComfyUI node output."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.relu(Y)


NODE_CLASS_MAPPINGS["CdlResidual"] = CdlResidual
NODE_DISPLAY_NAME_MAPPINGS["CdlResidual"] = "Residual Block"


class CdlResNeXtBlock:
    """The ResNeXt block.

    d2lcore: ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv, strides)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_channels": ("INT", {"default": 64, "min": 1, "max": 2048, "step": 1}),
                "groups": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "bot_mul": ("FLOAT", {"default": 0.5, "min": 0.125, "max": 2.0, "step": 0.125}),
                "use_1x1conv": ("BOOLEAN", {"default": False}),
                "strides": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlModel",)
    RETURN_NAMES = ("block",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/CV Models"

    def execute(self, num_channels, groups, bot_mul, use_1x1conv, strides):
        block = _ResNeXtModule(num_channels, groups, bot_mul, use_1x1conv, strides)
        return (block,)


class _ResNeXtModule(nn.Module):
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False, strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3, stride=strides, padding=1,
                                    groups=bot_channels // groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = torch.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return torch.relu(Y + X)


NODE_CLASS_MAPPINGS["CdlResNeXtBlock"] = CdlResNeXtBlock
NODE_DISPLAY_NAME_MAPPINGS["CdlResNeXtBlock"] = "ResNeXt Block"
