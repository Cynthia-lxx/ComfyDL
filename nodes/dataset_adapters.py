"""
Dataset adapters — bridge the ``DATASET`` type to the existing ComfyDL data
conduits (raw ``TENSOR`` x/y and the ``cdlDataloader`` iterable).

    TENSOR x,y  <-->  DATASET  -->  cdlDataloader

These keep the new universal type interoperable with the older d2l-style nodes
and the image/dataset pipeline.
"""

import torch

from .data_types import CdlDataset
from .datasets import CdlLoadArray

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlTensorsToDataset:
    """Bundle TENSOR features (and optional labels) into a single ``DATASET``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("TENSOR",),
            },
            "optional": {
                "y": ("TENSOR",),
            },
        }

    RETURN_TYPES = ("DATASET",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, X, y=None):
        return (CdlDataset.from_tensors(X, y),)


class CdlDatasetToTensors:
    """Split a ``DATASET`` back into its ``X`` / ``y`` TENSORs.

    When the dataset has no labels, ``y`` is an empty tensor so downstream
    nodes always receive a consistent tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("X", "y")
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset):
        y = dataset.labels if dataset.labels is not None else torch.empty(0)
        return (dataset.features, y)


class CdlDatasetToLoader:
    """Convert a ``DATASET`` into a ``cdlDataloader`` (reuses ``CdlLoadArray``).

    The produced DataLoader yields the same batch structure as the native
    ``CdlLoadArray`` node, so it is compatible with the existing dataset
    statistics / preview nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET", {"forceInput": True}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 4096, "step": 1}),
                "shuffle": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("cdlDataloader",)
    RETURN_NAMES = ("dataloader",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Datasets"

    def execute(self, dataset, batch_size, shuffle):
        if dataset is None:
            raise ValueError("CdlDatasetToLoader: a DATASET input is required.")
        return CdlLoadArray().execute(
            batch_size, shuffle, features=dataset.features, labels=dataset.labels
        )


NODE_CLASS_MAPPINGS.update({
    "CdlTensorsToDataset": CdlTensorsToDataset,
    "CdlDatasetToTensors": CdlDatasetToTensors,
    "CdlDatasetToLoader": CdlDatasetToLoader,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "CdlTensorsToDataset": "Tensors → Dataset",
    "CdlDatasetToTensors": "Dataset → Tensors",
    "CdlDatasetToLoader": "Dataset → DataLoader",
})
