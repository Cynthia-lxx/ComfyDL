"""
d2lcore/Datasets - Download, manage, import, and visualize datasets.

d2lcore functions:
  - load_array(data_arrays, batch_size, is_train)    : Wrap tensors into DataLoader
  - load_data_fashion_mnist(batch_size, resize)       : Fashion-MNIST dataset
  - load_data_bananas(batch_size)                     : Banana detection dataset
  - load_data_voc(batch_size, crop_size)              : VOC semantic segmentation
  - download(url, folder, sha1_hash)                  : Download with cache
  - download_extract(name, folder)                    : Download and extract

Node categories:
  - ComfyDL/Datasets: All dataset-related nodes (download, load, preview, stats)
"""

import torch
import os
import io
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ---------------------------------------------------------------------------
# Import d2lcore functions
# ---------------------------------------------------------------------------
try:
    from ..src.d2lcore.torch import download, download_extract, load_array, DATA_HUB
    from ..src.d2lcore.torch import load_data_fashion_mnist, load_data_bananas, load_data_voc
except ImportError:
    from src.d2lcore.torch import download, download_extract, load_array, DATA_HUB
    from src.d2lcore.torch import load_data_fashion_mnist, load_data_bananas, load_data_voc


# ---------------------------------------------------------------------------
# Helper: convert matplotlib figure to IMAGE tensor [1, H, W, C]
# ---------------------------------------------------------------------------
def _fig_to_image_tensor(fig):
    """Convert matplotlib figure to ComfyUI IMAGE tensor [1, H, W, C]."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = PILImage.open(buf).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]
    plt.close(fig)
    return img_tensor


# ================================================================
# 1. CdlLoadArray — tensor(s) → DataLoader
# ================================================================

class CdlLoadArray:
    """Wrap tensors into a PyTorch DataLoader.

    d2lcore: load_array(data_arrays, batch_size, is_train)

    Connect features and labels tensors (cdlTensor) to inputs, and this node
    outputs a cdlDataloader that downstream nodes can use for training/evaluation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 4096, "step": 1}),
                "shuffle": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "features": ("cdlTensor",),
                "labels": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlDataloader",)
    RETURN_NAMES = ("dataloader",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, batch_size, shuffle, features=None, labels=None):
        arrays = []
        if features is not None:
            arrays.append(features)
        if labels is not None:
            arrays.append(labels)
        if not arrays:
            raise ValueError("CdlLoadArray: at least one of features or labels must be connected.")
        loader = load_array(arrays, batch_size, is_train=shuffle)
        return (loader,)


NODE_CLASS_MAPPINGS["CdlLoadArray"] = CdlLoadArray
NODE_DISPLAY_NAME_MAPPINGS["CdlLoadArray"] = "Load Array → DataLoader"


# ================================================================
# 2. CdlDataLoaderInfo — inspect DataLoader properties
# ================================================================

class CdlDataLoaderInfo:
    """Inspect properties of a cdlDataloader.

    Iterates the DataLoader to report:
      - num_batches : total number of batches
      - batch_size  : number of samples per batch
      - dataset_size: total number of samples in the dataset

    Accepts a cdlDataloader slot input and outputs scalar INT values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("cdlDataloader",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("num_batches", "batch_size", "dataset_size")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, dataloader):
        num_batches = len(dataloader)
        # Peek the first batch to infer batch_size
        first_batch = next(iter(dataloader))
        if isinstance(first_batch, (list, tuple)):
            sample = first_batch[0]
        else:
            sample = first_batch
        batch_size = sample.shape[0] if hasattr(sample, 'shape') else -1
        dataset_size = batch_size * num_batches if batch_size > 0 else -1
        return (num_batches, batch_size, dataset_size)


NODE_CLASS_MAPPINGS["CdlDataLoaderInfo"] = CdlDataLoaderInfo
NODE_DISPLAY_NAME_MAPPINGS["CdlDataLoaderInfo"] = "DataLoader Info"


# ================================================================
# 3. CdlDownload — general file download with SHA1 cache check
# ================================================================

class CdlDownload:
    """Download a file from a URL with SHA1-based cache checking.

    d2lcore: download(url, folder, sha1_hash)

    If the file already exists locally and its SHA1 matches, the download
    is skipped. Returns the local file path.

    Inputs:
      - url      : URL of the file to download
      - save_dir : directory to save the file (default '../data')
      - sha1_hash: expected SHA1 hash for cache validation (optional)

    Output:
      - file_path: local path to the downloaded/cached file
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING",
                        {"default": "https://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip",
                         "multiline": False, "placeholder": "https://..."}),
            },
            "optional": {
                "save_dir": ("STRING", {"default": "../data", "multiline": False}),
                "sha1_hash": ("STRING", {"default": "", "placeholder": "optional SHA1 hash"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    @classmethod
    def IS_CHANGED(cls, url, save_dir="../data", sha1_hash=""):
        return url + sha1_hash  # Re-download only when URL or hash changes

    def execute(self, url, save_dir="../data", sha1_hash=""):
        if not url:
            return ("",)
        sha1 = sha1_hash if sha1_hash else None
        fname = download(url, folder=save_dir, sha1_hash=sha1)
        return (fname,)


NODE_CLASS_MAPPINGS["CdlDownload"] = CdlDownload
NODE_DISPLAY_NAME_MAPPINGS["CdlDownload"] = "Download"


# ================================================================
# 4. CdlDownloadExtract — download & extract via DATA_HUB registry
# ================================================================

class CdlDownloadExtract:
    """Download and extract a dataset registered in the d2l DATA_HUB.

    d2lcore: download_extract(name, folder)

    Select from pre-registered datasets. The file is downloaded (with cache)
    and extracted automatically. Returns the extraction directory.

    Inputs:
      - name     : dataset key (chosen from DATA_HUB dropdown)
      - subfolder: optional subfolder to return after extraction

    Output:
      - extract_dir: path to the extracted dataset directory
    """

    @classmethod
    def INPUT_TYPES(cls):
        hub_keys = list(DATA_HUB.keys()) if DATA_HUB else ["banana-detection", "voc2012"]
        return {
            "required": {
                "name": (hub_keys, {"default": hub_keys[0]}),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "", "placeholder": "optional subfolder inside archive"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extract_dir",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    @classmethod
    def IS_CHANGED(cls, name, subfolder=""):
        return name + subfolder  # Re-extract only when selection changes

    def execute(self, name, subfolder=""):
        data_dir = download_extract(name, folder=subfolder if subfolder else None)
        return (data_dir,)


NODE_CLASS_MAPPINGS["CdlDownloadExtract"] = CdlDownloadExtract
NODE_DISPLAY_NAME_MAPPINGS["CdlDownloadExtract"] = "Download + Extract"


# ================================================================
# 5. CdlFashionMNIST — Fashion-MNIST dataset loader
# ================================================================

class CdlFashionMNIST:
    """Load the Fashion-MNIST image classification dataset.

    d2lcore: load_data_fashion_mnist(batch_size, resize)

    10 classes: t-shirt, trouser, pullover, dress, coat,
               sandal, shirt, sneaker, bag, ankle boot.

    Downloads automatically on first use (~30 MB).

    Inputs:
      - batch_size : samples per batch
      - resize     : resize images to (resize x resize), 0 = no resize

    Outputs:
      - train_loader : DataLoader for training set (60,000 images)
      - test_loader  : DataLoader for test set (10,000 images)
      - class_names  : newline-separated list of class names
    """

    FASHION_MNIST_CLASSES = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 64, "min": 1, "max": 2048, "step": 1}),
                "resize": ("INT", {"default": 28, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlDataloader", "cdlDataloader", "STRING")
    RETURN_NAMES = ("train_loader", "test_loader", "class_names")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, batch_size, resize):
        rs = (resize, resize) if resize > 0 else None
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=rs)
        class_names = '\n'.join(self.FASHION_MNIST_CLASSES)
        return (train_iter, test_iter, class_names)


NODE_CLASS_MAPPINGS["CdlFashionMNIST"] = CdlFashionMNIST
NODE_DISPLAY_NAME_MAPPINGS["CdlFashionMNIST"] = "Fashion-MNIST"


# ================================================================
# 6. CdlBananasDetection — banana detection dataset loader
# ================================================================

class CdlBananasDetection:
    """Load the banana detection dataset for object detection.

    d2lcore: load_data_bananas(batch_size)

    Contains images of bananas with bounding box annotations.
    All images have a single class (banana, index 0).
    Downloads automatically on first use.

    Inputs:
      - batch_size: samples per batch

    Outputs:
      - train_loader: DataLoader for training set
      - val_loader  : DataLoader for validation set
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("cdlDataloader", "cdlDataloader")
    RETURN_NAMES = ("train_loader", "val_loader")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, batch_size):
        train_iter, val_iter = load_data_bananas(batch_size)
        return (train_iter, val_iter)


NODE_CLASS_MAPPINGS["CdlBananasDetection"] = CdlBananasDetection
NODE_DISPLAY_NAME_MAPPINGS["CdlBananasDetection"] = "Bananas Detection"


# ================================================================
# 7. CdlVOCSegmentation — VOC2012 semantic segmentation dataset
# ================================================================

class CdlVOCSegmentation:
    """Load the VOC2012 semantic segmentation dataset.

    d2lcore: load_data_voc(batch_size, crop_size)

    21 classes (background + 20 object categories).
    Downloads and extracts automatically on first use (~2 GB).

    Inputs:
      - batch_size : samples per batch
      - crop_height: random crop height
      - crop_width : random crop width

    Outputs:
      - train_loader: DataLoader for training set
      - test_loader : DataLoader for validation set
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "crop_height": ("INT", {"default": 320, "min": 64, "max": 1024, "step": 16}),
                "crop_width": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("cdlDataloader", "cdlDataloader")
    RETURN_NAMES = ("train_loader", "test_loader")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, batch_size, crop_height, crop_width):
        train_iter, test_iter = load_data_voc(batch_size, (crop_height, crop_width))
        return (train_iter, test_iter)


NODE_CLASS_MAPPINGS["CdlVOCSegmentation"] = CdlVOCSegmentation
NODE_DISPLAY_NAME_MAPPINGS["CdlVOCSegmentation"] = "VOC Segmentation"


# ================================================================
# 8. CdlDataLoaderPreview — batch preview as IMAGE output
# ================================================================

class CdlDataLoaderPreview:
    """Preview a batch of data from a DataLoader as an image grid.

    Takes one batch from the cdlDataloader and renders a grid of images.
    Automatically adapts to different data formats:
      - Image classification: renders images with class labels
      - Object detection: renders images with bounding boxes
      - Semantic segmentation: renders image/label pairs

    Inputs:
      - dataloader  : cdlDataloader to sample from
      - num_rows    : rows in the grid
      - num_cols    : columns in the grid
      - max_samples : maximum images to show (capped at num_rows * num_cols)

    Output:
      - image: IMAGE tensor for downstream processing or preview
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("cdlDataloader",),
                "num_rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "num_cols": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
                "max_samples": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, dataloader, num_rows, num_cols, max_samples=32):
        # Gather a batch
        batch = next(iter(dataloader))
        n_show = min(num_rows * num_cols, max_samples)

        # Determine data format
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, Y = batch
        else:
            X = batch
            Y = None

        n = min(X.shape[0], n_show)
        scale = 2.0
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
        if num_rows * num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(n_show):
            ax = axes[i]
            if i < n:
                img = X[i]
                # Handle different tensor shapes
                if img.ndim == 3:
                    # [C, H, W] → [H, W, C]
                    if img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0)
                    # Normalize to [0, 1] for display
                    img_disp = img.float()
                    if img_disp.max() > 1.0:
                        img_disp = img_disp / 255.0
                    img_disp = img_disp.clamp(0, 1)
                    if img_disp.shape[-1] == 1:
                        img_disp = img_disp.squeeze(-1)
                    ax.imshow(img_disp.cpu().numpy() if img_disp.ndim == 3 else img_disp.cpu().numpy(), cmap='gray' if img_disp.ndim == 2 else None)
                elif img.ndim == 2:
                    ax.imshow(img.cpu().numpy(), cmap='gray')
                else:
                    ax.text(0.5, 0.5, f'shape: {img.shape}', ha='center', va='center')

                if Y is not None and i < len(Y):
                    label = Y[i]
                    if hasattr(label, 'item'):
                        ax.set_title(f'label: {label.item()}', fontsize=8)
                    elif hasattr(label, 'shape') and label.ndim <= 1:
                        ax.set_title(f'label: {label.tolist()}', fontsize=6)
            ax.axis('off')

        plt.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlDataLoaderPreview"] = CdlDataLoaderPreview
NODE_DISPLAY_NAME_MAPPINGS["CdlDataLoaderPreview"] = "DataLoader Preview"


# ================================================================
# 9. CdlDataLoaderPreviewOutput — batch preview as OUTPUT_NODE
# ================================================================

class CdlDataLoaderPreviewOutput:
    """Preview a batch of data from a DataLoader (OUTPUT_NODE variant).

    Same as CdlDataLoaderPreview but registered as OUTPUT_NODE so the
    rendered image grid is displayed directly in the UI. Use this for
    quick inspection.

    Inputs:
      - dataloader  : cdlDataloader to sample from
      - num_rows    : rows in the grid
      - num_cols    : columns in the grid
      - max_samples : maximum images to show

    Output:
      - image: rendered preview grid as IMAGE tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("cdlDataloader",),
                "num_rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "num_cols": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
                "max_samples": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, dataloader, num_rows, num_cols, max_samples=32):
        # Delegate to CdlDataLoaderPreview logic
        preview_node = CdlDataLoaderPreview()
        image_result, = preview_node.execute(dataloader, num_rows, num_cols, max_samples)
        return (image_result,)


NODE_CLASS_MAPPINGS["CdlDataLoaderPreviewOutput"] = CdlDataLoaderPreviewOutput
NODE_DISPLAY_NAME_MAPPINGS["CdlDataLoaderPreviewOutput"] = "DataLoader Preview (Output)"


# ================================================================
# 10. CdlDataLoaderStats — dataset statistics
# ================================================================

class CdlDataLoaderStats:
    """Compute and display dataset class distribution statistics.

    Iterates over the cdlDataloader and counts how many samples belong
    to each class. Renders a bar chart showing the distribution.

    Inputs:
      - dataloader   : cdlDataloader to analyze
      - num_classes  : expected number of classes
      - class_names  : comma-separated class names (optional)

    Outputs:
      - stats_text  : formatted text summary of class counts
      - stats_image : bar chart image as IMAGE tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataloader": ("cdlDataloader",),
                "num_classes": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
            },
            "optional": {
                "class_names": ("STRING", {"default": "", "multiline": True, "placeholder": "comma-separated class names"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("stats_text", "stats_image")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Datasets"

    def execute(self, dataloader, num_classes, class_names=""):
        counts = torch.zeros(num_classes, dtype=torch.int64)

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                _, labels = batch
            else:
                continue

            if labels.ndim > 1:
                # Multi-dimensional labels (e.g., object detection): skip counting
                continue

            labels_flat = labels.view(-1).long()
            for lbl in labels_flat:
                if 0 <= lbl < num_classes:
                    counts[lbl] += 1

        # Build text summary
        total = counts.sum().item()
        if class_names:
            names = [n.strip() for n in class_names.split(',')]
        else:
            names = [f'Class {i}' for i in range(num_classes)]

        lines = [f'Total samples: {total}']
        for i in range(min(num_classes, len(names))):
            cnt = counts[i].item()
            pct = f'({cnt / total * 100:.1f}%)' if total > 0 else '(0.0%)'
            lines.append(f'  {names[i]}: {cnt} {pct}')
        stats_text = '\n'.join(lines)

        # Build bar chart
        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.5), 4))
        x = range(num_classes)
        bars = ax.bar(x, counts.numpy(), color='steelblue', edgecolor='white')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Count')
        ax.set_title('Dataset Class Distribution')
        ax.set_xticks(list(x))
        if len(names) <= 20:
            ax.set_xticklabels([n[:8] for n in names], rotation=45, ha='right', fontsize=8)

        # Add count labels on bars
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        str(cnt.item()), ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        stats_image = _fig_to_image_tensor(fig)
        return (stats_text, stats_image)


NODE_CLASS_MAPPINGS["CdlDataLoaderStats"] = CdlDataLoaderStats
NODE_DISPLAY_NAME_MAPPINGS["CdlDataLoaderStats"] = "Dataset Stats"
