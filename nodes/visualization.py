"""
d2lcore/Visualization - Plot and display functions.

d2lcore functions:
  - plot(X, Y, xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, figsize, axes)
  - show_images(imgs, num_rows, num_cols, titles, scale)
  - show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)
  - show_bboxes(axes, bboxes, labels, colors)
  - show_trace_2d(f, results)
  - show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist)
  - annotate(text, xy, xytext)

Each visualization function has TWO node variants:
  1. OUTPUT_NODE variant: renders interactive plot in node
  2. IMAGE output variant: converts plot to image tensor for downstream use
"""

import torch
import numpy as np
import io
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


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


# ============================================================
# show_images nodes
# ============================================================

class CdlShowImagesOutput:
    """Display images in a grid. OUTPUT_NODE variant.

    d2lcore: show_images(imgs, num_rows, num_cols, titles, scale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_rows": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "num_cols": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "titles": ("STRING", {"default": "", "multiline": True, "placeholder": "comma-separated titles"}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, images, num_rows, num_cols, scale, titles=None):
        n = min(images.shape[0], num_rows * num_cols)
        figsize = (num_cols * scale, num_rows * scale)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows * num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        titles_list = []
        if titles and titles.strip():
            titles_list = [t.strip() for t in titles.split(',')]

        for i in range(n):
            ax = axes[i]
            img = images[i].cpu().numpy()
            # Clip to valid range
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if i < len(titles_list):
                ax.set_title(titles_list[i])

        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        return ()


NODE_CLASS_MAPPINGS["CdlShowImagesOutput"] = CdlShowImagesOutput
NODE_DISPLAY_NAME_MAPPINGS["CdlShowImagesOutput"] = "Show Images (Output)"


class CdlShowImages:
    """Display images in a grid. IMAGE output variant.

    d2lcore: show_images(imgs, num_rows, num_cols, titles, scale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_rows": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "num_cols": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "titles": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, images, num_rows, num_cols, scale, titles=None):
        n = min(images.shape[0], num_rows * num_cols)
        figsize = (num_cols * scale, num_rows * scale)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows * num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        titles_list = []
        if titles and titles.strip():
            titles_list = [t.strip() for t in titles.split(',')]

        for i in range(n):
            ax = axes[i]
            img = images[i].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if i < len(titles_list):
                ax.set_title(titles_list[i])

        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlShowImages"] = CdlShowImages
NODE_DISPLAY_NAME_MAPPINGS["CdlShowImages"] = "Show Images"


# ============================================================
# show_heatmaps nodes
# ============================================================

class CdlShowHeatmapsOutput:
    """Show heatmaps of matrices. OUTPUT_NODE variant.

    d2lcore: show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)
    Input: matrices as [num_rows, num_cols, H, W] tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matrices": ("cdlTensor",),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "cmap": ("STRING", {"default": "Reds", "placeholder": "matplotlib colormap name"}),
            },
            "optional": {
                "titles": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, matrices, xlabel, ylabel, figsize_w, figsize_h, cmap, titles=None):
        m = matrices.cpu().numpy()
        if m.ndim == 2:
            m = m[np.newaxis, np.newaxis, :, :]
        elif m.ndim == 3:
            m = m[:, np.newaxis, :, :]

        num_rows, num_cols = m.shape[0], m.shape[1]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_w * num_cols, figsize_h * num_rows),
                                 sharex=True, sharey=True, squeeze=False)

        titles_list = []
        if titles and titles.strip():
            titles_list = [t.strip() for t in titles.split(',')]

        for i in range(num_rows):
            for j in range(num_cols):
                ax = axes[i][j]
                pcm = ax.imshow(m[i, j], cmap=cmap)
                if i == num_rows - 1 and xlabel:
                    ax.set_xlabel(xlabel)
                if j == 0 and ylabel:
                    ax.set_ylabel(ylabel)
                if titles_list and j < len(titles_list):
                    ax.set_title(titles_list[j])

        fig.colorbar(pcm, ax=axes, shrink=0.6)
        fig.canvas.draw()
        return ()


NODE_CLASS_MAPPINGS["CdlShowHeatmapsOutput"] = CdlShowHeatmapsOutput
NODE_DISPLAY_NAME_MAPPINGS["CdlShowHeatmapsOutput"] = "Show Heatmaps (Output)"


class CdlShowHeatmaps:
    """Show heatmaps of matrices. IMAGE output variant.

    d2lcore: show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matrices": ("cdlTensor",),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "cmap": ("STRING", {"default": "Reds", "placeholder": "matplotlib colormap name"}),
            },
            "optional": {
                "titles": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, matrices, xlabel, ylabel, figsize_w, figsize_h, cmap, titles=None):
        m = matrices.cpu().numpy()
        if m.ndim == 2:
            m = m[np.newaxis, np.newaxis, :, :]
        elif m.ndim == 3:
            m = m[:, np.newaxis, :, :]

        num_rows, num_cols = m.shape[0], m.shape[1]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_w * num_cols, figsize_h * num_rows),
                                 sharex=True, sharey=True, squeeze=False)

        titles_list = []
        if titles and titles.strip():
            titles_list = [t.strip() for t in titles.split(',')]

        for i in range(num_rows):
            for j in range(num_cols):
                ax = axes[i][j]
                pcm = ax.imshow(m[i, j], cmap=cmap)
                if i == num_rows - 1 and xlabel:
                    ax.set_xlabel(xlabel)
                if j == 0 and ylabel:
                    ax.set_ylabel(ylabel)
                if titles_list and j < len(titles_list):
                    ax.set_title(titles_list[j])

        fig.colorbar(pcm, ax=axes, shrink=0.6)
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlShowHeatmaps"] = CdlShowHeatmaps
NODE_DISPLAY_NAME_MAPPINGS["CdlShowHeatmaps"] = "Show Heatmaps"


# ============================================================
# plot nodes
# ============================================================

class CdlPlot:
    """Plot data as IMAGE output.

    d2lcore: plot(X, Y, xlabel, ylabel, legend, xlim, ylim, xscale, yscale, fmts, figsize, axes)
    Input: X and Y as tensors or comma-separated strings
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xlabel": ("STRING", {"default": "x", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "y", "placeholder": "y-axis label"}),
                "xscale": (["linear", "log"], {"default": "linear"}),
                "yscale": (["linear", "log"], {"default": "linear"}),
                "figsize_w": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5}),
            },
            "optional": {
                "X": ("cdlTensor",),  # [N,] or [L,N] for multiple curves
                "Y": ("cdlTensor",),
                "legend": ("STRING", {"default": "", "placeholder": "comma-separated legend labels"}),
                "xlim_min": ("FLOAT", {"default": -1.0, "min": -1e9, "max": 1e9, "step": 0.1}),
                "xlim_max": ("FLOAT", {"default": -1.0, "min": -1e9, "max": 1e9, "step": 0.1}),
                "ylim_min": ("FLOAT", {"default": -1.0, "min": -1e9, "max": 1e9, "step": 0.1}),
                "ylim_max": ("FLOAT", {"default": -1.0, "min": -1e9, "max": 1e9, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, xlabel, ylabel, xscale, yscale, figsize_w, figsize_h,
                X=None, Y=None, legend=None, xlim_min=-1.0, xlim_max=-1.0,
                ylim_min=-1.0, ylim_max=-1.0):
        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        fmts = ('-', 'm--', 'g-.', 'r:')

        if X is None and Y is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        elif X is not None:
            x_data = X.cpu().numpy()
            if Y is not None:
                y_data = Y.cpu().numpy()
                # Squeeze trailing singleton dim: [N, 1] -> [N]
                if y_data.ndim == 2 and y_data.shape[1] == 1:
                    y_data = y_data.squeeze(-1)
                if x_data.ndim == 1:
                    ax.plot(x_data, y_data, fmts[0])
                else:
                    for i in range(min(x_data.shape[0], len(fmts))):
                        yi = y_data if y_data.ndim == 1 else y_data[i]
                        ax.plot(x_data[i], yi, fmts[i])
            else:
                if x_data.ndim == 1:
                    ax.plot(x_data, fmts[0])
                else:
                    for i in range(min(x_data.shape[0], len(fmts))):
                        ax.plot(x_data[i], fmts[i])

        ax.set_xlabel(xlabel) if xlabel else None
        ax.set_ylabel(ylabel) if ylabel else None
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if xlim_min < xlim_max:
            ax.set_xlim(xlim_min, xlim_max)
        if ylim_min < ylim_max:
            ax.set_ylim(ylim_min, ylim_max)

        if legend and legend.strip():
            legend_list = [l.strip() for l in legend.split(',')]
            ax.legend(legend_list)

        ax.grid(True)
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlPlot"] = CdlPlot
NODE_DISPLAY_NAME_MAPPINGS["CdlPlot"] = "Plot"


# ============================================================
# show_trace_2d (IMAGE output only - too complex for OUTPUT_NODE)
# ============================================================

class CdlShowTrace2D:
    """Show 2D optimization trace. IMAGE output variant.

    d2lcore: show_trace_2d(f, results)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("cdlTensor",),  # [N, 2] tensor of (x1, x2) points
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, results):
        pts = results.cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(pts[:, 0], pts[:, 1], '-o', color='#ff7f0e')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(True)
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlShowTrace2D"] = CdlShowTrace2D
NODE_DISPLAY_NAME_MAPPINGS["CdlShowTrace2D"] = "Show Trace 2D"


# ============================================================
# show_bboxes (IMAGE output variant)
# ============================================================

class CdlShowBboxes:
    """Show bounding boxes on an image. IMAGE output variant.

    d2lcore: show_bboxes(axes, bboxes, labels, colors)
    Input: image [B,H,W,C], bboxes [N,4] (x1,y1,x2,y2 normalized 0-1)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("cdlTensor",),
            },
            "optional": {
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated labels"}),
                "colors": ("STRING", {"default": "b,g,r,m,c", "placeholder": "comma-separated matplotlib colors"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, image, bboxes, labels=None, colors="b,g,r,m,c"):
        color_list = [c.strip() for c in colors.split(',') if c.strip()]
        label_list = []
        if labels and labels.strip():
            label_list = [l.strip() for l in labels.split(',')]

        # Take first image
        img = image[0].cpu().numpy()  # [H, W, C]
        H, W = img.shape[0], img.shape[1]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')

        b = bboxes.cpu().numpy()
        for i in range(min(b.shape[0], 200)):  # Limit for performance
            color = color_list[i % len(color_list)]
            x1, y1, x2, y2 = b[i, :4]
            x1_px, y1_px = x1 * W, y1 * H
            w_px, h_px = (x2 - x1) * W, (y2 - y1) * H

            rect = plt.Rectangle((x1_px, y1_px), w_px, h_px,
                                  fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            if label_list and i < len(label_list):
                ax.text(x1_px, y1_px, label_list[i],
                         va='bottom', ha='left', fontsize=9,
                         color='white' if color != 'w' else 'black',
                         bbox=dict(facecolor=color, alpha=0.7, lw=0))

        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlShowBboxes"] = CdlShowBboxes
NODE_DISPLAY_NAME_MAPPINGS["CdlShowBboxes"] = "Show BBoxes"
