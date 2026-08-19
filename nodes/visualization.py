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


# ============================================================
# Histogram
# ============================================================

class CdlHistogram:
    """Draw a histogram of tensor value distribution with optional density curve.

    Uses ``matplotlib.pyplot.hist`` for bin-based distribution visualisation.
    Supports density-normalised overlay and semitransparent face colour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("cdlTensor",),
                "bins": ("INT", {"default": 30, "min": 5, "max": 200, "step": 1}),
                "density": ("BOOLEAN", {"default": False}),
                "color": ("STRING", {"default": "#4673a6", "placeholder": "bar face color"}),
                "alpha": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05}),
                "title": ("STRING", {"default": "", "placeholder": "plot title"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, tensor, bins, density, color, alpha, title, xlabel, ylabel):
        data = tensor.cpu().numpy().flatten()
        if data.size == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=bins, density=density, color=color, alpha=alpha,
                edgecolor='white', linewidth=0.5)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlHistogram"] = CdlHistogram
NODE_DISPLAY_NAME_MAPPINGS["CdlHistogram"] = "Histogram"


# ============================================================
# Bar Chart
# ============================================================

class CdlBarChart:
    """Draw a bar chart (vertical or horizontal) with optional value annotations.

    Uses ``matplotlib.pyplot.bar`` / ``barh``.  Values tensor supplies bar
    heights; labels provide category names.  A compact figure size avoids
    overlap on modest numbers of bars.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("cdlTensor",),
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated category labels"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "horizontal": ("BOOLEAN", {"default": False}),
                "color": ("STRING", {"default": "#4673a6", "placeholder": "bar face color"}),
                "annotate": ("BOOLEAN", {"default": True}),
                "figsize_w": ("FLOAT", {"default": 7.0, "min": 2.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, values, labels, xlabel, ylabel, horizontal, color,
                annotate, figsize_w, figsize_h):
        v = values.cpu().numpy().flatten()
        if v.size == 0:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        labs = [s.strip() for s in labels.split(',') if s.strip()] if labels and labels.strip() else []
        if labs:
            labs = labs[:len(v)]
        else:
            labs = [str(i) for i in range(len(v))]

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        xs = np.arange(len(v))

        if horizontal:
            bars = ax.barh(xs, v, color=color, alpha=0.85, edgecolor='white',
                           height=0.6, linewidth=0.5)
            ax.set_yticks(xs)
            ax.set_yticklabels(labs)
            if annotate:
                for bar_val, bar_patch in zip(v, bars):
                    w = bar_patch.get_width()
                    ax.text(w + max(abs(v)) * 0.01, bar_patch.get_y() + bar_patch.get_height() / 2,
                            f'{bar_val:.2f}'.rstrip('0').rstrip('.'),
                            va='center', ha='left', fontsize=8)
        else:
            bars = ax.bar(xs, v, color=color, alpha=0.85, edgecolor='white',
                          width=0.6, linewidth=0.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, rotation=45 if len(labs) > 6 else 0, ha='right' if len(labs) > 6 else 'center')
            if annotate:
                for bar_val, bar_patch in zip(v, bars):
                    h = bar_patch.get_height()
                    ax.text(bar_patch.get_x() + bar_patch.get_width() / 2, h + max(abs(v)) * 0.01,
                            f'{bar_val:.2f}'.rstrip('0').rstrip('.'),
                            ha='center', va='bottom', fontsize=8)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlBarChart"] = CdlBarChart
NODE_DISPLAY_NAME_MAPPINGS["CdlBarChart"] = "Bar Chart"


# ============================================================
# Scatter Plot
# ============================================================

class CdlScatter:
    """Draw a 2-D scatter plot with optional colour and size encodings.

    Uses ``matplotlib.pyplot.scatter``.  ``color_map`` drives the point colour
    via a colormap; ``size_map`` scales marker sizes between min/max values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "X": ("cdlTensor",),
                "Y": ("cdlTensor",),
                "alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
                "cmap": ("STRING", {"default": "viridis", "placeholder": "matplotlib colormap name"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "color_map": ("cdlTensor",),
                "size_map": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, X, Y, alpha, cmap, xlabel, ylabel, figsize_w, figsize_h,
                color_map=None, size_map=None):
        x = X.cpu().numpy().flatten()
        y = Y.cpu().numpy().flatten()
        n = min(len(x), len(y))
        if n == 0:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        x, y = x[:n], y[:n]

        c = None
        if color_map is not None:
            c = color_map.cpu().numpy().flatten()[:n]

        s = None
        min_marker, max_marker = 20, 200
        if size_map is not None:
            raw_s = size_map.cpu().numpy().flatten()[:n]
            if raw_s.max() - raw_s.min() > 1e-9:
                s = min_marker + (max_marker - min_marker) * (raw_s - raw_s.min()) / (raw_s.max() - raw_s.min())
            else:
                s = np.full_like(raw_s, (min_marker + max_marker) / 2.0)

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        sc = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap, edgecolors='none')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if c is not None:
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlScatter"] = CdlScatter
NODE_DISPLAY_NAME_MAPPINGS["CdlScatter"] = "Scatter"


# ============================================================
# Confusion Matrix
# ============================================================

class CdlConfusionMatrix:
    """Render a confusion matrix heatmap with per-cell value annotations.

    Uses ``matplotlib.pyplot.imshow`` for the colour grid and ``ax.text`` to
    print each number.  Supports normalisation and configurable number format.
    """

    LABEL_LIST = [".0f", ".1f", ".2f", ".3f"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matrix": ("cdlTensor",),
                "class_labels": ("STRING", {"default": "", "placeholder": "comma-separated class names"}),
                "cmap": ("STRING", {"default": "Blues", "placeholder": "matplotlib colormap name"}),
                "normalize": ("BOOLEAN", {"default": False}),
                "fmt": (cls.LABEL_LIST, {"default": ".1f"}),
                "figsize_w": ("FLOAT", {"default": 6.0, "min": 3.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 5.0, "min": 3.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, matrix, class_labels, cmap, normalize, fmt, figsize_w, figsize_h):
        m = matrix.cpu().numpy()
        if m.ndim == 1:
            n = int(np.sqrt(m.size))
            m = m[:n * n].reshape(n, n)
        if m.size == 0:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        if normalize:
            row_sums = m.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            m = m.astype(np.float64) / row_sums

        labs = [s.strip() for s in class_labels.split(',') if s.strip()] if class_labels and class_labels.strip() else []

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        im = ax.imshow(m, cmap=cmap, aspect='auto')

        n_rows, n_cols = m.shape
        threshold = (m.max() + m.min()) / 2.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = m[i, j]
                text_color = 'white' if val > threshold else 'black'
                ax.text(j, i, f"{val:{fmt}}", ha='center', va='center',
                        fontsize=9, color=text_color)

        if labs:
            tick_labs = labs[:max(n_rows, n_cols)]
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(tick_labs[:n_cols], rotation=45 if len(labs) > 6 else 0,
                               ha='right' if len(labs) > 6 else 'center')
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(tick_labs[:n_rows])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlConfusionMatrix"] = CdlConfusionMatrix
NODE_DISPLAY_NAME_MAPPINGS["CdlConfusionMatrix"] = "Confusion Matrix"


# ============================================================
# Pie Chart
# ============================================================

class CdlPieChart:
    """Draw a pie chart (regular or donut) with percentage labels.

    Uses ``matplotlib.pyplot.pie``.  The ``donut`` option hollows the centre
    via ``wedgeprops``.  ``explode`` expects a comma-separated list of 0/1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("cdlTensor",),
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated slice labels"}),
                "donut": ("BOOLEAN", {"default": False}),
                "explode": ("STRING", {"default": "", "placeholder": "comma-separated 0/1 per slice"}),
                "pctdistance": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.5, "step": 0.05}),
                "shadow": ("BOOLEAN", {"default": False}),
                "figsize_w": ("FLOAT", {"default": 6.0, "min": 3.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 6.0, "min": 3.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, values, labels, donut, explode, pctdistance, shadow,
                figsize_w, figsize_h):
        v = values.cpu().numpy().flatten()
        if v.size == 0:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        labs = [s.strip() for s in labels.split(',') if s.strip()] if labels and labels.strip() else None
        if labs:
            labs = labs[:len(v)]

        expl = None
        if explode and explode.strip():
            raw = [int(x.strip()) for x in explode.split(',') if x.strip().isdigit()]
            expl = [float(i) * 0.08 for i in raw[:len(v)]]

        wedges_kw = {}
        if donut:
            wedges_kw = {'width': 0.4, 'edgecolor': 'white'}

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        wedges, texts, autotexts = ax.pie(
            v, labels=labs, autopct='%1.1f%%', explode=expl,
            shadow=shadow, startangle=90, pctdistance=pctdistance,
            wedgeprops=wedges_kw
        )

        if donut:
            # Draw a centre circle to create donut look
            centre_circle = plt.Circle((0, 0), 0.4, fc='white', edgecolor='none')
            ax.add_artist(centre_circle)

        ax.axis('equal')
        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlPieChart"] = CdlPieChart
NODE_DISPLAY_NAME_MAPPINGS["CdlPieChart"] = "Pie Chart"


# ============================================================
# Area Chart
# ============================================================

class CdlAreaChart:
    """Draw a filled area chart (single series or stacked).

    Uses ``matplotlib.pyplot.fill_between`` for a single series, and
    ``plt.stackplot`` for multiple stacked series.  Transparent fills and
    muted default colours give a clean look.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Y": ("cdlTensor",),
                "stacked": ("BOOLEAN", {"default": False}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "color_palette": ("STRING", {"default": "tab10", "placeholder": "matplotlib palette name"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 7.0, "min": 3.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "X_vals": ("cdlTensor",),
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated series labels (for stacked)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/Visualization"

    def execute(self, Y, stacked, alpha, color_palette, xlabel, ylabel,
                figsize_w, figsize_h, X_vals=None, labels=None):
        y = Y.cpu().numpy()
        if y.size == 0:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            result = _fig_to_image_tensor(fig)
            return (result,)

        # Ensure y is 2-D
        if y.ndim == 1:
            y = y.reshape(1, -1)

        x = None
        if X_vals is not None:
            x = X_vals.cpu().numpy().flatten()
        else:
            x = np.arange(y.shape[1])

        n_series = min(y.shape[0], y.shape[1])
        if y.shape[0] <= y.shape[1]:
            y_data = y  # [series, T]
        else:
            y_data = y.T[:n_series]

        lbls = [s.strip() for s in labels.split(',') if s.strip()] if labels and labels.strip() else []
        colors = plt.get_cmap(color_palette)(np.linspace(0, 1, max(n_series, 1)))

        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

        if stacked:
            ax.stackplot(x, *[y_data[i] for i in range(y_data.shape[0])],
                         labels=lbls[:y_data.shape[0]] if lbls else None,
                         alpha=alpha, colors=colors, edgecolor='none')
        else:
            for i in range(y_data.shape[0]):
                lbl = lbls[i] if i < len(lbls) else f"series {i}"
                ax.fill_between(x, y_data[i], alpha=alpha,
                                color=colors[i % len(colors)],
                                label=lbl, linewidth=0)

        if lbls:
            ax.legend(fontsize=8, loc='upper left')

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(x[0], x[-1])

        fig.tight_layout()
        result = _fig_to_image_tensor(fig)
        return (result,)


NODE_CLASS_MAPPINGS["CdlAreaChart"] = CdlAreaChart
NODE_DISPLAY_NAME_MAPPINGS["CdlAreaChart"] = "Area Chart"
