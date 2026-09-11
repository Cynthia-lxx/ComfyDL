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
    CATEGORY = "d2l/Visualization"

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
#
# Shared by ``CdlShowHeatmaps`` / ``CdlShowHeatmapsOutput`` (this module) and
# ``CdlHeatmapsTo3D`` (``nodes/mesh3d.py``):
#
#   _heatmap_view()  sample -> choose axes -> reduce -> numpy array
#   _render_heatmap() 1-D barcode strip / 2-D plane / 3-D translucent cube
#
# Behaviour changes versus the pre-sampling implementation (documented here on
# purpose, they are meant to be visible in the docs):
#   * a tensor is strided down to ``max_samples`` elements before anything is
#     rendered, so a 1920x1080 image no longer costs ~2M pixels per figure;
#   * the old "grid of subplots" (num_rows x num_cols) is gone: the node now
#     renders at most three axes as a *single* view, every other axis is
#     reduced by ``axis_reduce`` (mean by default);
#   * ``titles`` therefore holds a single title (the first comma-separated
#     entry is used).

#: Sampling budget shared by all heatmap nodes. A tensor with more elements
#: than this is strided down before rendering; ``0`` disables sampling.
DEFAULT_MAX_SAMPLES = 262144

#: Widget choices. 'first' / 'mid' pick one index, 'mean' / 'max' reduce the
#: whole axis -- all four cost O(sampled elements), never O(input elements).
_REDUCE_MODES = ["mean", "max", "first", "mid"]
_AUTO_SELECT_MODES = ["first_n", "last_n", "most_informative_n", "least_informative_n"]
_ON_ERROR_MODES = ["error", "fallback_first_n"]

#: Per-axis cap of the 3-D branch: matplotlib builds one polygon per grid
#: cell, so 32 samples per axis is already far more than a readable cube needs.
_CUBE_AXIS_CAP = 32

#: Alpha of the 3-D mid-planes -- the cube must stay translucent.
_CUBE_ALPHA = 0.6


class HeatmapSpecError(ValueError):
    """The requested axes cannot be honoured; ``on_error`` decides what happens."""


def _shape_numel(shape):
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _kept_per_axis(shape, step):
    kept = 1
    for size in shape:
        kept *= -(-int(size) // step)
    return kept


def _sample_step(shape, max_samples):
    """One common stride for every axis, so sampling stays a pure view.

    ``total=10, max_samples=4`` -> ``step=3`` -> indices 0, 3, 6, 9. Returns
    ``1`` (no sampling) when the tensor already fits or sampling is disabled.
    """
    if not shape:
        return 1
    total = _shape_numel(shape)
    if max_samples <= 0 or total <= max_samples:
        return 1
    step = -(-total // max_samples)
    for _ in range(8):
        if _kept_per_axis(shape, step) <= max_samples:
            return step
        step += 1
    # Pathological shape (many one-element axes rounding up): one sample per
    # axis always fits and still keeps a huge tensor out of memory.
    return max([step] + [int(size) for size in shape])


def _sample_index(ndim, step):
    """Slice tuple implementing the uniform stride (pure view, no copy)."""
    if step <= 1:
        return tuple(slice(None) for _ in range(ndim))
    return tuple(slice(None, None, step) for _ in range(ndim))


def _parse_axes(text, ndim):
    """``'0,2'`` / ``'-1'`` -> ``[0, 2]`` / ``[ndim - 1]``; ``None`` = automatic."""
    raw = (text or "").strip()
    if not raw or raw.lower() == "auto":
        return None
    axes = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            axis = int(token)
        except ValueError:
            raise HeatmapSpecError(
                f"dims='{text}': '{token}' is not an axis index (use 'auto' or e.g. '0,1')"
            ) from None
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise HeatmapSpecError(f"dims='{text}': axis {token} does not exist in a {ndim}-D tensor")
        axes.append(axis)
    if not axes:
        raise HeatmapSpecError(f"dims='{text}': no axis listed")
    if len(axes) > 3:
        raise HeatmapSpecError(f"dims='{text}': at most 3 axes can be rendered, got {len(axes)}")
    if len(set(axes)) != len(axes):
        raise HeatmapSpecError(f"dims='{text}': duplicate axis")
    return axes


def _render_dim(auto_n, ndim):
    """How many axes to render; ``auto_n <= 0`` follows the input rank."""
    if auto_n is None or int(auto_n) <= 0:
        # 1-D -> barcode, 2-D -> plane, 3-D -> cube; bigger batches stay flat.
        return min(ndim, 3) if ndim <= 3 else 2
    return max(1, min(int(auto_n), 3))


def _auto_axes(shape, mode, n):
    """Pick ``n`` axes; 'information' is proxied by the axis length (elements)."""
    ndim = len(shape)
    if mode == "first_n":
        return list(range(n))
    if mode == "last_n":
        return list(range(ndim - n, ndim))
    by_length = sorted(range(ndim), key=lambda i: (int(shape[i]), i))
    if mode == "least_informative_n":
        return sorted(by_length[:n])
    return sorted(by_length[ndim - n:])  # most_informative_n (and the fallback)


def _resolve_axes(shape, dims, auto_select, auto_n, on_error):
    """``(axes, note)`` -- ``note`` is set when ``on_error`` degraded the request."""
    ndim = len(shape)
    n = _render_dim(auto_n, ndim)
    try:
        axes = _parse_axes(dims, ndim)
        if axes is None:
            axes = _auto_axes(shape, auto_select, n)
        return axes, ""
    except HeatmapSpecError as exc:
        if on_error != "fallback_first_n":
            raise
        axes = list(range(min(n, ndim)))
        return axes, f"{exc}; fell back to axis {axes}"


def _reduce_to_axes(tensor, keep_axes, how):
    """Collapse every axis outside ``keep_axes`` (mean / max / first / mid)."""
    keep = set(keep_axes)
    dropped = [axis for axis in range(tensor.dim()) if axis not in keep]
    if not dropped:
        return tensor
    if how == "mean":
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.float32)
        return tensor.mean(dim=dropped)
    if how == "max":
        return torch.amax(tensor, dim=dropped)
    index = [slice(None)] * tensor.dim()
    for axis in dropped:
        index[axis] = 0 if how == "first" else int(tensor.shape[axis]) // 2
    return tensor[tuple(index)]


def _heatmap_view(matrices, max_samples, dims, axis_reduce, auto_select, auto_n, on_error):
    """Sample, select and reduce ``matrices`` into a 1-/2-/3-D numpy array.

    Returns ``(data, step, note)``. Sampling is a strided **view**: nothing is
    materialised before numpy copies the already bounded result, so peak memory
    is O(max_samples) instead of O(input elements).
    """
    if not torch.is_tensor(matrices):
        matrices = torch.as_tensor(matrices)
    if matrices.dim() == 0:
        matrices = matrices.reshape(1)
    shape = tuple(int(size) for size in matrices.shape)

    step = _sample_step(shape, max_samples)
    sampled = matrices.detach()[_sample_index(matrices.dim(), step)]

    keep_axes, note = _resolve_axes(shape, dims, auto_select, auto_n, on_error)
    data = _reduce_to_axes(sampled, sorted(keep_axes), axis_reduce)
    return data.cpu().numpy(), step, note


def _resolve_cmap(name):
    """``(colormap, note)`` -- an unknown name falls back to Reds instead of crashing."""
    text = (name or "").strip() or "Reds"
    try:
        return matplotlib.colormaps[text], ""
    except (KeyError, ValueError):
        return matplotlib.colormaps["Reds"], f"cmap='{text}' is not a matplotlib colormap; using 'Reds'"


def _first_title(titles):
    """``titles`` is comma separated; the single view uses the first entry."""
    if not titles or not titles.strip():
        return ""
    return titles.split(",")[0].strip()


def _tick_labels(size, stride):
    """Tick positions / labels in *original* index space (after sampling)."""
    step = max(1, -(-int(size) // 6))
    positions = np.arange(0, int(size), step)
    return positions, [str(int(position * stride)) for position in positions]


def _cap_axis_samples(data, cap):
    """Nearest-neighbour cap so a mesh never explodes into too many polygons."""
    stride = max(1, max(-(-int(size) // cap) for size in data.shape))
    if stride == 1:
        return data, 1
    return data[(slice(None, None, stride),) * data.ndim], stride


def _draw_plane(ax, data, cmap_obj, stride):
    """1-D -> supermarket-barcode strip, 2-D -> plain heatmap; returns the mappable."""
    matrix = np.asarray(data)
    if matrix.ndim <= 1:
        row = matrix.reshape(1, -1)
        pcm = ax.imshow(row, cmap=cmap_obj, aspect="auto",
                        extent=(-0.5, row.shape[1] - 0.5, 0.0, 1.0))
        ax.set_ylim(-1.5, 2.5)  # float the strip instead of filling the axes
        ax.set_yticks([])
    else:
        pcm = ax.imshow(matrix, cmap=cmap_obj)
    if stride > 1:
        positions, labels = _tick_labels(matrix.shape[-1], stride)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        if matrix.ndim > 1:
            positions, labels = _tick_labels(matrix.shape[0], stride)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
    return pcm


def _plane_coords(mid, shape, axis):
    """Coordinates of the plane cutting ``axis`` at ``mid`` (the other two span it)."""
    others = [index for index in range(3) if index != axis]
    grid_a, grid_b = np.meshgrid(np.arange(shape[0], dtype=float),
                                 np.arange(shape[1], dtype=float), indexing="ij")
    coords = [None, None, None]
    coords[axis] = np.full(shape, float(mid))
    coords[others[0]] = grid_a
    coords[others[1]] = grid_b
    return coords


def _frame_segments(shape):
    """The 12 box edges as one NaN-separated polyline."""
    sizes = np.array(shape, dtype=float) - 1.0
    corners = [np.array([(bits >> 2) & 1, (bits >> 1) & 1, bits & 1], dtype=float) * sizes
               for bits in range(8)]
    xs, ys, zs = [], [], []
    for first in range(8):
        for second in range(first + 1, 8):
            if bin(first ^ second).count("1") != 1:
                continue
            xs += [corners[first][0], corners[second][0], np.nan]
            ys += [corners[first][1], corners[second][1], np.nan]
            zs += [corners[first][2], corners[second][2], np.nan]
    return xs, ys, zs


def _draw_cube(ax, data, cmap_obj):
    """Translucent cube: three orthogonal mid-planes inside a light frame.

    Only three planes are drawn (not the whole volume), which keeps the polygon
    count at ``3 * _CUBE_AXIS_CAP**2`` and still shows the interior.
    """
    cube, _ = _cap_axis_samples(np.asarray(data, dtype=np.float64), _CUBE_AXIS_CAP)
    finite = np.isfinite(cube)
    low = float(cube[finite].min()) if finite.any() else 0.0
    high = float(cube[finite].max()) if finite.any() else 1.0
    if high <= low:
        high = low + 1.0
    norm = matplotlib.colors.Normalize(vmin=low, vmax=high)

    for axis in range(3):
        mid = int(cube.shape[axis]) // 2
        plane = np.nan_to_num(np.take(cube, mid, axis=axis), nan=low)
        coords = _plane_coords(mid, plane.shape, axis)
        colors = cmap_obj(norm(plane))
        colors[..., 3] = _CUBE_ALPHA
        ax.plot_surface(coords[0], coords[1], coords[2], facecolors=colors,
                        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    xs, ys, zs = _frame_segments(cube.shape)
    ax.plot(xs, ys, zs, color="#9E9E9E", linewidth=0.6, alpha=0.9)
    ax.set_box_aspect(tuple(float(size) for size in cube.shape))
    ax.view_init(elev=22, azim=-55)


def _render_heatmap(data, *, xlabel, ylabel, figsize_w, figsize_h, cmap,
                    title="", note="", stride=1):
    """Build the 1-D / 2-D / 3-D figure for ``data`` (a numpy array) and return it."""
    cmap_obj, cmap_note = _resolve_cmap(cmap)
    notes = [part for part in (note, cmap_note) if part]

    array = np.asarray(data)
    if array.size == 0:
        array = np.full((1, 1), np.nan)
        notes.append("empty tensor")

    if array.ndim >= 3:
        fig = plt.figure(figsize=(figsize_w, figsize_h))
        ax = fig.add_subplot(111, projection="3d")
        _draw_cube(ax, array, cmap_obj)
    else:
        # Kept byte-for-byte compatible with the pre-sampling node so an
        # untouched 2-D input with max_samples=0 renders exactly as before.
        fig, axes = plt.subplots(1, 1, figsize=(figsize_w, figsize_h),
                                 sharex=True, sharey=True, squeeze=False)
        ax = axes[0][0]
        pcm = _draw_plane(ax, array, cmap_obj, stride)
        fig.colorbar(pcm, ax=axes, shrink=0.6)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if notes:
        ax.text(0.01, 0.99, "  ".join(notes), transform=ax.transAxes, va="top", ha="left",
                fontsize=6, color="#B00020")
    return fig


def _normalise_controls(max_samples, dims, axis_reduce, auto_select, auto_n, on_error):
    """Fall back to the widget defaults when an API prompt omits optional inputs."""
    return (
        DEFAULT_MAX_SAMPLES if max_samples is None else int(max_samples),
        "auto" if dims is None else dims,
        "mean" if axis_reduce in (None, "") else axis_reduce,
        "last_n" if auto_select in (None, "") else auto_select,
        0 if auto_n is None else int(auto_n),
        "fallback_first_n" if on_error in (None, "") else on_error,
    )


def _heatmap_input_types():
    """INPUT_TYPES shared by both Show Heatmaps variants.

    Existing widgets keep their position and defaults; every new control is
    appended *after* ``titles`` so workflows saved with the old five/six widget
    values still restore correctly (legacy nodes map widgets by index).
    """
    return {
        "required": {
            "matrices": ("TENSOR",),
            "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
            "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
            "figsize_w": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
            "figsize_h": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 20.0, "step": 0.5}),
            "cmap": ("STRING", {"default": "Reds", "placeholder": "matplotlib colormap name"}),
        },
        "optional": {
            "titles": ("STRING", {"default": "", "multiline": True}),
            "max_samples": ("INT", {"default": DEFAULT_MAX_SAMPLES, "min": 0, "max": 1073741824, "step": 1024}),
            "dims": ("STRING", {"default": "auto", "placeholder": "auto | 0,1 | -2,-1 (1-3 axes, negatives allowed)"}),
            "axis_reduce": (list(_REDUCE_MODES), {"default": "mean"}),
            "auto_select": (list(_AUTO_SELECT_MODES), {"default": "last_n"}),
            "auto_n": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            "on_error": (list(_ON_ERROR_MODES), {"default": "fallback_first_n", "advanced": True}),
        },
    }


class CdlShowHeatmapsOutput:
    """Show a heatmap of a tensor. OUTPUT_NODE variant (renders in the node).

    d2lcore: show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)

    Inputs:
        matrices (TENSOR): any rank. It is sampled down to ``max_samples``
            elements, the axes named by ``dims`` (or picked by ``auto_select`` /
            ``auto_n``) are kept and every other axis is collapsed by
            ``axis_reduce``. 1 kept axis renders a supermarket-barcode strip,
            2 a normal heatmap, 3 a translucent cube.
        xlabel / ylabel (STRING): axis labels, empty hides them.
        figsize_w / figsize_h (FLOAT): figure size in inches.
        cmap (STRING): matplotlib colormap name.
        titles (STRING): comma separated; the first entry is the title.
        max_samples (INT): element budget, ``0`` disables sampling.
        dims (STRING): ``auto``, or 1-3 axis indices such as ``0,1`` / ``-2,-1``.
        axis_reduce (COMBO): mean / max / first / mid reduction of the other axes.
        auto_select (COMBO): first_n / last_n / most_informative_n /
            least_informative_n -- 'information' is proxied by axis length.
        auto_n (INT): how many axes to render, ``0`` = follow the input rank.
        on_error (COMBO, advanced): ``error`` raises, ``fallback_first_n`` keeps
            working by rendering the first axes and notes the reason in the plot.

    Outputs:
        none (OUTPUT_NODE).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _heatmap_input_types()

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "d2l/Visualization"

    def execute(self, matrices, xlabel, ylabel, figsize_w, figsize_h, cmap, titles=None,
                max_samples=DEFAULT_MAX_SAMPLES, dims="auto", axis_reduce="mean",
                auto_select="last_n", auto_n=0, on_error="fallback_first_n"):
        max_samples, dims, axis_reduce, auto_select, auto_n, on_error = _normalise_controls(
            max_samples, dims, axis_reduce, auto_select, auto_n, on_error)
        data, stride, note = _heatmap_view(matrices, max_samples, dims, axis_reduce,
                                           auto_select, auto_n, on_error)
        fig = _render_heatmap(data, xlabel=xlabel, ylabel=ylabel, figsize_w=figsize_w,
                              figsize_h=figsize_h, cmap=cmap, title=_first_title(titles),
                              note=note, stride=stride)
        fig.canvas.draw()
        return ()


NODE_CLASS_MAPPINGS["CdlShowHeatmapsOutput"] = CdlShowHeatmapsOutput
NODE_DISPLAY_NAME_MAPPINGS["CdlShowHeatmapsOutput"] = "Show Heatmaps (Output)"


class CdlShowHeatmaps:
    """Show a heatmap of a tensor. IMAGE output variant.

    d2lcore: show_heatmaps(matrices, xlabel, ylabel, titles, figsize, cmap)

    Inputs:
        matrices (TENSOR): any rank. It is sampled down to ``max_samples``
            elements, the axes named by ``dims`` (or picked by ``auto_select`` /
            ``auto_n``) are kept and every other axis is collapsed by
            ``axis_reduce``. 1 kept axis renders a supermarket-barcode strip,
            2 a normal heatmap, 3 a translucent cube.
        xlabel / ylabel (STRING): axis labels, empty hides them.
        figsize_w / figsize_h (FLOAT): figure size in inches.
        cmap (STRING): matplotlib colormap name.
        titles (STRING): comma separated; the first entry is the title.
        max_samples (INT): element budget, ``0`` disables sampling.
        dims (STRING): ``auto``, or 1-3 axis indices such as ``0,1`` / ``-2,-1``.
        axis_reduce (COMBO): mean / max / first / mid reduction of the other axes.
        auto_select (COMBO): first_n / last_n / most_informative_n /
            least_informative_n -- 'information' is proxied by axis length.
        auto_n (INT): how many axes to render, ``0`` = follow the input rank.
        on_error (COMBO, advanced): ``error`` raises, ``fallback_first_n`` keeps
            working by rendering the first axes and notes the reason in the plot.

    Outputs:
        image (IMAGE): the rendered figure as a single RGB image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _heatmap_input_types()

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

    def execute(self, matrices, xlabel, ylabel, figsize_w, figsize_h, cmap, titles=None,
                max_samples=DEFAULT_MAX_SAMPLES, dims="auto", axis_reduce="mean",
                auto_select="last_n", auto_n=0, on_error="fallback_first_n"):
        max_samples, dims, axis_reduce, auto_select, auto_n, on_error = _normalise_controls(
            max_samples, dims, axis_reduce, auto_select, auto_n, on_error)
        data, stride, note = _heatmap_view(matrices, max_samples, dims, axis_reduce,
                                           auto_select, auto_n, on_error)
        fig = _render_heatmap(data, xlabel=xlabel, ylabel=ylabel, figsize_w=figsize_w,
                              figsize_h=figsize_h, cmap=cmap, title=_first_title(titles),
                              note=note, stride=stride)
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
                "X": ("TENSOR",),  # [N,] or [L,N] for multiple curves
                "Y": ("TENSOR",),
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
    CATEGORY = "d2l/Visualization"

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
                "results": ("TENSOR",),  # [N, 2] tensor of (x1, x2) points
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

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
                "bboxes": ("TENSOR",),
            },
            "optional": {
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated labels"}),
                "colors": ("STRING", {"default": "b,g,r,m,c", "placeholder": "comma-separated matplotlib colors"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

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
                "tensor": ("TENSOR",),
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
    CATEGORY = "d2l/Visualization"

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
                "values": ("TENSOR",),
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
    CATEGORY = "d2l/Visualization"

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
                "X": ("TENSOR",),
                "Y": ("TENSOR",),
                "alpha": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
                "cmap": ("STRING", {"default": "viridis", "placeholder": "matplotlib colormap name"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "color_map": ("TENSOR",),
                "size_map": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

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
                "matrix": ("TENSOR",),
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
    CATEGORY = "d2l/Visualization"

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
                "values": ("TENSOR",),
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
    CATEGORY = "d2l/Visualization"

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
                "Y": ("TENSOR",),
                "stacked": ("BOOLEAN", {"default": False}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "color_palette": ("STRING", {"default": "tab10", "placeholder": "matplotlib palette name"}),
                "xlabel": ("STRING", {"default": "", "placeholder": "x-axis label"}),
                "ylabel": ("STRING", {"default": "", "placeholder": "y-axis label"}),
                "figsize_w": ("FLOAT", {"default": 7.0, "min": 3.0, "max": 20.0, "step": 0.5}),
                "figsize_h": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "X_vals": ("TENSOR",),
                "labels": ("STRING", {"default": "", "placeholder": "comma-separated series labels (for stacked)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

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
