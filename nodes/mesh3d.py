"""
d2l/Visualization - Export a heatmap tensor as a real 3D model (OBJ + MTL).

``CdlHeatmapsTo3D`` reuses the sampling / axis-selection / reduction front end of
the Show Heatmaps nodes (``comfydl/nodes/visualization.py``) and turns the
resulting 1-/2-/3-D array into geometry the stock ``Preview3D`` node can display:

  * 1-D -> a flat, bar-code like colour ribbon with a real thickness, captioned
    "1D" on the ground plane in front of it;
  * 2-D -> the same thing in two dimensions: a flat, coloured, slightly thick
    plate captioned "2D";
  * 3-D -> a translucent cube: the six outer faces plus the three orthogonal
    mid-planes, so the interior stays readable.

Colour travels through the material library (one material per quantised palette
level) because OBJ vertex colours carry no alpha. ``d`` in the MTL is the only
place a viewer (three.js' ``MTLLoader``) reads transparency from, and
``Preview3D`` renames the file to ``preview3d_<uuid>.obj`` before saving it, so
the ``.mtl`` has to follow whatever path the viewer picks - hence the
``Types.File3D`` subclass below instead of a plain byte stream.

No 3D library is used: the OBJ / MTL text is written by hand, which keeps the
venv dependency-free and the output inspectable in a text editor.
"""

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, consistent with visualization.py
import matplotlib.colors
import numpy as np

from .visualization import (
    DEFAULT_MAX_SAMPLES,
    _AUTO_SELECT_MODES,
    _CUBE_AXIS_CAP,
    _ON_ERROR_MODES,
    _REDUCE_MODES,
    HeatmapSpecError,
    _cap_axis_samples,
    _heatmap_view,
    _normalise_controls,
    _resolve_cmap,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

#: Number of distinct materials the colormap is quantised into. One material per
#: level keeps the MTL small while the ramp still looks continuous.
_PALETTE_LEVELS = 64

#: Per-axis caps for the exported mesh. These are deliberately much smaller than
#: ``DEFAULT_MAX_SAMPLES``: a 1920x1080 heatmap would otherwise become millions of
#: quads. The sampled array is strided once more to respect them.
_RIBBON_CELL_CAP = 256
_PLATE_AXIS_CAP = 64

#: Palette slot reserved for the caption geometry, and its colour.
_LABEL_PALETTE = -1
_LABEL_RGB = (0.14, 0.14, 0.14)

#: 5x7 bitmap glyphs for the caption. Only the four characters the caption can
#: contain are kept, so the table stays trivial to verify by eye.
_GLYPHS = {
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
}


# ============================================================
# Small vector helpers (tuples of 3 floats - no numpy overhead per vertex)
# ============================================================

def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector, factor):
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ============================================================
# Mesh accumulation and OBJ / MTL serialisation
# ============================================================

def _material_name(palette):
    """Material id of a palette slot ('m_label' for the caption geometry)."""
    return "m_label" if palette == _LABEL_PALETTE else f"m{palette:02d}"


def _material_block(name, rgb, opacity):
    """One ``newmtl`` block; ``d`` < 1 is what makes a viewer render it see-through."""
    red, green, blue = rgb
    return "\n".join([
        f"newmtl {name}",
        f"Ka {red * 0.2:.6f} {green * 0.2:.6f} {blue * 0.2:.6f}",
        f"Kd {red:.6f} {green:.6f} {blue:.6f}",
        # No specular highlight: the palette colour must read as the value, not as
        # a reflection, and a shiny default would wash the ramp out.
        "Ks 0.000000 0.000000 0.000000",
        "Ns 1.000000",
        "illum 1",
        f"d {opacity:.4f}",
    ])


def _mtl_text(palette_rgb, opacity):
    """The whole ``.mtl`` file: one entry per palette level plus the caption."""
    blocks = [_material_block(_material_name(index), rgb, opacity)
              for index, rgb in enumerate(palette_rgb)]
    blocks.append(_material_block(_material_name(_LABEL_PALETTE), _LABEL_RGB, 1.0))
    return "\n\n".join(blocks) + "\n"


class _MeshBuilder:
    """Collects flat quads, grouped by palette slot, and writes OBJ text.

    Every quad is stored together with an *outward* direction and the winding is
    flipped when the naive one would face inwards, because three.js renders OBJ
    materials single sided (``MeshPhongMaterial.side`` defaults to ``FrontSide``):
    a wrongly wound face is not shaded oddly, it is invisible.
    """

    def __init__(self):
        self.vertices = []
        self._faces = {}
        self._order = []

    def _bucket(self, palette):
        faces = self._faces.get(palette)
        if faces is None:
            faces = self._faces[palette] = []
            self._order.append(palette)
        return faces

    def add_quad(self, palette, corners, outward=None, double=False):
        """Add one flat quad; ``outward=None`` means it is visible from both sides."""
        points = list(corners)
        if outward is not None:
            normal = _cross(
                (points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]),
                (points[2][0] - points[1][0], points[2][1] - points[1][1], points[2][2] - points[1][2]),
            )
            if _dot(normal, outward) < 0:
                points.reverse()
        base = len(self.vertices)
        self.vertices.extend(points)
        self._bucket(palette).append((base, base + 1, base + 2, base + 3))
        if double:
            self._bucket(palette).append((base + 3, base + 2, base + 1, base))

    def add_grid(self, palette_grid, origin, u_vec, v_vec, outward=None, double=False):
        """One quad per cell of a 2-D palette grid, sharing the grid vertices."""
        rows, cols = palette_grid.shape
        base = len(self.vertices)
        for row in range(rows + 1):
            for column in range(cols + 1):
                self.vertices.append(_add(_add(origin, _scale(u_vec, row)), _scale(v_vec, column)))
        flip = outward is not None and _dot(_cross(u_vec, v_vec), outward) < 0
        for row in range(rows):
            for column in range(cols):
                start = base + row * (cols + 1) + column
                quad = (start, start + cols + 1, start + cols + 2, start + 1)
                if flip:
                    quad = tuple(reversed(quad))
                self._bucket(int(palette_grid[row, column])).append(quad)
                if double:
                    self._bucket(int(palette_grid[row, column])).append(tuple(reversed(quad)))

    def obj_body(self):
        """The OBJ file *without* its ``mtllib`` line (only known at save time)."""
        lines = ["# ComfyDL heatmap -> 3D mesh"]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for palette in self._order:
            lines.append(f"usemtl {_material_name(palette)}")
            lines.append(f"g {_material_name(palette)}")
            for quad in self._faces[palette]:
                lines.append("f " + " ".join(str(index + 1) for index in quad))
        return "\n".join(lines) + "\n"


# ============================================================
# Geometry
# ============================================================

def _add_slab(mesh, palette, thickness):
    """A 2-D palette grid -> a flat plate of ``thickness`` with coloured side walls.

    The top and bottom faces carry the values; the four walls repeat the boundary
    cell colours, so the plate never shows an untextured edge.
    """
    rows, cols = palette.shape
    up = (0.0, 0.0, thickness)
    mesh.add_grid(palette, (0.0, 0.0, thickness), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    mesh.add_grid(palette, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    mesh.add_grid(palette[:, :1], (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), up, (-1.0, 0.0, 0.0))
    mesh.add_grid(palette[:, -1:], (float(cols), 0.0, 0.0), (0.0, 1.0, 0.0), up, (1.0, 0.0, 0.0))
    mesh.add_grid(palette[:1, :], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), up, (0.0, -1.0, 0.0))
    mesh.add_grid(palette[-1:, :], (0.0, float(rows), 0.0), (1.0, 0.0, 0.0), up, (0.0, 1.0, 0.0))


def _add_cube(mesh, palette):
    """A 3-D palette -> translucent shell (6 faces) plus the 3 orthogonal mid-planes.

    Only the shell and three slices are emitted, never the full volume: the polygon
    count stays at ``9 * cap**2`` instead of ``cap**3``. The mid-planes are added
    double sided because a single-sided plane would vanish from half of the orbit.
    """
    depth, height, width = palette.shape
    mesh.add_grid(palette[:, :, -1], (0.0, 0.0, float(width)), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    mesh.add_grid(palette[:, :, 0].T, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    mesh.add_grid(palette[0, :, :].T, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    mesh.add_grid(palette[-1, :, :], (float(depth), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    mesh.add_grid(palette[:, 0, :], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    mesh.add_grid(palette[:, -1, :].T, (0.0, float(height), 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    for axis in range(3):
        mid = int(palette.shape[axis]) // 2
        others = [index for index in range(3) if index != axis]
        origin = [0.0, 0.0, 0.0]
        origin[axis] = float(mid)
        u_vec, v_vec = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        u_vec[others[0]] = 1.0
        v_vec[others[1]] = 1.0
        mesh.add_grid(np.take(palette, mid, axis=axis), tuple(origin), tuple(u_vec), tuple(v_vec),
                      outward=None, double=True)


def _add_box(mesh, palette, x0, y0, z0, x1, y1, z1):
    """Axis-aligned box, used for the (few hundred) caption pixels."""
    faces = (
        (((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), (0.0, 0.0, 1.0)),
        (((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)), (0.0, 0.0, -1.0)),
        (((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)), (-1.0, 0.0, 0.0)),
        (((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)), (1.0, 0.0, 0.0)),
        (((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)), (0.0, -1.0, 0.0)),
        (((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)), (0.0, 1.0, 0.0)),
    )
    for corners, outward in faces:
        mesh.add_quad(palette, corners, outward)


def _add_caption(mesh, text, span, extent_x):
    """Extrude a "1D"/"2D"/"3D" caption onto the ground plane in front of the model.

    A 3D file cannot carry text, so the caption is drawn: each lit pixel of a 5x7
    bitmap becomes a small box. Two characters cost a few hundred quads, which is
    nothing next to the mesh itself, and it makes the exported dimensionality
    obvious without reading the filename.
    """
    scale = max(0.02 * span, 1e-6)
    glyph_width, gap, height = 5 * scale, scale, 7 * scale
    total_width = len(text) * glyph_width + (len(text) - 1) * gap
    origin_x = (extent_x - total_width) / 2.0
    origin_y = -(height + 0.05 * span)
    depth = 0.9 * scale
    for index, char in enumerate(text):
        glyph = _GLYPHS.get(char)
        if glyph is None:
            continue
        base_x = origin_x + index * (glyph_width + gap)
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit != "1":
                    continue
                x0 = base_x + column * scale
                y0 = origin_y + (6 - row) * scale
                _add_box(mesh, _LABEL_PALETTE, x0, y0, 0.0, x0 + scale, y0 + scale, depth)


# ============================================================
# Palette and mesh assembly
# ============================================================

def _palette(values, cmap_obj):
    """``(palette_indices, palette_rgb)`` - the array quantised onto the colormap.

    Quantising to ``_PALETTE_LEVELS`` levels is what keeps the MTL small: the
    alternative (one material per cell) would produce a megabyte of ``newmtl``
    blocks for a modest heatmap.
    """
    finite = np.isfinite(values)
    if finite.any():
        low = float(np.min(values[finite]))
        high = float(np.max(values[finite]))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    normalised = matplotlib.colors.Normalize(vmin=low, vmax=high)
    scaled = np.clip(np.nan_to_num(normalised(values), nan=0.0), 0.0, 1.0)
    indices = np.minimum(np.rint(scaled * (_PALETTE_LEVELS - 1)).astype(np.int64), _PALETTE_LEVELS - 1)
    rgb = [tuple(float(channel) for channel in cmap_obj(level / (_PALETTE_LEVELS - 1))[:3])
           for level in range(_PALETTE_LEVELS)]
    return indices, rgb


def _build_heatmap_mesh(data, cmap, opacity, thickness):
    """``(obj_body, mtl_text, note)`` for a 1-/2-/3-D heatmap array."""
    cmap_obj, cmap_note = _resolve_cmap(cmap)
    array = np.asarray(data, dtype=np.float64)
    if array.size == 0:
        array = np.zeros((1, 1))
    notes = [cmap_note] if cmap_note else []

    mesh = _MeshBuilder()
    if array.ndim == 1:
        capped, stride = _cap_axis_samples(array, _RIBBON_CELL_CAP)
        palette, rgb = _palette(capped, cmap_obj)
        _add_slab(mesh, palette.reshape(1, -1), thickness)
        caption, extent = "1D", (float(palette.size), 1.0, float(thickness))
    elif array.ndim == 2:
        capped, stride = _cap_axis_samples(array, _PLATE_AXIS_CAP)
        palette, rgb = _palette(capped, cmap_obj)
        _add_slab(mesh, palette, thickness)
        caption, extent = "2D", (float(palette.shape[1]), float(palette.shape[0]), float(thickness))
    else:
        capped, stride = _cap_axis_samples(array, _CUBE_AXIS_CAP)
        palette, rgb = _palette(capped, cmap_obj)
        _add_cube(mesh, palette)
        caption, extent = "3D", tuple(float(size) for size in palette.shape)

    _add_caption(mesh, caption, max(extent), extent[0])
    if stride > 1:
        notes.append(f"mesh strided by {stride} to stay inside the polygon budget")
    return mesh.obj_body(), _mtl_text(rgb, opacity), "; ".join(notes)


# ============================================================
# File3D wrapper
# ============================================================

_OBJ_WITH_MTL_CLASS = None


def _obj_file_class():
    """Build (once) the ``Types.File3D`` subclass that carries its own ``.mtl``.

    ``Preview3D`` saves whatever it receives under ``preview3d_<uuid>.obj``; a plain
    ``File3D`` would copy only the OBJ, so the material file - and with it every
    colour and the requested transparency - would be lost.
    """
    global _OBJ_WITH_MTL_CLASS
    if _OBJ_WITH_MTL_CLASS is None:
        from comfy_api.latest import Types

        class _ObjWithMtl(Types.File3D):
            """OBJ file that writes its sibling ``.mtl`` next to whatever path it is saved to."""

            def __init__(self, obj_body, mtl_text):
                self._obj_body = obj_body
                self._mtl_text = mtl_text
                super().__init__(io.BytesIO(obj_body.encode("utf-8")), file_format="obj")

            def save_to(self, path):
                destination = Path(path)
                mtl_path = destination.with_suffix(".mtl")
                # The OBJ must name the .mtl by the name it will actually have.
                payload = f"mtllib {mtl_path.name}\n{self._obj_body}"
                self._source = io.BytesIO(payload.encode("utf-8"))
                saved = super().save_to(str(destination))
                mtl_path.write_text(self._mtl_text, encoding="utf-8")
                return saved

        _OBJ_WITH_MTL_CLASS = _ObjWithMtl
    return _OBJ_WITH_MTL_CLASS


# ============================================================
# Node
# ============================================================

class CdlHeatmapsTo3D:
    """Export a heatmap tensor as a real 3D model (OBJ + MTL).

    The front end is identical to Show Heatmaps: the tensor is strided down to
    ``max_samples`` elements, the axes named by ``dims`` (or picked by
    ``auto_select`` / ``auto_n``) are kept and every other axis is collapsed by
    ``axis_reduce``. What comes out is always a 1-, 2- or 3-D array:

      * 1-D -> bar-code like colour ribbon lying flat, with a real thickness;
      * 2-D -> flat coloured plate of the same thickness;
      * 3-D -> translucent cube (six faces + three orthogonal mid-planes).

    Colour is quantised into 64 materials, so the whole palette ships as one small
    MTL file. Feed the result into the stock ``Preview3D`` node - the frontend
    only binds its 3D canvas to that hard-coded node id, so a custom node cannot
    display 3D itself.

    Inputs:
        matrices (TENSOR): any rank. Sampled to ``max_samples`` elements, reduced
            to at most three axes, then strided again to the mesh polygon budget
            (256 cells for 1-D, 64 per axis for 2-D, 32 per axis for 3-D).
        cmap (STRING): matplotlib colormap name; an unknown name falls back to
            "Reds" instead of failing.
        opacity (FLOAT): material alpha (``d`` in the MTL). Keep it below 1.0 to
            see the inside of the 3-D cube.
        thickness (FLOAT): thickness of the 1-D / 2-D ribbon in cell units.
        max_samples (INT): element budget, ``0`` disables sampling.
        dims (STRING): ``auto``, or 1-3 axis indices such as ``0,1`` / ``-2,-1``.
        axis_reduce (COMBO): mean / max / first / mid reduction of the other axes.
        auto_select (COMBO): first_n / last_n / most_informative_n /
            least_informative_n -- 'information' is proxied by axis length.
        auto_n (INT): how many axes to keep, ``0`` = follow the input rank.
        on_error (COMBO, advanced): ``error`` raises, ``fallback_first_n`` keeps
            working by using the first axes.

    Outputs:
        model_3d (FILE_3D_OBJ): OBJ mesh plus a sibling MTL; connect it to
            ``Preview3D`` to look at it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matrices": ("TENSOR",),
                "cmap": ("STRING", {"default": "Reds", "placeholder": "matplotlib colormap name"}),
                "opacity": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 1.0, "step": 0.05}),
                "thickness": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "max_samples": ("INT", {"default": DEFAULT_MAX_SAMPLES, "min": 0, "max": 1073741824, "step": 1024}),
                "dims": ("STRING", {"default": "auto", "placeholder": "auto | 0,1 | -2,-1 (1-3 axes, negatives allowed)"}),
                "axis_reduce": (list(_REDUCE_MODES), {"default": "mean"}),
                "auto_select": (list(_AUTO_SELECT_MODES), {"default": "last_n"}),
                "auto_n": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "on_error": (list(_ON_ERROR_MODES), {"default": "fallback_first_n", "advanced": True}),
            },
        }

    RETURN_TYPES = ("FILE_3D_OBJ",)
    RETURN_NAMES = ("model_3d",)
    FUNCTION = "execute"
    CATEGORY = "d2l/Visualization"

    def execute(self, matrices, cmap, opacity, thickness, max_samples=DEFAULT_MAX_SAMPLES,
                dims="auto", axis_reduce="mean", auto_select="last_n", auto_n=0,
                on_error="fallback_first_n"):
        max_samples, dims, axis_reduce, auto_select, auto_n, on_error = _normalise_controls(
            max_samples, dims, axis_reduce, auto_select, auto_n, on_error)
        data, _, note = _heatmap_view(matrices, max_samples, dims, axis_reduce,
                                      auto_select, auto_n, on_error)
        obj_body, mtl_text, mesh_note = _build_heatmap_mesh(data, cmap, opacity, thickness)
        notes = "; ".join(part for part in (note, mesh_note) if part)
        if notes:
            logging.info("[ComfyDL] %s -> 3D (%s): %s", tuple(np.shape(data)), len(mtl_text), notes)
        return (_obj_file_class()(obj_body, mtl_text),)


NODE_CLASS_MAPPINGS["CdlHeatmapsTo3D"] = CdlHeatmapsTo3D
NODE_DISPLAY_NAME_MAPPINGS["CdlHeatmapsTo3D"] = "Heatmaps to 3D"
