"""
Shared visualisation palette and helpers.

One place defines the colours so every figure, the notebook, and the widgets agree.

Palette
-------
- Soft / average consensus maps: sequential `magma_r` composited *over the image*
  with a value-driven alpha (high consensus = dark, near-zero = transparent). Drawing
  the map over the image (rather than on a bare background) keeps a soft map and a
  hard map the same visible size on the page, and `magma_r` puts the salient high
  end at dark, which reads on both a light and a dark page.
- Individual raters: translucent per-rater fills (alpha ~0.25) plus a solid 2px
  contour in the same colour, so every region is faintly tinted yet its boundary is
  crisp. Fills and contours share the `FILL_PALETTE` colour for each rater.
- Hard consensus contour: white (distinct from every rater hue, high contrast on the
  image).
- Vote map: sequential `Blues` (a different hue from magma so the two quantities
  are not confused).

Figures are written as a single PNG each (not two theme variants): everything is
drawn over the image, and panel titles are white glyphs with a dark halo, so one
file is legible on both a light and a dark page and matches the notebook exactly.
"""

import json
import os

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# Visual knobs live in viz_config.json next to this file, so the colormap, the
# soft-map opacity cap, the fill alpha and the contour widths can be tuned in one
# place and picked up by both the notebook figures and the widgets. Missing file or
# key falls back to the defaults below.
_CFG_DEFAULTS = {
    "soft_cmap": "magma_r",
    "soft_gamma": 0.6,
    "soft_max_alpha": 0.8,
    "vote_cmap": "Blues",
    "fill_alpha": 0.25,
    "rater_contour_px": 1,
    "hard_contour_lw": 1.0,
}


def _load_cfg():
    """
    Load the configuration from the JSON file.
    """
    cfg = dict(_CFG_DEFAULTS)
    path = os.path.join(os.path.dirname(__file__), "viz_config.json")
    try:
        with open(path) as f:
            raw = json.load(f)
        cfg.update({k: v for k, v in raw.items() if k in _CFG_DEFAULTS})
    except (OSError, ValueError):
        pass

    return cfg


_CFG = _load_cfg()

__all__ = [
    "OKABE_ITO",
    "FILL_PALETTE",
    "SOFT_CMAP",
    "SOFT_GAMMA",
    "SOFT_MAX_ALPHA",
    "VOTE_CMAP",
    "HARD_COLOR",
    "ATTENTION_COLOR",
    "PLOT_INK",
    "CONTOUR_LW",
    "RATER_LW",
    "RATER_CONTOUR_PX",
    "FILL_ALPHA",
    "rater_hex",
    "rater_rgb",
    "fill_hex",
    "fill_rgb",
    "content_crop",
    "apply_crop",
    "crop_case",
    "ring_px",
    "soft_alpha_ramp",
    "soft_to_rgba",
    "soft_over_image_rgba",
    "vote_to_rgba",
    "gray_to_rgba",
    "rgb_to_rgba",
    "contour_rgba",
    "hard_over_image_rgba",
    "save_fig",
    "panel_title",
    "gray_bg_alpha",
    "raters_fill_over_image_rgba",
    "raters_overlay_rgba",
]

# Okabe-Ito, reordered so the first five (used by the 5-rater IMA++ case) stay
# clear of dermoscopy skin tone; mauve (skin-blender) is last and only reached by
# QUBIQ's 7th rater on grey MRI.
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#F0E442",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
]

# Palette for translucent rater FILLS + their solid contours (same colour per rater).
# This is the canonical Okabe-Ito set: the dataviz validator (all-pairs, CVD-simulated)
# gives worst-pair deutan dE 7.6 -- inside the 6-8 band that is legal WITH the
# secondary encoding we already have (solid contours, R-labels, per-rater toggles).
# An earlier #1FA187 green dropped that pair to 5.2 (a FAIL); the canonical #009E73 is
# strictly better. ColorBrewer Set1 was rejected: its blue/purple collapse to dE 3.5
# (below the floor) and its orange/pink/red blend with dermoscopy skin.
FILL_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#F0E442",
    "#56B4E9",
    "#D55E00",
    "#CC79A7",
]

SOFT_CMAP = _CFG[
    "soft_cmap"
]  # high consensus -> dark end (config: try plasma_r/viridis_r)
SOFT_GAMMA = _CFG["soft_gamma"]  # value->opacity ramp exponent
SOFT_MAX_ALPHA = _CFG[
    "soft_max_alpha"
]  # opacity cap at full consensus (<1 -> image shows through)
VOTE_CMAP = _CFG["vote_cmap"]
HARD_COLOR = "white"
ATTENTION_COLOR = (
    "#FF3B30"  # for annotations (e.g. circling an isolated speck)
)
CONTOUR_LW = _CFG[
    "hard_contour_lw"
]  # hard-consensus contour width (matplotlib points)
RATER_CONTOUR_PX = int(
    _CFG["rater_contour_px"]
)  # rater contour width in native pixels
RATER_LW = RATER_CONTOUR_PX  # back-compat alias
FILL_ALPHA = _CFG["fill_alpha"]  # translucent rater fill

# Single neutral ink for the one non-image plot (the sens/spec scatter), chosen to
# stay legible on both a light and a dark page so that figure also needs one version.
PLOT_INK = "#7f7f7f"


def rater_hex(k: int) -> str:
    """
    Get the hex colour for the rater.
    """
    return OKABE_ITO[k % len(OKABE_ITO)]


def rater_rgb(k: int):
    """
    Get the RGB colour for the rater.
    """
    h = rater_hex(k).lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def fill_hex(k: int) -> str:
    """
    Get the hex colour for the fill.
    """
    return FILL_PALETTE[k % len(FILL_PALETTE)]


def fill_rgb(k: int):
    """
    Get the RGB colour for the fill.
    """
    h = fill_hex(k).lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# --------------------------------------------------------------- display crop


def content_crop(
    image: np.ndarray, margin: int = 10, thresh_frac: float = 0.02
):
    """
    Bounding-box slices of the non-background content, for display cropping.

    QUBIQ frames are ~90% black padding with the brain in a corner; cropping
    to the content makes the structure fill the panel. Returns (row_slice, col_slice).
    Falls back to the full frame when the image is (nearly) constant.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:  # RGB -> luma
        img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return slice(None), slice(None)
    fg = img > lo + thresh_frac * (hi - lo)
    if not fg.any():
        return slice(None), slice(None)
    ys, xs = np.where(fg)
    H, W = img.shape
    r0, r1 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    c0, c1 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)

    return slice(r0, r1), slice(c0, c1)


def apply_crop(arr: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    """
    Apply (row_slice, col_slice) to the last two spatial axes of arr.
    """
    if arr.ndim == 2:
        return arr[rows, cols]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # (H, W, C) RGB(A)
        return arr[rows, cols, :]
    if arr.ndim == 3:  # (K, H, W)
        return arr[:, rows, cols]
    raise ValueError(f"cannot crop array of shape {arr.shape}")


def crop_case(image, masks, mode: str, margin: int = 10):
    """
    Return (image, masks, (rows, cols)) cropped for display.

    For 'nifti' (QUBIQ) crop to the brain content; for 'image' (IMA++) leave as is
    (the dermoscopy image already fills the frame).
    """
    if mode != "nifti":
        return image, masks, (slice(None), slice(None))
    rows, cols = content_crop(image, margin=margin)

    return (
        apply_crop(image, rows, cols),
        apply_crop(masks, rows, cols),
        (rows, cols),
    )


# --------------------------------------------------------------- array -> rgba
# (used by the widget baker; matplotlib figures use the mpl helpers below)


def _rgba_u8(rgba_float):
    """
    Convert a float RGBA array to uint8 RGBA.
    """
    return (np.clip(rgba_float, 0, 1) * 255).astype(np.uint8)


def ring_px(core, px):
    """
    A `px`-pixel-wide contour ring around a binary `core`, centred on the
    boundary. `px=1` gives the thinnest (1-pixel) line. Config drives `px` via
    `rater_contour_px` so contour thickness is tunable without code changes.
    """
    from scipy import ndimage as ndi

    core = np.asarray(core) > 0
    px = max(1, int(px))
    out = int(np.ceil(px / 2))
    inn = px // 2
    outer = ndi.binary_dilation(core, iterations=out) if out else core
    inner = ndi.binary_erosion(core, iterations=inn) if inn else core

    return outer & ~inner


def soft_alpha_ramp(soft):
    """
    Per-pixel opacity for a soft map so that value 0 is transparent.

    Soft consensus maps are drawn with this alpha so the image shows through the
    near-zero background instead of a fixed block. The exponent `SOFT_GAMMA` shapes
    the ramp, and `SOFT_MAX_ALPHA` caps the opacity at full consensus so the darkest
    end is not a pure opaque block (both are config knobs).
    """
    return np.clip(soft, 0, 1) ** SOFT_GAMMA * SOFT_MAX_ALPHA


def soft_to_rgba(soft, cmap=SOFT_CMAP, ramp_alpha=True):
    """
    Soft map -> uint8 RGBA. With ramp_alpha, background fades to transparent.
    """
    rgba = cm.get_cmap(cmap)(Normalize(0, 1)(np.clip(soft, 0, 1)))
    if ramp_alpha:
        rgba[..., 3] = soft_alpha_ramp(soft)

    return _rgba_u8(rgba)


def soft_over_image_rgba(image, soft, is_rgb=False, cmap=SOFT_CMAP):
    """
    Composite a soft consensus map over the image (uint8 RGBA).

    The map is coloured with `cmap` (`magma_r`: high = dark) and given a
    value-driven opacity, so near-zero consensus is transparent (the image shows
    through) and high consensus is an opaque dark wash. Because the image is the
    background of *every* panel, a soft map and a hard map occupy the same visible
    footprint on the page (fixing the "the average looks smaller than the vote"
    mismatch).
    """
    if is_rgb:
        base = np.clip(image, 0, 1).astype(np.float32)[:, :, :3].copy()
        a_img = np.ones(base.shape[:2], np.float32)
    else:
        g = np.clip(image, 0, 1).astype(np.float32)
        base = np.dstack([g, g, g])
        a_img = gray_bg_alpha(image)
    sa = soft_alpha_ramp(soft)
    scol = cm.get_cmap(cmap)(Normalize(0, 1)(np.clip(soft, 0, 1)))[
        ..., :3
    ].astype(np.float32)
    out = base * (1 - sa)[..., None] + scol * sa[..., None]
    a = np.maximum(a_img, sa)

    return _rgba_u8(np.concatenate([out, a[..., None]], axis=-1))


def vote_to_rgba(counts, kmax, cmap=VOTE_CMAP):
    """
    Convert a vote count array to uint8 RGBA.
    """
    return _rgba_u8(cm.get_cmap(cmap)(Normalize(0, kmax)(counts)))


def gray_to_rgba(image01):
    """
    Convert a greyscale image array to uint8 RGBA.
    """
    g = _rgba_u8(np.dstack([image01] * 3))
    a = np.full(g.shape[:2] + (1,), 255, np.uint8)

    return np.concatenate([g, a], axis=-1)


def rgb_to_rgba(rgb01):
    """
    Convert an RGB image array to uint8 RGBA.
    """
    rgb = _rgba_u8(rgb01)
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8)

    return np.concatenate([rgb, a], axis=-1)


def contour_rgba(mask, rgb, width=RATER_CONTOUR_PX):
    """
    Convert a mask array to uint8 RGBA.
    """
    ring = ring_px(mask, width)
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[ring] = (*rgb, 255)

    return out


def hard_over_image_rgba(
    image, mask, rgb=(255, 255, 255), is_rgb=False, width=RATER_CONTOUR_PX
):
    """
    Composite: image (greyscale or RGB) with a coloured hard-consensus ring burned in.
    """
    if is_rgb:
        base = _rgba_u8(image)[:, :, :3].copy()
    else:
        g = _rgba_u8(np.dstack([image] * 3))
        base = g.copy()
    base[ring_px(mask, width)] = rgb
    a = np.full(base.shape[:2] + (1,), 255, np.uint8)

    return np.concatenate([base, a], axis=-1)


def gray_bg_alpha(image, gain=6.0):
    """
    Opacity for a greyscale panel so the near-zero background (e.g. the black MRI
    padding outside the head) fades to transparent and the themed page shows through.
    RGB images are fully opaque (no padding to hide).
    """
    return np.clip(np.clip(image, 0, 1) * gain, 0, 1).astype(np.float32)


def raters_fill_over_image_rgba(image, masks, is_rgb=False, alpha=FILL_ALPHA):
    """
    Composite translucent per-rater fills over the image (RGBA uint8).

    Each rater's foreground is tinted with its FILL_PALETTE colour at `alpha` and
    left fully transparent elsewhere; overlaps blend, so darker regions read as
    stronger agreement. For greyscale panels the background padding is made
    transparent so the figure adapts to a light or dark page.
    """
    if is_rgb:
        base = np.clip(image, 0, 1).astype(np.float32)[:, :, :3].copy()
        a = np.ones(base.shape[:2], np.float32)
    else:
        base = np.dstack([np.clip(image, 0, 1).astype(np.float32)] * 3)
        a = gray_bg_alpha(image)
    for k in range(len(masks)):
        col = np.array(fill_rgb(k), np.float32) / 255.0
        m = masks[k] > 0
        base = np.where(m[:, :, None], (1 - alpha) * base + alpha * col, base)
        a = np.maximum(
            a, (m.astype(np.float32) * alpha)
        )  # fills stay translucent over padding

    return _rgba_u8(np.concatenate([base, a[:, :, None]], axis=-1))


def raters_overlay_rgba(
    image,
    masks,
    is_rgb=False,
    alpha=FILL_ALPHA,
    lw=RATER_CONTOUR_PX,
    opaque=False,
):
    """
    Translucent per-rater fills PLUS a solid same-colour contour, over the image.

    This is the segmentation-mask-overlay convention: every rater's
    region is faintly tinted (alpha `alpha`) so overlaps read as agreement, and
    every rater's boundary is a crisp opaque `lw`-pixel contour in the same colour,
    so the raters stay individually distinguishable. Contours are drawn after all
    fills so a boundary is never hidden under another rater's wash.

    `opaque` keeps the greyscale background fully visible (for a full-frame CT/MRI
    where the dark anatomy is real content); the default fades near-black padding to
    transparent (for a small structure in a mostly black frame, e.g. the brain).
    """
    if is_rgb:
        base = np.clip(image, 0, 1).astype(np.float32)[:, :, :3].copy()
        a = np.ones(base.shape[:2], np.float32)
    else:
        base = np.dstack([np.clip(image, 0, 1).astype(np.float32)] * 3)
        a = (
            np.ones(base.shape[:2], np.float32)
            if opaque
            else gray_bg_alpha(image)
        )
    for k in range(len(masks)):  # fills first
        col = np.array(fill_rgb(k), np.float32) / 255.0
        m = masks[k] > 0
        base = np.where(m[:, :, None], (1 - alpha) * base + alpha * col, base)
        a = np.maximum(a, m.astype(np.float32) * alpha)
    for k in range(len(masks)):  # solid contours on top
        col = np.array(fill_rgb(k), np.float32) / 255.0
        ring = ring_px(masks[k], lw)
        base[ring] = col
        a[ring] = 1.0

    return _rgba_u8(np.concatenate([base, a[:, :, None]], axis=-1))


# --------------------------------------------------------------- matplotlib


def panel_title(ax, text, y=0.965, fontsize=11, rotation=0):
    """
    Draw a title as white text on a small translucent dark pill.

    The pill (rather than a stroke halo, which glowed on a light page) gives
    clean contrast on any image and on both a light and a dark page, so one rendering
    suffices. Used for panel titles, column headers and the all-methods row labels.
    """
    ha = "center" if rotation == 0 else "right"
    x = 0.5 if rotation == 0 else -0.02

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        rotation=rotation,
        color="white",
        fontsize=fontsize,
        fontweight="medium",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#111111",
            edgecolor="none",
            alpha=0.55,
        ),
    )


def save_fig(fig, stem, pad=0.06, bg=None):
    """
    Save a figure as a single PNG `{stem}.png` and return the path.

    One version, not two. With `bg=None` the figure is transparent for panels
    drawn over the image, whose content is mode-independent and whose titles carry
    their own halo. Pass `bg` (a hex colour) for the one plot with no image to sit
    on (the sens/spec scatter): it becomes a self-contained opaque card in that
    neutral colour, which reads on both a light and a dark page.
    """
    path = stem if stem.endswith(".png") else stem + ".png"
    if bg is None:
        fig.savefig(
            path, transparent=True, bbox_inches="tight", pad_inches=pad
        )
    else:
        fig.patch.set_facecolor(bg)
        fig.savefig(
            path,
            facecolor=bg,
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=pad,
        )
    return path
