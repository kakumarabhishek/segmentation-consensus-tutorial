"""
Loading multi-annotator cases from QUBIQ 2021 and IMA++.

Both datasets end up in the same dictionary layout so every consensus method
downstream can ignore which one it is looking at:

    {
        "image":    float32 in [0, 1], (D, H, W) for 'nifti', (H, W) for 'image'
        "masks":    uint8 {0, 1},      (K, D, H, W)        or (K, H, W)
        "n_raters": int,
        "modality": str,
    }

Dataset layouts
---------------
QUBIQ 2021, brain-growth task: one directory per case::

    brain-growth/Training/case01/
        image.nii.gz          T1 MRI, one 256x256 axial slice stored as NIfTI
        task01_seg01.nii.gz   rater 1 binary mask (256x256)
        ...
        task01_seg07.nii.gz   rater 7

    NOTE: although the original QUBIQ challenge distributes brain-growth as 3D
    volumes, the copy used here is already a single 2D slice per case (verified:
    all 34 training cases are (256, 256)). `load_case(modality='nifti')` returns
    it as 2D directly, so `get_reference_slice`, which selects a slice out of
    a volume, is not needed for this data. Case selection across the 34 cases
    (`rank_cases_by_disagreement`) takes the role the slice choice would have.

IMA++: flat directories, images and masks separated::

    imaplusplus/
        images/   one .JPG per case, named ISIC_<id>.JPG
        segs/     PNG masks named ISIC_<id>_A<rater>_T<trial>_S<scale>_<hash>.png

    The rater identity is the `A<nn>` annotator code (e.g. A02, A06), NOT the
    trailing hash. Most images have a single annotator; only ~250 have two or
    more, and just two cases have five. `find_ima_cases` keeps one mask per
    annotator (the first by filename when an annotator segmented a case more than
    once at different trial/scale settings) and returns multi-rater cases only.
"""

import glob
import os
import re

import numpy as np

__all__ = [
    "load_case",
    "get_reference_slice",
    "find_qubiq_cases",
    "find_ima_cases",
    "rank_cases_by_disagreement",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _normalise_intensity(volume: np.ndarray) -> np.ndarray:
    """Clip to the 1st-99th percentile and rescale to [0, 1].

    Percentile clipping rather than min/max so that a few bright voxels (scanner
    artefacts, specular highlights) cannot compress the rest of the range.
    """
    volume = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(volume, [1.0, 99.0])
    if hi <= lo:  # constant image
        return np.zeros_like(volume, dtype=np.float32)

    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_case(
    image_dir: str,
    image_path: str,
    segs_dir: str,
    seg_paths: list,
    modality: str,
) -> dict:
    """
    Load one case: an image plus one binary mask per rater.

    Args:
        image_dir / image_path: the image, path given relative to image_dir.
        segs_dir / seg_paths:   the rater masks, relative to segs_dir, one per rater.
        modality:               'nifti' (3D QUBIQ) or 'image' (2D IMA++).

    For 'image' the returned `image` is greyscale (H, W) so it is a drop-in
    for the NIfTI path; the original RGB is kept alongside under `image_rgb`
    for display, since dermoscopy figures are far more readable in colour.
    """
    if modality not in ("nifti", "image"):
        raise ValueError(
            f"modality must be 'nifti' or 'image', got {modality!r}"
        )
    if len(seg_paths) == 0:
        raise ValueError("seg_paths is empty; a case needs at least one rater")

    full_image = os.path.join(image_dir, image_path)
    full_segs = [os.path.join(segs_dir, p) for p in seg_paths]
    for p in [full_image] + full_segs:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    case = {"modality": modality, "n_raters": len(seg_paths)}

    if modality == "nifti":
        import nibabel as nib

        image = np.asarray(nib.load(full_image).dataobj, dtype=np.float32)
        masks = [
            (np.asarray(nib.load(p).dataobj) > 0.5).astype(np.uint8)
            for p in full_segs
        ]
        case["image"] = _normalise_intensity(image)
    else:
        from PIL import Image

        rgb = (
            np.asarray(Image.open(full_image).convert("RGB"), dtype=np.float32)
            / 255.0
        )
        masks = []
        for p in full_segs:
            m = (
                np.asarray(Image.open(p).convert("L"), dtype=np.float32)
                / 255.0
            )
            masks.append((m > 0.5).astype(np.uint8))
        case["image_rgb"] = rgb.astype(np.float32)
        # ITU-R BT.601 luma; the standard greyscale reduction.
        case["image"] = _normalise_intensity(
            rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        )

    shapes = {m.shape for m in masks}
    if len(shapes) != 1:
        raise ValueError(f"rater masks disagree on shape: {shapes}")
    if masks[0].shape != case["image"].shape:
        raise ValueError(
            f"mask shape {masks[0].shape} does not match image shape {case['image'].shape}"
        )

    case["masks"] = np.stack(masks, axis=0).astype(np.uint8)

    return case


def get_reference_slice(case: dict, axis: int = 0) -> dict:
    """
    Pick the 3D slice where the raters disagree most, and return it as a 2D case.

    Disagreement is scored as the standard deviation, across raters, of the
    per-rater foreground pixel count in that slice. A slice every rater outlines
    identically scores 0; a slice one rater calls large and another calls small
    scores high. Slices no rater annotated also score 0, so they are never
    selected as long as some annotated slice exists.

    Args:
        case: the case to get the reference slice from.
        axis: the axis to get the reference slice from
              (0 for axial, 1 for sagittal, 2 for coronal).

    Returns:
        The same structure as `load_case` with the image reduced to (H, W)
        and the masks to (K, H, W), plus `slice_index` and `slice_axis`.
    """
    if case["modality"] != "nifti":
        raise ValueError(
            "get_reference_slice is only meaningful for 3D 'nifti' cases"
        )
    masks = case["masks"]  # (K, D, H, W)
    if masks.ndim != 4:
        raise ValueError(f"expected (K, D, H, W) masks, got {masks.shape}")

    # Foreground count per (rater, slice) along the chosen volume axis.
    vol_axis = axis + 1  # +1 to skip the rater axis
    other_axes = tuple(a for a in range(1, masks.ndim) if a != vol_axis)
    counts = masks.sum(axis=other_axes)  # (K, n_slices)
    disagreement = counts.std(axis=0)
    if not disagreement.any():
        raise ValueError("no slice shows any inter-rater disagreement")
    idx = int(np.argmax(disagreement))

    ref = dict(case)
    ref["image"] = np.take(case["image"], idx, axis=axis)
    ref["masks"] = np.take(masks, idx, axis=vol_axis)
    ref["slice_index"] = idx
    ref["slice_axis"] = axis

    return ref


# --------------------------------------------------------------------------
# case discovery
# --------------------------------------------------------------------------


def find_qubiq_cases(training_dir: str, task: int = 1) -> list:
    """
    List QUBIQ cases as `(case_dir, image_name, seg_names)` tuples.

    Rater masks follow `task{task:02d}_seg{rater:02d}.nii.gz`.

    Args:
        training_dir: the directory containing the QUBIQ training cases.
        task: the task number to load.

    Returns:
        A list of tuples, each containing `(case_dir, image_name, seg_names)`.
    """
    out = []
    for case_dir in sorted(glob.glob(os.path.join(training_dir, "case*"))):
        segs = sorted(
            os.path.basename(p)
            for p in glob.glob(
                os.path.join(case_dir, f"task{task:02d}_seg*.nii.gz")
            )
        )
        if segs and os.path.exists(os.path.join(case_dir, "image.nii.gz")):
            out.append((case_dir, "image.nii.gz", segs))

    return out


# IMA++ mask names: ISIC_<id>_A<rater>_T<trial>_S<scale>_<hash>.png
_IMA_SEG_RE = re.compile(r"^(ISIC_\d+)_A(\d+)_T\d+_S\d+_", re.IGNORECASE)


def find_ima_cases(
    images_dir: str, segs_dir: str, min_raters: int = 2
) -> list:
    """Group IMA++ masks by case and annotator.

    Returns ``(image_name, [seg_names])`` tuples for every case segmented by at
    least ``min_raters`` distinct annotators, sorted by annotator count
    descending (most-annotated first) so the notebook can grab the richest case
    without loading anything. Masks within a case are ordered by annotator code.

    An annotator who segmented the same image more than once (different trial or
    superpixel-scale settings) contributes a single mask, the first by filename
    ; so the rater count is a count of people, not of files.
    """
    # Map ISIC id -> {annotator code -> chosen seg filename}.
    per_case = {}
    for p in sorted(glob.glob(os.path.join(segs_dir, "*.png"))):
        name = os.path.basename(p)
        m = _IMA_SEG_RE.match(name)
        if not m:
            continue
        case_id, annotator = m.group(1), m.group(2)
        per_case.setdefault(case_id, {}).setdefault(annotator, name)

    # Resolve the image filename for each case (extension case varies).
    def image_for(case_id: str):
        for ext in (".JPG", ".jpg", ".jpeg", ".png"):
            cand = os.path.join(images_dir, case_id + ext)
            if os.path.exists(cand):
                return case_id + ext
        return None

    out = []
    for case_id, ann_to_seg in per_case.items():
        if len(ann_to_seg) < min_raters:
            continue
        image_name = image_for(case_id)
        if image_name is None:
            continue
        seg_names = [ann_to_seg[a] for a in sorted(ann_to_seg)]
        out.append((image_name, seg_names))

    out.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    return out


def rank_cases_by_disagreement(cases: list) -> list:
    """Order loaded 2D cases from most to least inter-rater disagreement.

    Args:
        cases: a list of dicts from `load_case` (2D only).

    Returns:
        A list of tuples, each containing `(index, score)`.

    Disagreement is `1 - avg_pairwise_dice` over the rater masks, so a case every rater
    outlines identically scores 0. Returns `(index, score)` pairs, highest
    score first; the notebook picks `[0]` for its worked example.

    This plays the role that slice selection would for 3D data: it chooses the
    case that best shows why consensus is non-trivial.
    """
    from .metrics import avg_pairwise_dice

    scored = []
    for i, case in enumerate(cases):
        masks = case["masks"]
        if masks.ndim != 3:
            raise ValueError(
                f"case {i}: expected 2D (K, H, W) masks, got {masks.shape}"
            )
        scored.append((i, 1.0 - avg_pairwise_dice(masks)))
    scored.sort(key=lambda pair: -pair[1])

    return scored
