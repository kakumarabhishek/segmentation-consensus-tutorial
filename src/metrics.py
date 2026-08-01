"""
Basic consensus operators and overlap metrics.

Everything here works on a stack of rater masks with the rater axis first:
`(K, H, W)` for 2D cases and `(K, D, H, W)` for 3D volumes. The spatial
rank never matters; all reductions are over axis 0, so the same functions
serve the QUBIQ volumes and the IMA++ images.
"""

from itertools import combinations

import numpy as np

__all__ = [
    "vote_map",
    "majority_vote",
    "mask_average",
    "dice",
    "dice_distance",
    "avg_pairwise_dice",
]


def _as_mask_stack(masks: np.ndarray) -> np.ndarray:
    """
    Validate a rater-first mask stack and return it as uint8 {0, 1}.

    Args:
        masks: the mask stack to validate.

    Returns:
        The mask stack as uint8 {0, 1}.
    """
    masks = np.asarray(masks)
    if masks.ndim < 2:
        raise ValueError(
            f"masks must have a rater axis plus >=1 spatial axis, got shape {masks.shape}"
        )
    if masks.shape[0] < 1:
        raise ValueError("masks must contain at least one rater")

    return (masks > 0).astype(np.uint8)


def vote_map(masks: np.ndarray) -> np.ndarray:
    """
    Per-voxel count of raters labelling it foreground.

    Args:
        masks: the mask stack to validate.

    Returns:
        The vote map as int, values in 0..K.
    """
    return _as_mask_stack(masks).sum(axis=0).astype(np.int64)


def majority_vote(masks: np.ndarray) -> np.ndarray:
    """
    Hard majority-vote consensus.

    Args:
        masks: the mask stack to validate.

    Returns:
        The majority vote as uint8 {0, 1}.

    Tie-breaking: foreground wins. A voxel is foreground when at least K/2
    raters selected it (`>= K/2`, not `> K/2`), so with an even K = 6 a
    3-3 split resolves to foreground. This is the inclusive convention; it
    biases the consensus slightly towards larger structures, which matters
    when raters are few and the target is small.
    """
    masks = _as_mask_stack(masks)
    k = masks.shape[0]
    return (masks.sum(axis=0) >= (k / 2.0)).astype(np.uint8)


def mask_average(masks: np.ndarray) -> np.ndarray:
    """
    Soft consensus: the per-voxel fraction of raters voting foreground.

    Args:
        masks: the mask stack to validate.

    Returns:
        The mask average as float32 in [0, 1].
    """
    return (
        _as_mask_stack(masks).mean(axis=0, dtype=np.float64).astype(np.float32)
    )


def dice(A: np.ndarray, B: np.ndarray) -> float:
    """
    Dice similarity coefficient between two binary arrays of equal shape.

    Args:
        A: the first binary array.
        B: the second binary array.

    Returns:
        The Dice similarity coefficient as float in [0, 1].

    Returns 0.0 when both arrays are empty. (The other common convention is
    1.0 -- two empty masks do agree perfectly -- but 0.0 is the choice made
    throughout this tutorial, so an empty-vs-empty pair never inflates a mean
    agreement score.)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    A = A > 0
    B = B > 0
    denom = A.sum() + B.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(A, B).sum() / denom)


def dice_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    1 - dice(A, B).

    Args:
        A: the first binary array.
        B: the second binary array.

    Returns:
        The Dice distance as float in [0, 1].
    """
    return 1.0 - dice(A, B)


def avg_pairwise_dice(masks: np.ndarray) -> float:
    """
    Mean Dice over all unique rater pairs -- a scalar summary of agreement.

    Args:
        masks: the mask stack to validate.

    Returns:
        The average Dice score as float in [0, 1].

    masks: (K, ...) -> float. Requires K >= 2.
    """
    masks = _as_mask_stack(masks)
    k = masks.shape[0]
    if k < 2:
        raise ValueError("avg_pairwise_dice needs at least two raters")
    scores = [dice(masks[i], masks[j]) for i, j in combinations(range(k), 2)]

    return float(np.mean(scores))
