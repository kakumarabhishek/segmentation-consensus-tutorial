"""
Soft-STAPLE (Kats et al., MICCAI 2019); a simplified variant.

Two pieces live here:

`generate_soft_labels`  turns a binary rater mask into a soft annotation by
    dilating it and giving the added rim a fractional weight.
`soft_staple`           runs the modified EM of Kats et al. (Eq. 10-13),
    which accepts soft annotations in [0, 1] and stays linear in the number of
    raters.

The EM itself is modality-agnostic; only the soft-label construction makes any
assumption about the images, and the one used here (dilation only) makes none.
"""

import numpy as np
from scipy.special import logsumexp
from skimage.morphology import binary_dilation, disk

__all__ = ["generate_soft_labels", "soft_staple"]

_EPS = 1e-12


def generate_soft_labels(
    binary_mask: np.ndarray,
    dilation_radius: int = 3,
    soft_weight: float = 0.3,
) -> np.ndarray:
    """

    1. Interior voxels (`binary_mask == 1`) get weight 1.0.
    2. The mask is dilated by ``dilation_radius`` using a disk element.
    3. The rim added by dilation (dilated XOR original) gets ``soft_weight``.
    4. Everything else gets 0.0.

    binary_mask: (H, W) uint8 -> (H, W) float32 in [0, 1]

    Note: Kats et al. additionally filtered the rim by FLAIR intensity, keeping
    only hyper-intense voxels, since they worked on MS lesions. We dilate only,
    which makes the construction modality-agnostic, so the same call works for
    T1 MRI and for dermoscopy. The cost is that the rim is uniformly uncertain
    rather than image-aware.
    """
    if not 0.0 <= soft_weight <= 1.0:
        raise ValueError(f"soft_weight must be in [0, 1], got {soft_weight}")
    if dilation_radius < 0:
        raise ValueError(
            f"dilation_radius must be non-negative, got {dilation_radius}"
        )

    core = np.asarray(binary_mask) > 0
    soft = core.astype(np.float32)
    if dilation_radius == 0 or not core.any():
        # radius 0 reproduces the binary input exactly, which is what the
        # notebook's "soft-STAPLE with binary inputs" equivalence check needs.
        return soft

    dilated = binary_dilation(core, disk(dilation_radius))
    soft[np.logical_xor(dilated, core)] = soft_weight

    return soft


def soft_staple(
    soft_annotations: np.ndarray,
    prior: float = None,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """
    Simplified soft-STAPLE EM (Kats et al., MICCAI 2019).

    soft_annotations: (K, ...) float in [0, 1], rater axis first.

    E-step (tractable, linear in K):
        p(z_it | x_t=1) = q_it * th_i1 + (1 - q_it) * (1 - th_i1)
        p(z_it | x_t=0) = q_it * (1 - th_i0) + (1 - q_it) * th_i0
        w_t(1) = prior * prod_i p(z_it|1)
                 / [prior * prod_i p(z_it|1) + (1-prior) * prod_i p(z_it|0)]

    M-step (identical to standard STAPLE):
        th_i1 = sum_t q_it * w_t(1) / sum_t w_t(1)
        th_i0 = sum_t (1 - q_it) * w_t(0) / sum_t w_t(0)

    The products over K raters underflow quickly in float, so both are
    accumulated in log space and normalised with ``logsumexp``.

    Returns:
        soft_consensus: (...) float32 -- the converged w_t(1) map
        sensitivities:  (K,) float    -- th_i1
        specificities:  (K,) float    -- th_i0
    """
    q = np.asarray(soft_annotations, dtype=np.float64)
    if q.ndim < 2:
        raise ValueError(
            f"soft_annotations needs a rater axis plus spatial axes, got {q.shape}"
        )
    if q.min() < 0.0 or q.max() > 1.0:
        raise ValueError("soft_annotations must lie in [0, 1]")

    spatial_shape = q.shape[1:]
    k = q.shape[0]
    q = q.reshape(k, -1)  # (K, N)

    if prior is None:
        prior = float(q.mean())
    prior = float(np.clip(prior, _EPS, 1.0 - _EPS))
    log_prior = np.log(prior)
    log_prior_c = np.log1p(-prior)

    # Same initialisation as standard STAPLE: assume every rater is very good.
    theta1 = np.full(k, 0.99)
    theta0 = np.full(k, 0.99)

    w1 = None
    for _ in range(max_iter):
        # --- E-step, in log space ---
        p1 = q * theta1[:, None] + (1.0 - q) * (1.0 - theta1[:, None])
        p0 = q * (1.0 - theta0[:, None]) + (1.0 - q) * theta0[:, None]
        log_num1 = log_prior + np.log(np.clip(p1, _EPS, None)).sum(axis=0)
        log_num0 = log_prior_c + np.log(np.clip(p0, _EPS, None)).sum(axis=0)
        w1 = np.exp(
            log_num1 - logsumexp(np.stack([log_num1, log_num0]), axis=0)
        )
        w0 = 1.0 - w1

        # --- M-step ---
        sum_w1 = w1.sum()
        sum_w0 = w0.sum()
        new_theta1 = (q * w1).sum(axis=1) / max(sum_w1, _EPS)
        new_theta0 = ((1.0 - q) * w0).sum(axis=1) / max(sum_w0, _EPS)
        # Keep parameters off the 0/1 boundary so the next log stays finite.
        new_theta1 = np.clip(new_theta1, _EPS, 1.0 - _EPS)
        new_theta0 = np.clip(new_theta0, _EPS, 1.0 - _EPS)

        delta = max(
            np.abs(new_theta1 - theta1).max(),
            np.abs(new_theta0 - theta0).max(),
        )
        theta1, theta0 = new_theta1, new_theta0
        if delta < tol:
            break

    return (
        w1.reshape(spatial_shape).astype(np.float32),
        theta1.copy(),
        theta0.copy(),
    )
