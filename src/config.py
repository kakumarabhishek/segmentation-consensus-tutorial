"""
Worked-example case selection for the tutorial.

The two cases below were chosen by ranking every case by inter-rater
disagreement (1 - avg pairwise Dice) and are fixed here so the notebook, the
figure scripts, and the widget baker all use the same data.

Paths are given relative to a data root so callers can point at their own copy.
"""

# QUBIQ brain-growth (7 raters, 256x256, already 2D on disk).
# case21 has the highest inter-rater disagreement of the 34 training cases (0.219).
QUBIQ_CASE = {
    "name": "case21",
    "rel_dir": "qubiq2021/brain-growth/Training/case21",
    "image_name": "image.nii.gz",
    "seg_names": [f"task01_seg{k:02d}.nii.gz" for k in range(1, 8)],
    "modality": "nifti",
}

# IMA++ dermoscopy (5 raters; most in the dataset; disagreement 0.224).
IMA_CASE = {
    "name": "ISIC_0010183",
    "images_rel_dir": "imaplusplus/images",
    "segs_rel_dir": "imaplusplus/segs",
    "image_name": "ISIC_0010183.JPG",
    "seg_names": [
        "ISIC_0010183_A02_T3_S1_57c054699fc3c158f2bd0e5e.png",
        "ISIC_0010183_A03_T1_S1_57be14599fc3c1434763a353.png",
        "ISIC_0010183_A04_T3_S2_55a9384a9fc3c156bd715c1b.png",
        "ISIC_0010183_A06_T1_S1_57b712089fc3c1565f9bf376.png",
        "ISIC_0010183_A11_T1_S1_55d4fa989fc3c1490e1f6607.png",
    ],
    "modality": "image",
}

# Goal: "draw your own 3rd rater" widget (IMA++, exactly 2 raters so the reader is
# the 3rd). ISIC_0000432 is a faint, diffuse lesion whose edge fades gradually into
# normal skin; the two experts drew in the SAME smooth style but their contours CROSS
# rather than nest, so each claims a stretch of border the other omits (areas 28.5%
# A04 / 24.1% A06; pairwise Dice 0.77; ~26% / ~12% of the union is exclusive to one
# rater). This two-directional, non-subset disagreement is a better teaching demo than
# a nested pair (the earlier ISIC_0014129, where one seg was almost a subset of the
# other), and it still avoids the smooth-vs-jagged tracing-style confound.
DRAW_CASE = {
    "name": "ISIC_0000432",
    "images_rel_dir": "imaplusplus/images",
    "segs_rel_dir": "imaplusplus/segs",
    "image_name": "ISIC_0000432.JPG",
    "seg_names": [
        "ISIC_0000432_A04_T1_S2_5451505bbae47821f8801fbc.png",  # R1
        "ISIC_0000432_A06_T1_S1_54cda622bae47819d8e4cd8c.png",  # R2
    ],
    "modality": "image",
}

# Extra "problem illustration" cases shown ONLY in the first vote-map widget, to add
# more modalities to the opening picture of inter-rater disagreement. They are NOT part
# of the method analysis (which stays on QUBIQ brain-growth + IMA++). Both are 2D on
# disk (kidney 497x497; prostate 960x640x1, squeezed). Cases were picked as the
# highest-disagreement Training case with no empty rater.

# QUBIQ kidney (CT, 3 raters, task01). case11: disagreement 0.088.
KIDNEY_CASE = {
    "name": "case11",
    "rel_dir": "qubiq2021/kidney/Training/case11",
    "image_name": "image.nii.gz",
    "seg_names": [f"task01_seg{k:02d}.nii.gz" for k in range(1, 4)],
    "modality": "nifti",
    "label": "kidney (CT, 3 raters)",
}

# QUBIQ prostate (MRI, 6 raters). task02 (disagreement 0.202 on case24) is used over
# task01 (~0.10) because it shows the disagreement band far more clearly.
PROSTATE_CASE = {
    "name": "case24",
    "rel_dir": "qubiq2021/prostate/Training/case24",
    "image_name": "image.nii.gz",
    "seg_names": [f"task02_seg{k:02d}.nii.gz" for k in range(1, 7)],
    "modality": "nifti",
    "label": "prostate (MRI, 6 raters)",
}
