# Segmentation Consensus Tutorial

Code for the MICCAI Educational Challenge 2026 submission: **"Many Raters, One Mask? A Practical Guide to Consensus Aggregation for Multi-Annotator Medical Image Segmentation."**

The tutorial walks through five families of consensus-aggregation methods for combining multiple expert segmentations into a single mask, applied to two multi-annotator datasets (QUBIQ 2021 brain-growth MRI and IMA++ dermoscopy):

1. **Majority Voting (MV)** and **Mask Averaging (MA)**.
2. **[STAPLE](https://doi.org/10.1109/TMI.2004.828354)**: Warfield et al., IEEE TMI 2004.
3. **[SIMPLE](https://doi.org/10.1109/TMI.2010.2057442)**: Langerak et al., IEEE TMI 2010
4. **[MACCHIatO-D](https://doi.org/10.59275/j.melba.2023-219c)**: Hamzaoui et al., MELBA 2023.
5. **[Soft-STAPLE](https://doi.org/10.1007/978-3-030-32248-9_57)**: Kats et al., MICCAI 2019.

Everything runs on CPU in a couple of minutes.

## Repository layout

```
.
├── src/
│   ├── config.py           # The two worked-example cases used in the notebook
│   ├── data_loading.py     # Load QUBIQ / IMA++ cases into a common dict; case selection
│   ├── metrics.py          # Vote map, majority vote, mask average, Dice, pairwise Dice
│   └── soft_staple.py      # Soft-label construction + simplified soft-STAPLE EM
├── notebooks/
│   └── consensus_tutorial.ipynb  # Runs every method, produces every figure
└── requirements.txt        # Pinned, tested dependencies (Python 3.10)
```

STAPLE, SIMPLE, and MACCHIatO are called through existing libraries: [SimpleITK](https://simpleitk.readthedocs.io/en/master/gettingStarted.html), [FeTS `LabelFusion`](https://github.com/FeTS-AI/LabelFusion), and [`macchiato`](https://gitlab.inria.fr/dhamzaou/jaccardmap) directly inside the notebook, so they must be installed before running the notebook (see below). Other simple operators, soft-STAPLE, and utilities are implemented in `src/`.

## Data

The datasets are **not** redistributed here. To run the notebook, please obtain your own local copy of [QUBIQ 2021](https://qubiq21.grand-challenge.org/) (brain-growth task) and [IMA++](https://doi.org/10.5281/zenodo.14201692), and lay them out as:

<!-- ```
<data-root>/
  qubiq2021/brain-growth/Training/caseNN/{image.nii.gz, task01_segKK.nii.gz}
  imaplusplus/images/ISIC_*.JPG
  imaplusplus/segs/ISIC_*_A??_T?_S?_*.png
``` -->
```
<data-root>/
├── qubiq2021/
│   └── brain-growth/
│       └── Training/
│           └── case<xx>/
│               ├── image.nii.gz
│               └── task01_seg<yy>.nii.gz
└── imaplusplus/
    ├── images/
    │   └── ISIC_<xxxxxxx>.JPG
    └── segs/
        └── ISIC_<xxxxxxx>_A<aa>_T<tt>_S<ss>.png
```

Then point the notebook at it:

```bash
export CONSENSUS_DATA_ROOT=/path/to/data-root
jupyter nbconvert --to notebook --execute --inplace notebooks/consensus_tutorial.ipynb
```

The worked-example cases (`case21` for QUBIQ, `ISIC_0010183` for IMA++) are specified in `src/config.py`; edit that file to use different cases.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

I use a hybrid `conda` + `uv` setup as described [here](https://pixels-and-predictions.pages.dev/posts/notes_uv_ruff/#my-workflow-a-hybrid-conda--uv-approach), but you can use whatever workflow you prefer.

The `macchiato` package installs `connected-components-3d`, so make sure to install `connected-components-3d` explicitly if MACCHIatO import fails.

<!-- ## Citation

If you use this code in your research, please cite our MICCAI Educational Challenge 2026 submission. -->

<!-- ```bibtex
@article{abhishek2026many,
  title={Many Raters, One Mask? A Practical Guide to Consensus Aggregation for Multi-Annotator Medical Image Segmentation},
  author={Kumar Abhishek},
  year={2026}
} -->
