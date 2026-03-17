# NTIRE 2026 ImageSR x4 - VAI-GM

This repository is prepared for the NTIRE 2026 organizer-side reproduction pipeline.
It follows the official structure and uses a lightweight wrapper over a fresh upstream PFT-SR codebase.

## 1) Model used for this submission

- Team: VAI-GM
- Method: Progressive Focused Transformer (PFT-SR)
- Scale: x4
- Training strategy: fine-tune only the last 2 layers/blocks (+ SR tail)
- Training data: Flickr2K
- Training iterations: 25,000

## Colab alignment (exact project setup)

This package matches the training recipe used in `PFT_Baseline_Colab.ipynb`:

- Training mode: custom `PFTFinetuneModel` that freezes backbone and trains only last layers
- Key knob: `finetune_last_n_blocks = 2`
- Total iterations used for this final model: `25000`
- Training set: `Flickr2K`

At inference time, this package loads the final trained checkpoint and reproduces the submitted outputs.

## 2) Repository layout

```text
NTIRE2026_ImageSR_x4_submission/
├── test.py
├── requirements.txt
├── README.md
├── models/
│   └── team11_vaigm_last2/
│       ├── __init__.py
│       └── io.py
├── model_zoo/
│   └── team11_vaigm_last2/
│       ├── team11_vaigm_last2.pth
│       └── team11_vaigm_last2.txt
└── external/
        └── PFT-SR/
                ├── basicsr/
                ├── ops_smm/
                ├── requirements.txt
                ├── setup.py
                ├── VERSION
                └── ...
```

## 3) Environment and installation

Recommended: Python 3.9+ and CUDA GPU.

```bash
pip install -r requirements.txt
cd external/PFT-SR/ops_smm
python setup.py install
cd ../../..
```

Notes:
- `requirements.txt` in root delegates to `external/PFT-SR/requirements.txt`.
- `ops_smm` build is required for efficient/compatible sparse attention execution.

## 4) Checkpoint placement

Place your final model at:

```text
model_zoo/team11_vaigm_last2/team11_vaigm_last2.pth
```

The loader supports checkpoints containing one of these keys:
- `params_ema`
- `params`
- `state_dict`
- or raw state dict directly

If file size >100MB for organizer submission constraints, put a public download link in:

```text
model_zoo/team11_vaigm_last2/team11_vaigm_last2.txt
```

## 5) Inference command (official-style entrypoint)

Test set:

```bash
python test.py --test_dir /path/to/LQ --save_dir /path/to/output --model_id 0
```

Validation set:

```bash
python test.py --valid_dir /path/to/LQ --save_dir /path/to/output --model_id 0
```

Both together:

```bash
python test.py --valid_dir /path/to/val_LQ --test_dir /path/to/test_LQ --save_dir /path/to/output --model_id 0
```

## 6) Input/output conventions

- Input: LR image folder (PNG/JPG accepted)
- Output: SR images are saved with the same filenames in:

```text
<save_dir>/11_vaigm_last2/test/
```

or

```text
<save_dir>/11_vaigm_last2/valid/
```

## 7) Reproducibility notes

- External code is copied from a fresh upstream PFT-SR clone.
- Wrapper file `models/team11_vaigm_last2/io.py` only handles:
    - model construction
    - checkpoint loading
    - folder-level inference loop
- No training scripts, datasets, or generated outputs are required in this repo.

## 8) Final packaging checklist

- Keep this single folder as the upload source.
- Ensure checkpoint exists (or valid download link provided).
- Remove any local cache/output images before final zip.
- Confirm `python -m py_compile test.py models/team11_vaigm_last2/io.py` passes.
