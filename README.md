# NTIRE 2026 ImageSR x4 - VAI-GM

This repository is prepared for NTIRE 2026 organizer-side reproducibility.
Base model used: PFT-SR.

## 1) Method summary

- Team: VAI-GM (Team ID: 11)
- Task: Image Super-Resolution x4
- Training strategy: fine-tune only the last 2 layers/blocks (+ SR tail)
- Training data: Flickr2K
- Training iterations: 25,000
- Inference: single model checkpoint

## 2) Repository layout

```text
NTIRE2026_ImageSR_x4_submission/
├── test.py
├── requirements.txt
├── README.md
├── setup.py
├── VERSION
├── basicsr/
├── ops_smm/
├── models/
│   └── team11_vaigm_last2/
│       ├── __init__.py
│       └── io.py
└── model_zoo/
    └── team11_vaigm_last2/
        ├── team11_vaigm_last2.pth
        └── team11_vaigm_last2.txt
```

## 3) Environment setup

Recommended: Python 3.9+ with CUDA-capable GPU.

```bash
pip install -r requirements.txt
cd ops_smm
python setup.py install
cd ..
```

## 4) Checkpoint placement

Place final checkpoint at:

```text
model_zoo/team11_vaigm_last2/team11_vaigm_last2.pth
```

If checkpoint size exceeds organizer limit, put a public download link in:

```text
model_zoo/team11_vaigm_last2/team11_vaigm_last2.txt
```

## 5) Run commands

Test set:

```bash
python test.py --test_dir /path/to/LQ --save_dir /path/to/output --model_id 0
```

Validation set:

```bash
python test.py --valid_dir /path/to/LQ --save_dir /path/to/output --model_id 0
```

Both:

```bash
python test.py --valid_dir /path/to/val_LQ --test_dir /path/to/test_LQ --save_dir /path/to/output --model_id 0
```

## 6) Input/output conventions

- Input: LR images folder (PNG/JPG)
- Output folder:

```text
<save_dir>/11_vaigm_last2/test/
<save_dir>/11_vaigm_last2/valid/
```

Output images keep the same filenames as input images.

## 7) Colab alignment

This package matches the project notebook pipeline:
- `finetune_last_n_blocks = 2`
- final model corresponds to the 25,000-iteration run
- Flickr2K training setup

## 8) Final checklist before submission

- Ensure checkpoint exists (or link provided)
- Ensure commands run in terminal as-is
- Do not include datasets or generated result images in this repo
- Keep this repository as the single code package for organizers
