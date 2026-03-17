# NTIRE 2026 Challenge on Image Super-Resolution (x4) Team11 VAI-GM

Official submission repository for Team11 (VAI-GM).

## Method

- Team: VAI-GM (Team ID: 11)
- Task: Image Super-Resolution x4
- Base model: PFT-SR
- Fine-tuning setup: last 2 layers/blocks + reconstruction tail
- Data: Flickr2K
- Iterations: 25,000

## Inference

1. Clone repository:

    ```bash
    git clone https://github.com/imnihalk17/NTIRE2026_ImageSR_x4_submission.git
    cd NTIRE2026_ImageSR_x4_submission
    ```

2. Install environment:

    ```bash
    pip install -r requirements.txt
    cd ops_smm
    python setup.py install
    cd ..
    ```

3. Checkpoint source:

  The checkpoint download link is provided in:

   ```text
   ./model_zoo/PFT_SR_finetuned_VAIGM.txt
   ```

  Download the `.pth` file into `model_zoo/`.
  The inference script auto-detects a single checkpoint in that folder.

  Example checkpoint path:

   ```text
   ./model_zoo/PFT_SR_finetuned_VAIGM.pth
   ```

4. Run single-folder inference:

   ```bash
   python test.py --input_dir /path/to/LQ --output_dir /path/to/output
   ```

  Notes:
  - `--input_dir`: folder with LR images
  - `--output_dir`: folder where SR images are saved
  - Input and output filenames are identical

## Folder structure

```text
NTIRE2026_ImageSR_x4_submission/
├── test.py
├── README.md
├── requirements.txt
├── setup.py
├── VERSION
├── LICENSE
├── basicsr/
├── ops_smm/
├── models/
│   ├── __init__.py
│   └── io.py
└── model_zoo/
  ├── PFT_SR_finetuned_VAIGM.pth
  └── PFT_SR_finetuned_VAIGM.txt
```

## Input and output format

- Input: directory of LR images (PNG/JPG/JPEG)
- Output: directory of restored SR images with original file names

Example:

```text
/path/to/LQ/
├── 0001x4.png
├── 0002x4.png
└── ...

/path/to/output/
├── 0001x4.png
├── 0002x4.png
└── ...
```

## License and acknowledgement

This repository is released under the [MIT License](LICENSE).
