# [NTIRE 2026 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

Team submission repository for Team11 (VAI-GM).

## Notice

This repository follows the official NTIRE 2026 Image SR x4 submission format.

## Team and Method

- Team name: VAI-GM
- Team ID: 11
- Codabench username: `nihalk17`
- Model name: PFT-SR
- Fine-tuning: last 2 transformer blocks + reconstruction tail
- Training data: Flickr2K
- Iterations: 25,000

## How to test our model?

1. Clone the repository:

  ```bash
  git clone https://github.com/imnihalk17/NTIRE2026_ImageSR_x4_submission.git
  cd NTIRE2026_ImageSR_x4_submission
  ```

2. Install dependencies:

  ```bash
  pip install -r requirements.txt
  cd utils
  python setup.py install
  cd ..
  ```

3. Download checkpoint:

  Download link file: `./model_zoo/team11_pft_sr/team11_pft_sr.txt`
  Put one `.pth` checkpoint under `./model_zoo/team11_pft_sr/`

4. Run inference:

  ```bash
  python test.py --input_dir /path/to/LQ --output_dir /path/to/output
  ```

The script auto-detects one checkpoint in `model_zoo/**`. If no checkpoint is found, it reports the link from the `.txt` file.

## Folder Structure

```text
NTIRE2026_ImageSR_x4_submission/
├── factsheet/
│   └── NTIRE2026_Image_Super_Resolution_X4_factsheet.pdf
├── model_zoo/
│   └── team11_pft_sr/
│       ├── PFT_SR_finetuned_VAIGM.pth
│       └── team11_pft_sr.txt
├── models/
│   ├── __init__.py
│   └── team11_pft_sr/
│       ├── __init__.py
│       └── io.py
├── basicsr/
├── utils/
├── test.py
├── requirements.txt
├── setup.py
├── VERSION
└── LICENSE
```

## Input and Output Folder Structure

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

## License and Acknowledgement

This repository is released under the [MIT License](LICENSE).
