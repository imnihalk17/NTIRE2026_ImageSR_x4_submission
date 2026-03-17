# NTIRE 2026 Challenge on Image Super-Resolution (x4) Team11 VAI-GM

This repository is the official reproducibility package for Team11 (VAI-GM) in the NTIRE 2026 Image Super-Resolution x4 challenge.

## Method summary

- **Team**: VAI-GM (Team ID: 11)
- **Task**: Image Super-Resolution (x4)
- **Backbone**: PFT-SR
- **Training strategy**: fine-tune only the last 2 transformer blocks/layers (+ SR tail), freeze the rest
- **Training data**: Flickr2K
- **Training iterations**: 25,000
- **Inference**: single checkpoint, single-model inference

## How to test the submitted model

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

3. Place checkpoint at:

    ```text
    ./model_zoo/team11_vaigm_last2/team11_vaigm_last2.pth
    ```

    If the checkpoint is too large for direct hosting, put a download link in:

    ```text
    ./model_zoo/team11_vaigm_last2/team11_vaigm_last2.txt
    ```

4. Run inference using `test.py`:

    - Test set only:
      ```bash
      python test.py --test_dir /path/to/test/LQ --save_dir /path/to/output --model_id 0
      ```

    - Validation set only:
      ```bash
      python test.py --valid_dir /path/to/valid/LQ --save_dir /path/to/output --model_id 0
      ```

    - Both validation and test:
      ```bash
      python test.py --valid_dir /path/to/valid/LQ --test_dir /path/to/test/LQ --save_dir /path/to/output --model_id 0
      ```

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
│   └── team11_vaigm_last2/
│       ├── __init__.py
│       └── io.py
└── model_zoo/
     └── team11_vaigm_last2/
          ├── team11_vaigm_last2.pth
          └── team11_vaigm_last2.txt
```

Expected inference outputs:

```text
<save_dir>/11_vaigm_last2/valid/
<save_dir>/11_vaigm_last2/test/
```

Input images can be PNG/JPG/JPEG and output filenames are preserved.

## How to add/replace a model in this baseline format

> Submissions that do not follow the official structure may be rejected.

1. Add model code in:

    ```text
    ./models/[teamID_modelname]/
    ```

2. Add model checkpoint (or link file) in:

    ```text
    ./model_zoo/[teamID_modelname]/
    ```

3. Implement a callable entrypoint:

    ```python
    def main(model_dir, input_path, output_path, device):
         ...
    ```

    Your `main` function must accept exactly these 4 arguments:
    - `model_dir`: checkpoint path
    - `input_path`: directory containing LQ images
    - `output_path`: directory to save restored images
    - `device`: computation device

4. Register the model in `test.py` by adding a `model_id` branch inside `select_model()`.

## Reproducibility notes

- This package corresponds to the final Team11 run using the **last-2-blocks fine-tuning** setup.
- The architecture loaded in `models/team11_vaigm_last2/io.py` is the official PFT x4 variant used in the run:
  - `depths = [4, 4, 4, 6, 6, 6]`
  - `num_heads = 6`
  - 30-element `num_topk` schedule
- Checkpoint loading supports common keys: `params_ema`, `params`, `state_dict`, or raw state dict.

## Submission checklist

- Confirm `test.py` runs directly from repo root.
- Confirm checkpoint exists at the expected path (or link file is valid).
- Confirm output images are generated under `<save_dir>/11_vaigm_last2/{valid|test}`.
- Do not include datasets or temporary result folders in the final package.
- Provide organizers with the clone command:

  ```bash
  git clone https://github.com/imnihalk17/NTIRE2026_ImageSR_x4_submission.git
  ```

## License and acknowledgement

This repository is released under the [MIT License](LICENSE).

Base implementation relies on the PFT-SR / BasicSR codebase components included in this submission package.
