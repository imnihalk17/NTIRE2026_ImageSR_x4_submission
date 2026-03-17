# NTIRE 2026 Challenge on Image Super-Resolution (x4) Team11 VAI-GM

Official submission repository for Team11 (VAI-GM).

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

     - You can use `--valid_dir`, or `--test_dir`, or both.

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

## License and acknowledgement

This repository is released under the [MIT License](LICENSE).
