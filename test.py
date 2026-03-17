import argparse
import os
import torch
from glob import glob


def select_model():
    from models import main as pft_main
    model_func = pft_main

    model_zoo = "model_zoo"
    pth_files = sorted(glob(os.path.join(model_zoo, "*.pth")))
    txt_files = sorted(glob(os.path.join(model_zoo, "*.txt")))

    if len(pth_files) > 1:
        raise RuntimeError(
            "Multiple checkpoints found in model_zoo. Keep only one .pth file for inference."
        )

    if len(pth_files) == 1:
        model_path = pth_files[0]
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_link_path = os.path.join(model_zoo, f"{model_name}.txt")
    else:
        if len(txt_files) == 0:
            raise FileNotFoundError(
                "No checkpoint (.pth) or link (.txt) found in model_zoo."
            )
        if len(txt_files) > 1:
            raise RuntimeError(
                "Multiple .txt link files found in model_zoo. Keep only one link file."
            )
        model_link_path = txt_files[0]
        model_name = os.path.splitext(os.path.basename(model_link_path))[0]
        model_path = os.path.join(model_zoo, f"{model_name}.pth")

    return model_func, model_name, model_path, model_link_path


def run(model_func, model_path, model_link_path, args):
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    if not os.path.isfile(model_path):
        if os.path.isfile(model_link_path):
            with open(model_link_path, "r", encoding="utf-8") as f:
                model_link = f.read().strip()
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}\n"
                f"Download the checkpoint from: {model_link}"
            )
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_func(model_dir=model_path, input_path=args.input_dir, output_path=save_path, device=device)
    print(f"Saved outputs: {save_path}")


def main(args):
    model_func, _, model_path, model_link_path = select_model()
    run(model_func, model_path, model_link_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-ImageSRx4")
    parser.add_argument("--input_dir", required=True, type=str, help="Path to input LR image folder")
    parser.add_argument("--output_dir", default="results", type=str, help="Path to output SR image folder")

    args = parser.parse_args()
    main(args)
