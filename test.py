import argparse
import os
import torch


def select_model():
    from models import main as pft_main
    model_func = pft_main
    model_name = "PFT_SR_finetuned_VAIGM"
    model_path = os.path.join("model_zoo", f"{model_name}.pth")
    model_link_path = os.path.join("model_zoo", f"{model_name}.txt")
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
