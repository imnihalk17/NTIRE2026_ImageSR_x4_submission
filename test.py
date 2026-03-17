import argparse
import os
import torch


def select_model(args):
    model_id = args.model_id
    if model_id == 0:
        from models.team11_vaigm_last2 import main as pft_main
        model_name = "11_vaigm_last2"
        model_func = pft_main
        model_path = os.path.join("model_zoo", "team11_vaigm_last2", "team11_vaigm_last2.pth")
    else:
        raise NotImplementedError(f"Model ID {model_id} is not implemented")
    return model_func, model_name, model_path


def run(model_func, model_name, model_path, args, mode="test"):
    if mode == "valid":
        data_path = args.valid_dir
    else:
        data_path = args.test_dir

    save_path = os.path.join(args.save_dir, model_name, mode)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    print(f"Saved outputs: {save_path}")


def main(args):
    model_func, model_name, model_path = select_model(args)

    if args.valid_dir is not None:
        run(model_func, model_name, model_path, args, mode="valid")

    if args.test_dir is not None:
        run(model_func, model_name, model_path, args, mode="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-ImageSRx4")
    parser.add_argument("--valid_dir", default=None, type=str)
    parser.add_argument("--test_dir", default=None, type=str)
    parser.add_argument("--save_dir", default="results", type=str)
    parser.add_argument("--model_id", default=0, type=int)

    args = parser.parse_args()
    main(args)
