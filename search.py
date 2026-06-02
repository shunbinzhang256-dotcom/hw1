from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from train import run_training


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_search(config: dict) -> list[dict]:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = _parse_float_list(config["learning_rates"])
    hidden_dims = _parse_int_list(config["hidden_dims"])
    weight_decays = _parse_float_list(config["weight_decays"])
    activations = _parse_str_list(config["activations"])

    combinations = list(itertools.product(learning_rates, hidden_dims, weight_decays, activations))
    if config["max_combinations"]:
        combinations = combinations[: config["max_combinations"]]

    results: list[dict] = []
    for index, (lr, hidden_dim, weight_decay, activation) in enumerate(combinations, start=1):
        run_name = f"run_{index:03d}_lr{lr:g}_h{hidden_dim}_wd{weight_decay:g}_{activation}"
        run_config = {
            "data_root": config["data_root"],
            "out_dir": str(out_dir / run_name),
            "split_file": config.get("split_file"),
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "hidden_dim": hidden_dim,
            "activation": activation,
            "lr": lr,
            "lr_decay": config["lr_decay"],
            "lr_decay_every": config["lr_decay_every"],
            "weight_decay": weight_decay,
            "dropout": config["dropout"],
            "val_ratio": config["val_ratio"],
            "test_ratio": config["test_ratio"],
            "image_size": config["image_size"],
            "seed": config["seed"],
            "max_per_class": config["max_per_class"],
            "verbose": config["verbose"],
            "backend": config["backend"],
            "device": config["device"],
            "cache_data": config["cache_data"],
            "normalize": config["normalize"],
            "augment": config["augment"],
            "aug_hflip": config["aug_hflip"],
            "aug_vflip": config["aug_vflip"],
            "aug_rot90": config["aug_rot90"],
            "aug_brightness": config["aug_brightness"],
            "aug_contrast": config["aug_contrast"],
        }
        print(f"[{index}/{len(combinations)}] {run_name}")
        result = run_training(run_config)
        row = {
            "run": run_name,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "activation": activation,
            "best_epoch": result["best_epoch"],
            "best_val_accuracy": result["best_val_accuracy"],
            "out_dir": result["out_dir"],
        }
        results.append(row)

        with (out_dir / "search_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        (out_dir / "search_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    results.sort(key=lambda item: item["best_val_accuracy"], reverse=True)
    (out_dir / "search_results_sorted.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("best:", results[0] if results else None)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search for the NumPy MLP.")
    parser.add_argument("--data-root", default="/root/hw1/EuroSAT_RGB")
    parser.add_argument("--out-dir", default="/root/hw1_solution/outputs/search")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rates", default="0.1,0.05,0.01")
    parser.add_argument("--hidden-dims", default="128,256,512")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activations", default="relu,tanh")
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--lr-decay-every", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-data", choices=["auto", "ram", "none"], default="auto")
    parser.add_argument("--normalize", choices=["none", "channel", "feature"], default="none")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-aug-hflip", action="store_false", dest="aug_hflip")
    parser.add_argument("--no-aug-vflip", action="store_false", dest="aug_vflip")
    parser.add_argument("--no-aug-rot90", action="store_false", dest="aug_rot90")
    parser.add_argument("--aug-brightness", type=float, default=0.0)
    parser.add_argument("--aug-contrast", type=float, default=0.0)
    parser.set_defaults(aug_hflip=True, aug_vflip=True, aug_rot90=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    config = vars(parser.parse_args())
    config["verbose"] = not config.pop("quiet")
    run_search(config)


if __name__ == "__main__":
    main()
