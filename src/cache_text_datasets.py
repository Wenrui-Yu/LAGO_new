from __future__ import annotations

import argparse

from data import load_text_splits
from utils import parse_csv


DEFAULT_DATASETS = (
    "yywwrr/mmarco_english,"
    "yywwrr/mmarco_french,"
    "yywwrr/mmarco_german,"
    "yywwrr/mmarco_italian,"
    "yywwrr/mmarco_portuguese,"
    "yywwrr/mmarco_spanish,"
    "yywwrr/mmarco_dutch"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Hugging Face text datasets as local txt splits.")
    parser.add_argument("--datasets", type=parse_csv, default=parse_csv(DEFAULT_DATASETS))
    parser.add_argument("--data_folder", default="datasets/finetuning_decoder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in args.datasets:
        splits = load_text_splits(dataset_name, data_folder=args.data_folder)
        print(
            f"{dataset_name}: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()
