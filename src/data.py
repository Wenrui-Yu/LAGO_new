from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import datasets
import torch
from torch.utils.data import Dataset


def load_text_splits(
    dataset_name: str,
    data_folder: str = "datasets/finetuning_decoder",
) -> Dict[str, List[str]]:
    if dataset_name.startswith("yiyic/") or dataset_name.startswith("yywwrr/"):
        print(f"Loading Hugging Face dataset {dataset_name}")
        dataset = datasets.load_dataset(dataset_name)
        val_split = "dev" if "dev" in dataset else "validation"
        test_split = "test" if "test" in dataset else val_split
        return {
            "train": list(dataset["train"]["text"]),
            "validation": list(dataset[val_split]["text"]),
            "test": list(dataset[test_split]["text"]),
        }

    dataset_dir = os.path.join(data_folder, dataset_name)
    with open(os.path.join(dataset_dir, "train.txt")) as f:
        train = [line.rstrip("\n") for line in f]
    with open(os.path.join(dataset_dir, "val.txt")) as f:
        validation = [line.rstrip("\n") for line in f]
    with open(os.path.join(dataset_dir, "test.txt")) as f:
        test = [line.rstrip("\n") for line in f]
    return {"train": train, "validation": validation, "test": test}


def sample_splits(
    dataset_name: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    data_folder: str = "datasets/finetuning_decoder",
) -> Dict[str, List[str]]:
    splits = load_text_splits(dataset_name, data_folder=data_folder)
    return {
        "train": splits["train"][:train_samples],
        "validation": splits["validation"][:val_samples],
        "test": splits["test"][:test_samples],
    }


def load_many_for_decoder(
    dataset_names: Sequence[str],
    train_samples: int,
    val_samples: int,
    data_folder: str,
) -> Dict[str, List[str]]:
    train_texts: List[str] = []
    val_texts: List[str] = []
    per_dataset_train = max(1, train_samples // max(1, len(dataset_names)))
    per_dataset_val = max(1, val_samples // max(1, len(dataset_names)))

    for dataset_name in dataset_names:
        splits = sample_splits(
            dataset_name,
            train_samples=per_dataset_train,
            val_samples=per_dataset_val,
            test_samples=1,
            data_folder=data_folder,
        )
        train_texts.extend(splits["train"])
        val_texts.extend(splits["validation"])

    return {
        "train": train_texts[:train_samples],
        "validation": val_texts[:val_samples],
    }


@dataclass
class DecoderBatch:
    hidden_states: torch.Tensor
    labels: torch.Tensor
    references: List[str]


class DecoderTextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class AlignmentDataset:
    def __init__(
        self,
        dataset_name: str,
        lang_key: str,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
    ):
        self.dataset_name = dataset_name
        self.lang_key = lang_key
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
