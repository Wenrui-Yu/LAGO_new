from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import datasets
import torch
from torch.utils.data import Dataset


def dataset_cache_dir(data_folder: str, dataset_name: str) -> str:
    return os.path.join(data_folder, dataset_name.replace("/", "_"))


def read_text_file(path: str) -> List[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def write_text_file(path: str, texts: Sequence[str]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")
    os.replace(tmp_path, path)


def local_split_paths(dataset_name: str, data_folder: str) -> Dict[str, str]:
    dataset_dir = dataset_cache_dir(data_folder, dataset_name)
    return {
        "train": os.path.join(dataset_dir, "train.txt"),
        "validation": os.path.join(dataset_dir, "val.txt"),
        "test": os.path.join(dataset_dir, "test.txt"),
    }


def has_local_text_splits(dataset_name: str, data_folder: str) -> bool:
    paths = local_split_paths(dataset_name, data_folder)
    return all(os.path.exists(path) for path in paths.values())


def load_local_text_splits(dataset_name: str, data_folder: str) -> Dict[str, List[str]]:
    paths = local_split_paths(dataset_name, data_folder)
    print(f"Loading cached text splits from {dataset_cache_dir(data_folder, dataset_name)}")
    return {
        "train": read_text_file(paths["train"]),
        "validation": read_text_file(paths["validation"]),
        "test": read_text_file(paths["test"]),
    }


def save_local_text_splits(
    dataset_name: str,
    data_folder: str,
    splits: Dict[str, Sequence[str]],
) -> None:
    dataset_dir = dataset_cache_dir(data_folder, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    paths = local_split_paths(dataset_name, data_folder)
    write_text_file(paths["train"], splits["train"])
    write_text_file(paths["validation"], splits["validation"])
    write_text_file(paths["test"], splits["test"])
    print(f"Cached text splits to {dataset_dir}")


def load_hf_dataset_with_retry(dataset_name: str):
    retries = int(os.environ.get("HF_DATASET_LOAD_RETRIES", "6"))
    sleep_seconds = int(os.environ.get("HF_DATASET_RETRY_SLEEP", "60"))
    for attempt in range(retries):
        try:
            return datasets.load_dataset(dataset_name)
        except Exception as exc:
            message = str(exc)
            if "429" not in message and "Too Many Requests" not in message:
                raise
            if attempt == retries - 1:
                raise
            delay = sleep_seconds * (attempt + 1)
            print(
                f"Hugging Face rate limit while loading {dataset_name}; "
                f"sleeping {delay}s before retry {attempt + 2}/{retries}."
            )
            time.sleep(delay)
    raise RuntimeError(f"Failed to load {dataset_name}")


def load_text_splits(
    dataset_name: str,
    data_folder: str = "datasets/finetuning_decoder",
) -> Dict[str, List[str]]:
    if has_local_text_splits(dataset_name, data_folder):
        return load_local_text_splits(dataset_name, data_folder)

    if dataset_name.startswith("yiyic/") or dataset_name.startswith("yywwrr/"):
        print(f"Loading Hugging Face dataset {dataset_name}")
        dataset = load_hf_dataset_with_retry(dataset_name)
        val_split = "dev" if "dev" in dataset else "validation"
        test_split = "test" if "test" in dataset else val_split
        splits = {
            "train": list(dataset["train"]["text"]),
            "validation": list(dataset[val_split]["text"]),
            "test": list(dataset[test_split]["text"]),
        }
        save_local_text_splits(dataset_name, data_folder, splits)
        return splits

    dataset_dir = dataset_cache_dir(data_folder, dataset_name)
    train = read_text_file(os.path.join(dataset_dir, "train.txt"))
    validation = read_text_file(os.path.join(dataset_dir, "val.txt"))
    test = read_text_file(os.path.join(dataset_dir, "test.txt"))
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
