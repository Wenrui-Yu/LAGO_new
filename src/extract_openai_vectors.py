from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from data import sample_splits
from utils import fill_special_tokens, parse_csv


DEFAULT_DATASETS = (
    "yywwrr/mmarco_english,"
    "yywwrr/mmarco_french,"
    "yywwrr/mmarco_german,"
    "yywwrr/mmarco_italian,"
    "yywwrr/mmarco_portuguese,"
    "yywwrr/mmarco_spanish,"
    "yywwrr/mmarco_dutch"
)


def slug(name: str) -> str:
    return name.replace("/", "_")


def canonicalize_texts(texts: List[str], tokenizer, max_length: int) -> List[str]:
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    decoded = tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def embed_batch_with_retry(
    client,
    texts: List[str],
    model: str,
    max_retries: int,
    retry_min_sleep: float,
    retry_max_sleep: float,
) -> np.ndarray:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                encoding_format="float",
            )
            vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
            return np.stack(vectors, axis=0)
        except Exception as exc:  # OpenAI errors vary across client versions.
            last_error = exc
            message = str(exc)
            if "insufficient_quota" in message or "invalid_api_key" in message:
                raise RuntimeError(f"OpenAI embedding request cannot continue: {exc}") from exc
            sleep_s = min(retry_max_sleep, retry_min_sleep * (2**attempt))
            print(f"OpenAI embedding request failed on attempt {attempt + 1}: {exc}")
            if attempt + 1 < max_retries:
                print(f"Retrying in {sleep_s:.1f}s")
                time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI embedding request failed after {max_retries} attempts") from last_error


def embed_texts(
    client,
    texts: List[str],
    model: str,
    batch_size: int,
    max_retries: int,
    retry_min_sleep: float,
    retry_max_sleep: float,
) -> np.ndarray:
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {model}"):
        batch = texts[start : start + batch_size]
        chunks.append(
            embed_batch_with_retry(
                client,
                batch,
                model,
                max_retries=max_retries,
                retry_min_sleep=retry_min_sleep,
                retry_max_sleep=retry_max_sleep,
            )
        )
    return np.concatenate(chunks, axis=0)


def output_path(vector_root: str, openai_model: str, target_model: str, dataset_name: str, max_length: int) -> str:
    return os.path.join(
        vector_root,
        slug(openai_model),
        slug(target_model),
        slug(dataset_name),
        f"vecs_maxlength{max_length}.npz",
    )


def extract_dataset(client, tokenizer, dataset_name: str, args) -> None:
    path = output_path(
        args.output_dir,
        args.openai_model,
        args.target_model,
        dataset_name,
        args.max_length,
    )
    if os.path.exists(path) and not args.overwrite:
        print(f"Skipping existing vectors: {path}")
        return

    splits = sample_splits(
        dataset_name,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        data_folder=args.data_folder,
    )
    train_texts = canonicalize_texts(splits["train"], tokenizer, args.max_length)
    val_texts = canonicalize_texts(splits["validation"], tokenizer, args.max_length)
    test_texts = canonicalize_texts(splits["test"], tokenizer, args.max_length)

    print(
        f"Extracting {args.openai_model} vectors for {dataset_name}: "
        f"train={len(train_texts)}, dev={len(val_texts)}, test={len(test_texts)}"
    )
    train_vectors = embed_texts(
        client,
        train_texts,
        args.openai_model,
        args.batch_size,
        args.max_retries,
        args.retry_min_sleep,
        args.retry_max_sleep,
    )
    val_vectors = embed_texts(
        client,
        val_texts,
        args.openai_model,
        args.batch_size,
        args.max_retries,
        args.retry_min_sleep,
        args.retry_max_sleep,
    )
    test_vectors = embed_texts(
        client,
        test_texts,
        args.openai_model,
        args.batch_size,
        args.max_retries,
        args.retry_min_sleep,
        args.retry_max_sleep,
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, train=train_vectors, dev=val_vectors, test=test_vectors)
    print(
        f"Saved {path}: train={train_vectors.shape}, "
        f"dev={val_vectors.shape}, test={test_vectors.shape}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract OpenAI victim embeddings for LAGO_new ablations.")
    parser.add_argument("--openai_model", default="text-embedding-ada-002")
    parser.add_argument("--target_model", default="google/mt5-small")
    parser.add_argument("--datasets", type=parse_csv, default=parse_csv(DEFAULT_DATASETS))
    parser.add_argument("--output_dir", default="datasets/vectors")
    parser.add_argument("--data_folder", default="datasets/finetuning_decoder")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--retry_min_sleep", type=float, default=1.0)
    parser.add_argument("--retry_max_sleep", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Install requirements.txt or add openai to the runtime image."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=False)
    fill_special_tokens(tokenizer)
    client = OpenAI()

    for dataset_name in args.datasets:
        extract_dataset(client, tokenizer, dataset_name, args)


if __name__ == "__main__":
    main()
