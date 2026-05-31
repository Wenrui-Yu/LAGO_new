from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    jsonable = {}
    for key, value in data.items():
        try:
            json.dumps(value)
            jsonable[key] = value
        except TypeError:
            jsonable[key] = str(value)
    return jsonable


def fill_special_tokens(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return tokenizer


def load_seq2seq_model_and_tokenizer(model_name: str, device: torch.device):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_special_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model, tokenizer


def load_encoder_model_and_tokenizer(model_name: str, device: torch.device):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fill_special_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model, tokenizer


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention_mask = attention_mask.to(hidden_states.device)
    pooled = hidden_states * attention_mask[..., None]
    pooled = pooled.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)[:, None]
    return pooled


def encode_texts(
    texts: List[str],
    tokenizer,
    encoder,
    device: torch.device,
    max_length: int,
    batch_size: int,
    normalize: bool = True,
) -> torch.Tensor:
    chunks = []
    encoder.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            if getattr(encoder.config, "is_encoder_decoder", False):
                outputs = encoder.encoder(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
            else:
                outputs = encoder(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
            embeddings = mean_pool(outputs.last_hidden_state, tokens.attention_mask)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        chunks.append(embeddings.cpu())
    return torch.cat(chunks, dim=0)


def check_normalization(embeddings: torch.Tensor, name: str) -> None:
    norms = embeddings.norm(p=2, dim=1)
    print(f"{name}: mean norm={norms.mean().item():.4f}, std={norms.std().item():.4f}")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def batched(iterable: Iterable[Any], batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
