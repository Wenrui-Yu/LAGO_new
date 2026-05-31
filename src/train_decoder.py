from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DecoderTextDataset, load_many_for_decoder
from decoder_model import AttackDecoderModel
from metrics import generation_metrics
from utils import (
    encode_texts,
    ensure_jsonable,
    get_device,
    parse_csv,
    save_json,
    set_seed,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_labels(texts: List[str], tokenizer, max_length: int):
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokens.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels_for_decode = labels.clone()
    labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
    references = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    references = [text.strip() for text in references]
    return labels, references


class DecoderCollator:
    def __init__(self, model: AttackDecoderModel, max_length: int, encode_batch_size: int):
        self.model = model
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size

    def __call__(self, texts: List[str]) -> Dict:
        labels, references = make_labels(texts, self.model.tokenizer, self.max_length)
        embeddings = encode_texts(
            texts,
            self.model.tokenizer,
            self.model.encoder_decoder.encoder,
            self.model.device,
            self.max_length,
            self.encode_batch_size,
            normalize=True,
        )
        return {
            "hidden_states": embeddings,
            "labels": labels,
            "text": references,
        }


def evaluate(model: AttackDecoderModel, loader: DataLoader, max_length: int):
    model.eval()
    total_loss = 0.0
    predictions: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = {
                "hidden_states": batch["hidden_states"].to(model.device),
                "labels": batch["labels"].to(model.device),
            }
            outputs = model(inputs)
            total_loss += outputs.loss.item()
            generated = model.generate(inputs, max_length=max_length)
            decoded = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([text.strip() for text in decoded])
            references.extend(batch["text"])

    metrics = generation_metrics(predictions, references)
    metrics["loss"] = total_loss / max(1, len(loader))
    print("decoded:", predictions[:4])
    print("true:", references[:4])
    return metrics, predictions, references


def output_subdir(args) -> str:
    dataset_slug = "_".join(name.replace("/", "_") for name in args.train_datasets)
    return os.path.join(
        args.output_dir,
        args.model_name.replace("/", "_"),
        f"{dataset_slug}_maxlength{args.max_length}_train{args.train_samples}"
        f"_batch{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}"
        f"_epochs{args.num_epochs}",
    )


def train(args) -> None:
    set_seed(args.seed)
    device = get_device()
    run_dir = output_subdir(args)
    os.makedirs(run_dir, exist_ok=True)

    model = AttackDecoderModel(
        model_name=args.model_name,
        prompt_length=args.prompt_length,
        max_length=args.max_length,
        normalize_input=True,
        device=device,
    )

    splits = load_many_for_decoder(
        args.train_datasets,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        data_folder=args.data_folder,
    )
    train_dataset = DecoderTextDataset(splits["train"])
    val_dataset = DecoderTextDataset(splits["validation"])
    collator = DecoderCollator(model, args.max_length, args.encode_batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_rouge = -1.0
    best_models = []
    history = []

    save_json(
        os.path.join(run_dir, "decoder_config.json"),
        {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "prompt_length": args.prompt_length,
            "input_dim": model.input_dim,
            "decoder_hidden_dim": model.decoder_hidden_dim,
        },
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            inputs = {
                "hidden_states": batch["hidden_states"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model(inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics, predictions, references = evaluate(model, val_loader, args.max_length)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        history.append(record)
        print(record)

        if val_metrics["rougeL"] > best_rouge:
            best_rouge = val_metrics["rougeL"]
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "decoder_config": {
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "prompt_length": args.prompt_length,
                        "input_dim": model.input_dim,
                    },
                    "training_args": ensure_jsonable(vars(args)),
                },
                checkpoint_path,
            )
            best_models.append((val_metrics["rougeL"], val_metrics, checkpoint_path))
            best_models.sort(key=lambda item: item[0], reverse=True)
            for _, _, old_path in best_models[args.keep_best :]:
                if os.path.exists(old_path):
                    os.remove(old_path)
            best_models = best_models[: args.keep_best]
            save_json(
                os.path.join(run_dir, "training_args_and_best_models.json"),
                {
                    "training_args": ensure_jsonable(vars(args)),
                    "decoder_config": {
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "prompt_length": args.prompt_length,
                        "input_dim": model.input_dim,
                    },
                    "best_models": best_models,
                    "history": history,
                },
            )
            save_json(
                os.path.join(run_dir, "best_model.json"),
                {"checkpoint_path": checkpoint_path, "val_metrics": val_metrics},
            )

    print(f"Finished training. Output directory: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an mT5-based ALGEN attack decoder.")
    parser.add_argument("--model_name", default="google/mt5-small")
    parser.add_argument(
        "--train_datasets",
        type=parse_csv,
        default=parse_csv("yywwrr/mmarco_english_500k"),
        help="Comma-separated datasets for attack decoder fine-tuning.",
    )
    parser.add_argument("--output_dir", default="outputs/decoders")
    parser.add_argument("--data_folder", default="datasets/finetuning_decoder")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--prompt_length", type=int, default=32)
    parser.add_argument("--train_samples", type=int, default=450000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--keep_best", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
