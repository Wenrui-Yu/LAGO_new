from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data import sample_splits
from decoder_model import AttackDecoderModel
from graph_alignment import build_graph, infer_lang_key, lago_pdmm, ridge_alignment
from metrics import eval_embeddings, generation_metrics
from utils import (
    check_normalization,
    encode_texts,
    ensure_jsonable,
    get_device,
    load_encoder_model_and_tokenizer,
    parse_csv,
    read_json,
    save_json,
    set_seed,
)


def resolve_checkpoint(checkpoint_path: str) -> Tuple[str, Dict]:
    if os.path.isdir(checkpoint_path):
        metadata_path = os.path.join(checkpoint_path, "training_args_and_best_models.json")
        metadata = read_json(metadata_path)
        best_models = metadata["best_models"]
        if not best_models:
            raise ValueError(f"No best model found in {metadata_path}")
        return best_models[0][2], metadata
    return checkpoint_path, {}


def load_decoder(checkpoint_path: str, device: torch.device) -> AttackDecoderModel:
    checkpoint_file, metadata = resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    decoder_config = checkpoint.get("decoder_config") or metadata.get("decoder_config")
    if decoder_config is None:
        raise ValueError("Checkpoint is missing decoder_config.")

    model = AttackDecoderModel(
        model_name=decoder_config["model_name"],
        input_dim=decoder_config.get("input_dim"),
        prompt_length=decoder_config.get("prompt_length", decoder_config.get("max_length", 32)),
        max_length=decoder_config.get("max_length", 32),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded decoder checkpoint: {checkpoint_file}")
    return model


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


def load_precomputed_vectors(
    source_model_name: str,
    target_model_name: str,
    dataset_name: str,
    vector_root: str,
    max_length: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_slug = source_model_name.replace("/", "_")
    target_slug = target_model_name.replace("/", "_")
    dataset_slug = dataset_name.replace("/", "_")
    path = os.path.join(
        vector_root,
        source_slug,
        target_slug,
        dataset_slug,
        f"vecs_maxlength{max_length}.npz",
    )
    print(f"Loading precomputed vectors from {path}")
    data = np.load(path)
    return (
        torch.tensor(data["train"][:train_samples], dtype=torch.float32, device=device),
        torch.tensor(data["dev"][:val_samples], dtype=torch.float32, device=device),
        torch.tensor(data["test"][:test_samples], dtype=torch.float32, device=device),
    )


def prepare_language_data(
    dataset_name: str,
    lang_key: str,
    args,
    decoder: AttackDecoderModel,
    source_encoder,
    source_tokenizer,
    device: torch.device,
) -> Dict:
    splits = sample_splits(
        dataset_name,
        train_samples=args.align_train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        data_folder=args.data_folder,
    )

    train_texts = canonicalize_texts(splits["train"], decoder.tokenizer, decoder.max_length)
    val_texts = canonicalize_texts(splits["validation"], decoder.tokenizer, decoder.max_length)
    test_texts = canonicalize_texts(splits["test"], decoder.tokenizer, decoder.max_length)

    y_train = encode_texts(
        train_texts,
        decoder.tokenizer,
        decoder.encoder_decoder.encoder,
        device,
        decoder.max_length,
        args.encode_batch_size,
        normalize=True,
    ).to(device)
    y_val = encode_texts(
        val_texts,
        decoder.tokenizer,
        decoder.encoder_decoder.encoder,
        device,
        decoder.max_length,
        args.encode_batch_size,
        normalize=True,
    ).to(device)
    y_test = encode_texts(
        test_texts,
        decoder.tokenizer,
        decoder.encoder_decoder.encoder,
        device,
        decoder.max_length,
        args.encode_batch_size,
        normalize=True,
    ).to(device)

    if args.source_model_name == "random":
        source_dim = args.random_source_dim
        x_train = torch.randn(len(train_texts), source_dim, device=device)
        x_val = torch.randn(len(val_texts), source_dim, device=device)
        x_test = torch.randn(len(test_texts), source_dim, device=device)
        x_train = torch.nn.functional.normalize(x_train, p=2, dim=1)
        x_val = torch.nn.functional.normalize(x_val, p=2, dim=1)
        x_test = torch.nn.functional.normalize(x_test, p=2, dim=1)
    elif args.source_model_name.startswith("text-embedding-"):
        x_train, x_val, x_test = load_precomputed_vectors(
            args.source_model_name,
            decoder.model_name,
            dataset_name,
            args.vector_root,
            decoder.max_length,
            args.align_train_samples,
            args.val_samples,
            args.test_samples,
            device,
        )
    else:
        x_train = encode_texts(
            train_texts,
            source_tokenizer,
            source_encoder,
            device,
            args.source_max_length,
            args.encode_batch_size,
            normalize=True,
        ).to(device)
        x_val = encode_texts(
            val_texts,
            source_tokenizer,
            source_encoder,
            device,
            args.source_max_length,
            args.encode_batch_size,
            normalize=True,
        ).to(device)
        x_test = encode_texts(
            test_texts,
            source_tokenizer,
            source_encoder,
            device,
            args.source_max_length,
            args.encode_batch_size,
            normalize=True,
        ).to(device)

    check_normalization(x_train, f"X train {lang_key}")
    check_normalization(y_train, f"Y train {lang_key}")
    return {
        "dataset_name": dataset_name,
        "lang_key": lang_key,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "test_texts": test_texts,
    }


def generate_predictions(
    decoder: AttackDecoderModel,
    hidden_states: torch.Tensor,
    batch_size: int,
) -> List[str]:
    predictions: List[str] = []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, hidden_states.shape[0], batch_size):
            batch = hidden_states[start : start + batch_size].to(decoder.device)
            generated = decoder.generate({"hidden_states": batch})
            decoded = decoder.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([text.strip() for text in decoded])
    return predictions


def evaluate_alignment(decoder: AttackDecoderModel, lang_data: Dict, transform: torch.Tensor, args):
    x_train_aligned = lang_data["x_train"] @ transform
    x_val_aligned = lang_data["x_val"] @ transform
    x_test_aligned = lang_data["x_test"] @ transform

    train_cos, train_mse = eval_embeddings(x_train_aligned, lang_data["y_train"])
    val_cos, val_mse = eval_embeddings(x_val_aligned, lang_data["y_val"])
    test_cos, test_mse = eval_embeddings(x_test_aligned, lang_data["y_test"])
    predictions = generate_predictions(decoder, x_test_aligned, args.decode_batch_size)
    gen_metrics = generation_metrics(predictions, lang_data["test_texts"])

    metrics = {
        "alignment": {
            "train_cos": train_cos.item(),
            "train_mse": train_mse.item(),
            "val_cos": val_cos.item(),
            "val_mse": val_mse.item(),
            "test_cos": test_cos.item(),
            "test_mse": test_mse.item(),
        },
        "generation": gen_metrics,
    }
    return metrics, predictions


def macro_average(per_lang: Dict[str, Dict]) -> Dict[str, float]:
    generation_keys = list(next(iter(per_lang.values()))["generation"].keys())
    alignment_keys = list(next(iter(per_lang.values()))["alignment"].keys())
    return {
        **{
            f"generation_{key}": float(np.mean([v["generation"][key] for v in per_lang.values()]))
            for key in generation_keys
        },
        **{
            f"alignment_{key}": float(np.mean([v["alignment"][key] for v in per_lang.values()]))
            for key in alignment_keys
        },
    }


def output_dir(args, decoder: AttackDecoderModel) -> str:
    decoder_slug = decoder.model_name.replace("/", "_")
    source_slug = args.source_model_name.replace("/", "_")
    return os.path.join(
        args.output_dir,
        decoder_slug,
        source_slug,
        f"{args.graph_type}_{args.constraint_mode}_train{args.align_train_samples}"
        f"_ridge{args.reg_lambda}_eps{args.epsilon}_seed{args.seed}",
    )


def write_prediction_csv(path: str, predictions: List[str], references: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prediction", "reference"])
        writer.writeheader()
        for prediction, reference in zip(predictions, references):
            writer.writerow({"prediction": prediction, "reference": reference})


def run(args) -> None:
    set_seed(args.seed)
    device = get_device()
    decoder = load_decoder(args.decoder_checkpoint_path, device)

    if args.lang_keys is None:
        args.lang_keys = [infer_lang_key(dataset_name) for dataset_name in args.datasets]
    if len(args.lang_keys) != len(args.datasets):
        raise ValueError("--lang_keys must match --datasets length.")

    source_encoder = None
    source_tokenizer = None
    if args.source_model_name not in {"random"} and not args.source_model_name.startswith("text-embedding-"):
        source_encoder, source_tokenizer = load_encoder_model_and_tokenizer(
            args.source_model_name, device
        )

    lang_items = []
    for dataset_name, lang_key in zip(args.datasets, args.lang_keys):
        lang_items.append(
            prepare_language_data(
                dataset_name,
                lang_key,
                args,
                decoder,
                source_encoder,
                source_tokenizer,
                device,
            )
        )

    graph = build_graph(args.graph_type, args.lang_keys, args.graph_file, args.seed)
    if graph is not None:
        print(f"Graph type={args.graph_type}\n{graph}")

    xs = [item["x_train"] for item in lang_items]
    ys = [item["y_train"] for item in lang_items]
    if graph is None:
        transforms = [ridge_alignment(x, y, args.reg_lambda) for x, y in zip(xs, ys)]
    else:
        transforms = lago_pdmm(
            graph,
            xs,
            ys,
            reg_lambda=args.reg_lambda,
            constraint_mode=args.constraint_mode,
            epsilon=args.epsilon,
            num_iter=args.num_iter,
            c=args.pdmm_c,
        )

    out_dir = output_dir(args, decoder)
    os.makedirs(out_dir, exist_ok=True)

    per_lang_metrics = {}
    for item, transform in tqdm(list(zip(lang_items, transforms)), desc="Evaluating languages"):
        metrics, predictions = evaluate_alignment(decoder, item, transform, args)
        lang_key = item["lang_key"]
        per_lang_metrics[lang_key] = metrics
        write_prediction_csv(
            os.path.join(out_dir, f"{lang_key}_predictions.csv"),
            predictions,
            item["test_texts"],
        )
        torch.save(transform.cpu(), os.path.join(out_dir, f"{lang_key}_W.pt"))

    summary = {
        "args": ensure_jsonable(vars(args)),
        "decoder_model": decoder.model_name,
        "macro": macro_average(per_lang_metrics),
        "per_lang": per_lang_metrics,
        "graph": None if graph is None else graph.cpu().tolist(),
    }
    save_json(os.path.join(out_dir, "results.json"), summary)
    print(summary["macro"])
    print(f"Wrote results to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run no-graph/lang2vec/ajsp/random-graph alignment ablations.")
    parser.add_argument("--decoder_checkpoint_path", required=True)
    parser.add_argument("--source_model_name", default="google/mt5-base")
    parser.add_argument(
        "--datasets",
        type=parse_csv,
        default=parse_csv(
            "yywwrr/mmarco_english,yywwrr/mmarco_french,yywwrr/mmarco_german,"
            "yywwrr/mmarco_italian,yywwrr/mmarco_portuguese,"
            "yywwrr/mmarco_spanish,yywwrr/mmarco_dutch"
        ),
    )
    parser.add_argument("--lang_keys", type=parse_csv, default=None)
    parser.add_argument("--data_folder", default="datasets/finetuning_decoder")
    parser.add_argument("--output_dir", default="outputs/lago_ablation")
    parser.add_argument("--vector_root", default="datasets/vectors")
    parser.add_argument(
        "--graph_type",
        choices=[
            "none",
            "syntactic",
            "asjp",
            "lang2vec",
            "ajsp",
            "lago",
            "random",
            "random_syntactic",
            "random_lang2vec",
            "random_asjp",
            "random_ajsp",
        ],
        default="none",
    )
    parser.add_argument(
        "--constraint_mode",
        choices=["none", "equality", "inequality", "conic", "totalvariation"],
        default="totalvariation",
    )
    parser.add_argument("--graph_file", default=None)
    parser.add_argument("--align_train_samples", type=int, default=100)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--source_max_length", type=int, default=32)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--decode_batch_size", type=int, default=128)
    parser.add_argument("--reg_lambda", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--pdmm_c", type=float, default=0.4)
    parser.add_argument("--random_source_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
