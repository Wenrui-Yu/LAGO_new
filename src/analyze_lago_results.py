from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_ROOTS = [
    "server_results_json_20260531/outputs/lago_ablation",
    "server_results_json_20260531/outputs/lago_ablation_multilingual_decoder",
]

GENERATION_METRICS = [
    "generation_rougeL",
    "generation_rouge1",
    "generation_rouge2",
    "generation_exact_match",
    "generation_bleu",
    "generation_bleu1",
    "generation_bleu2",
    "generation_bleu3",
    "generation_bleu4",
]

ALIGNMENT_METRICS = [
    "alignment_train_cos",
    "alignment_train_mse",
    "alignment_val_cos",
    "alignment_val_mse",
    "alignment_test_cos",
    "alignment_test_mse",
]


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def iter_result_files(roots: Iterable[str]) -> Iterable[str]:
    for root in roots:
        if not os.path.isdir(root):
            print(f"Skipping missing root: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            if "results.json" in filenames:
                yield os.path.join(dirpath, "results.json")


def infer_decoder_setup(path: str) -> str:
    parts = set(path.split(os.sep))
    if "lago_ablation_multilingual_decoder" in parts:
        return "multilingual_decoder"
    if "lago_ablation" in parts:
        return "english_decoder"
    return "unknown_decoder"


def parse_condition_from_dir(path: str) -> Dict[str, Any]:
    folder = os.path.basename(os.path.dirname(path))
    match = re.match(
        r"(?P<graph_type>.+)_(?P<constraint_mode>none|equality|inequality|conic|totalvariation)"
        r"_train(?P<align_train_samples>\d+)"
        r"_ridge(?P<reg_lambda>[^_]+)"
        r"_eps(?P<epsilon>[^_]+)"
        r"_seed(?P<seed>\d+)$",
        folder,
    )
    if not match:
        return {}
    parsed: Dict[str, Any] = match.groupdict()
    parsed["align_train_samples"] = int(parsed["align_train_samples"])
    parsed["seed"] = int(parsed["seed"])
    for key in ["reg_lambda", "epsilon"]:
        try:
            parsed[key] = float(parsed[key])
        except ValueError:
            pass
    return parsed


def flatten_metric_group(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    row = {}
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value
    return row


def build_rows(results_path: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = load_json(results_path)
    args = payload.get("args", {})
    condition = parse_condition_from_dir(results_path)

    base = {
        "decoder_setup": infer_decoder_setup(results_path),
        "decoder_model": payload.get("decoder_model"),
        "source_model_name": args.get("source_model_name"),
        "graph_type": args.get("graph_type") or condition.get("graph_type"),
        "constraint_mode": args.get("constraint_mode") or condition.get("constraint_mode"),
        "align_train_samples": args.get("align_train_samples") or condition.get("align_train_samples"),
        "reg_lambda": args.get("reg_lambda") or condition.get("reg_lambda"),
        "epsilon": args.get("epsilon") or condition.get("epsilon"),
        "seed": args.get("seed") or condition.get("seed"),
        "results_path": results_path,
    }

    macro_row = dict(base)
    macro_row.update(payload.get("macro", {}))

    per_lang_rows = []
    for lang, metrics in payload.get("per_lang", {}).items():
        row = dict(base)
        row["lang"] = lang
        row.update(flatten_metric_group("generation", metrics.get("generation", {})))
        row.update(flatten_metric_group("alignment", metrics.get("alignment", {})))
        per_lang_rows.append(row)

    return macro_row, per_lang_rows


def collect_results(roots: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_rows: List[Dict[str, Any]] = []
    per_lang_rows: List[Dict[str, Any]] = []
    for results_path in iter_result_files(roots):
        macro_row, rows = build_rows(results_path)
        macro_rows.append(macro_row)
        per_lang_rows.extend(rows)
    macro_df = pd.DataFrame(macro_rows)
    per_lang_df = pd.DataFrame(per_lang_rows)
    if not macro_df.empty:
        macro_df = macro_df.sort_values(
            ["decoder_setup", "source_model_name", "align_train_samples", "graph_type"],
            kind="stable",
        )
    if not per_lang_df.empty:
        per_lang_df = per_lang_df.sort_values(
            ["decoder_setup", "source_model_name", "align_train_samples", "graph_type", "lang"],
            kind="stable",
        )
    return macro_df, per_lang_df


def compare_against_baselines(macro_df: pd.DataFrame) -> pd.DataFrame:
    if macro_df.empty:
        return pd.DataFrame()

    group_cols = [
        "decoder_setup",
        "source_model_name",
        "align_train_samples",
        "reg_lambda",
        "epsilon",
        "seed",
    ]
    metrics = [metric for metric in GENERATION_METRICS + ALIGNMENT_METRICS if metric in macro_df.columns]
    rows = []

    for _, group in macro_df.groupby(group_cols, dropna=False):
        by_graph = {row["graph_type"]: row for _, row in group.iterrows()}
        none = by_graph.get("none")
        pairs = [
            ("lang2vec_vs_none", "lang2vec", none),
            ("ajsp_vs_none", "ajsp", none),
            ("random_lang2vec_vs_none", "random_lang2vec", none),
            ("random_ajsp_vs_none", "random_ajsp", none),
            ("lang2vec_vs_random", "lang2vec", by_graph.get("random_lang2vec")),
            ("ajsp_vs_random", "ajsp", by_graph.get("random_ajsp")),
        ]
        for comparison, graph, baseline in pairs:
            current = by_graph.get(graph)
            if current is None or baseline is None:
                continue
            row = {
                "comparison": comparison,
                "graph_type": graph,
                "baseline_graph_type": baseline["graph_type"],
                "constraint_mode": current["constraint_mode"],
                "baseline_constraint_mode": baseline["constraint_mode"],
            }
            for col in group_cols:
                row[col] = current[col]
            for metric in metrics:
                current_value = current.get(metric)
                baseline_value = baseline.get(metric)
                row[f"{metric}_current"] = current_value
                row[f"{metric}_baseline"] = baseline_value
                row[f"{metric}_delta"] = current_value - baseline_value
                if baseline_value not in (0, None):
                    row[f"{metric}_rel_delta"] = (current_value - baseline_value) / abs(baseline_value)
            rows.append(row)

    return pd.DataFrame(rows)


def best_by_group(macro_df: pd.DataFrame) -> pd.DataFrame:
    if macro_df.empty or "generation_rougeL" not in macro_df.columns:
        return pd.DataFrame()
    group_cols = ["decoder_setup", "source_model_name", "align_train_samples"]
    idx = macro_df.groupby(group_cols, dropna=False)["generation_rougeL"].idxmax()
    cols = group_cols + [
        "graph_type",
        "constraint_mode",
        "generation_rougeL",
        "generation_rouge1",
        "generation_bleu",
        "alignment_test_cos",
        "alignment_test_mse",
        "results_path",
    ]
    cols = [col for col in cols if col in macro_df.columns]
    return macro_df.loc[idx, cols].sort_values(group_cols, kind="stable")


def compare_decoder_setups(macro_df: pd.DataFrame) -> pd.DataFrame:
    if macro_df.empty:
        return pd.DataFrame()

    group_cols = [
        "source_model_name",
        "graph_type",
        "constraint_mode",
        "align_train_samples",
        "reg_lambda",
        "epsilon",
        "seed",
    ]
    metrics = [metric for metric in GENERATION_METRICS + ALIGNMENT_METRICS if metric in macro_df.columns]
    rows = []

    for _, group in macro_df.groupby(group_cols, dropna=False):
        by_setup = {row["decoder_setup"]: row for _, row in group.iterrows()}
        english = by_setup.get("english_decoder")
        multilingual = by_setup.get("multilingual_decoder")
        if english is None or multilingual is None:
            continue
        row = {"comparison": "multilingual_vs_english"}
        for col in group_cols:
            row[col] = multilingual[col]
        for metric in metrics:
            multilingual_value = multilingual.get(metric)
            english_value = english.get(metric)
            row[f"{metric}_multilingual"] = multilingual_value
            row[f"{metric}_english"] = english_value
            row[f"{metric}_delta"] = multilingual_value - english_value
            if english_value not in (0, None):
                row[f"{metric}_rel_delta"] = (multilingual_value - english_value) / abs(english_value)
        rows.append(row)

    return pd.DataFrame(rows)


def write_report(
    output_dir: str,
    macro_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_df: pd.DataFrame,
    decoder_comparison_df: pd.DataFrame,
) -> None:
    def table_text(df: pd.DataFrame) -> str:
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines = ["# LAGO Result Analysis", ""]
    lines.append(f"- Macro result rows: {len(macro_df)}")
    lines.append(f"- Comparison rows: {len(comparison_df)}")
    lines.append(f"- Decoder comparison rows: {len(decoder_comparison_df)}")
    lines.append("")

    if not macro_df.empty:
        lines.append("## Result Coverage")
        coverage = (
            macro_df.groupby(["decoder_setup", "source_model_name"])["results_path"]
            .count()
            .reset_index(name="num_results")
        )
        lines.append(table_text(coverage))
        lines.append("")

    if not best_df.empty:
        lines.append("## Best Graph by Rouge-L")
        preview_cols = [
            "decoder_setup",
            "source_model_name",
            "align_train_samples",
            "graph_type",
            "generation_rougeL",
            "alignment_test_cos",
        ]
        preview_cols = [col for col in preview_cols if col in best_df.columns]
        lines.append(table_text(best_df[preview_cols]))
        lines.append("")

    if not comparison_df.empty and "generation_rougeL_delta" in comparison_df.columns:
        lines.append("## Average Rouge-L Deltas")
        delta = (
            comparison_df.groupby(["decoder_setup", "source_model_name", "comparison"])[
                "generation_rougeL_delta"
            ]
            .mean()
            .reset_index()
            .sort_values(["decoder_setup", "source_model_name", "comparison"], kind="stable")
        )
        lines.append(table_text(delta))
        lines.append("")

    if not decoder_comparison_df.empty and "generation_rougeL_delta" in decoder_comparison_df.columns:
        lines.append("## Multilingual Decoder Gain")
        gain = (
            decoder_comparison_df.groupby(["source_model_name"])[
                ["generation_rougeL_delta", "generation_bleu_delta", "alignment_test_cos_delta"]
            ]
            .mean()
            .reset_index()
            .sort_values("source_model_name", kind="stable")
        )
        lines.append(table_text(gain))
        lines.append("")

    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LAGO_new result folders.")
    parser.add_argument("--roots", type=parse_csv_arg, default=DEFAULT_ROOTS)
    parser.add_argument("--output_dir", default="analysis/lago_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    macro_df, per_lang_df = collect_results(args.roots)
    comparison_df = compare_against_baselines(macro_df)
    best_df = best_by_group(macro_df)
    decoder_comparison_df = compare_decoder_setups(macro_df)

    macro_df.to_csv(os.path.join(args.output_dir, "summary_macro.csv"), index=False)
    per_lang_df.to_csv(os.path.join(args.output_dir, "per_language.csv"), index=False)
    comparison_df.to_csv(os.path.join(args.output_dir, "comparisons.csv"), index=False)
    best_df.to_csv(os.path.join(args.output_dir, "best_by_group.csv"), index=False)
    decoder_comparison_df.to_csv(os.path.join(args.output_dir, "decoder_comparisons.csv"), index=False)
    write_report(args.output_dir, macro_df, comparison_df, best_df, decoder_comparison_df)

    print(f"Read {len(macro_df)} result files")
    print(f"Wrote analysis to {args.output_dir}")
    if not best_df.empty:
        print(best_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
