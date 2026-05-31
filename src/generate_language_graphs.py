from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from typing import Dict, Iterable, List, Sequence

import numpy as np


LANGUAGE_ORDER = [
    "en",
    "zh",
    "fr",
    "de",
    "id",
    "it",
    "pt",
    "ru",
    "es",
    "ar",
    "nl",
    "hi",
    "ja",
    "vi",
]

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "id": "Indonesian",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "ar": "Arabic",
    "nl": "Dutch",
    "hi": "Hindi",
    "ja": "Japanese",
    "vi": "Vietnamese",
}

NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}

MMARCO_7_LANGS = ["en", "fr", "de", "it", "pt", "es", "nl"]

LANG2VEC_CODES = {
    "en": "eng",
    "zh": "zho",
    "fr": "fra",
    "de": "deu",
    "id": "ind",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "es": "spa",
    "ar": "ara",
    "nl": "nld",
    "hi": "hin",
    "ja": "jpn",
    "vi": "vie",
}

ASJP_LABEL_TO_CODE = {
    "ENGLISH": "en",
    "MANDARIN": "zh",
    "FRENCH": "fr",
    "STANDARD_GERMAN": "de",
    "INDONESIAN": "id",
    "ITALIAN": "it",
    "PORTUGUESE": "pt",
    "RUSSIAN": "ru",
    "SPANISH": "es",
    "CAIRO_ARABIC": "ar",
    "DUTCH": "nl",
    "HINDI": "hi",
    "JAPANESE": "ja",
    "VIETNAMESE": "vi",
}

DEFAULT_ASJP_OUTPUT_PATH = "resources/asjp/output.txt"
DEFAULT_LANG2VEC_PACKAGE_DIR = "../lang2vec-master"
DEFAULT_LANG2VEC_DISTANCE_PATH = "resources/lang2vec/syntactic_distances.csv"


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> List[float]:
    return [float(item) for item in parse_csv(value)]


def parse_langs(value: str) -> List[str]:
    langs = []
    for raw in parse_csv(value):
        key = raw.lower()
        code = NAME_TO_CODE.get(key, key)
        if code not in LANGUAGE_ORDER:
            raise ValueError(f"Unsupported language '{raw}'. Supported codes: {LANGUAGE_ORDER}")
        langs.append(code)
    return langs


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(project_root, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find path: {path}")


def parse_asjp_label(line: str) -> str:
    label = line.split("{", 1)[0].strip()
    label = label.split()[0].strip()
    return label


def parse_float_token(token: str) -> float:
    if token.startswith("."):
        token = "0" + token
    return float(token)


def load_asjp_ldnd_matrix(path: str) -> np.ndarray:
    resolved_path = resolve_path(path)
    rows: List[tuple[str, List[float]]] = []
    in_table = False
    with open(resolved_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("LDND"):
                if in_table:
                    break
                in_table = True
                continue
            if not in_table or not stripped:
                continue
            label = parse_asjp_label(line)
            if label not in ASJP_LABEL_TO_CODE:
                continue
            values = [parse_float_token(token) for token in re.findall(r"(?:\d+\.\d+|\.\d+)", line)]
            rows.append((ASJP_LABEL_TO_CODE[label], values))

    if not rows:
        raise ValueError(f"No ASJP LDND rows parsed from {resolved_path}")

    parsed_codes = [code for code, _ in rows]
    size = len(rows)
    matrix_by_output_order = np.zeros((size, size), dtype=np.float32)
    for row_idx, (_, values) in enumerate(rows):
        if len(values) != row_idx + 1:
            raise ValueError(
                f"Expected {row_idx + 1} LDND values on row {row_idx}, got {len(values)}"
            )
        for col_idx, value in enumerate(values):
            matrix_by_output_order[row_idx, col_idx] = value
            matrix_by_output_order[col_idx, row_idx] = value

    missing = [code for code in LANGUAGE_ORDER if code not in parsed_codes]
    if missing:
        raise ValueError(f"ASJP output is missing languages: {missing}")
    output_idx = {code: idx for idx, code in enumerate(parsed_codes)}
    ordered_idx = [output_idx[code] for code in LANGUAGE_ORDER]
    return matrix_by_output_order[np.ix_(ordered_idx, ordered_idx)]


def load_matrix_csv(path: str) -> np.ndarray:
    resolved_path = resolve_path(path)
    with open(resolved_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        codes = [cell.strip() for cell in header[1:]]
        rows = []
        row_codes = []
        for row in reader:
            if not row:
                continue
            row_codes.append(row[0].strip())
            rows.append([float(value) for value in row[1:]])
    if codes != row_codes:
        raise ValueError(f"CSV row/column language codes do not match in {resolved_path}")
    missing = [code for code in LANGUAGE_ORDER if code not in codes]
    if missing:
        raise ValueError(f"Distance CSV is missing languages: {missing}")
    matrix = np.array(rows, dtype=np.float32)
    idx = [codes.index(code) for code in LANGUAGE_ORDER]
    return matrix[np.ix_(idx, idx)]


def try_load_lang2vec_package_matrix(package_dir: str) -> tuple[np.ndarray, str] | None:
    if not package_dir:
        return None
    try:
        resolved_package_dir = resolve_path(package_dir)
    except FileNotFoundError:
        return None

    if resolved_package_dir not in sys.path:
        sys.path.insert(0, resolved_package_dir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lang2vec.lang2vec as l2v  # type: ignore
    except Exception as exc:
        print(f"Could not import lang2vec from {resolved_package_dir}: {exc}")
        return None

    lang2vec_codes = [LANG2VEC_CODES[code] for code in LANGUAGE_ORDER]
    matrix = np.asarray(l2v.distance("syntactic", lang2vec_codes), dtype=np.float32)
    return matrix, resolved_package_dir


def load_lang2vec_syntactic_matrix(
    package_dir: str,
    fallback_distance_path: str,
) -> tuple[np.ndarray, str]:
    package_result = try_load_lang2vec_package_matrix(package_dir)
    if package_result is not None:
        return package_result
    return load_matrix_csv(fallback_distance_path), fallback_distance_path


def subset(matrix: np.ndarray, languages: Sequence[str]) -> np.ndarray:
    idx = [LANGUAGE_ORDER.index(lang) for lang in languages]
    return matrix[np.ix_(idx, idx)]


def adjacency_from_distances(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return ((matrix < threshold) & (matrix > 0)).astype(np.int64)


def signed_from_adjacency(
    adjacency: np.ndarray,
    graph_name: str,
    languages: Sequence[str],
    sign_mode: str,
) -> np.ndarray:
    signed = np.zeros(adjacency.shape, dtype=np.int64)
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            if adjacency[i, j] or adjacency[j, i]:
                signed[i, j] = 1
                signed[j, i] = -1
    if sign_mode == "legacy" and graph_name == "syntactic":
        if "pt" in languages and "es" in languages:
            pt_idx = languages.index("pt")
            es_idx = languages.index("es")
            if adjacency[pt_idx, es_idx] or adjacency[es_idx, pt_idx]:
                signed[pt_idx, es_idx] = -1
                signed[es_idx, pt_idx] = -1
    return signed


def edge_count(adjacency: np.ndarray) -> int:
    return int(np.triu(adjacency, k=1).sum())


def write_csv(path: str, matrix: np.ndarray, languages: Sequence[str]) -> None:
    names = [LANGUAGE_NAMES[lang] for lang in languages]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix.tolist()):
            writer.writerow([name] + row)


def write_json(path: str, matrix: np.ndarray) -> None:
    with open(path, "w") as f:
        json.dump(matrix.astype(int).tolist(), f, indent=2)
        f.write("\n")


def write_metadata(
    path: str,
    graph_name: str,
    threshold: float,
    languages: Sequence[str],
    sign_mode: str,
    source: str,
    adjacency: np.ndarray,
) -> None:
    metadata = {
        "graph_name": graph_name,
        "threshold": threshold,
        "language_codes": list(languages),
        "language_names": [LANGUAGE_NAMES[lang] for lang in languages],
        "edge_count": edge_count(adjacency),
        "sign_mode": sign_mode,
        "source": source,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def threshold_label(threshold: float) -> str:
    return str(int(threshold)) if float(threshold).is_integer() else str(threshold).rstrip("0").rstrip(".")


def generate_graph(
    output_dir: str,
    graph_name: str,
    distance_matrix: np.ndarray,
    threshold: float,
    languages: Sequence[str],
    sign_mode: str,
    source: str,
    aliases: Iterable[str] = (),
) -> None:
    distances = subset(distance_matrix, languages)
    adjacency = adjacency_from_distances(distances, threshold)
    signed = signed_from_adjacency(adjacency, graph_name, languages, sign_mode)
    label = threshold_label(threshold)
    names = [graph_name, *aliases]

    for name in names:
        write_csv(os.path.join(output_dir, f"adjacency_matrix_{name}_{label}.csv"), adjacency, languages)
        write_json(os.path.join(output_dir, f"graph_{name}_{label}_signed.json"), signed)
        write_metadata(
            os.path.join(output_dir, f"graph_{name}_{label}_metadata.json"),
            name,
            threshold,
            languages,
            sign_mode,
            source,
            adjacency,
        )

    print(
        f"{graph_name}@{label}: languages={','.join(languages)}, "
        f"edges={edge_count(adjacency)}, sign_mode={sign_mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LAGO language topology graphs.")
    parser.add_argument("--output_dir", default="graphs/mmarco_7")
    parser.add_argument("--languages", type=parse_langs, default=MMARCO_7_LANGS)
    parser.add_argument("--syntactic_thresholds", type=parse_float_csv, default=[0.45])
    parser.add_argument("--ajsp_thresholds", type=parse_float_csv, default=[90.0])
    parser.add_argument(
        "--lang2vec_package_dir",
        default=DEFAULT_LANG2VEC_PACKAGE_DIR,
        help="Local lang2vec source directory used to compute syntactic distances.",
    )
    parser.add_argument(
        "--lang2vec_distance_path",
        default=DEFAULT_LANG2VEC_DISTANCE_PATH,
        help="Fallback cached lang2vec syntactic distance CSV.",
    )
    parser.add_argument(
        "--asjp_output_path",
        default=DEFAULT_ASJP_OUTPUT_PATH,
        help="ASJP62 LDND output.txt used to derive AJSP distances.",
    )
    parser.add_argument(
        "--sign_mode",
        choices=["legacy", "antisymmetric"],
        default="legacy",
        help="legacy reproduces the ALGEN/LAGO hard-coded graph signs.",
    )
    parser.add_argument(
        "--skip_aliases",
        action="store_true",
        help="Do not write lang2vec alias files for the syntactic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    syntactic_matrix, syntactic_source = load_lang2vec_syntactic_matrix(
        args.lang2vec_package_dir,
        args.lang2vec_distance_path,
    )
    for threshold in args.syntactic_thresholds:
        aliases = [] if args.skip_aliases else ["lang2vec"]
        generate_graph(
            args.output_dir,
            "syntactic",
            syntactic_matrix,
            threshold,
            args.languages,
            args.sign_mode,
            syntactic_source,
            aliases=aliases,
        )
    ajsp_matrix = load_asjp_ldnd_matrix(args.asjp_output_path)
    for threshold in args.ajsp_thresholds:
        generate_graph(
            args.output_dir,
            "ajsp",
            ajsp_matrix,
            threshold,
            args.languages,
            args.sign_mode,
            args.asjp_output_path,
        )


if __name__ == "__main__":
    main()
