from __future__ import annotations

import json
import csv
from typing import Any, Dict, List, Optional

import numpy as np
import torch


DEFAULT_LANG_ORDER = ["en", "fr", "de", "it", "pt", "es", "nl"]

# Lang2vec graph from ALGEN/src/attacker_pdmm.py.
LANG2VEC_GRAPH = torch.tensor(
    [
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, -1, 0, -1, 0],
        [-1, -1, 0, 0, -1, 0, 0],
        [0, 0, -1, -1, 0, 0, 0],
    ],
    dtype=torch.float32,
)

# AJSP graph from ALGEN's "ajsp" branch.
AJSP_GRAPH = torch.tensor(
    [
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [-1, 0, 0, 1, 0, 0, 1],
        [-1, -1, -1, 0, 1, 1, 1],
        [0, -1, 0, -1, 0, 1, 0],
        [0, -1, 0, -1, -1, 0, 0],
        [-1, 0, -1, -1, 0, 0, 0],
    ],
    dtype=torch.float32,
)

DEFAULT_GRAPHS: Dict[str, torch.Tensor] = {
    "lang2vec": LANG2VEC_GRAPH,
    "ajsp": AJSP_GRAPH,
}

GRAPH_ALIASES = {
    "lago": "lang2vec",
    "syntactic": "lang2vec",
    "asjp": "ajsp",
}

RANDOM_GRAPH_BASES = {
    "random": "lang2vec",
    "random_lang2vec": "lang2vec",
    "random_syntactic": "lang2vec",
    "random_ajsp": "ajsp",
    "random_asjp": "ajsp",
}


def infer_lang_key(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    aliases = {
        "english": "en",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "spanish": "es",
        "dutch": "nl",
        "xnli_en": "en",
        "xnli_fr": "fr",
        "xnli_de": "de",
        "xnli_es": "es",
    }
    for needle, key in aliases.items():
        if needle in lowered:
            return key
    return dataset_name.replace("/", "_")


def load_graph(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        return torch.tensor(np.load(path), dtype=torch.float32)
    if path.endswith(".npz"):
        data = np.load(path)
        first_key = sorted(data.files)[0]
        return torch.tensor(data[first_key], dtype=torch.float32)
    if path.endswith(".csv"):
        names = []
        rows = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    names.append(row[0].strip().lower())
                    rows.append([float(value) for value in row[1:]])
        data = np.array(rows, dtype=np.float32)
        if np.all(data >= 0) and np.allclose(data, data.T):
            signed = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[1]):
                    if data[i, j] or data[j, i]:
                        signed[i, j] = 1.0
                        signed[j, i] = -1.0
            lowered_path = path.lower()
            if ("syntactic" in lowered_path or "lang2vec" in lowered_path) and "antisymmetric" not in lowered_path:
                if "portuguese" in names and "spanish" in names:
                    pt_idx = names.index("portuguese")
                    es_idx = names.index("spanish")
                    if data[pt_idx, es_idx] or data[es_idx, pt_idx]:
                        signed[pt_idx, es_idx] = -1.0
                        signed[es_idx, pt_idx] = -1.0
            data = signed
        return torch.tensor(data, dtype=torch.float32)
    with open(path, "r") as f:
        return torch.tensor(json.load(f), dtype=torch.float32)


def signed_default_graph(graph_name: str, lang_keys: List[str]) -> torch.Tensor:
    graph_name = GRAPH_ALIASES.get(graph_name, graph_name)
    if graph_name not in DEFAULT_GRAPHS:
        raise ValueError(
            f"Unknown default graph={graph_name}. "
            f"Available graphs: {sorted(DEFAULT_GRAPHS)}."
        )
    missing = [lang for lang in lang_keys if lang not in DEFAULT_LANG_ORDER]
    if missing:
        raise ValueError(
            f"Default graph supports {DEFAULT_LANG_ORDER}, got unsupported keys {missing}. "
            "Pass --graph_file for a custom graph."
        )
    idx = [DEFAULT_LANG_ORDER.index(lang) for lang in lang_keys]
    return DEFAULT_GRAPHS[graph_name][idx][:, idx].clone()


def edge_count(graph: torch.Tensor) -> int:
    return int((graph.abs().triu(diagonal=1) > 0).sum().item())


def connected_components(graph: torch.Tensor) -> List[List[int]]:
    adjacency = (graph.abs() > 0).cpu()
    visited = set()
    components = []
    for start in range(adjacency.shape[0]):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = torch.nonzero(adjacency[node], as_tuple=False).view(-1).tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def graph_stats(graph: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if graph is None:
        return None
    adjacency = (graph.abs() > 0).cpu()
    num_nodes = int(adjacency.shape[0])
    possible_edges = num_nodes * (num_nodes - 1) / 2
    edges = edge_count(graph)
    degrees = adjacency.sum(dim=1).to(torch.int64).tolist()
    components = connected_components(graph)
    return {
        "num_nodes": num_nodes,
        "edge_count": edges,
        "density": 0.0 if possible_edges == 0 else edges / possible_edges,
        "degrees": [int(degree) for degree in degrees],
        "degree_min": int(min(degrees)) if degrees else 0,
        "degree_max": int(max(degrees)) if degrees else 0,
        "degree_mean": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "num_components": len(components),
        "is_connected": len(components) == 1,
        "components": components,
    }


def random_signed_graph(num_nodes: int, num_edges: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    order = torch.randperm(len(pairs), generator=generator).tolist()
    graph = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for pair_idx in order[:num_edges]:
        i, j = pairs[pair_idx]
        graph[i, j] = 1.0
        graph[j, i] = -1.0
    return graph


def build_graph(
    graph_type: str,
    lang_keys: List[str],
    graph_file: Optional[str] = None,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    if graph_type == "none":
        return None

    if graph_file:
        graph = load_graph(graph_file)
    else:
        graph_name = GRAPH_ALIASES.get(graph_type, graph_type)
        if graph_type in RANDOM_GRAPH_BASES:
            graph_name = RANDOM_GRAPH_BASES[graph_type]
        graph = signed_default_graph(graph_name, lang_keys)

    expected_shape = (len(lang_keys), len(lang_keys))
    if tuple(graph.shape) != expected_shape:
        raise ValueError(
            f"Graph shape {tuple(graph.shape)} does not match {len(lang_keys)} languages. "
            f"Expected {expected_shape}. graph_file={graph_file}"
        )

    if graph_type in DEFAULT_GRAPHS or graph_type in GRAPH_ALIASES:
        return graph
    if graph_type in RANDOM_GRAPH_BASES:
        return random_signed_graph(len(lang_keys), edge_count(graph), seed)
    raise ValueError(f"Unknown graph_type={graph_type}")


def ridge_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
    reg_lambda: float = 1e-2,
) -> torch.Tensor:
    lhs = x.T @ x
    rhs = x.T @ y
    eye = torch.eye(lhs.shape[0], dtype=x.dtype, device=x.device)
    return torch.linalg.pinv(lhs + reg_lambda * eye) @ rhs


def projection_negative_dual_cone(epsilon: float, x: torch.Tensor) -> torch.Tensor:
    norm_x = torch.norm(x, "fro")
    if norm_x <= epsilon:
        return -x
    return -x * (epsilon / norm_x)


def lago_pdmm(
    graph: torch.Tensor,
    xs: List[torch.Tensor],
    ys: List[torch.Tensor],
    reg_lambda: float = 1e-2,
    constraint_mode: str = "totalvariation",
    epsilon: float = 1e-2,
    num_iter: int = 500,
    c: float = 0.4,
    theta: float = 1.0,
) -> List[torch.Tensor]:
    """Graph-constrained alignment copied from the LAGO paper implementation."""
    num_client = len(xs)
    m = xs[0].shape[1]
    n = ys[0].shape[1]
    device = xs[0].device
    dtype = xs[0].dtype
    graph = graph.to(device=device, dtype=dtype)

    ts = [torch.zeros(m, n, device=device, dtype=dtype) for _ in range(num_client)]
    eye_m = torch.eye(m, device=device, dtype=dtype)

    if constraint_mode == "none":
        return [ridge_alignment(x, y, reg_lambda=reg_lambda) for x, y in zip(xs, ys)]

    if constraint_mode == "equality":
        zs = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        yijs = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        for _ in range(num_iter):
            for i in range(num_client):
                degree = torch.sum(torch.abs(graph[i]))
                inv_term = xs[i].T @ xs[i] + (c * degree + reg_lambda) * eye_m
                rhs = xs[i].T @ ys[i] - sum(graph[i][j] * zs[i][j] for j in range(num_client))
                ts[i] = torch.linalg.solve(inv_term, rhs)
            for i in range(num_client):
                for j in range(num_client):
                    yijs[i][j] = zs[i][j] + 2 * c * graph[i][j] * ts[i]
            for i in range(num_client):
                for j in range(num_client):
                    zs[j][i] = (1 - theta) * zs[j][i] + theta * yijs[i][j]

    elif constraint_mode == "inequality":
        zs_1 = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        zs_2 = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        yijs_1 = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        yijs_2 = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        for _ in range(num_iter):
            for i in range(num_client):
                degree = torch.sum(torch.abs(graph[i]))
                inv_term = xs[i].T @ xs[i] + (2 * c * degree + reg_lambda) * eye_m
                rhs = xs[i].T @ ys[i] - sum(
                    graph[i][j] * (zs_1[i][j] - zs_2[i][j]) for j in range(num_client)
                )
                ts[i] = torch.linalg.pinv(inv_term) @ rhs

            for i in range(num_client):
                for j in range(num_client):
                    yijs_1[i][j] = zs_1[i][j] + 2 * c * graph[i][j] * ts[i] - c * epsilon
                    yijs_2[i][j] = zs_2[i][j] - 2 * c * graph[i][j] * ts[i] - c * epsilon
            for i in range(num_client):
                for j in range(num_client):
                    zs_1[i][j] = torch.where(
                        (yijs_1[i][j] + yijs_1[j][i]) > 0,
                        (1 - theta) * zs_1[i][j] + theta * yijs_1[j][i],
                        (1 - theta) * zs_1[i][j] - theta * yijs_1[i][j],
                    )
                    zs_2[i][j] = torch.where(
                        (yijs_2[i][j] + yijs_2[j][i]) > 0,
                        (1 - theta) * zs_2[i][j] + theta * yijs_2[j][i],
                        (1 - theta) * zs_2[i][j] - theta * yijs_2[i][j],
                    )

    elif constraint_mode == "conic":
        zs = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        yijs = [[torch.zeros_like(ts[0]) for _ in range(num_client)] for _ in range(num_client)]
        for _ in range(num_iter):
            for i in range(num_client):
                degree = torch.sum(torch.abs(graph[i]))
                inv_term = xs[i].T @ xs[i] + (2 * c * degree + reg_lambda) * eye_m
                rhs = xs[i].T @ ys[i] - sum(graph[i][j] * zs[i][j] for j in range(num_client))
                ts[i] = torch.linalg.solve(inv_term, rhs)
            for i in range(num_client):
                for j in range(num_client):
                    yijs[i][j] = zs[i][j] + 2 * c * graph[i][j] * ts[i] - c * epsilon
            for i in range(num_client):
                for j in range(num_client):
                    new = projection_negative_dual_cone(epsilon, yijs[j][i] + yijs[i][j])
                    zs[i][j] = (1 - theta) * zs[i][j] + theta * (new - yijs[i][j])

    elif constraint_mode == "totalvariation":
        for k in range(num_iter):
            alpha = 0.01 / ((k + 1) ** 0.5)
            for i in range(num_client):
                diff_sum = torch.zeros_like(ts[i])
                for j in range(num_client):
                    if graph[i][j] != 0:
                        diff_sum += torch.sign(ts[i] - ts[j])
                grad = (
                    -xs[i].T @ ys[i]
                    + xs[i].T @ xs[i] @ ts[i]
                    + reg_lambda * ts[i]
                    + epsilon * diff_sum
                )
                ts[i] = ts[i] - alpha * grad
    else:
        raise ValueError(f"Unknown constraint_mode={constraint_mode}")

    return ts
