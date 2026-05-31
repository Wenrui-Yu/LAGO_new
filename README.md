# LAGO New: mT5-Based Attack Decoder and Graph Alignment

This folder is a clean experiment scaffold for the topology-enhanced embedding
inversion paper. It keeps the ALGEN/LAGO framework but replaces the
English-centric attack decoder with an mT5-based decoder backbone.

The contribution remains graph-aware few-shot alignment:

```text
victim embedding -> W_i alignment -> attack decoder embedding space -> mT5 attack decoder -> reconstructed text
```

## Main Entry Points

Train the mT5 attack decoder:

```bash
cd LAGO_new
sbatch run_train_mt5_decoder.sh
```

Train a multilingual fine-tuned mT5 attack decoder:

```bash
cd LAGO_new
sbatch run_train_mt5_multilingual_decoder.sh
```

Run no-graph / Lang2vec-graph / AJSP-graph / random-graph alignment with
the same decoder checkpoint:

```bash
cd LAGO_new
sbatch run_cache_text_datasets.sh
sbatch run_lago_ablation.sh
```

To submit a single condition:

```bash
cd LAGO_new
GRAPH_TYPES=lang2vec ALIGN_TRAIN_SAMPLES=100 sbatch run_lago_ablation.sh
```

`lago` and `syntactic` are kept only as backward-compatible aliases for the
built-in `lang2vec` graph. Prefer explicit graph names in new runs:

```bash
GRAPH_TYPES=lang2vec ALIGN_TRAIN_SAMPLES=100 sbatch run_lago_ablation.sh
GRAPH_TYPES=ajsp ALIGN_TRAIN_SAMPLES=100 sbatch run_lago_ablation.sh
```

Extract OpenAI victim vectors for `text-embedding-ada-002`:

```bash
cd LAGO_new
export OPENAI_API_KEY=...
sbatch run_extract_openai_vectors.sh
```

This writes vectors under:

```text
datasets/vectors/text-embedding-ada-002/google_mt5-small/<dataset>/vecs_maxlength32.npz
```

Generate the language topology files used by the graph ablations:

```bash
cd LAGO_new
sbatch run_generate_language_graphs.sh
```

This writes adjacency CSV files and signed JSON graph files under:

```text
graphs/mmarco_7/
```

The default generated files reproduce the built-in 7-language graph settings:

```text
syntactic/lang2vec threshold 0.45 from lang2vec syntactic distances
ajsp threshold 90 from ASJP62 LDND distances
legacy ALGEN/LAGO signed-edge convention
```

The same command also writes top-k graph variants for the connectivity
ablation:

```text
lang2vec_topk3, lang2vec_topk6, ..., lang2vec_topk18
ajsp_topk3, ajsp_topk6, ..., ajsp_topk18
```

Each top-k graph keeps the k shortest language-distance edges, so the
experiment can separate "does this topology help?" from "does any graph with
this density help?". The generator writes `connectivity_summary.csv` with edge
counts, density, degree range, and connected components.

To run an ablation with an explicitly generated graph file:

```bash
GRAPH_TYPES=lang2vec sbatch run_lago_ablation.sh
# or directly:
python src/run_alignment_ablation.py ... --graph_type lang2vec --graph_file graphs/mmarco_7/graph_lang2vec_0.45_signed.json --graph_label lang2vec_0.45
```

For a strictly antisymmetric signed graph instead of the legacy ALGEN/LAGO
sign convention:

```bash
SIGN_MODE=antisymmetric OUTPUT_DIR=graphs/mmarco_7_antisymmetric sbatch run_generate_language_graphs.sh
```

## Default Experiments

`run_train_mt5_decoder.sh` trains:

- attack decoder: `google/mt5-small`
- decoder training data: `yywwrr/mmarco_english_500k`
- architecture: ALGEN-style embedding projector + seq2seq generator

To train a multilingual attack decoder instead of the English-only decoder:

```bash
sbatch run_train_mt5_multilingual_decoder.sh
```

This keeps the total training budget at `450000` samples by default and splits
it approximately evenly over:

```text
English, French, German, Italian, Portuguese, Spanish, Dutch
```

Use this as a stronger decoder/backbone diagnostic, not as the primary LAGO
contribution.

`run_lago_ablation.sh` evaluates:

- victim/source encoder: `google/mt5-base`
- languages: English, French, German, Italian, Portuguese, Spanish, Dutch
- graph conditions: `none`, `lang2vec`, `random_lang2vec`, `ajsp`, `random_ajsp`
- sample sizes: `10 30 100 300 500 1000`

The important comparison is under the same mT5 decoder checkpoint:

```text
mT5 + no graph
mT5 + Lang2vec graph
mT5 + random Lang2vec graph
mT5 + AJSP graph
mT5 + random AJSP graph
```

For the multilingual fine-tuned decoder, run:

```bash
sbatch run_lago_multilingual_decoder_ablation.sh
```

To test graph connectivity instead of a single fixed topology:

```bash
sbatch run_cache_text_datasets.sh
sbatch run_generate_language_graphs.sh
sbatch run_lago_connectivity_ablation.sh
```

This runs:

```text
no graph
real lang2vec top-k graphs
random graphs with the same number of edges as each lang2vec top-k graph
real AJSP top-k graphs
random graphs with the same number of edges as each AJSP top-k graph
```

The default output is separate from the fixed-graph ablation:

```text
outputs/lago_ablation_connectivity/google_mt5-small/google_mt5-base/<graph_label>_<constraint>_train.../
```

For a smaller connectivity smoke test:

```bash
ALIGN_TRAIN_SAMPLES=100 \
GRAPH_FILES="graphs/mmarco_7/graph_lang2vec_topk6_signed.json graphs/mmarco_7/graph_ajsp_topk6_signed.json" \
sbatch run_lago_connectivity_ablation.sh
```

For the multilingual decoder, reuse the same script with the multilingual
checkpoint and a different output directory:

```bash
DECODER_CHECKPOINT_PATH=outputs/decoders/google_mt5-small/yywwrr_mmarco_english_yywwrr_mmarco_french_yywwrr_mmarco_german_yywwrr_mmarco_italian_yywwrr_mmarco_portuguese_yywwrr_mmarco_spanish_yywwrr_mmarco_dutch_maxlength32_train450000_batch128_lr0.0001_wd0.0001_epochs100 \
OUTPUT_DIR=outputs/lago_ablation_multilingual_decoder_connectivity \
sbatch run_lago_connectivity_ablation.sh
```

## Outputs

Decoder checkpoints:

```text
outputs/decoders/google_mt5-small/<dataset>_maxlength32_train.../
```

Alignment results:

```text
outputs/lago_ablation/google_mt5-small/google_mt5-base/<graph>_<constraint>_train.../
outputs/lago_ablation_connectivity/google_mt5-small/google_mt5-base/<graph_label>_<constraint>_train.../
```

Each alignment folder contains:

- `results.json`
- per-language `*_predictions.csv`
- per-language learned alignment matrices `*_W.pt`

## Notes

- This code removes hard-coded Hugging Face tokens from the old scripts.
- The built-in `lang2vec` graph is the 7-language graph at threshold `0.45`
  from lang2vec's precomputed syntactic distance matrix. The generator uses
  `../lang2vec-master` when available, and falls back to
  `resources/lang2vec/syntactic_distances.csv`.
- The built-in `ajsp` graph uses the same topology as the ALGEN AJSP branch;
  `asjp` is accepted as a compatibility alias. The AJSP graph is generated by
  parsing `resources/asjp/output.txt`, the ASJP62 LDND output.
- `random_lang2vec` and `random_ajsp` use the same number of edges as the
  corresponding real graph but rewire them randomly.
- Connectivity ablations should use `--graph_label` or
  `run_lago_connectivity_ablation.sh`; otherwise different `--graph_file`
  values with the same `graph_type` can be hard to distinguish.
- `run_cache_text_datasets.sh` materializes Hugging Face text datasets under
  `datasets/finetuning_decoder/<dataset_slug>/`. Run it once before large
  ablations to avoid repeated Hugging Face API calls and rate-limit failures.
