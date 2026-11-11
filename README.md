# PolyABench-HM

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.8-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20x86__64-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-Zenodo-orange)](https://doi.org/10.5281/zenodo.22282470)

A standalone, production-ready APA/polyA site prediction tool for **Human** and **Mouse**, built on the `apa-polyA-benchmark` deep-learning models. Provides the `apa-polya-predict` command-line tool and a Python API.

## Features

- **Input**: any FASTA file of DNA/RNA sequences. Sequences longer than 240 nt are cropped to a 240-nt centered window; shorter ones are padded with `N`.
- **Models**:
  - Traditional one-hot/k-mer: `deeppolya`, `deerect`, `polyaid`, `deepgsr`, `pasnet`
  - RNA-structure: `deeppasta` (requires the bundled `RNAshapes` binary, Linux only)
  - Large language models: `DNABERT1`, `DNABERT2`, `RNABERT`, `UTRLM`, `GENA-LM-base`
- **Ensemble**: `-m ensemble` uses a greedy-forward-selected best model subset with gradient boosting.
- **Output**: tab-delimited file with per-model probabilities, binary labels, and ensemble prediction.

## Installation

Requires **Python ≥ 3.8** (tested on 3.9) and **Linux x86_64** (the bundled `RNAshapes` binary is Linux-only). A CUDA-capable GPU is optional but significantly speeds up LLM inference.

### Option 1: pip (recommended for users)

```bash
cd PolyABench-HM

# Virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

For GPU support, use `requirements-cuda.txt` instead (CUDA 11.8 build, matching `environment.yml`):

```bash
pip install -r requirements-cuda.txt
pip install -e .
```

### Option 2: conda (recommended for GPU)

```bash
conda env create -f environment.yml
conda activate apa-polya
pip install -e .
```

Core dependencies (`numpy`, `pandas`, `scikit-learn`, `scipy`, `torch >= 2.0`, `biopython`, `transformers`, etc.) are declared in `pyproject.toml`; the `requirements*.txt` files pin tested versions. The full hardware/software configuration used in the accompanying study is documented in [`ENVIRONMENT.md`](ENVIRONMENT.md).

> **Note on DNABERT2**: the optional flash-attention kernel is disabled (it fails with several Triton versions and on CPU). DNABERT2 runs with standard eager attention — numerically equivalent, works on CPU and GPU, only slightly slower. No action needed.

## Weights

Pre-trained weights (~5 GB) are **not** included in this repository:

**[⬇ Download pre-trained weights (Zenodo)](https://doi.org/10.5281/zenodo.22282470)** — the record contains both `PolyABench-HM-weights.tar.gz` and `PolyABench-HM-dataset.tar.gz`.

Place them under `weights/`:

```
weights/
├── Human/
│   ├── deeppolya_best_model.pt  ... (5 traditional models)
│   ├── llm/                     ... (5 LLM checkpoints)
│   └── gb_ensemble_best.joblib  # greedy-forward-selected best ensemble
├── Mouse/                       ... (same structure)
└── pretrained/llm/              # dnabert1, dnabert2, rnabert, utrlm, gena_lm_base
```

## Quick start

An example FASTA file is provided at `examples/example.fa` (6 Human sequences: 3 positive, 3 negative polyA sites).

```bash
# Best ensemble (recommended)
apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt

# Default: 5 traditional models
apa-polya-predict -i examples/example.fa -s Human -o predictions.txt

# Mouse
apa-polya-predict -i examples/example.fa -s Mouse -o predictions.txt

# Single model
apa-polya-predict -i examples/example.fa -s Human -m deeppolya -o predictions.txt

# All 11 models
apa-polya-predict -i examples/example.fa -s Human -m all+llm -o predictions.txt

# Include deeppasta (RNAshapes in PATH, or pass --rnashapes_bin)
apa-polya-predict -i examples/example.fa -s Human -m all+llm,deeppasta --rnashapes_bin /path/to/RNAshapes -o predictions.txt

# Force CPU
apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt --device cpu
```

## CLI options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input FASTA file (required) |
| `-s, --species` | `Human` or `Mouse` (required) |
| `-m, --models` | `all` (default, 5 traditional models), `all+llm` (all 11), `ensemble`, or comma-separated list |
| `-t, --threshold` | Decision threshold (default: 0.5; overrides the tuned threshold for `-m ensemble`) |
| `-o, --output` | Output file path (required) |
| `--weights_dir` | Custom weights directory |
| `--batch_size` | Batch size for traditional models (default: 256) |
| `--llm_batch_size` | Batch size for LLMs (default: 32) |
| `--device` | `cpu` or `cuda` (auto-detected if omitted) |
| `--rnashapes_bin` | Path to the RNAshapes binary (searched in PATH if omitted) |
| `--output_format` | `tsv` (default) or `csv` |

## Decision thresholds

All models use a default threshold of 0.5. As an optional reference, [`optimal_thresholds.txt`](optimal_thresholds.txt) lists the validation-MCC-optimal threshold for each model and species; apply them by binarising the `*_prob` columns at the listed cutoffs. For `-m ensemble`, the tuned threshold stored in `gb_ensemble_best.joblib` is used automatically unless `-t` is given.

## Output format

One row per input sequence:

```
header  deeppolya_prob  deeppolya_pred  deerect_prob  deerect_pred  ...  ensemble_prob  ensemble_pred
seq1    0.0034          0               0.0003        0             ...  0.2498         0
```

- `*_prob`: probability of being a positive poly(A) site.
- `*_pred`: binary label (`1` if `*_prob >= threshold`).
- `ensemble_prob` / `ensemble_pred`: ensemble outputs (only with `-m ensemble`).

## Programmatic use

```python
import joblib
from apa_polya_predictor import Predictor
from apa_polya_predictor.cli import apply_best_ensemble

predictor = Predictor(
    weights_dir="weights",
    species="Human",
    models=["ensemble"],
    device="cuda",
)

df = predictor.predict_fasta("input.fa")
meta_bundle = joblib.load("weights/Human/gb_ensemble_best.joblib")
df, threshold = apply_best_ensemble(df, meta_bundle)
print(df[["header", "ensemble_prob", "ensemble_pred"]])
```

## Best ensemble selection

The `-m ensemble` subsets, chosen by greedy-forward selection (anchored at DeepPASTA, the best single model by test AUPRC) with gradient boosting:

- **Human** (7 models): `deeppasta`, `DNABERT2`, `pasnet`, `GENA-LM-base`, `polyaid`, `UTRLM`, `DNABERT1`
- **Mouse** (8 models): `deeppasta`, `pasnet`, `DNABERT2`, `polyaid`, `DNABERT1`, `UTRLM`, `deeppolya`, `deepgsr`

Meta-learners are stored as `weights/{Human,Mouse}/gb_ensemble_best.joblib`, with tuned thresholds inside each bundle.

## Updating the best ensemble (maintainers only)

```bash
python scripts/export_best_ensemble.py \
  --benchmark_root /path/to/apa-polyA-benchmark \
  --species Human \
  --output_dir weights
```

## License

[MIT](LICENSE)
