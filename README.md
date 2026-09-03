# apa-polya-predictor

A standalone, production-ready APA/polyA site prediction tool for **Human** and **Mouse**, based on the `apa-polyA-benchmark` deep-learning models.

## Features

- **Input**: any FASTA file of DNA/RNA sequences.
- **Preprocessing**: sequences longer than 240 nt are cropped to a 240-nt window centered on the input; shorter sequences are padded with `N`.
- **Models**:
  - Traditional one-hot/k-mer models: `deeppolya`, `deerect`, `polyaid`, `deepgsr`, `pasnet`.
  - RNA-structure model: `deeppasta` (requires the `RNAshapes` binary; automatically called when selected).
  - Large language models: `DNABERT1`, `DNABERT2`, `RNABERT`, `UTRLM`, `GENA-LM-base`.
- **Ensemble**: `ensemble` keyword uses the greedy-forward-selected best model subset + gradient boosting.
- **Output**: tab-delimited text file with per-model probabilities, per-model binary labels, and ensemble prediction (when `-m ensemble`).

## Installation

The package requires **Python >= 3.8** (tested on Python 3.9).

### Option 1: pip install (recommended for users)

```bash
# Clone or copy the package
cd apa-polya-predictor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package itself
pip install -e .
```

For GPU support, use the CUDA requirements file instead:

```bash
pip install -r requirements-cuda.txt
pip install -e .
```

### Option 2: conda install (recommended for GPU)

```bash
conda env create -f environment.yml
conda activate apa-polya
pip install -e .
```

### Dependencies

Core runtime dependencies are listed in `pyproject.toml` and pinned in `requirements.txt`:

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `torch` (>= 2.0)
- `joblib`, `biopython`
- `transformers`, `tokenizers`, `huggingface-hub`, `accelerate`
- `tqdm`, `attrdict`, `packaging`

If you only install via `pip install -e .`, pip will pull compatible versions of these packages automatically. The `requirements.txt` files provide exact versions that have been tested.

## System requirements

- **OS**: Linux x86_64 (the bundled `RNAshapes` binary is compiled for Linux).
- **Python**: >= 3.8 (tested on Python 3.9).
- **GPU**: Optional; CUDA-capable GPU speeds up LLM inference significantly.

> **Note on DNABERT2**: DNABERT2's optional flash-attention kernel requires a
> specific Triton nightly and fails with several released Triton versions (or on
> CPU). This package therefore always runs DNABERT2 with standard (eager)
> attention, which is numerically equivalent and works on both CPU and GPU; the
> only cost is slightly slower DNABERT2 inference. No action is needed from the
> user.

## Weights

Pre-trained weights (~5 GB) are **not** included in this repository. Download them from:

**[Download pre-trained weights](https://example.com/PolyABench-HM-weights)** (placeholder link, to be updated)

After downloading, the weights are expected under the `weights/` directory:

```
weights/
├── Human/
│   ├── deeppolya_best_model.pt
│   ├── deerect_best_model.pt
│   ├── polyaid_best_model.pt
│   ├── deepgsr_best_model.pt
│   ├── pasnet_best_model.pt
│   ├── llm/
│   │   ├── DNABERT1_best_model.pt
│   │   ├── DNABERT2_best_model.pt
│   │   ├── RNABERT_best_model.pt
│   │   ├── UTRLM_best_model.pt
│   │   └── GENA-LM-base_best_model.pt
│   └── gb_ensemble_best.joblib       # greedy-forward-selected best ensemble
├── Mouse/
│   └── ... (same structure)
└── pretrained/llm/
    ├── dnabert1/
    ├── dnabert2/
    ├── rnabert/
    ├── utrlm/
    └── gena_lm_base/
```

> **Note for distribution**: the full weight directory is ~5 GB. It is not intended to be committed to Git; provide it as a separate download or shared folder.

## Quick start

An example FASTA file is provided in `examples/example.fa` (6 Human sequences, 3 positive and 3 negative polyA sites).

```bash
# Best ensemble (greedy-forward-selected subset + gradient boosting)
apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt

# Default: 5 traditional models, per-model probabilities and labels
apa-polya-predict -i examples/example.fa -s Human -o predictions.txt

# Mouse
apa-polya-predict -i examples/example.fa -s Mouse -o predictions.txt

# Single model
apa-polya-predict -i examples/example.fa -s Human -m deeppolya -o predictions.txt

# All 11 models, per-model probabilities
apa-polya-predict -i examples/example.fa -s Human -m all+llm -o predictions.txt

# Include deeppasta (RNAshapes must be installed and in PATH, or use --rnashapes_bin)
apa-polya-predict -i examples/example.fa -s Human -m all+llm,deeppasta -o predictions.txt
apa-polya-predict -i examples/example.fa -s Human -m all+llm,deeppasta --rnashapes_bin /path/to/RNAshapes -o predictions.txt

# Force CPU / GPU
apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt --device cpu
```

## CLI options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input FASTA file (required) |
| `-s, --species` | `Human` or `Mouse` (required) |
| `-m, --models` | `ensemble` (greedy-forward-selected best subset + GB), `all`, `all+llm`, or comma-separated list |
| `-t, --threshold` | Decision threshold (default: 0.5; for `-m ensemble`, defaults to the validation-tuned threshold) |
| `-o, --output` | Output file path (required) |
| `--weights_dir` | Custom weights directory |
| `--batch_size` | Batch size for traditional models (default: 256) |
| `--llm_batch_size` | Batch size for LLMs (default: 32) |
| `--device` | `cpu` or `cuda` (auto-detected) |
| `--output_format` | `tsv` (default) or `csv` |

## Output format

A tab-delimited file with one row per input sequence:

```
header  deeppolya_prob  deeppolya_pred  deerect_prob  deerect_pred  ...  ensemble_prob  ensemble_pred
seq1    0.0034          0               0.0003        0             ...  0.2498         0
```

- `*_prob`: probability of being a positive poly(A) site from each model.
- `*_pred`: binary prediction from each model (`1` if `*_prob >= threshold`, else `0`).
- `ensemble_prob`: aggregated probability from the greedy-forward-selected best ensemble (only when `-m ensemble`).
- `ensemble_pred`: binary prediction from the ensemble (`1` if `ensemble_prob >= threshold`, else `0`).

## Programmatic use

```python
import joblib
from apa_polya_predictor import Predictor
from apa_polya_predictor.cli import apply_best_ensemble

predictor = Predictor(
    weights_dir="weights",
    species="Human",
    models=["ensemble"],  # best greedy-forward-selected subset
    device="cuda",
)

df = predictor.predict_fasta("input.fa")

# Apply the shipped best-ensemble meta-learner
meta_bundle = joblib.load("weights/Human/gb_ensemble_best.joblib")
df, threshold = apply_best_ensemble(df, meta_bundle)
print(df[["header", "ensemble_prob", "ensemble_pred"]])
```

## Best ensemble selection

The default `-m ensemble` uses the greedy-forward-selected best model subset for each species:

- **Human** (7 models): `deeppasta`, `DNABERT2`, `pasnet`, `GENA-LM-base`, `polyaid`, `UTRLM`, `DNABERT1`
- **Mouse** (8 models): `deeppasta`, `pasnet`, `DNABERT2`, `polyaid`, `DNABERT1`, `UTRLM`, `deeppolya`, `deepgsr`

These subsets were chosen by greedy-forward selection with gradient boosting (`script/V07_DL_Star/07_01_18_run_forward_selection_v2.sh` in the APA_yangqingxin project): selection was anchored at DeepPASTA (the best single model by test AUPRC), and members and subset size were selected by validation AUPRC. The corresponding gradient-boosting meta-learners are saved as `weights/{Human,Mouse}/gb_ensemble_best.joblib`.

## Updating the best ensemble (maintainers only)

If you update the base models or re-run the forward-selection analysis, regenerate the best-ensemble meta-learner:

```bash
python scripts/export_best_ensemble.py \
  --species Human \
  --output_dir weights
```

## License

MIT
