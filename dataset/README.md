# Datasets

Training/validation/test benchmark datasets and the eight independent external
evaluation datasets used in the APA benchmark study, organized by species.

The datasets (~9 GB) are **not** included in this repository. Download them from:

**[Download datasets](https://doi.org/10.5281/zenodo.22282470)** (download `PolyABench-HM-dataset.tar.gz`; link activates when the record is published)

All files are plain FASTA. The last colon-separated field of each header is the
class label (`1` = positive PAS sequence, `0` = negative). Each sequence is a
240-nt window centred on the candidate PAS.

## Directory layout

```
dataset/
├── Human/
│   ├── Benchmark/        # train_1.fasta, val.fasta, test.fasta (8:1:1 group-level split)
│   ├── PolyAID/          # All.fasta
│   ├── PolyAseq/         # All.fasta
│   ├── DeepGSR/          # All.fasta
│   ├── Dragon_PolyA/     # All.fasta
│   └── Omni_PolyA/       # All.fasta
└── Mouse/
    ├── Benchmark/        # train_1.fasta, val.fasta, test.fasta
    ├── DeepGSR/          # All.fasta
    ├── SPRET_EiJ/        # All.fasta (SP)
    └── C57BL_6J/         # All.fasta (BL)
```

## Benchmark datasets (Human/Mouse `Benchmark/`)

Constructed from the stringent cross-database PAS sets (supported by 3–4 of
PolyASite_sc, PolyASite_bk, PolyaDB and GENCODE) after internal-priming and
N-base filtering. Positive examples were augmented by shifting the window 1–25 nt
around authentic PASs (training set only); negative examples were sampled from
random intergenic regions, shuffled sequences and PAS-flanking regions shifted
60–150 nt away from annotated PASs. PASs were partitioned into
training/validation/test at an 8:1:1 ratio by split groups (same gene or
physically clustered intergenic PASs stay in the same split).

- `train_1.fasta`: full augmented training set (used to train all released models)
- `val.fasta`: validation set (model selection)
- `test.fasta`: held-out test set (internal benchmark)

## External evaluation datasets

Eight independent datasets from previously published PAS studies, used only for
external validation (never for training):

| Folder | Dataset | Species |
|---|---|---|
| `Human/PolyAID` | PolyAID dataset | Human |
| `Human/PolyAseq` | PolyA-Seq dataset | Human |
| `Human/DeepGSR` | DeepGSR human dataset | Human |
| `Mouse/DeepGSR` | DeepGSR mouse dataset | Mouse |
| `Human/Dragon_PolyA` | Dragon human poly(A) dataset (DeeReCT-PolyA) | Human |
| `Human/Omni_PolyA` | Omni human poly(A) dataset (DeeReCT-PolyA) | Human |
| `Mouse/SPRET_EiJ` | SPRET/EiJ (SP) mouse dataset (DeeReCT-PolyA) | Mouse |
| `Mouse/C57BL_6J` | C57BL/6J (BL) mouse dataset (DeeReCT-PolyA) | Mouse |

> Note: these files are derived from the corresponding publications' released
> data, reformatted to labelled FASTA. Redistribution is subject to the original
> sources' terms.
