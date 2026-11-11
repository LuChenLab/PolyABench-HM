# Compute environment

All model training, benchmarking and analyses reported in the accompanying
study were reproduced and run on the following server configuration.
The package itself has no specific hardware requirement beyond what is listed
in the README (Linux x86_64; GPU optional); this file is provided for
transparency and reproducibility.

## Hardware

| Component | Specification |
|---|---|
| OS | Ubuntu 22.04.5 LTS (kernel 6.8.0-117-generic, x86_64) |
| CPU | 2 × Intel Xeon E5-2650 v4 @ 2.20 GHz (12 cores/socket, 2 threads/core; 24 cores / 48 threads total) |
| RAM | 251 GB |
| GPU | 1 × NVIDIA GeForce RTX 3090, 24 GB VRAM |
| GPU driver | NVIDIA driver 580.159.03 (CUDA 13.0 capable) |

## Software environment (benchmark/analysis conda environment)

| Package | Version |
|---|---|
| Python | 3.9.0 |
| PyTorch | 2.0.1+cu117 (CUDA runtime 11.7) |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |
| SciPy | 1.13.1 |
| Transformers | 4.41.0 |
| RNAshapes | bundled Linux x86_64 binary (DeepPASTA structure features) |

Exact pinned dependencies for installation are given in `requirements.txt`
(default PyPI torch build) and `requirements-cuda.txt` / `environment.yml`
(CUDA 11.8 toolchain, matching the environment in which the models were
originally trained). All reported results are robust to the choice between
the cu117 and cu118 torch builds; the shipped `gb_ensemble_best.joblib`
meta-learners require scikit-learn 1.5.x for exact probability reproduction.

## Notes

- GPU inference of the LLM-based models (DNABERT1/2, RNABERT, UTR-LM,
  GENA-LM-base) is substantially faster on CUDA; all models also run on CPU.
- DeepPASTA requires the bundled `RNAshapes` binary, which is compiled for
  Linux x86_64; other operating systems are not supported.
