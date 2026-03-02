# NovaMod

## Project background

NovaMod is a research codebase for **unsupervised detection of DNA/RNA modifications** from Oxford Nanopore sequencing data.

Nanopore signals (ionic current traces) are sensitive to chemical modifications such as DNA 5mC and RNA m6A, but many existing methods are supervised and depend on curated labels. In practice, label generation is expensive and model performance can shift across sequencing chemistry, basecaller versions, and experimental domains.

Following the manuscript, this project treats modification discovery as an **anomaly detection** task: learn a generative model of canonical (unmodified) signal, then score deviations as candidate modified events.

## Methods summary

The implemented workflow uses a CNN–Transformer variational autoencoder (VAE) and a config-driven training/validation pipeline:

1. **Train on unmodified proxy data** (e.g., WGA DNA or IVT RNA) to learn a reference distribution of canonical signal patterns.
2. **Score per-instance anomalies** using reconstruction-based error metrics from the trained VAE.
3. **Aggregate read-level evidence to site level** for downstream ranking, enrichment analysis, and candidate prioritization.

This is intended as a label-light discovery framework for candidate nomination and regional pattern analysis, rather than a fully supervised end-to-end caller.

## Data and code availability

This repository contains the code, configurations, and model artifacts used to reproduce the manuscript’s training and evaluation workflows.

### Repository structure

- `training/`
  - `train.py` — main training entrypoint (config-driven).
  - `val.py` — validation/evaluation and anomaly-scoring entrypoint (config-driven).
  - `train.sh`, `val.sh` — batch wrappers for cluster execution.
  - `dataset_utils.py`, `feature_utils.py`, `bam_utils.py` — data loading, feature extraction, and BAM/signal processing utilities.
  - `data_manifest.csv` — dataset bookkeeping used by training/evaluation workflows.
  - `configs/` — JSON experiment configurations for training and validation runs.
  - `models/` — model definitions (including CNN–Transformer VAE implementation).
  - `state_dicts/` — saved model checkpoints and related exported artifacts.
  - `evaluation.ipynb`, `supervised-baseline.ipynb` — analysis notebooks.
- `scripts/`
  - preprocessing and workflow scripts for basecalling, alignment, feature generation, and utility conversions.
- `paper/`
  - manuscript sources and figures.

### Where to find models and scripts

- **Model code:** `training/models/`
- **Trained checkpoints/artifacts:** `training/state_dicts/`
- **Runnable training/evaluation scripts:** `training/train.py`, `training/val.py`, `training/train.sh`, `training/val.sh`
- **Data-preparation and pipeline helpers:** `scripts/`

## Reproducibility quick start

From the repository root:

```bash
cd training
python train.py --config configs/train.example.json
python val.py --config configs/val.example.json
```

Cluster examples:

```bash
cd training
sbatch train.sh configs/train.example.json
sbatch val.sh configs/val.example.json
```

## Notes

- The repository is organized around **config-driven experiments**.
- JSON files in `training/configs/` define dataset paths, model settings, and run parameters for each experiment variant.
