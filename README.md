# novamod

This repository contains model training and validation workflows for signal-based modification modeling.

## Repository layout

- `training/training-online.py`: config-driven online training entrypoint.
- `training/configs/`: explicit JSON configs for experiments.
- `training/validation-online_evalOnly.py`: evaluation and anomaly scoring workflow.
- `training/dataset_utils.py`, `training/feature_utils.py`, `training/bam_utils.py`: core data and feature utilities.
- `training/models/`: model definitions.
- `training/state_dicts/`: saved checkpoints (large artifacts).
- `training/*.ipynb`: exploratory and analysis notebooks.

## Implemented organization changes

This repo now uses **explicit experiment configs** for training:

- Training parameters are no longer hard-coded inside `training-online.py`.
- Use `training/configs/train_online.example.json` as a template for runs.
- Slurm launcher `training/train.sh` accepts a config path argument and passes it through.

Example:

```bash
cd training
python training-online.py --config configs/train_online.example.json
```

## Next practical improvements

1. Separate source code and generated artifacts into distinct folders.
2. Standardize script naming (`train_online.py`, `validate_online.py`) while keeping backwards-compatible wrappers.
3. Add CI checks for syntax and basic utility tests.
