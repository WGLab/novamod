# NovaMod

## Project background

NovaMod is a research codebase for **unsupervised detection of DNA/RNA modifications** from Oxford Nanopore sequencing data.

Nanopore signals (ionic current traces) are sensitive to chemical modifications such as DNA 5mC and RNA m6A, but many existing methods are supervised and depend on curated labels. In practice, label generation is expensive and model performance can shift across sequencing chemistry, basecaller versions, and experimental domains.

This project treats modification discovery as an **anomaly detection** task: learn a generative model of canonical (unmodified) signal, then score deviations as candidate modified events.

## Methods summary

The implemented workflow uses a CNN–Transformer variational autoencoder (VAE) and a config-driven streaming training/validation pipeline:

1. **Train on unmodified proxy data** (e.g., WGA DNA or IVT RNA) to learn a reference distribution of canonical signal patterns.
2. **Score per-instance anomalies** using reconstruction-based error metrics from the trained VAE.
3. **Aggregate read-level evidence to site level** for downstream ranking, enrichment analysis, and candidate prioritization.

This is intended as a label-light discovery framework for candidate nomination and regional pattern analysis, rather than a fully supervised end-to-end caller.

## Citation

If you use NovaMod, please cite:

> Zou, Y.; Ahsan, M.U.; Wang, K. Unsupervised Reference Modeling of Nanopore Signals for DNA/RNA Modification Detection. *Genes* **2026**, *17*(5), 525. https://doi.org/10.3390/genes17050525

## Data and code availability

This repository contains the code, configurations, and model artifacts used to reproduce the manuscript’s training and evaluation workflows. It will be maintained and versioned as the project develops. At present, the release is intended as a reproducible research framework with scripts, configurations, and trained models rather than a fully mature end-user package. More user-friendly packaging and broader dataset support are planned for future development.

Raw sequencing data availability is described in the manuscript; this repository ships code, configurations, and trained model artifacts only.

## Installation

```bash
conda env create -f environment.yml
conda activate novamod
```

`environment.yml` pins the versions actually used for the manuscript runs (Python 3.12.9, PyTorch 2.6.0+cu124, pysam 0.23.0, pod5 0.3.23, blosc2 3.3.1, numba 0.61.2, NumPy 2.2.5). A pip-only alternative is in `requirements.txt`, but it does not install `samtools`/`minimap2`, which `scripts/*.sh` require.

Two external tools are installed separately and are only needed to regenerate data from raw signal:

| Tool | Version used | Needed for |
| --- | --- | --- |
| [dorado](https://github.com/nanoporetech/dorado) | 1.0.2 | `scripts/basecall-*.sh` |
| [modkit](https://github.com/nanoporetech/modkit) | — | `scripts/modkit_extract.sh` |

## Data: the SignalBAM format

Everything under `training/` consumes **SignalBAM** files: ordinary coordinate-sorted BAMs in which each primary alignment additionally carries the read's *raw* signal in a custom tag. `training/bam_utils.get_read_info` reads these tags:

| Tag | Written by | Contents |
| --- | --- | --- |
| `SG` | `scripts/pod5_to_bam.py` | blosc2/ZSTD-compressed `int16` raw signal |
| `mv`, `ts` | dorado (`--emit-moves`) | move table and signal start offset |
| `sm`, `sd`, `sv` | dorado | signal shift, scale, and normalisation type |
| `co`, `cs`, `pf`, `pc` | `scripts/pod5_to_bam.py` | POD5 calibration / predicted scaling |

Reads missing `mv`/`ts`/`sm`/`sd`/`sv` are silently skipped by the dataset classes, so the move table and scaling tags must survive basecalling *and* alignment.

### Generating SignalBAM files

The four steps below produce the `*.final.bam` files listed in `training/data_manifest.csv`.

```bash
# 1. Basecall with move tables (dorado). Keeps mv/ts/sm/sd/sv.
#    basecall-wt.sh emits mod calls (for orthogonal labels);
#    basecall-ivt.sh omits them (unmodified proxy data).
sbatch scripts/basecall-ivt.sh

# 2. Align, carrying all tags through (samtools fastq -T "*", minimap2 -y),
#    then strip MM/ML so the signal model never sees basecaller mod calls.
sbatch scripts/align_nomod.sh

# 3. Embed raw signal into the aligned BAM -> SignalBAM.
sbatch scripts/generate_features.sh <sample> <aligned_bam> <pod5_dir> <outdir> [threads]

# 4. (optional) Keep a MM/ML-bearing copy for modkit-derived labels.
sbatch scripts/align_mod.sh
```

Step 3 is the one that creates the `SG` tag. It runs `scripts/pod5_to_bam.py`, which writes one chunk BAM per 100k reads; `generate_features.sh` then merges and sorts them with `samtools cat | samtools sort` into the final indexed SignalBAM.

### Labels

Validation needs labels, in one of two forms selected by each dataset's `method`:

- `method: "site"` — a per-site **BED6** file; the score column (column 5) is the label and column 6 is the strand. Build these with `scripts/make_motif_bed.py`, or convert an existing BED4 with `scripts/bed4-to-bed6.sh`.
- `method: "nt"` — a per-read TSV from `scripts/modkit_extract.sh`, keyed by `read_id` / `forward_read_position` with `mod_qual` as the label.
- `method: "region"` — no labels; scores every position in a `chrom,start,end` window (requires a `.bai` index).

### Dataset manifest

`training/data_manifest.csv` maps a short `data_id` to a SignalBAM, its reference FASTA, and an optional label file. **All paths in the committed manifest are absolute paths on the lab cluster** and must be repointed at your own copies before anything will run. Columns:

`data_id, condition (ref|unmod|mod), reference (data_id of the ref row), type (dna|rna), size, file_path, label_path, notes`


## Repository structure

- `training/`
  - `train.py` — training entrypoint (config-driven).
  - `val.py` — validation / anomaly-scoring entrypoint; writes Parquet.
  - `train.sh`, `val.sh` — SLURM wrappers (`sbatch train.sh <config>`).
  - `dataset_utils.py` — streaming datasets (training + the three validation modes).
  - `bam_utils.py` — SignalBAM parsing, reference preprocessing, `Read` class.
  - `feature_utils.py` — numba k-mer signal statistics and k-mer encode/decode.
  - `data_manifest.csv` — dataset bookkeeping.
  - `configs/` — JSON experiment configs (`train.*.json`, `val.*.json`).
  - `models/model_v1_cnn_t.py` — CNN–Transformer VAE.
  - `state_dicts/` — trained checkpoints, one per run and epoch.
  - `metrics.ipynb` — ROC/PRC/k-mer analysis; produces `figures/`.
  - `supervised-baseline.ipynb` — supervised comparison baseline.
  - `val-*-archive-dont-change.ipynb` — archived per-dataset validation analyses.
  - `figures/` — figures generated by the notebooks.
- `scripts/` — preprocessing: basecalling, alignment, SignalBAM generation, label preparation.

## Reproducibility quick start

From `training/`:

```bash
cd training
python train.py --config configs/train.example.json   # -> state_dicts/model_v1_cnn_t/example-epoch{1..20}.pt
python val.py   --config configs/val.example.json     # -> validation/example-*.pq
```

Cluster equivalents:

```bash
cd training
sbatch train.sh configs/train.example.json
sbatch val.sh   configs/val.example.json
```

`train.example.json` uses run name `example` specifically so it cannot overwrite the released checkpoints. It otherwise mirrors the `online_test9` recipe (9-mers, `d_model=256`, NLL reconstruction, β warmed to 1e-3 over 10 epochs). Before running, edit `configs/val.example.json` to point `spec` at a real label BED.

To score with a **released** checkpoint instead of training your own, use the run-specific configs, e.g.:

```bash
python val.py --config configs/val.test9.json          # online_test9, epoch 20
python val.py --config configs/val.test10-hg004.json   # online_test10, epoch 1
```

### Config invariants

- `data.sampling.kmer_len` must equal `model.kwargs.seq_len`, and `kmer_len == 2 * flank + 1`. `train.py`/`val.py` raise on a mismatch.
- `kmer_len` must be **odd** (a centre base must exist) and **≤ 15**: k-mer codes are packed base-4 into signed `int32`.
- A validation config's `model.kwargs` must match the checkpoint it loads, or `load_state_dict` fails on shape.
- `run.run_name` + `val.checkpoint_epoch` select `state_dicts/<model>/<run_name>-epoch<N>.pt`. Training **overwrites** that path, so give new runs new names.

### Run naming

See `training/README.txt` for what each `online_test*` / `static_test*` run was.

## Notes

- Checkpoints in `training/state_dicts/` are committed directly, so a clone is large (~4.6 GB of history). Cloning with `--depth 1` is much faster if you only need the current files.
- `val.py` writes one Parquet file per validation dataset with `score_recon`, `score_kl`, `labels`, `kmer`, `embeddings`, plus position metadata.
