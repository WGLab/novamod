#!/usr/bin/env python
"""Validation/inference entrypoint for VAE models using JSON config."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dataset_utils as du
import feature_utils as fu
from train import LOGVAR_MAX, LOGVAR_MIN, load_json, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAE validation on configured datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/val.example.json"),
        help="Path to validation config JSON",
    )
    return parser.parse_args()


def build_model(cfg: Dict[str, Any]) -> tuple[str, torch.nn.Module]:
    sampling_k = int(cfg["data"]["sampling"]["kmer_len"])
    model_seq_len = int(cfg["model"]["kwargs"].get("seq_len", sampling_k))
    if model_seq_len != sampling_k:
        raise ValueError(
            f"Config mismatch: data.sampling.kmer_len={sampling_k} but model.kwargs.seq_len={model_seq_len}."
        )

    module = importlib.import_module(cfg["model"]["module"])
    importlib.reload(module)
    model = module.VAE(**cfg["model"]["kwargs"])
    print(f"Total parameters: {sum([x.numel() for x in model.parameters()])}")
    return module.META.name, model


def build_val_loader(cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> DataLoader:
    bam, ref, seq_type = load_manifest(cfg["data"]["manifest_path"], dataset_cfg["data_id"])

    method = dataset_cfg["method"]
    sampling_cfg = cfg["data"]["sampling"]
    kmer_len = int(sampling_cfg["kmer_len"])
    normalize_signal = bool(sampling_cfg.get("normalize_signal", True))
    max_lines = int(dataset_cfg.get("max_lines", 10000))

    if method == "site":
        ds = du.SignalBAMRefPosValidationDataset(
            bam_path=bam,
            ref_fasta_path=ref,
            seq_type=seq_type,
            pos_list=dataset_cfg["spec"],
            max_lines=max_lines,
            kmer_len=kmer_len,
            normalize_signal=normalize_signal,
        )
    elif method == "nt":
        ds = du.SignalBAMkmerValidationDataset(
            bam_path=bam,
            ref_fasta_path=ref,
            seq_type=seq_type,
            labels_path_tsv=dataset_cfg["spec"],
            max_lines=max_lines,
            kmer_len=kmer_len,
            normalize_signal=normalize_signal,
        )
    elif method == "region":
        chrom, start, end = dataset_cfg["spec"]
        ds = du.SignalBAMRegionDataset(
            bam_path=bam,
            ref_fasta_path=ref,
            seq_type=seq_type,
            chrom=chrom,
            start=int(start),
            end=int(end),
            kmer_len=kmer_len,
            normalize_signal=normalize_signal,
            fake_label=float(dataset_cfg.get("fake_label", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported validation method: {method}")

    loader_cfg = cfg.get("val", {}).get("loader", {})
    return DataLoader(
        ds,
        batch_size=int(loader_cfg.get("batch_size", 256)),
        num_workers=int(loader_cfg.get("num_workers", 10)),
    )


def reconstruction_score_per_sample(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    recon: str,
    huber_delta: float,
    nll_var: float,
    kmer_weight_center_base: float,
    kmer_weight_sd: float,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [B, T, F], got {x.shape}.")

    seq_len = x.shape[1]
    kmer_pos = torch.arange(seq_len, device=x.device, dtype=x.dtype) - (seq_len - 1) / 2.0
    if kmer_weight_sd <= 0:
        raise ValueError("kmer_weight_sd must be > 0.")
    kmer_weights = torch.exp(-0.5 * ((kmer_pos - kmer_weight_center_base) / kmer_weight_sd) ** 2)
    kmer_weights = kmer_weights / kmer_weights.sum()

    if recon == "mse":
        pointwise = F.mse_loss(x_hat, x, reduction="none")
    elif recon == "l1":
        pointwise = F.l1_loss(x_hat, x, reduction="none")
    elif recon == "huber":
        pointwise = F.huber_loss(x_hat, x, reduction="none", delta=huber_delta)
    elif recon == "nll":
        if nll_var <= 0:
            raise ValueError("nll_var must be > 0.")
        pointwise = -torch.distributions.Normal(x_hat, nll_var**0.5).log_prob(x)
    else:
        raise ValueError(f"Unsupported recon loss: {recon}")

    recon_per_timestep = pointwise.mean(dim=2)
    return torch.sum(recon_per_timestep * kmer_weights.unsqueeze(0), dim=1)


def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar_safe = logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    return 0.5 * torch.sum(torch.exp(logvar_safe) + mu**2 - 1.0 - logvar_safe, dim=1)


def _pos_rows_from_batch(pos: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(batch_size):
        row: Dict[str, Any] = {}
        for key, values in pos.items():
            if isinstance(values, torch.Tensor):
                row[key] = values[i].item()
            else:
                row[key] = values[i]
        rows.append(row)
    return rows


def anomaly_score(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    recon_cfg: Dict[str, Any],
) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    with torch.no_grad():
        for x, kmer, pos, y in loader:
            x = x.to(device)
            out = model(x)

            scores = reconstruction_score_per_sample(
                x=x,
                x_hat=out["x_hat"],
                recon=str(recon_cfg.get("recon", "mse")),
                huber_delta=float(recon_cfg.get("huber_delta", 1.0)),
                nll_var=float(recon_cfg.get("nll_var", 1.0)),
                kmer_weight_center_base=float(recon_cfg.get("kmer_weight_center_base", 0.0)),
                kmer_weight_sd=float(recon_cfg.get("kmer_weight_sd", 1.0)),
            ).detach().cpu().numpy()
            kl_vals = kl_per_sample(out["mu"], out["logvar"]).detach().cpu().numpy()

            embs = out["mu"].detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            kmers = [fu.decode_kmer(km) for km in kmer.detach().cpu()]
            pos_df = pd.json_normalize(_pos_rows_from_batch(pos, batch_size=x.shape[0]))

            batch_df = pd.DataFrame(
                {
                    "score_recon": scores,
                    "score_kl": kl_vals,
                    "labels": labels,
                    "kmer": kmers,
                    "embeddings": list(embs),
                }
            )
            batch_df = pd.concat([batch_df, pos_df], axis=1)
            table = pa.Table.from_pandas(batch_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    val_cfg = cfg.get("val", {})
    datasets = val_cfg.get("datasets", [])
    if not datasets:
        raise ValueError("Validation config requires val.datasets with one or more dataset entries.")

    checkpoint_epoch = int(val_cfg["checkpoint_epoch"])
    output_dir = Path(val_cfg.get("output_dir", "validation"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    model_name, model = build_model(cfg)
    run_name = cfg["run"]["run_name"]

    state_dict = Path("state_dicts") / model_name / f"{run_name}-epoch{checkpoint_epoch}.pt"
    print(f"Loading checkpoint: {state_dict}")
    pt = torch.load(state_dict, map_location=device)
    model.load_state_dict(pt["model_state_dict"])
    model = model.to(device)

    for ds_cfg in datasets:
        dataset_name = ds_cfg["name"]
        loader = build_val_loader(cfg, ds_cfg)
        out_path = output_dir / f"{run_name}-{dataset_name}.pq"
        print(f"Scoring dataset '{dataset_name}' -> {out_path}")
        anomaly_score(model, loader, device, out_path, recon_cfg=cfg.get("train", {}))


if __name__ == "__main__":
    main()
