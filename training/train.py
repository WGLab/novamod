#!/usr/bin/env python
"""
Online training entrypoint for VAE models.

"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import dataset_utils as du


LOGVAR_MIN = -20.0
LOGVAR_MAX = 10.0


@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    beta_max: float = 1.0
    beta_warmup_epochs: int = 10
    grad_clip: float = 1.0
    recon: str = "mse"
    huber_delta: float = 1.0
    nll_var: float = 1.0
    kmer_weight_center_base: float = 0.0
    kmer_weight_sd: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE model on online iterable dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_online.example.json"),
        help="Path to training config JSON",
    )
    return parser.parse_args()


def load_manifest(path: str, data_id: str) -> tuple[str, str, str]:
    import pandas as pd

    df = pd.read_csv(path).set_index("data_id")
    row = df.loc[data_id]
    if row.condition == "ref":
        raise ValueError("Reference file provided as training input.")
    return row.file_path, df.loc[row.reference].file_path, row.type


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_loader(cfg: Dict[str, Any]) -> tuple[Any, DataLoader]:
    manifest_path = cfg["data"]["manifest_path"]
    data_id = cfg["data"]["data_id"]
    bam, ref, seq_type = load_manifest(manifest_path, data_id)

    sampling_cfg = du.SamplingConfig(**cfg["data"]["sampling"])
    dataset = du.SignalBAMkmerIterableDataset(
        bam_path=bam,
        ref_fasta_path=ref,
        cfg=sampling_cfg,
        seq_type=seq_type,
        max_samples_per_epoch=cfg["data"].get("max_samples_per_epoch"),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["loader"].get("batch_size", 256),
        num_workers=cfg["loader"].get("num_workers", 20),
        persistent_workers=cfg["loader"].get("persistent_workers", True),
    )
    return dataset, loader


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
    return module.META.name, model


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon: str = "mse",
    huber_delta: float = 1.0,
    nll_var: float = 1.0,
    kmer_weight_center_base: float = 0.0,
    kmer_weight_sd: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if x.dim() != 3:
        raise ValueError(f"Expected reconstruction tensors shaped [B, T, F], got {x.shape}.")

    seq_len = x.shape[1]
    kmer_pos = torch.arange(seq_len, device=x.device, dtype=x.dtype) - (seq_len - 1) / 2.0
    if kmer_weight_sd <= 0:
        raise ValueError("kmer_weight_sd must be > 0.")
    kmer_weights = torch.exp(-0.5 * ((kmer_pos - kmer_weight_center_base) / kmer_weight_sd) ** 2)
    kmer_weights = kmer_weights / kmer_weights.sum()

    if recon == "mse":
        pointwise_recon = F.mse_loss(x_hat, x, reduction="none")
    elif recon == "l1":
        pointwise_recon = F.l1_loss(x_hat, x, reduction="none")
    elif recon == "huber":
        pointwise_recon = F.huber_loss(x_hat, x, reduction="none", delta=huber_delta)
    elif recon == "nll":
        if nll_var <= 0:
            raise ValueError("nll_var must be > 0.")
        pointwise_recon = -torch.distributions.Normal(x_hat, nll_var**0.5).log_prob(x)
    else:
        raise ValueError(f"Unsupported recon loss: {recon}")

    recon_per_timestep = pointwise_recon.mean(dim=2)
    recon_loss = torch.sum(recon_per_timestep * kmer_weights.unsqueeze(0), dim=1).mean()

    logvar_safe = logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar_safe) + mu**2 - 1.0 - logvar_safe, dim=1))
    loss = recon_loss + beta * kl
    return {"loss": loss, "recon": recon_loss, "kl": kl}


def train_vae(
    model: torch.nn.Module,
    dataset: Any,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    run_name: str,
    cfg: TrainConfig,
) -> torch.nn.Module:
    model.to(device)
    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(parameters, lr=cfg.lr)
    skipped_nonfinite_batches = 0

    for epoch in range(1, cfg.epochs + 1):
        dataset.set_epoch(epoch)
        beta = (
            cfg.beta_max * min(1.0, epoch / cfg.beta_warmup_epochs)
            if cfg.beta_warmup_epochs > 0
            else cfg.beta_max
        )

        model.train()
        tr_total = tr_recon = tr_kl = 0.0
        n_batches = 0

        for x, _, _ in loader:
            x = x.to(device)
            if not torch.isfinite(x).all():
                skipped_nonfinite_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            if not all(torch.isfinite(out[key]).all() for key in ("x_hat", "mu", "logvar")):
                skipped_nonfinite_batches += 1
                continue
            losses = vae_loss(
                x,
                out["x_hat"],
                out["mu"],
                out["logvar"],
                beta=beta,
                recon=cfg.recon,
                huber_delta=cfg.huber_delta,
                nll_var=cfg.nll_var,
                kmer_weight_center_base=cfg.kmer_weight_center_base,
                kmer_weight_sd=cfg.kmer_weight_sd,
            )
            losses["loss"].backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            tr_total += float(losses["loss"].detach().cpu())
            tr_recon += float(losses["recon"].detach().cpu())
            tr_kl += float(losses["kl"].detach().cpu())
            n_batches += 1

        tr_total /= max(1, n_batches)
        tr_recon /= max(1, n_batches)
        tr_kl /= max(1, n_batches)

        print(
            f"Epoch {epoch:03d} | beta={beta:.4f} | "
            f"train total {tr_total:.5f} (recon {tr_recon:.5f}, kl {tr_kl:.5f})"
        )
        if skipped_nonfinite_batches:
            print(f"  skipped {skipped_nonfinite_batches} batch(es) with non-finite tensors so far")

        store_path = Path("state_dicts") / model_name / f"{run_name}-epoch{epoch}.pt"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "beta": beta,
                "epoch": epoch,
                "model": model_name,
                "run": run_name,
            },
            store_path,
        )

    return model


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    dataset, loader = build_loader(cfg)
    model_name, model = build_model(cfg)

    train_cfg = TrainConfig(**cfg["train"])
    run_name = cfg["run"]["run_name"]

    print(f"Training {model_name} with run '{run_name}' ...")
    train_vae(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        model_name=model_name,
        run_name=run_name,
        cfg=train_cfg,
    )


if __name__ == "__main__":
    main()
