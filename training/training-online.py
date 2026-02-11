#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bam_utils import *

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")


# ### Loading Data

# In[2]:


from torch.utils.data import DataLoader
import dataset_utils as du

dna_hs = "/mnt/isilon/wang_lab/umair/deepmod/model_features/r10_5khz_drd.v1_full_bam/HG002_WGA/HG002_WGA.final.bam"
rna_hs = "/home/zouy1/projects/RNAmod/VAE/data/RNA/20240411_HEK293T_IVT/20240411_HEK293T_IVT.final.bam"
ref_hs = "/home/zouy1/software/reference/GRCh38.primary_assembly.genome.fa"

#dna_oligos = "/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/control_rep1/control_rep1.subset.final.bam"
dna_oligos = "/mnt/isilon/wang_lab/umair/projects/5hmC/ONT_oligos/features/full_data/full_read/test/CG/control_rep1/control_rep1_CG.bam"
rna_oligos = "/home/zouy1/projects/RNAmod/VAE/data/oligos/RNA/control_rep1/control_rep1.subset.final.bam"
ref_oligos_dna = '/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/all_5mers.fa'
ref_oligos_rna = '/home/zouy1/projects/RNAmod/VAE/data/oligos/RNA/sampled_context_strands.fa'

#specify training data
ds = du.SignalBAMkmerIterableDataset(
    bam_path=dna_oligos,           # data
    ref_fasta_path=ref_oligos_dna, # reference
    cfg=du.SamplingConfig(
        kmer_len=7,
        flank=3,
        positions_per_read=1000,    # 10
        max_kmer_count=50_000,
        normalize_signal=True,
        seed=42,
    ),
    seq_type="dna",                # type
    max_samples_per_epoch=None,
)

loader = DataLoader(
    ds,
    batch_size=256,
    num_workers=20,
    persistent_workers=True,
)


# ### Training

# In[3]:


# Loss function
def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0, recon: str = "mse") -> Dict[str, torch.Tensor]:
    if recon == "mse":
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    elif recon == "l1":
        recon_loss = F.l1_loss(x_hat, x, reduction="mean")

    # KL divergence for diagonal Gaussian q(z|x) vs N(0,1)
    kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
    loss = recon_loss + beta * kl
    return {"loss": loss, "recon": recon_loss, "kl": kl}


# In[4]:


from torch.optim import AdamW

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    beta_max: float = 1.0        # final beta for KL
    beta_warmup_epochs: int = 10 # linear warmup for KL
    grad_clip: float = 1.0
    recon: str = "mse"


def train_vae(model, dataset, loader, device, model_name, run_name, cfg: TrainConfig):
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        dataset.set_epoch(epoch)   # change rng seed for data sampling
        beta = cfg.beta_max * min(1.0, epoch / cfg.beta_warmup_epochs) if cfg.beta_warmup_epochs > 0 else cfg.beta_max

        #train
        model.train()
        tr_total = tr_vae = tr_recon = tr_kl = tr_msm = 0.0
        n_batches = 0

        for x, _, _ in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)

            out = model(x)
            vae_losses = vae_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=beta, recon=cfg.recon)

            total_loss = vae_losses["loss"]
            total_loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            tr_total += float(total_loss.detach().cpu())
            tr_vae   += float(vae_losses["loss"].detach().cpu())
            tr_recon += float(vae_losses["recon"].detach().cpu())
            tr_kl    += float(vae_losses["kl"].detach().cpu())
            n_batches += 1

        tr_total /= max(1, n_batches)
        tr_vae   /= max(1, n_batches)
        tr_recon /= max(1, n_batches)
        tr_kl    /= max(1, n_batches)

        print(
            f"Epoch {epoch:03d} | beta={beta:.4f} | "
            f"train total {tr_total:.5f} (vae {tr_vae:.5f}, recon {tr_recon:.5f}, kl {tr_kl:.5f})"
        )

        #for early stop
        store_path = f"state_dicts/{model_name}/{run_name}-epoch{epoch}.pt"
        torch.save({
            'model_state_dict': model.state_dict(), 
            'beta': beta, 
            'epoch': epoch,
            'model': model_name,
            'run': run_name
        }, store_path)

    return model


# In[ ]:


# Import model
import importlib
import models.model_v1_cnn_t as m  # change model here
importlib.reload(m)
model_name = m.META.name

model = m.VAE(
    n_features=11,
    seq_len=7,
    d_model=256,
    cnn_channels=128,
    n_heads=4,
    n_layers=3,
    latent_dim=16,
    dropout=0.1,
    use_cls_token=True,
)
#print(model)

# Change run configurations here
run_name = "online_test8"
n_epoch = 20

cfg = TrainConfig(epochs=n_epoch, beta_max=1e-4, weight_decay=1e-4, beta_warmup_epochs=10, lr=1e-4, recon="mse")
print(f"Training {model_name} with {n_epoch} epochs ...")
model = train_vae(model, dataset=ds, loader=loader, device=device, model_name=model_name, run_name=run_name, cfg=cfg)


# In[ ]:




