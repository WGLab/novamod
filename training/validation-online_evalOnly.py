#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bam_utils import *

import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")


# ### Loading Data

# In[2]:


from torch.utils.data import DataLoader
import dataset_utils as du

def valData(bam, ref, seq_type, method, spec, num_workers=20):
    if method == "region":
        #spec: [chrom, start, end]
        ds = du.SignalBAMRegionDataset(
            bam_path=bam,
            ref_fasta_path=ref,
            seq_type=seq_type,
            chrom=spec[0],
            start=spec[1],
            end=spec[2],
            kmer_len=7,
        )
    if method == "nt":
        #spec: label_path
        ds = du.SignalBAMkmerValidationDataset(
            bam_path=bam,
            ref_fasta_path=ref,
            seq_type=seq_type, 
            labels_path_tsv=spec,
            max_lines=50000,
        )
    if method == "site":
        #spec: pos_list
        ds = du.SignalBAMRefPosValidationDataset(
            bam_path=c_bam,
            ref_fasta_path=ref,
            seq_type=seq_type,
            pos_list=spec,
            max_lines=50000,
        )
    loader = DataLoader(ds, batch_size=256, num_workers=num_workers)
    return loader


# In[ ]:


#Human
wt = "20240321_directRNA_HEK293T"
ref_hs = "/home/zouy1/software/reference/GRCh38.primary_assembly.genome.fa"
wt_bam = f"/home/zouy1/projects/RNAmod/VAE/data/RNA/{wt}/{wt}.final.bam"
label_path = f"/home/zouy1/projects/RNAmod/VAE/data/RNA/{wt}/{wt}.modkit.extract.chr20.bed"

hg_bam = "/mnt/isilon/wang_lab/umair/deepmod/model_features/r10_5khz_drd.v1_full_bam/HG002/HG002.1_100_chunks.bam"
wga_bam = "/mnt/isilon/wang_lab/umair/deepmod/model_features/r10_5khz_drd.v1_full_bam/HG002_WGA/HG002_WGA.final.bam"

#Oligos
mod = "5mC"
ref_oligos = '/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/all_5mers.fa'
c_bam = "/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/control_rep1/control_rep1.subset.final.bam"
mc_bam = f"/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/{mod}_rep1/{mod}_rep1.subset.final.bam"
pos_list = f"/home/zouy1/projects/RNAmod/VAE/data/oligos/DNA/labels/all_5mers_{mod}_sites.bed"

#Region
region1 = ["chr20", 58837550, 58890800]
region2 = ["chr22", 25343801, 25393801]

val_loader = valData(hg_bam, ref_hs, "dna", method="region", spec=region2)


# ### Evaluation

# In[3]:


# Import model
import importlib
import models.model_v1_cnn_t as m  # change model here
importlib.reload(m)
model_name = m.META.name

model = m.VAE(
    n_features=11,
    seq_len=7,
    d_model=256, #256
    cnn_channels=128, #128
    n_heads=4,
    n_layers=3,
    latent_dim=16,
    dropout=0.1,
    use_cls_token=True,
).to(device)

run_name = "online_test3"
n_epoch = "3"
state_dict = f"state_dicts/{model_name}/{run_name}-epoch{n_epoch}.pt"
#state_dict = f"state_dicts/{model_name}/model_vae_cnn_be-3_e50-wga.pt"
pt = torch.load(state_dict, map_location=device)
model.load_state_dict(pt['model_state_dict'])
last_beta = pt['beta']
#check params
num_nonzero = sum((p.abs() > 1e-6).sum().item() for p in model.parameters())
nparam = sum([p.numel() for p in model.parameters()])
print(f"non-zero params: {100*num_nonzero/nparam}%")


# In[4]:


import pandas as pd
import os
import feature_utils as fu
import pyarrow as pa
import pyarrow.parquet as pq

def anomaly_score(model, loader, device, out_path):
    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    writer = None
    with torch.no_grad():
        for x, kmer, pos, y in loader:
            x = x.to(device)
            out = model(x)
            recon_score = ((out["x_hat"] - x) ** 2).mean(dim=(1, 2))  # [B]

            # tensors
            embs = out["mu"].detach().cpu().numpy()      # [B, D]
            scores = recon_score.detach().cpu().numpy()  # [B]
            labels = y.detach().cpu().numpy()            # [B] or [B, ...]
            # metadata
            kmers = [fu.decode_kmer(km) for km in kmer.detach().cpu()]
            pos_rows = []
            for i in range(x.shape[0]):
                pos_rows.append({k: (v[i] if isinstance(v, (list, tuple)) else v[i].item())
                                 for k, v in pos.items()})
            
            # build batch dataframe
            batch_df = pd.DataFrame({
                "scores": scores,
                "labels": labels,
                "kmer": kmers,
                "embeddings": list(embs)
            })
            pos_df = pd.json_normalize(pos_rows)
            batch_df = pd.concat([batch_df, pos_df], axis=1)
            
            table = pa.Table.from_pandas(batch_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

        print("mu std:", out["mu"].std(dim=0).mean().item())
        print("logvar mean:", out["logvar"].mean().item())

    if writer is not None:
        writer.close()
        
    return out_path

#Run analysis
anomaly_score(model, val_loader, device, out_path="./validation/validation_result3-region2-wt.pq")

