Training notes
==============

Dataset / feature setup
-----------------------
- data: unmodified Nanopore data (SignalBAM; see ../README.md)
- feature count: 11 per k-mer position
    [0:4]  one-hot reference base (A,C,G,T)
    [4:10] signal stats: winsorised mean, q1, q2, q3, central value, RMS
    [10]   base quality (error probability)
- model family: CNN-Transformer VAE (models/model_v1_cnn_t.py)

Config-driven training
----------------------
Training reads all run settings from JSON config files.

- example config: configs/train.example.json
- launch locally: python train.py --config configs/train.example.json
- slurm launch:   sbatch train.sh configs/train.example.json

Checkpoints are written to:
    state_dicts/<model_name>/<run_name>-epoch<N>.pt
one per epoch. Training OVERWRITES an existing file at that path, so always
give a new run a new run_name -- otherwise you will silently destroy a
released checkpoint. configs/train.example.json uses run_name "example"
for exactly this reason.

Config-driven validation
------------------------
Validation reads model/checkpoint/dataset settings from JSON config files.

- example config: configs/val.example.json
- launch locally: python val.py --config configs/val.example.json
- slurm launch:   sbatch val.sh configs/val.example.json

Output: one parquet per dataset in <output_dir>/<run_name>-<dataset>.pq with
columns score_recon, score_kl, labels, kmer, embeddings + position metadata.

Constraints worth remembering
-----------------------------
- kmer_len must be odd, must equal 2*flank+1, and must equal model seq_len.
- kmer_len <= 15: k-mer codes are packed base-4 into signed int32.
- A val config's model.kwargs must match the checkpoint being loaded.

Experiment naming reference
---------------------------
(beta values below are beta_max, warmed up linearly over beta_warmup_epochs.)

online-
- test0: DNA, HG002-WGA random subset, small model (64,64), static beta=1e-3
- test1: RNA, IVT, large model (256,128), static beta=1e-4
- test2: RNA, IVT, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test3: DNA, HG002-WGA, large model (256,128), warmup beta to 1e-3 over 10 epochs, 7-mers
- test4: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test5: RNA, unmodified RNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test6: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs, increased sampling rate
- test7: RNA, unmodified RNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs, increased sampling rate
- test8: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-4 over 10 epochs, increased sampling rate
- test9: DNA, shuffled unmodified DNA oligos, large model, NLL weighted loss, warmup beta to 1e-3 over 10 epochs
- test10: DNA, HG002-WGA, large model, NLL weighted loss, warmup beta to 1e-3
- test11: DNA, HG002-WGA, large model, NLL weighted loss, warmup beta to 1e-3, training optimized

static-
- test0: DNA oligos, static, small model (64,64), static beta=1e-3
- test1: RNA oligos, static, small model (64,64), static beta=1e-3

Note: the test9 entry previously read "warmup beta to 1e-4". The released
checkpoint state_dicts/model_v1_cnn_t/online_test9-epoch20.pt stores
beta=0.001 at epoch 20 (warmup 10 epochs), and configs/train.test9.json sets
beta_max=0.001, so the run used 1e-3. Corrected above.
