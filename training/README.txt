Training notes
==============

Dataset / feature setup
-----------------------
- data: IVT
- feature count: 11
- model family: CNN-Transformer

Config-driven training (implemented)
------------------------------------
Training now reads all run settings from JSON config files.

- example config: `configs/train_online.example.json`
- launch locally: `python training-online.py --config configs/train_online.example.json`
- slurm launch: `sbatch train.sh configs/train_online.example.json`

Experiment naming reference
---------------------------
online-
- test0: DNA, HG002-WGA random subset, small model (64,64), static beta=1e-3
- test1: RNA, IVT, large model (256,128), static beta=1e-4
- test2: RNA, IVT, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test3: DNA, HG002-WGA, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test4: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test5: RNA, unmodified RNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs
- test6: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs, increased sampling rate
- test7: RNA, unmodified RNA oligos, large model (256,128), warmup beta to 1e-3 over 10 epochs, increased sampling rate
- test8: DNA, unmodified DNA oligos, large model (256,128), warmup beta to 1e-4 over 10 epochs, increased sampling rate
- test9: DNA, shuffled unmodified DNA oligos, large model (256,128), warmup beta to 1e-4 over 10 epochs

static-
- test0: DNA oligos, static, small model (64,64), static beta=1e-3
- test1: RNA oligos, static, small model (64,64), static beta=1e-3
