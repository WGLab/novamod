#!/bin/bash
#SBATCH --mem=32G --time=3-00:00:00 --out=feature_generate.out --job-name=ftDNA

python -u feature_generate.py
