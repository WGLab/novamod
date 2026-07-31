#!/bin/bash
#
# Step 3 of SignalBAM generation: embed raw signal into the aligned BAM.
#
#   basecall-{wt,ivt}.sh  ->  align_nomod.sh  ->  generate_features.sh
#
# Produces the "*.final.bam" SignalBAM files referenced by
# training/data_manifest.csv. See README.md ("Generating SignalBAM files").
#
# Usage:
#   ./scripts/generate_features.sh <sample> <aligned_bam> <pod5_dir> <outdir> [threads]
#   sbatch scripts/generate_features.sh 20240321_directRNA_HEK293T \
#       /data/20240321_directRNA_HEK293T.aligned.GRCh38.splice.dorado.bam \
#       /data/pod5/20240321_directRNA_HEK293T /data/out

#SBATCH --time=1-00:00:00
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -J gen.signalbam
#SBATCH --output=convert_%A_%a.out
#SBATCH --error=convert_%A_%a.out

set -euo pipefail

SAMPLE=${1:?"usage: $0 <sample> <aligned_bam> <pod5_dir> <outdir> [threads]"}
ALIGNED_BAM=${2:?"usage: $0 <sample> <aligned_bam> <pod5_dir> <outdir> [threads]"}
POD5_DIR=${3:?"usage: $0 <sample> <aligned_bam> <pod5_dir> <outdir> [threads]"}
OUTDIR=${4:?"usage: $0 <sample> <aligned_bam> <pod5_dir> <outdir> [threads]"}
THREADS=${5:-12}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

# Attach blosc2-compressed raw signal to each primary alignment as tag SG.
# The input BAM must already carry dorado's mv/ts/sm/sd/sv tags.
python "$SCRIPT_DIR/pod5_to_bam.py" \
    --bam "$ALIGNED_BAM" \
    --input "$POD5_DIR" \
    --file_type pod5 \
    --output "$OUTDIR" \
    --prefix converted \
    --threads "$THREADS"

# pod5_to_bam.py writes one chunk per 100k reads; merge them into the
# single coordinate-sorted, indexed SignalBAM used by training/ and val.
ls -1 "$OUTDIR"/converted*bam > "$OUTDIR/bam_files"

samtools cat -b "$OUTDIR/bam_files" -o - \
  | samtools sort -m 8G -O BAM -o "$OUTDIR/$SAMPLE.final.bam" \
      --write-index --threads "$THREADS"

echo "SignalBAM written to $OUTDIR/$SAMPLE.final.bam"
echo "Chunk files ($OUTDIR/converted*.bam) can be removed once verified."
