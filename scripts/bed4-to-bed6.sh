#!/bin/bash

# Convert a BED4-like file:
# chrom  position  strand  score
# into BED6:
# chrom  start  end  name  score  strand

# Usage:
#   ./bed4_to_bed6.sh input.bed4 output.bed6 [zero|one]
#
#   zero → positions are already 0-based (default)
#   one  → positions are 1-based (will subtract 1)

if [ $# -lt 2 ]; then
    echo "Usage: $0 input.bed4 output.bed6 [zero|one]"
    exit 1
fi

INPUT=$1
OUTPUT=$2
COORD=${3:-zero}

if [ "$COORD" = "one" ]; then
    echo "Assuming 1-based positions → converting to 0-based BED"
    awk 'BEGIN{OFS="\t"} {print $1, $2-1, $2, ".", $4, $3}' "$INPUT" > "$OUTPUT"
else
    echo "Assuming 0-based positions → creating 1bp intervals"
    awk 'BEGIN{OFS="\t"} {print $1, $2, $2+1, ".", $4, $3}' "$INPUT" > "$OUTPUT"
fi

echo "Done. Output written to $OUTPUT"
