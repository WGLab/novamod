#!/usr/bin/env python3
"""
Generate BED6 of single-base CpG cytosine positions from a genome FASTA.

BED fields:
  chrom  start  end  name  score  strand

Coordinates:
- BED is 0-based, half-open.
- '+' strand CpG cytosine: the 'C' in "CG" => [i, i+1)
- '-' strand CpG cytosine: cytosine on reverse strand corresponds to the 'G' in "GC"
  on the reference, and its genomic coordinate is the second base of "GC" => [i+1, i+2)
"""

from __future__ import annotations
import argparse
import gzip
import sys
from typing import Iterator, Tuple, TextIO


def open_maybe_gzip(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def fasta_records(handle: TextIO) -> Iterator[Tuple[str, str]]:
    name = None
    seq_chunks = []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq_chunks)
            name = line[1:].split()[0]
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if name is not None:
        yield name, "".join(seq_chunks)


def write_cpg_c_bed(
    chrom: str,
    seq: str,
    out: TextIO,
    plus_only: bool = False,
    minus_only: bool = False,
    name_prefix: str = "CpG_C",
) -> None:
    s = seq.upper()

    # '+' strand: "CG" -> cytosine is at i
    if not minus_only:
        i = 0
        while True:
            j = s.find("CG", i)
            if j == -1:
                break
            start = j
            end = j + 1
            name = f"{name_prefix}_{chrom}_{start}_plus"
            out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t+\n")
            i = j + 1

    # '-' strand: reverse CpG is "GC" on reference
    # cytosine (on '-') aligns to the reference 'G' position, which is the 2nd base of "GC"
    if not plus_only:
        i = 0
        while True:
            j = s.find("GC", i)
            if j == -1:
                break
            start = j + 1
            end = j + 2
            name = f"{name_prefix}_{chrom}_{start}_minus"
            out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t-\n")
            i = j + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate BED of single-base CpG cytosine positions from a genome FASTA."
    )
    ap.add_argument("fasta", help="Genome FASTA (can be .gz). Use '-' for stdin.")
    ap.add_argument("-o", "--out", default="-", help="Output BED path (default: stdout).")

    strand = ap.add_mutually_exclusive_group()
    strand.add_argument("--plus-only", action="store_true",
                        help="Only output '+' strand CpG cytosines (C in 'CG').")
    strand.add_argument("--minus-only", action="store_true",
                        help="Only output '-' strand CpG cytosines (C on '-', i.e. G in 'GC').")

    ap.add_argument("--name-prefix", default="CpG_C", help="Prefix for BED name field.")
    args = ap.parse_args()

    fin = open_maybe_gzip(args.fasta)
    fout = sys.stdout if args.out == "-" else open(args.out, "w")

    try:
        for chrom, seq in fasta_records(fin):
            write_cpg_c_bed(
                chrom=chrom,
                seq=seq,
                out=fout,
                plus_only=args.plus_only,
                minus_only=args.minus_only,
                name_prefix=args.name_prefix,
            )
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()


if __name__ == "__main__":
    main()