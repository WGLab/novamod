#!/usr/bin/env python3
"""
Generate BED6 of single-base motif-center positions from a genome FASTA.

BED fields:
  chrom  start  end  name  score  strand

Coordinates:
- BED is 0-based, half-open.
- '+' strand entries are reported directly from motif matches on the reference.
- '-' strand entries are reported from reverse-complement motif matches, with center
  coordinates remapped back to reference coordinates.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from typing import Dict, Iterator, List, Tuple, TextIO


IUPAC: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}

COMPLEMENT: Dict[str, str] = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "D": "H",
    "H": "D",
    "V": "B",
    "N": "N",
}


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


def reverse_complement_iupac(seq: str) -> str:
    try:
        return "".join(COMPLEMENT[base] for base in reversed(seq.upper()))
    except KeyError as exc:
        raise ValueError(f"Unsupported IUPAC symbol in motif: {exc.args[0]}") from exc


def find_iupac_matches(seq: str, motif: str) -> Iterator[int]:
    seq_u = seq.upper()
    motif_u = motif.upper()
    mlen = len(motif_u)

    allowed_per_pos: List[set[str]] = []
    for symbol in motif_u:
        if symbol not in IUPAC:
            raise ValueError(f"Unsupported IUPAC symbol in motif: {symbol}")
        allowed_per_pos.append(set(IUPAC[symbol]))

    for start in range(0, len(seq_u) - mlen + 1):
        window = seq_u[start : start + mlen]
        if all(base in allowed_per_pos[i] for i, base in enumerate(window)):
            yield start


def center_offsets_for_motif(motif: str, center_base: str) -> List[int]:
    motif_u = motif.upper()
    center_u = center_base.upper()

    if center_u not in {"A", "C", "G", "T"}:
        raise ValueError("--center-base must be one of A/C/G/T")

    offsets = [
        i for i, symbol in enumerate(motif_u)
        if symbol in IUPAC and center_u in IUPAC[symbol]
    ]
    if not offsets:
        raise ValueError(
            f"Center base '{center_u}' cannot occur in motif '{motif_u}' with IUPAC rules"
        )
    return offsets


def write_motif_center_bed(
    chrom: str,
    seq: str,
    out: TextIO,
    motif: str,
    center_base: str,
    plus_only: bool = False,
    minus_only: bool = False,
    name_prefix: str = "motif_center",
) -> None:
    motif_u = motif.upper()
    mlen = len(motif_u)

    plus_offsets = center_offsets_for_motif(motif_u, center_base)

    if not minus_only:
        emitted_plus = set()
        for j in find_iupac_matches(seq, motif_u):
            for offset in plus_offsets:
                if seq[j + offset].upper() != center_base.upper():
                    continue
                start = j + offset
                if start in emitted_plus:
                    continue
                emitted_plus.add(start)
                end = start + 1
                name = f"{name_prefix}_{chrom}_{start}_plus"
                out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t+\n")

    if not plus_only:
        rc_motif = reverse_complement_iupac(motif_u)
        minus_ref_base = COMPLEMENT[center_base.upper()]
        minus_offsets = center_offsets_for_motif(rc_motif, minus_ref_base)

        emitted_minus = set()
        for j in find_iupac_matches(seq, rc_motif):
            for offset in minus_offsets:
                if seq[j + offset].upper() != minus_ref_base:
                    continue
                start = j + offset
                if start in emitted_minus:
                    continue
                emitted_minus.add(start)
                end = start + 1
                name = f"{name_prefix}_{chrom}_{start}_minus"
                out.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t-\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate BED of single-base center positions for motif matches from a genome FASTA."
        )
    )
    ap.add_argument("fasta", help="Genome FASTA (can be .gz). Use '-' for stdin.")
    ap.add_argument("-o", "--out", default="-", help="Output BED path (default: stdout).")
    ap.add_argument(
        "--motif",
        default="CG",
        help="Motif in IUPAC symbols (default: CG).",
    )
    ap.add_argument(
        "--center-base",
        default="C",
        help="Base to emit within motif matches, one of A/C/G/T (default: C).",
    )

    strand = ap.add_mutually_exclusive_group()
    strand.add_argument("--plus-only", action="store_true", help="Only output '+' strand centers.")
    strand.add_argument("--minus-only", action="store_true", help="Only output '-' strand centers.")

    ap.add_argument("--name-prefix", default="motif_center", help="Prefix for BED name field.")
    args = ap.parse_args()

    if not args.motif:
        raise SystemExit("--motif cannot be empty")

    fin = open_maybe_gzip(args.fasta)
    fout = sys.stdout if args.out == "-" else open(args.out, "w")

    try:
        for chrom, seq in fasta_records(fin):
            write_motif_center_bed(
                chrom=chrom,
                seq=seq,
                out=fout,
                motif=args.motif,
                center_base=args.center_base,
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
