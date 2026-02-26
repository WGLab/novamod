from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Any, List
import csv

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import pysam
import bisect

import bam_utils
import feature_utils

import torch.distributed as dist


@dataclass
class SamplingConfig:
    kmer_len: int = 7
    flank: int = 3                      # (kmer_len-1)//2 for 7-mer
    positions_per_read: int = 10        # K positions sampled per read
    max_kmer_count: int = 50_000        # soft cap C for online balancing
    min_ref_window_ok: bool = True      # require full ref window in bounds + no N
    require_mapped: bool = True
    include_insertions: bool = False    # if False, require read_pos maps to ref_pos
    normalize_signal: bool = True       # apply Read.normalize()
    seed: int = 0                       # base seed; worker/epoch can offset externally

    def __post_init__(self) -> None:
        self.kmer_len = int(self.kmer_len)
        self.flank = int(self.flank)
        if self.kmer_len <= 0:
            raise ValueError("kmer_len must be positive.")
        if self.kmer_len % 2 == 0:
            raise ValueError("kmer_len must be odd so a center base exists.")
        if self.kmer_len != (2 * self.flank + 1):
            raise ValueError("kmer_len must equal 2*flank+1.")
        # k-mer codes are currently stored in signed int32 tensors.
        # Keep a strict bound so packed base-4 codes never overflow.
        if self.kmer_len > 15:
            raise ValueError("kmer_len > 15 is unsupported with int32 k-mer codes.")


def _onehot_base_num(base_num: int) -> np.ndarray:
    # base_num is 0..3 for A/C/G/T; other -> all zeros
    out = np.zeros(4, dtype=np.float32)
    if 0 <= base_num <= 3:
        out[base_num] = 1.0
    return out


def _build_read_to_ref_map(aligned_pairs: np.ndarray, read_len: int) -> np.ndarray:
    """
    aligned_pairs: shape (R,2) where col0=read_pos, col1=ref_pos, -1 indicates gap.
    Returns: array of length read_len with ref_pos or -1 if not aligned to ref at that base.
    """
    m = np.full(read_len, -1, dtype=np.int64)
    # Only positions where both read and ref exist
    mask = (aligned_pairs[:, 0] != -1) & (aligned_pairs[:, 1] != -1)
    rp = aligned_pairs[mask, 0].astype(np.int64)
    rf = aligned_pairs[mask, 1].astype(np.int64)
    m[rp] = rf
    return m

def _build_ref_to_read_map(aligned_pairs: np.ndarray) -> Dict[int, int]:
    """
    aligned_pairs: shape (R,2) where col0=read_pos, col1=ref_pos, -1 indicates gap.
    Returns dict: ref_pos -> read_pos for aligned (match/mismatch) bases.
    If multiple read positions map to same ref (rare), last one wins.
    """
    ref_to_read: Dict[int, int] = {}
    mask = (aligned_pairs[:, 0] != -1) & (aligned_pairs[:, 1] != -1)
    rp = aligned_pairs[mask, 0].astype(np.int64)
    rf = aligned_pairs[mask, 1].astype(np.int64)
    for rpos, fpos in zip(rp, rf):
        ref_to_read[int(fpos)] = int(rpos)
    return ref_to_read


def _get_signal_vector_for_base(read_obj: bam_utils.Read, base_idx: int) -> np.ndarray:
    """
    Returns 1-D float array for the signal corresponding to one base.
    Your Read stores:
      - read_obj.signal: (len(move_table), stride) int16/float after normalize
      - read_obj.segments: (num_bases,2) start/end indices into signal rows
    We flatten all rows for that base into one vector.
    """
    s0, s1 = read_obj.segments[base_idx]
    # signal chunk: rows s0..s1-1, each row has 'stride' samples
    chunk = read_obj.signal[s0:s1, :].reshape(-1).astype(np.float64, copy=False)
    return chunk




# For training
class SignalBAMkmerIterableDataset(IterableDataset):
    """
    Streaming dataset:
      yields (features_7x11, kmer_code_uint32, pos_info_dict)

    features_7x11:
      columns = [onehot(4), wmean, q1, q2, q3, central, rms, base_qual]  -> 11 dims
    """

    def __init__(
        self,
        bam_path: str,
        ref_fasta_path: str,
        cfg: SamplingConfig = SamplingConfig(),
        seq_type: str = "rna",
        chrom_list: Optional[List[str]] = None,
        allow_secondary: bool = False,
        allow_supplementary: bool = False,
        max_samples_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.bam_path = bam_path
        self.ref_fasta_path = ref_fasta_path
        self.cfg = cfg
        self.seq_type = seq_type
        self.chrom_list = chrom_list or []
        self.allow_secondary = allow_secondary
        self.allow_supplementary = allow_supplementary
        self.max_samples_per_epoch = max_samples_per_epoch

        # Pre-load reference numeric arrays once per Dataset instance.
        # Uses bam_utils.pre_process to build ref_seq_dict. :contentReference[oaicite:4]{index=4}
        self.ref_seq_dict, _, _ = bam_utils.pre_process(
            ref=self.ref_fasta_path,
            seq_type=self.seq_type,
            pos_list=None,
            motif=None,
            chrom_list=self.chrom_list,
            bam=self.bam_path,
        )

        # Online k-mer seen counts (streaming balancing). Per-worker instance keeps its own counts.
        self._seen: Dict[int, int] = {}

        # For reproducibility inside __iter__ (worker-aware)
        self._base_seed = int(self.cfg.seed)
        
    def _worker_rng(self):
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        epoch = getattr(self, "_epoch", 0)
    
        rank = dist.get_rank() if dist.is_initialized() else 0
    
        return random.Random(
            self._base_seed
            + 1_000_000 * epoch
            + 10_000 * rank
            + 100 * wid
        )
        
    def _worker_shard_ok(self, read_idx: int) -> bool:
        wi = get_worker_info()
        if wi is None:
            return True
        return (read_idx % wi.num_workers) == wi.id

    def _accept_kmer(self, kmer_code: int, rng: random.Random) -> bool:
        """
        Streaming k-mer balancing:
            p = min(1, C / (seen+1))
        """
        C = int(self.cfg.max_kmer_count)
        seen = self._seen.get(kmer_code, 0)
        p = 1.0 if seen < C else (C / float(seen + 1))
        if rng.random() < p:
            self._seen[kmer_code] = seen + 1
            return True
        return False

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        rng = self._worker_rng()

        k = int(self.cfg.kmer_len)
        flank = int(self.cfg.flank)
        assert k == 2 * flank + 1, "kmer_len must equal 2*flank+1 (e.g. 7 and 3)"

        bam = pysam.AlignmentFile(self.bam_path, "rb")

        emitted = 0
        # Iterate reads in a streaming manner; this works for unsorted BAM too.
        for read_idx, bam_read in enumerate(bam.fetch(until_eof=True)):
            if not self._worker_shard_ok(read_idx):
                continue

            if self.cfg.require_mapped and (not bam_read.is_mapped):
                continue
            if (not self.allow_secondary) and bam_read.is_secondary:
                continue
            if (not self.allow_supplementary) and bam_read.is_supplementary:
                continue

            try:
                (
                    move_table,
                    mat,
                    base_qual,
                    full_seq,
                    aligned_pairs,
                    read_name,
                    ref_name,
                    shift,
                    scale,
                    norm_type,
                    seq_type,
                    div,
                    flag,
                    ref_start,
                    ref_end,
                    cigarstring,
                ) = bam_utils.get_read_info(bam_read, self.ref_seq_dict, self.seq_type)  # :contentReference[oaicite:5]{index=5}
            except Exception:
                # If a read is malformed or missing tags, skip it
                continue

            read_obj = bam_utils.Read(
                move_table=move_table,
                signal=mat,
                base_qual=base_qual,
                full_seq=full_seq,
                aligned_pairs=aligned_pairs,
                read_name=read_name,
                ref_name=ref_name,
                shift=shift,
                scale=scale,
                norm_type=norm_type,
                seq_type=seq_type,
                div=div,
                flag=flag,
                reference_start=ref_start,
                reference_end=ref_end,
                cigarstring=cigarstring,
            )

            if self.cfg.normalize_signal:
                read_obj.normalize()

            L = int(read_obj.length)
            if L < (2 * flank + 1):
                continue

            # Map read base index -> ref pos (or -1 if insertion/unmapped at that base)
            read_to_ref = _build_read_to_ref_map(read_obj.aligned_pairs, L)

            # Sample K centers per read from [flank, L-flank-1]
            lo = flank
            hi = L - flank
            if hi <= lo:
                continue

            # If the read is short, sample without replacement; otherwise with replacement is fine.
            K = int(self.cfg.positions_per_read)
            if K <= 0:
                continue

            if (hi - lo) >= K:
                centers = rng.sample(range(lo, hi), k=K)
            else:
                centers = [rng.randrange(lo, hi) for _ in range(K)]

            # Determine which ref strand column to use for k-mer from ref_seq_dict
            # ref_seq_dict[chrom] is shape (len_ref+1,2) from get_ref_to_num :contentReference[oaicite:6]{index=6}
            ref_col = 1 if read_obj.is_reverse else 0

            ref_arr = self.ref_seq_dict.get(ref_name, None)
            if ref_arr is None:
                continue
            ref_len = int(ref_arr.shape[0]) - 1  # last entry is sentinel N in get_ref_to_num :contentReference[oaicite:7]{index=7}

            for center in centers:
                ref_pos = int(read_to_ref[center])
                if (not self.cfg.include_insertions) and ref_pos < 0:
                    continue

                # Require the 7-mer window to be in reference bounds
                if self.cfg.min_ref_window_ok:
                    if ref_pos - flank < 0 or ref_pos + flank >= ref_len:
                        continue

                # Extract 7-mer numeric bases from reference in read orientation
                kmer_nums = ref_arr[ref_pos - flank : ref_pos + flank + 1, ref_col].astype(np.int64, copy=False)

                # Optionally require no 'N' in k-mer (N is encoded as 4 in bam_utils.get_ref_to_num) :contentReference[oaicite:8]{index=8}
                if self.cfg.min_ref_window_ok and np.any(kmer_nums > 3):
                    continue

                # Encode k-mer (string A/C/G/T) with feature_utils.encode_kmer :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
                kmer_str = "".join(bam_utils.num_to_base_map[int(x)] for x in kmer_nums)
                try:
                    kmer_code = int(feature_utils.encode_kmer(kmer_str, k=k))  # :contentReference[oaicite:11]{index=11}
                except Exception:
                    continue

                # Online k-mer balancing (streaming)
                if not self._accept_kmer(kmer_code, rng):
                    continue

                # Build list of 7 signal vectors (one per base in the 7-mer window in read coords)
                # We assume read base indices are contiguous; we take read-centered window.
                # (This is per-read feature generation as requested.)
                chunk = []
                for j in range(center - flank, center + flank + 1):
                    chunk.append(_get_signal_vector_for_base(read_obj, j))

                # Extract 7x6 signal stats: [wmean,q1,q2,q3,central,rms] :contentReference[oaicite:12]{index=12}
                stats_7x6 = feature_utils.extract_kmer_features(chunk, k)  # (7,6)

                # Base qualities for 7 bases
                qual_7 = read_obj.base_qual[center - flank : center + flank + 1].astype(np.float32, copy=False)  # (7,)

                # One-hot bases for 7 positions (use reference 7-mer context)
                onehot_7x4 = np.stack([_onehot_base_num(int(b)) for b in kmer_nums], axis=0)  # (7,4)

                # Assemble final 7x11 feature matrix
                # columns = [4 onehot, 6 stats, 1 base_qual]
                feats = np.concatenate(
                    [onehot_7x4, stats_7x6.astype(np.float32, copy=False), qual_7[:, None]],
                    axis=1,
                ).astype(np.float32, copy=False)  # (7,11)

                pos_info = {
                    "chrom": ref_name,
                    "ref_pos_0based": ref_pos,
                    "strand": "-" if read_obj.is_reverse else "+",
                    "read_name": read_obj.read_name,
                    "read_pos_0based": int(center),
                    "cigar": read_obj.cigarstring,
                    "ref_start_0based": int(read_obj.ref_start),
                    "ref_end_0based_excl": int(read_obj.ref_end),
                }

                yield (
                    torch.from_numpy(feats),                           # (7,11) float32
                    torch.tensor(kmer_code, dtype=torch.int32),        # packed 7-mer code
                    pos_info,                                          # metadata dict
                )
                
                emitted += 1
                # set max samples
                if self.max_samples_per_epoch is not None and emitted >= self.max_samples_per_epoch:
                    return

        bam.close()
        
    def set_epoch(self, epoch: int):
        """
        change rng seed every different epoch
        """
        self._epoch = int(epoch)




# Validation set with per-nt labels
def load_labels_tsv(path: str, max_lines: int) -> Dict[str, List[Tuple[int, float]]]:
    """
    TSV columns: read_id, read_pos_0based, mod_score
    """
    out: Dict[str, List[Tuple[int, float]]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        lines_processed = 0
        for row in r:
            rid = row["read_id"]
            pos = int(row["forward_read_position"])
            score = float(row["mod_qual"])
            out.setdefault(rid, []).append((pos, score))
            lines_processed += 1
            if lines_processed > max_lines: 
                break
    return out


class SignalBAMkmerValidationDataset(IterableDataset):
    """
    Yields:
      (features_7x11, kmer_code_int32, pos_info_dict, mod_score_float32)

    Uses WT SignalBAM + an external label list keyed by (read_id, read_pos).
    """

    def __init__(
        self,
        bam_path: str,
        ref_fasta_path: str,
        labels_path_tsv: str, 
        max_lines: int = 100000,
        kmer_len: int = 7,
        normalize_signal: bool = True,
        seq_type: str = "rna",
        chrom_list: Optional[List[str]] = None,
        allow_secondary: bool = False,
        allow_supplementary: bool = False,
    ):
        super().__init__()
        self.bam_path = bam_path
        self.ref_fasta_path = ref_fasta_path
        self.labels_by_read = load_labels_tsv(labels_path_tsv, max_lines)

        self.k = int(kmer_len)
        self.flank = (self.k - 1) // 2
        self.normalize_signal = normalize_signal

        self.seq_type = seq_type
        self.chrom_list = chrom_list or []
        self.allow_secondary = allow_secondary
        self.allow_supplementary = allow_supplementary

        self.ref_seq_dict, _, _ = bam_utils.pre_process(
            ref=self.ref_fasta_path,
            seq_type=self.seq_type,
            pos_list=None,
            motif=None,
            chrom_list=self.chrom_list,
            bam=self.bam_path,
        )

    def _worker_shard_ok(self, read_idx: int) -> bool:
        wi = get_worker_info()
        if wi is None:
            return True
        return (read_idx % wi.num_workers) == wi.id

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]]:
        bam = pysam.AlignmentFile(self.bam_path, "rb")

        for read_idx, bam_read in enumerate(bam.fetch(until_eof=True)):
            if not self._worker_shard_ok(read_idx):
                continue

            if not bam_read.is_mapped:
                continue
            if (not self.allow_secondary) and bam_read.is_secondary:
                continue
            if (not self.allow_supplementary) and bam_read.is_supplementary:
                continue

            read_name = bam_read.query_name
            if read_name not in self.labels_by_read:
                continue

            try:
                (
                    move_table, mat, base_qual, full_seq, aligned_pairs,
                    read_name2, ref_name, shift, scale, norm_type,
                    seq_type, div, flag, ref_start, ref_end, cigarstring
                ) = bam_utils.get_read_info(bam_read, self.ref_seq_dict, self.seq_type)
            except Exception:
                continue

            read_obj = bam_utils.Read(
                move_table=move_table,
                signal=mat,
                base_qual=base_qual,
                full_seq=full_seq,
                aligned_pairs=aligned_pairs,
                read_name=read_name2,
                ref_name=ref_name,
                shift=shift,
                scale=scale,
                norm_type=norm_type,
                seq_type=seq_type,
                div=div,
                flag=flag,
                reference_start=ref_start,
                reference_end=ref_end,
                cigarstring=cigarstring,
            )
            if self.normalize_signal:
                read_obj.normalize()

            L = int(read_obj.length)
            if L < self.k:
                continue

            read_to_ref = _build_read_to_ref_map(read_obj.aligned_pairs, L)
            ref_arr = self.ref_seq_dict.get(ref_name, None)
            if ref_arr is None:
                continue
            ref_col = 1 if read_obj.is_reverse else 0
            ref_len = int(ref_arr.shape[0]) - 1

            for center, mod_score in self.labels_by_read[read_name]:
                if center < self.flank or center >= (L - self.flank):
                    continue

                ref_pos = int(read_to_ref[center])
                if ref_pos < 0:
                    continue
                if ref_pos - self.flank < 0 or ref_pos + self.flank >= ref_len:
                    continue

                kmer_nums = ref_arr[ref_pos - self.flank : ref_pos + self.flank + 1, ref_col].astype(np.int64, copy=False)
                if np.any(kmer_nums > 3):
                    continue

                kmer_str = "".join(bam_utils.num_to_base_map[int(x)] for x in kmer_nums)
                try:
                    kmer_code = int(feature_utils.encode_kmer(kmer_str, k=self.k))
                except Exception:
                    continue

                # per-read signals for the 7 bases around center (read coords)
                chunk = []
                for j in range(center - self.flank, center + self.flank + 1):
                    chunk.append(_get_signal_vector_for_base(read_obj, j))

                stats_7x6 = feature_utils.extract_kmer_features(chunk, self.k)  # (7,6)
                qual_7 = read_obj.base_qual[center - self.flank : center + self.flank + 1].astype(np.float32, copy=False)
                onehot_7x4 = np.stack([_onehot_base_num(int(b)) for b in kmer_nums], axis=0)

                feats = np.concatenate(
                    [onehot_7x4, stats_7x6.astype(np.float32, copy=False), qual_7[:, None]],
                    axis=1,
                ).astype(np.float32, copy=False)

                pos_info = {
                    "chrom": ref_name,
                    "ref_pos_0based": ref_pos,
                    "strand": "-" if read_obj.is_reverse else "+",
                    "read_name": read_obj.read_name,
                    "read_pos_0based": int(center),
                }

                yield (
                    torch.from_numpy(feats),                          # (7,11)
                    torch.tensor(kmer_code, dtype=torch.int32),       # scalar
                    pos_info,
                    torch.tensor(mod_score, dtype=torch.float32),     # scalar
                )

        bam.close()


    

#Validation set with reference (per-site) labels
def _range_slice_positions(sorted_pos_label: List[Tuple[int, float]], start: int, end: int) -> List[Tuple[int, float]]:
    """
    Get (pos,label) with start <= pos < end from a list sorted by pos.
    """
    positions = [p for p, _ in sorted_pos_label]
    i0 = bisect.bisect_left(positions, start)
    i1 = bisect.bisect_left(positions, end)
    return sorted_pos_label[i0:i1]

@dataclass
class ValStreamConfig:
    kmer_len: int = 7
    normalize_signal: bool = True
    require_mapped: bool = True
    allow_secondary: bool = False
    allow_supplementary: bool = False


class SignalBAMRefPosValidationDataset(IterableDataset):
    """
    Streams WT SignalBAM and yields samples only at labeled reference sites.

    Yields:
      (features_7x11, kmer_code_int32, pos_info_dict, label_float32)
    """

    def __init__(
        self,
        bam_path: str,
        ref_fasta_path: str,
        pos_list: str,
        seq_type: str = "rna",
        chrom_list: Optional[List[str]] = None,
        cfg: ValStreamConfig = ValStreamConfig(),
        max_lines: int = 100000,
    ):
        super().__init__()
        self.bam_path = bam_path
        self.ref_fasta_path = ref_fasta_path
        self.seq_type = seq_type
        self.pos_list = pos_list
        self.chrom_list = chrom_list or []
        self.cfg = cfg
        self.max_lines = max_lines

        self.k = int(cfg.kmer_len)
        self.flank = (self.k - 1) // 2

        # Preload reference numeric encoding using your utility
        self.ref_seq_dict, _, self.labelled_pos_list = bam_utils.pre_process(
            ref=self.ref_fasta_path,
            seq_type=self.seq_type,
            pos_list=self.pos_list,
            motif=None,
            chrom_list=self.chrom_list,
            bam=self.bam_path,
        )

    def _worker_shard_ok(self, read_idx: int) -> bool:
        wi = get_worker_info()
        if wi is None:
            return True
        return (read_idx % wi.num_workers) == wi.id

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]]:
        bam = pysam.AlignmentFile(self.bam_path, "rb")
        emitted = 0
        for read_idx, bam_read in enumerate(bam.fetch(until_eof=True)):
            if not self._worker_shard_ok(read_idx):
                continue
            if self.cfg.require_mapped and (not bam_read.is_mapped):
                continue
            if (not self.cfg.allow_secondary) and bam_read.is_secondary:
                continue
            if (not self.cfg.allow_supplementary) and bam_read.is_supplementary:
                continue

            # Build Read object via your SignalBAM helpers
            try:
                (
                    move_table,
                    mat,
                    base_qual,
                    full_seq,
                    aligned_pairs,
                    read_name,
                    ref_name,
                    shift,
                    scale,
                    norm_type,
                    seq_type,
                    div,
                    flag,
                    ref_start,
                    ref_end,
                    cigarstring,
                ) = bam_utils.get_read_info(bam_read, self.ref_seq_dict, self.seq_type)
            except Exception:
                continue

            read_obj = bam_utils.Read(
                move_table=move_table,
                signal=mat,
                base_qual=base_qual,
                full_seq=full_seq,
                aligned_pairs=aligned_pairs,
                read_name=read_name,
                ref_name=ref_name,
                shift=shift,
                scale=scale,
                norm_type=norm_type,
                seq_type=seq_type,
                div=div,
                flag=flag,
                reference_start=ref_start,
                reference_end=ref_end,
                cigarstring=cigarstring,
            )

            if self.cfg.normalize_signal:
                read_obj.normalize()

            # Determine read strand and pick the correct label bucket
            strand = "-" if read_obj.is_reverse else "+"
            key = (ref_name, strand)

            # Fast range query for labeled ref positions overlapping this read
            # Note: read_obj.ref_start/ref_end are 0-based; end is exclusive as stored.
            # We also need the ref position to be at least flank away from ends for a full 7-mer.
            ref_lo = int(read_obj.ref_start) + self.flank
            ref_hi = int(read_obj.ref_end) - self.flank
            if ref_hi <= ref_lo:
                continue

            candidates = _range_slice_positions(list(self.labelled_pos_list[ref_name][0].items()), ref_lo, ref_hi)
            if not candidates:
                continue
            
            # Map ref_pos -> read_pos for aligned bases
            ref_to_read = _build_ref_to_read_map(read_obj.aligned_pairs)

            # Reference numeric array for kmer extraction
            ref_arr = self.ref_seq_dict.get(ref_name, None)
            if ref_arr is None:
                continue
            ref_col = 1 if read_obj.is_reverse else 0
            ref_len = int(ref_arr.shape[0]) - 1

            for ref_pos, label in candidates:
                # Must map to a read base position (skip deletions)
                if ref_pos not in ref_to_read:
                    continue
                center = ref_to_read[ref_pos]

                # Must have full 7-mer window in read coords (for per-read features)
                if center < self.flank or center >= (int(read_obj.length) - self.flank):
                    continue

                # Must have full 7-mer window in reference bounds
                if ref_pos - self.flank < 0 or ref_pos + self.flank >= ref_len:
                    continue

                # Extract reference 7-mer in read orientation
                kmer_nums = ref_arr[ref_pos - self.flank : ref_pos + self.flank + 1, ref_col].astype(np.int64, copy=False)
                if np.any(kmer_nums > 3):
                    continue

                kmer_str = "".join(bam_utils.num_to_base_map[int(x)] for x in kmer_nums)
                try:
                    kmer_code = int(feature_utils.encode_kmer(kmer_str, k=self.k))
                except Exception:
                    continue

                # Per-read signal chunks for each of the 7 bases around center
                chunk = []
                for j in range(center - self.flank, center + self.flank + 1):
                    chunk.append(_get_signal_vector_for_base(read_obj, j))

                # (7,6): [wmean,q1,q2,q3,central,rms]
                stats_7x6 = feature_utils.extract_kmer_features(chunk, self.k)

                qual_7 = read_obj.base_qual[center - self.flank : center + self.flank + 1].astype(np.float32, copy=False)
                onehot_7x4 = np.stack([_onehot_base_num(int(b)) for b in kmer_nums], axis=0)

                feats = np.concatenate(
                    [onehot_7x4, stats_7x6.astype(np.float32, copy=False), qual_7[:, None]],
                    axis=1,
                ).astype(np.float32, copy=False)  # (7,11)

                pos_info = {
                    "chrom": ref_name,
                    "ref_pos_0based": int(ref_pos),
                    "strand": strand,
                    "read_name": read_obj.read_name,
                    "read_pos_0based": int(center),
                    "ref_start_0based": int(read_obj.ref_start),
                    "ref_end_0based_excl": int(read_obj.ref_end),
                }

                emitted += 1
                yield (
                    torch.from_numpy(feats),                    # (7,11)
                    torch.tensor(kmer_code, dtype=torch.int32),  # scalar
                    pos_info,
                    torch.tensor(label, dtype=torch.float32),    # scalar
                )
                if emitted > self.max_lines:
                    break

        bam.close()




#validation with region
class SignalBAMRegionDataset(IterableDataset):
    def __init__(self, bam_path, ref_fasta_path, chrom, start, end, kmer_len=7, seq_type="rna",
                 normalize_signal=True, fake_label=1.0):
        super().__init__()
        self.bam_path=bam_path
        self.ref_fasta_path=ref_fasta_path
        self.chrom=chrom
        self.start=int(start)
        self.end=int(end)
        self.k=int(kmer_len)
        self.flank=(self.k-1)//2
        self.seq_type=seq_type
        self.normalize_signal=normalize_signal
        self.fake_label=fake_label

        self.ref_seq_dict, _, _ = bam_utils.pre_process(
            ref=self.ref_fasta_path,
            seq_type=self.seq_type,
            pos_list=None, motif=None, chrom_list=[self.chrom], bam=self.bam_path
        )

    def __iter__(self):
        bam = pysam.AlignmentFile(self.bam_path, "rb")
        # Requires BAM index (.bai). If missing, index it first.
        for bam_read in bam.fetch(self.chrom, self.start, self.end):
            if not bam_read.is_mapped:
                continue
            try:
                (move_table, mat, base_qual, full_seq, aligned_pairs, read_name, ref_name,
                 shift, scale, norm_type, seq_type, div, flag, ref_start, ref_end, cigarstring
                ) = bam_utils.get_read_info(bam_read, self.ref_seq_dict, self.seq_type)
            except Exception:
                continue

            read_obj = bam_utils.Read(
                move_table=move_table, signal=mat, base_qual=base_qual, full_seq=full_seq,
                aligned_pairs=aligned_pairs, read_name=read_name, ref_name=ref_name,
                shift=shift, scale=scale, norm_type=norm_type, seq_type=seq_type, div=div,
                flag=flag, reference_start=ref_start, reference_end=ref_end, cigarstring=cigarstring
            )
            if self.normalize_signal:
                read_obj.normalize()

            # read overlaps region by construction, but limit to the requested window
            lo = max(self.start, int(read_obj.ref_start))
            hi = min(self.end, int(read_obj.ref_end))  # end is exclusive-like

            # Need full 7-mer on reference
            lo_k = lo + self.flank
            hi_k = hi - self.flank
            if hi_k <= lo_k:
                continue

            strand = "-" if read_obj.is_reverse else "+"
            ref_arr = self.ref_seq_dict.get(ref_name, None)
            if ref_arr is None:
                continue
            ref_col = 1 if read_obj.is_reverse else 0
            ref_len = int(ref_arr.shape[0]) - 1

            ref_to_read = _build_ref_to_read_map(read_obj.aligned_pairs)

            for ref_pos in range(lo_k, hi_k):
                # must map to a read base (skip deletions)
                if ref_pos not in ref_to_read:
                    continue
                center = ref_to_read[ref_pos]
                if center < self.flank or center >= (int(read_obj.length) - self.flank):
                    continue
                if ref_pos - self.flank < 0 or ref_pos + self.flank >= ref_len:
                    continue

                kmer_nums = ref_arr[ref_pos-self.flank:ref_pos+self.flank+1, ref_col].astype(np.int64, copy=False)
                if np.any(kmer_nums > 3):
                    continue
                kmer_str = "".join(bam_utils.num_to_base_map[int(x)] for x in kmer_nums)
                kmer_code = int(feature_utils.encode_kmer(kmer_str, k=self.k))

                chunk = [_get_signal_vector_for_base(read_obj, j)
                         for j in range(center-self.flank, center+self.flank+1)]
                stats_7x6 = feature_utils.extract_kmer_features(chunk, self.k)
                qual_7 = read_obj.base_qual[center-self.flank:center+self.flank+1].astype(np.float32, copy=False)
                onehot_7x4 = np.stack([_onehot_base_num(int(b)) for b in kmer_nums], axis=0)
                feats = np.concatenate([onehot_7x4, stats_7x6.astype(np.float32, copy=False), qual_7[:,None]], axis=1)

                pos_info = {
                    "chrom": ref_name,
                    "ref_pos_0based": ref_pos,
                    "strand": strand,
                    "read_name": read_obj.read_name,
                    "read_pos_0based": int(center),
                }

                yield torch.from_numpy(feats), torch.tensor(kmer_code, dtype=torch.int32), pos_info, self.fake_label

        bam.close()
