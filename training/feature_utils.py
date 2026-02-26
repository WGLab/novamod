import numpy as np
from numba import njit
import random

@njit
def percentile_1d(a, p):
    """p in [0,100] — numba-safe percentile for 1-D float64 array."""
    n  = len(a)
    if n == 0:
        return 0.0                 # or np.nan
    k  = (n-1) * p / 100.0
    f  = int(np.floor(k))
    c  = int(np.ceil(k))
    a_sorted = np.sort(a)          # numba supports np.sort 1-D
    if f == c:
        return a_sorted[f]
    return a_sorted[f] + (a_sorted[c]-a_sorted[f])*(k-f)

@njit
def winsorised_mean(a, lp=1.0, up=99.0):
    if len(a) == 0:
        return 0.0
    lo = percentile_1d(a, lp)
    hi = percentile_1d(a, up)
    s  = 0.0
    for v in a:
        if v < lo:
            s += lo
        elif v > hi:
            s += hi
        else:
            s += v
    return s / len(a)

@njit
def central_value(a):
    n = len(a)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return a[mid]
    else:
        return 0.5 * (a[mid] + a[mid-1])

@njit
def rms_energy(a):
    if len(a) == 0:
        return 0.0
    s = 0.0
    for v in a:
        s += v * v
    return np.sqrt(s / len(a))

@njit
def extract_kmer_features(chunk, k):
    """
    chunk : list length k, each element is np.ndarray(float64, 1-D)
    returns: k×6 np.ndarray(float64)
    """
    out = np.empty((k, 6), dtype=np.float64)
    for i in range(k):
        x = chunk[i]
        out[i, 0] = winsorised_mean(x)
        out[i, 1] = percentile_1d(x, 25.0)
        out[i, 2] = percentile_1d(x, 50.0)
        out[i, 3] = percentile_1d(x, 75.0)
        out[i, 4] = central_value(x)
        out[i, 5] = rms_energy(x)
    return out

import numpy as np

def encode_kmer(kmer, k: int = 9) -> np.uint32:
    # LUT: ASCII → 0/1/2/3 (A/C/G/T); everything else = 255
    lut = np.full(256, 255, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3
    # ---- normalise input ---------------------------------------------------
    if isinstance(kmer, bytes):
        b = kmer.strip()                  # drop trailing spaces in S9 dtype
    else:
        b = str(kmer).strip().encode("ascii")
    # ---- ASCII → base‑4 digits in one vectorised step ----------------------
    ascii_vals = np.frombuffer(b, dtype=np.uint8)
    if ascii_vals.size != k:
        raise ValueError(f"Expected a {k}-mer, got length {ascii_vals.size}.")
    digits = lut[ascii_vals]
    if np.any(digits > 3):
        raise ValueError(f"kmer contains non-ACGT bases: {kmer}")
    # ---- pack k × 2 bits into one integer ----------------------------------
    code = 0
    for d in digits:
        code = (code << 2) | int(d)
    if code >= (1 << 2*k):           # 2k‑bit ceiling for any k‑mer
        raise ValueError(
            f"Internal check failed: kmer {kmer} exceeds {2*k}-bit range, "
            "which means this input length does not match the configured k."
        )
        
    return np.uint32(code)

def encode_base(kmer, k: int = 9) -> np.uint32:
    # LUT: ASCII → 0/1/2/3 (A/C/G/T); everything else = 255
    lut = np.full(256, 255, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3
    # ---- normalise input ---------------------------------------------------
    if isinstance(kmer, bytes):
        b = kmer.strip()                  # drop trailing spaces in S9 dtype
    else:
        b = str(kmer).strip().encode("ascii")
    # ---- ASCII → base‑4 digits in one vectorised step ----------------------
    ascii_vals = np.frombuffer(b, dtype=np.uint8)
    if ascii_vals.size != k:
        raise ValueError(f"Expected a {k}-mer, got length {ascii_vals.size}.")
    digits = lut[ascii_vals]
    if np.any(digits > 3):
        raise ValueError(f"kmer contains non-ACGT bases: {kmer}")
    base = digits[int((k-1)/2)]
        
    return np.uint32(base)

def decode_kmer(code: int, k: int = 9) -> str:
    if code >= (1 << (2 * k)):
        raise ValueError(f"Code {code} exceeds the {k}-mer range")

    rev = "ACGT"          # index 0→A, 1→C, 2→G, 3→T
    seq = [""] * k

    # fill from the least‑significant 2 bits upward
    for i in range(k - 1, -1, -1):
        seq[i] = rev[code & 0b11]   # take the last 2 bits
        code >>= 2                  # shift to the next base
        
    return "".join(seq)

def decode_base(code: int, k: int = 9) -> np.uint32:
    if code >= (1 << (2 * k)):
        raise ValueError(f"Code {code} exceeds the {k}-mer range")
        
    seq = [0] * k

    # fill from the least‑significant 2 bits upward
    for i in range(k - 1, -1, -1):
        seq[i] = code & 0b11   # take the last 2 bits
        code >>= 2                  # shift to the next base
    base = seq[int((k-1)/2)]
    
    return np.uint32(base)
