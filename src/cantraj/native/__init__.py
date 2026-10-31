"""Native kernels: C sources compiled with gcc on first use and loaded with ctypes, with pure-Python references.

dtw.c holds the DTW of the SPy / RoCuMa baseline (cantraj/baselines/spy.py): `dtw_batch` scores many runs of one
curvature array against a steering section (each run z-scored from its own mean and std, a shared best-so-far cutoff
for exact early abandoning, a DP-cell budget) and `dtw_path` returns the warping path of one pair (optionally with a
free start and / or a free end). The shared library is built into this directory under a name carrying the hash of the
source and the flags, compiled to a temporary name and moved with os.replace (concurrent workers may race; every one of
them produces the same file). When it cannot be built or loaded, the entry points raise RuntimeError: the references
below give identical results bit for bit but run about three orders of magnitude slower, so they are used only when
the environment variable CANTRAJ_PY_DTW=1 allows it (or with native=False); the tests compare both on random inputs.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import warnings

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "dtw.c")
_FLAGS = ("-O3", "-shared", "-fPIC", "-ffp-contract=off")      # no FMA contraction: same bits as the references
_BIG = 2 ** 62
_lib, _err, _tried = None, None, False


def _load():
    """The compiled kernel (built on first use) or None when it cannot be built or loaded."""
    global _lib, _err, _tried
    if _tried:
        return _lib
    _tried = True
    try:
        with open(_SRC, "rb") as f:
            tag = hashlib.sha256(f.read() + " ".join(_FLAGS).encode()).hexdigest()[:12]
        so = os.path.join(_DIR, f"_dtw_{tag}.so")
        if not os.path.exists(so):
            tmp = os.path.join(_DIR, f".{os.path.basename(so)}.{os.getpid()}.tmp")
            try:
                subprocess.run(["gcc", *_FLAGS, "-o", tmp, _SRC, "-lm"], check=True, capture_output=True, text=True)
                os.replace(tmp, so)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            for f in os.listdir(_DIR):                                  # builds of older sources
                if f.startswith("_dtw_") and f.endswith(".so") and f != os.path.basename(so):
                    try:
                        os.unlink(os.path.join(_DIR, f))
                    except OSError:
                        pass
        lib = ctypes.CDLL(so)
        dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        i64 = ctypes.c_int64
        lib.spy_dtw_batch.argtypes = [dp, i64, dp, ip, i64, dp, ip, dp, ip, i64]
        lib.spy_dtw_batch.restype = ctypes.c_int
        lib.spy_dtw_path.argtypes = [dp, i64, dp, i64, i64, i64, ip, ip, dp, ip]
        lib.spy_dtw_path.restype = i64
        _lib = lib
    except (OSError, subprocess.SubprocessError) as e:
        _err = (getattr(e, "stderr", None) or str(e)).strip()[:300]
        warnings.warn(f"cantraj.native: DTW kernel unavailable ({_err})", RuntimeWarning)
    return _lib


def _kernel(native):
    """The compiled kernel, or None for the references: asked for (native=False), or allowed by CANTRAJ_PY_DTW=1
    when the kernel cannot be built or loaded; otherwise that is an error (no silent fallback: the references would
    turn a budget-bound episode into hours)."""
    if not native:
        return None
    lib = _load()
    if lib is None and os.environ.get("CANTRAJ_PY_DTW") != "1":
        raise RuntimeError(f"native DTW kernel unavailable ({_err}); CANTRAJ_PY_DTW=1 allows the slow Python reference")
    return lib


def available():
    """True when the compiled kernel is in use."""
    return _load() is not None


def build_error():
    _load()
    return _err


# ---- references (same arithmetic order as dtw.c)
def zscore(x):
    """(x - mean) / std with ddof = 1 (MATLAB zscore); std < 1e-9 (or a single sample) -> 1. Sums run left to right
    (np.cumsum) as in the C kernel."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mu = np.cumsum(x)[-1] / n
    d = x - mu
    sd = math.sqrt(np.cumsum(d * d)[-1] / (n - 1)) if n > 1 else 0.0
    if not sd >= 1e-9:
        sd = 1.0
    return (x - mu) / sd


def band_limits(i, m, n, w):
    """Columns [lo, hi] of row i under the band |j (m-1) - i (n-1)| <= w (m-1); every column for w < 0 or m = 1."""
    if w < 0 or m == 1:
        return 0, n - 1
    c, r = i * (n - 1), w * (m - 1)
    return max(0, -((r - c) // (m - 1))), min(n - 1, (c + r) // (m - 1))


def dtw_batch_ref(a, b, runs, best=math.inf, cells=0, cap=None):
    za = zscore(a).tolist()
    m = len(za)
    cap = _BIG if cap is None else int(cap)
    runs = np.asarray(runs, dtype=np.int64).reshape(-1, 3).tolist()
    out = np.full(len(runs), math.inf)
    bi, hit = -1, False
    for r, (s, e, w) in enumerate(runs):
        if cells >= cap:
            hit = True
            break
        zb = zscore(b[s:e]).tolist()
        n = e - s
        prev, d = None, math.inf
        for i in range(m):
            lo, hi = band_limits(i, m, n, w)
            cur = [math.inf] * n
            ai = za[i]
            for j in range(lo, hi + 1):
                mn = 0.0 if i == 0 and j == 0 else math.inf
                if i > 0:
                    mn = min(mn, prev[j], prev[j - 1] if j > 0 else math.inf)
                if j > 0:
                    mn = min(mn, cur[j - 1])
                cur[j] = abs(ai - zb[j]) + mn
            cells += max(hi - lo + 1, 0)
            rmin = min(cur[lo:hi + 1], default=math.inf)
            if not rmin <= best or rmin == math.inf:
                break
            if i == m - 1:
                d = cur[n - 1]
                break
            if cells >= cap:
                hit = True
                break
            prev = cur
        if hit:
            break
        out[r] = d
        if d < best:
            best, bi = d, r
    return out, best, bi, cells, hit


def dtw_path_ref(a, b, w=-1, free_start=False, free_end=False):
    za, zb = zscore(a).tolist(), zscore(b).tolist()
    m, n = len(za), len(zb)
    D = [[math.inf] * n for _ in range(m)]
    B = [[2] * n for _ in range(m)]
    cells, none = 0, (np.zeros(0, np.int64), np.zeros(0, np.int64))
    for i in range(m):
        lo, hi = band_limits(i, m, n, w)
        for j in range(lo, hi + 1):
            mn, k = (0.0 if i == 0 and (j == 0 or free_start) else math.inf), 2
            if i > 0 and j > 0 and D[i - 1][j - 1] < mn:
                mn, k = D[i - 1][j - 1], 0
            if i > 0 and D[i - 1][j] < mn:
                mn, k = D[i - 1][j], 1
            if j > 0 and D[i][j - 1] < mn:
                mn, k = D[i][j - 1], 2
            D[i][j] = abs(za[i] - zb[j]) + mn
            B[i][j] = k
        cells += max(hi - lo + 1, 0)
        if not min(D[i][lo:hi + 1], default=math.inf) < math.inf:       # NaN (overflowing input) counts as no path
            return (math.inf, *none, cells)
    je = n - 1
    if free_end:                                                        # the first column of the last row's minimum
        je = lo
        for j in range(lo + 1, hi + 1):
            if D[m - 1][j] < D[m - 1][je]:
                je = j
    if not D[m - 1][je] < math.inf:
        return (math.inf, *none, cells)
    pi, pj = [m - 1], [je]
    i, j = m - 1, je
    while i or (j and not free_start):
        k = B[i][j]
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        pi.append(i)
        pj.append(j)
    return D[m - 1][je], np.array(pi[::-1], dtype=np.int64), np.array(pj[::-1], dtype=np.int64), 2 * cells


# ---- entry points
def dtw_batch(a, b, runs, best=math.inf, cells=0, cap=None, native=True):
    """Score runs (start, end exclusive, band; band < 0: none) of b against a: DTW(zscore(a), zscore(b[start:end])).
    A run is abandoned (score inf) as soon as a finished DP row's minimum exceeds the best score so far (exact, costs are
    >= 0); the best moves only on a strictly lower score, so ties keep the earliest run. The cell counter `cells` is
    checked against `cap` after every row; reaching it stops the batch. Returns (scores (R,), best, index of the run that
    set the best or -1, cells, budget_hit)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    runs = np.ascontiguousarray(np.asarray(runs, dtype=np.int64).reshape(-1, 3))
    if len(a) < 1 or (len(runs) and (runs[:, 0].min() < 0 or runs[:, 1].max() > len(b) or (runs[:, 1] <= runs[:, 0]).any())):
        raise ValueError("dtw_batch: empty section or a run outside b")
    cap = _BIG if cap is None else int(cap)
    lib = _kernel(native)
    if lib is None:
        return dtw_batch_ref(a, b, runs, best, cells, cap)
    out = np.empty(len(runs))
    bst, bi, cl = np.array([best], dtype=np.float64), np.array([-1], dtype=np.int64), np.array([cells], dtype=np.int64)
    rc = lib.spy_dtw_batch(a, len(a), b if len(b) else np.zeros(1), runs if len(runs) else np.zeros((1, 3), np.int64),
                           len(runs), bst, bi, out if len(out) else np.zeros(1), cl, cap)
    if rc < 0:
        raise MemoryError("native DTW batch: allocation failed")
    return out, float(bst[0]), int(bi[0]), int(cl[0]), rc == 1


def dtw_path(a, b, w=-1, free_start=False, free_end=False, native=True):
    """Warping path of DTW(zscore(a), zscore(b)) with band w: (score, i (P,), j (P,), cells) from (0, 0) to
    (m-1, n-1), back-pointer ties resolved diagonal, then up, then left; (inf, empty, empty, cells) when no path is
    admissible. free_start: every cell of row 0 starts a path (D(0, j) = d(0, j)), so the path begins at (0, j0) for
    some j0; free_end: it ends at (m-1, argmin_j D(m-1, j)) (ties: the smallest j) and the score is that minimum.
    Non-finite inputs raise ValueError. Cells count the two passes of the native kernel (the reference reports the same
    number)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("dtw_path: empty input")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("dtw_path: non-finite input")
    lib = _kernel(native)
    if lib is None:
        return dtw_path_ref(a, b, w, free_start, free_end)
    m, n = len(a), len(b)
    pi, pj = np.empty(m + n - 1, dtype=np.int64), np.empty(m + n - 1, dtype=np.int64)
    sc, cl = np.array([math.inf]), np.array([0], dtype=np.int64)
    k = lib.spy_dtw_path(a, m, b, n, int(w), int(bool(free_start)) | 2 * int(bool(free_end)), pi, pj, sc, cl)
    if k == -2:
        raise MemoryError("native DTW path: allocation failed")
    if k < 0:
        return math.inf, pi[:0].copy(), pj[:0].copy(), int(cl[0])
    return float(sc[0]), pi[:k].copy(), pj[:k].copy(), int(cl[0])
