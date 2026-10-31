/* DTW kernels of the SPy / RoCuMa baseline (cantraj/baselines/spy.py), compiled and loaded by cantraj.native.

   Semantics (MATLAB dtw as inferred from Pese et al., PoPETs 2020, pp. 162-164): both series are z-scored (mean and
   std with ddof = 1 as MATLAB zscore, std < 1e-9 -> 1), local cost |a_i - b_j|, steps (1,0), (0,1), (1,1) of equal
   weight, D(i,j) = d(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1)), result D(m-1,n-1) unnormalised; an optional slanted
   band |j - i (n-1)/(m-1)| <= w (0-based, evaluated in integers as |j (m-1) - i (n-1)| <= w (m-1); none for w < 0 or
   m = 1). The arithmetic order is the one of the Python references in __init__.py (sequential sums, one rounding per
   operation; built with -ffp-contract=off), so both return identical bits. Unbanded rows are computed four at a time
   (rows4: the four dependency chains overlap, about 3x faster) but checked row by row, so results and cell counts are
   those of the row-by-row recurrence. The warping path can also be taken with a free start (every cell of row 0 starts
   a path) and / or a free end (the path ends at the last row's minimum): the window-edge sections of the timing. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { BP_DIAG = 0, BP_UP = 1, BP_LEFT = 2 };
enum { FREE_START = 1, FREE_END = 2 };      /* spy_dtw_path flags */

static int64_t fdiv(int64_t a, int64_t b)               /* floor(a / b) for b > 0 */
{
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

static void band(int64_t i, int64_t m, int64_t n, int64_t w, int64_t *lo, int64_t *hi)
{
    int64_t c, r;
    if (w < 0 || m == 1) {
        *lo = 0;
        *hi = n - 1;
        return;
    }
    c = i * (n - 1);
    r = w * (m - 1);
    *lo = -fdiv(r - c, m - 1);                           /* ceil((c - r) / (m - 1)) */
    *hi = fdiv(c + r, m - 1);
    if (*lo < 0)
        *lo = 0;
    if (*hi > n - 1)
        *hi = n - 1;
}

static void zscore(const double *x, int64_t n, double *z)
{
    double s = 0.0, v = 0.0, mu, sd;
    int64_t k;
    for (k = 0; k < n; k++)
        s += x[k];
    mu = s / (double)n;
    for (k = 0; k < n; k++) {
        double d = x[k] - mu;
        v += d * d;
    }
    sd = n > 1 ? sqrt(v / (double)(n - 1)) : 0.0;
    if (!(sd >= 1e-9))
        sd = 1.0;
    for (k = 0; k < n; k++)
        z[k] = (x[k] - mu) / sd;
}

/* Row i of the DP on columns [lo, hi] from the previous row (valid on [plo, phi]; bands only move right); returns the
   row minimum (INFINITY for an empty band row, possible only with w = 0). Columns outside a band are never read. */
static double row_fast(double ai, const double *b, int64_t i, int64_t lo, int64_t hi, const double *prev, int64_t plo,
                       int64_t phi, double *cur)
{
    double left, rmin, best;
    int64_t j, jm;
    if (lo > hi)
        return INFINITY;
    if (i == 0) {                                        /* lo = 0: only (0, j-1) */
        left = fabs(ai - b[0]);
        cur[0] = left;
        rmin = left;
        for (j = 1; j <= hi; j++) {
            left = fabs(ai - b[j]) + left;
            cur[j] = left;
            if (left < rmin)
                rmin = left;
        }
        return rmin;
    }
    best = INFINITY;                                     /* first cell: (i-1, lo) and (i-1, lo-1) */
    if (lo <= phi)
        best = prev[lo];
    if (lo - 1 >= plo && lo - 1 <= phi && prev[lo - 1] < best)
        best = prev[lo - 1];
    left = fabs(ai - b[lo]) + best;
    cur[lo] = left;
    rmin = left;
    jm = hi < phi ? hi : phi;                            /* lo < j <= jm: all three predecessors inside */
    for (j = lo + 1; j <= jm; j++) {
        double up = prev[j], dg = prev[j - 1];
        double mn = up < dg ? up : dg;
        if (left < mn)
            mn = left;
        left = fabs(ai - b[j]) + mn;
        cur[j] = left;
        if (left < rmin)
            rmin = left;
    }
    for (j = (jm > lo ? jm : lo) + 1; j <= hi; j++) {    /* right of the previous row's band */
        double mn = left;
        if (j - 1 <= phi && prev[j - 1] < mn)
            mn = prev[j - 1];
        left = fabs(ai - b[j]) + mn;
        cur[j] = left;
        if (left < rmin)
            rmin = left;
    }
    return rmin;
}

/* Unbanded rows i..i+3 (i >= 1) from the full previous row, interleaved so that their four dependency chains overlap
   (the cells and operations of row_fast, so the same bits). Row i+3 goes to out; rm and last get each row's minimum and
   last cell. */
static void rows4(const double *a, int64_t i, const double *b, int64_t n, const double *prev, double *out, double *rm,
                  double *last)
{
    double a0 = a[i], a1 = a[i + 1], a2 = a[i + 2], a3 = a[i + 3];
    double l0, l1, l2, l3, r0, r1, r2, r3;
    int64_t j;
    l0 = fabs(a0 - b[0]) + prev[0];
    l1 = fabs(a1 - b[0]) + l0;
    l2 = fabs(a2 - b[0]) + l1;
    l3 = fabs(a3 - b[0]) + l2;
    out[0] = l3;
    r0 = l0;
    r1 = l1;
    r2 = l2;
    r3 = l3;
    for (j = 1; j < n; j++) {
        double bj = b[j], up = prev[j], dg = prev[j - 1], mn, n0, n1, n2, n3;
        mn = up < dg ? up : dg;
        if (l0 < mn)
            mn = l0;
        n0 = fabs(a0 - bj) + mn;
        mn = n0 < l0 ? n0 : l0;                          /* row i+1: up = row i at j, diagonal = row i at j-1 */
        if (l1 < mn)
            mn = l1;
        n1 = fabs(a1 - bj) + mn;
        mn = n1 < l1 ? n1 : l1;
        if (l2 < mn)
            mn = l2;
        n2 = fabs(a2 - bj) + mn;
        mn = n2 < l2 ? n2 : l2;
        if (l3 < mn)
            mn = l3;
        n3 = fabs(a3 - bj) + mn;
        out[j] = n3;
        l0 = n0;
        l1 = n1;
        l2 = n2;
        l3 = n3;
        if (n0 < r0)
            r0 = n0;
        if (n1 < r1)
            r1 = n1;
        if (n2 < r2)
            r2 = n2;
        if (n3 < r3)
            r3 = n3;
    }
    rm[0] = r0;
    rm[1] = r1;
    rm[2] = r2;
    rm[3] = r3;
    last[0] = l0;
    last[1] = l1;
    last[2] = l2;
    last[3] = l3;
}

/* Row 0 with a free start: every cell starts a path, D(0, j) = |a_0 - b_j|; returns the row minimum. */
static double row_free(double a0, const double *b, int64_t lo, int64_t hi, double *cur)
{
    double rmin;
    int64_t j;
    if (lo > hi)
        return INFINITY;
    rmin = cur[lo] = fabs(a0 - b[lo]);
    for (j = lo + 1; j <= hi; j++) {
        cur[j] = fabs(a0 - b[j]);
        if (cur[j] < rmin)
            rmin = cur[j];
    }
    return rmin;
}

/* Same values as row_fast (row_free for row 0 with fs), plus back-pointers (ties: diagonal, then up, then left). */
static void row_bp(double ai, const double *b, int64_t i, int64_t lo, int64_t hi, const double *prev, int64_t plo,
                   int64_t phi, int fs, double *cur, uint8_t *bp)
{
    int64_t j;
    for (j = lo; j <= hi; j++) {
        double mn = INFINITY;
        uint8_t k = BP_LEFT;
        if (i == 0 && (j == 0 || fs))
            mn = 0.0;
        if (i > 0 && j - 1 >= plo && j - 1 <= phi && prev[j - 1] < mn) {
            mn = prev[j - 1];
            k = BP_DIAG;
        }
        if (i > 0 && j >= plo && j <= phi && prev[j] < mn) {
            mn = prev[j];
            k = BP_UP;
        }
        if (j > lo && cur[j - 1] < mn) {
            mn = cur[j - 1];
            k = BP_LEFT;
        }
        cur[j] = fabs(ai - b[j]) + mn;
        bp[j] = k;
    }
}

/* One DTW of z-scored a (m) and b (n). A finished row whose minimum exceeds `cutoff` (or is infinite: no admissible
   path) abandons the pair (*st = 1, INFINITY): exact, every path crosses every row and costs are >= 0. Cells are added
   to *cells; reaching `cap` with rows left stops the pair (*st = 2). */
static double dtw_cut(const double *a, int64_t m, const double *b, int64_t n, int64_t w, double cutoff, double *prev,
                      double *cur, int64_t *cells, int64_t cap, int *st)
{
    int64_t i, k, lo, hi, plo = 0, phi = -1;
    double rmin, *t, rm[4], last[4];
    *st = 0;
    for (i = 0; i < m; i++) {
        if (w < 0 && i > 0 && i + 3 < m) {                   /* unbanded: four rows at once, checked row by row */
            rows4(a, i, b, n, prev, cur, rm, last);
            for (k = 0; k < 4; k++) {
                *cells += n;
                if (!(rm[k] <= cutoff) || rm[k] == INFINITY) {
                    *st = 1;
                    return INFINITY;
                }
                if (i + k == m - 1)
                    return last[k];
                if (*cells >= cap) {
                    *st = 2;
                    return INFINITY;
                }
            }
            t = prev;
            prev = cur;
            cur = t;
            plo = 0;
            phi = n - 1;
            i += 3;
            continue;
        }
        band(i, m, n, w, &lo, &hi);
        rmin = row_fast(a[i], b, i, lo, hi, prev, plo, phi, cur);
        *cells += hi >= lo ? hi - lo + 1 : 0;
        if (!(rmin <= cutoff) || rmin == INFINITY) {
            *st = 1;
            return INFINITY;
        }
        if (i == m - 1)
            return cur[n - 1];
        if (*cells >= cap) {
            *st = 2;
            return INFINITY;
        }
        t = prev;
        prev = cur;
        cur = t;
        plo = lo;
        phi = hi;
    }
    return INFINITY;
}

/* Score runs r = 0..nr-1 (runs[3r] = start, runs[3r+1] = end (exclusive), runs[3r+2] = band) of b against a, each run
   z-scored from its own mean and std. *best is the shared cutoff (in: best so far; out: updated) and moves only on a
   strictly lower score, so ties keep the earliest run; *best_idx is set to the run that lowered it. out[r] = score, or
   INFINITY when abandoned / without admissible path / not evaluated. Returns 0, 1 when *cells reached cap (the
   remaining runs are not evaluated), -1 when memory is short. */
int spy_dtw_batch(const double *a, int64_t m, const double *b, const int64_t *runs, int64_t nr, double *best,
                  int64_t *best_idx, double *out, int64_t *cells, int64_t cap)
{
    int64_t r, nmax = 1;
    double *za, *zb, *p, *c;
    int ret = 0;
    for (r = 0; r < nr; r++) {
        int64_t n = runs[3 * r + 1] - runs[3 * r];
        if (n > nmax)
            nmax = n;
        out[r] = INFINITY;
    }
    za = malloc((size_t)m * sizeof(double));
    zb = malloc((size_t)nmax * sizeof(double));
    p = malloc((size_t)nmax * sizeof(double));
    c = malloc((size_t)nmax * sizeof(double));
    if (!za || !zb || !p || !c) {
        ret = -1;
        goto done;
    }
    zscore(a, m, za);
    for (r = 0; r < nr; r++) {
        int st;
        int64_t s = runs[3 * r], n = runs[3 * r + 1] - s;
        double d;
        if (*cells >= cap) {
            ret = 1;
            break;
        }
        zscore(b + s, n, zb);
        d = dtw_cut(za, m, zb, n, runs[3 * r + 2], *best, p, c, cells, cap, &st);
        if (st == 2) {
            ret = 1;
            break;
        }
        out[r] = d;
        if (d < *best) {
            *best = d;
            *best_idx = r;
        }
    }
done:
    free(za);
    free(zb);
    free(p);
    free(c);
    return ret;
}

/* Warping path of DTW(zscore(a), zscore(b)) with band w: cells (pi[k], pj[k]) from (0, 0) to (m-1, n-1) (capacity
   m + n - 1), *score = D(m-1, n-1). flags: FREE_START, every cell of row 0 starts a path (D(0, j) = |a_0 - b_j|) and
   the path begins where it reaches row 0; FREE_END, the path ends at the first column of the last row's minimum,
   *score = that minimum. Back-pointers are uint8, recomputed block by block (K ~ sqrt(8 m) rows) from row checkpoints
   of a first pass, so memory stays O(sqrt(m) n); every row is computed twice (*cells counts both). Returns the path
   length, -1 without admissible path (a row or the end without a finite cell: NaN from an overflowing input counts as
   none), -2 when memory is short. */
int64_t spy_dtw_path(const double *a, int64_t m, const double *b, int64_t n, int64_t w, int64_t flags, int64_t *pi,
                     int64_t *pj, double *score, int64_t *cells)
{
    int64_t K, nb, i, j, c, lo, hi, plo, phi, len = 0, ret = -1, je = n - 1;
    int fs = (flags & FREE_START) != 0, fe = (flags & FREE_END) != 0;
    double rmin, *za, *zb, *p, *q, *t, **ck;
    uint8_t *bp;
    K = (int64_t)ceil(sqrt(8.0 * (double)m));
    if (K > m)
        K = m;
    nb = (m + K - 1) / K;
    za = malloc((size_t)m * sizeof(double));
    zb = malloc((size_t)n * sizeof(double));
    p = malloc((size_t)n * sizeof(double));
    q = malloc((size_t)n * sizeof(double));
    ck = calloc((size_t)nb, sizeof(double *));
    bp = malloc((size_t)K * (size_t)n);
    if (!za || !zb || !p || !q || !ck || !bp) {
        ret = -2;
        goto done;
    }
    zscore(a, m, za);
    zscore(b, n, zb);
    plo = 0;                                             /* first pass: keep the last row of every block but the last */
    phi = -1;
    for (i = 0; i < m; i++) {
        band(i, m, n, w, &lo, &hi);
        *cells += hi >= lo ? hi - lo + 1 : 0;
        rmin = i == 0 && fs ? row_free(za[0], zb, lo, hi, q) : row_fast(za[i], zb, i, lo, hi, p, plo, phi, q);
        if (!(rmin < INFINITY))
            goto done;
        if (i % K == K - 1 && i < m - 1) {
            ck[i / K] = malloc((size_t)(hi - lo + 1) * sizeof(double));
            if (!ck[i / K]) {
                ret = -2;
                goto done;
            }
            memcpy(ck[i / K], q + lo, (size_t)(hi - lo + 1) * sizeof(double));
        }
        if (i == m - 1) {
            if (fe)                                      /* free end: the first column of the row minimum */
                for (je = lo, j = lo + 1; j <= hi; j++)
                    if (q[j] < q[je])
                        je = j;
            *score = q[je];
        }
        t = p;
        p = q;
        q = t;
        plo = lo;
        phi = hi;
    }
    if (!(*score < INFINITY))
        goto done;
    i = m - 1;                                           /* backtrack, last block first */
    j = je;
    for (c = nb - 1; c >= 0; c--) {
        int64_t r0 = c * K, r1 = (r0 + K < m ? r0 + K : m) - 1, r;
        plo = 0;
        phi = -1;
        if (c > 0) {
            band(r0 - 1, m, n, w, &plo, &phi);
            memcpy(p + plo, ck[c - 1], (size_t)(phi - plo + 1) * sizeof(double));
        }
        for (r = r0; r <= r1; r++) {
            band(r, m, n, w, &lo, &hi);
            row_bp(za[r], zb, r, lo, hi, p, plo, phi, fs, q, bp + (size_t)(r - r0) * (size_t)n);
            *cells += hi - lo + 1;
            t = p;
            p = q;
            q = t;
            plo = lo;
            phi = hi;
        }
        while (i >= r0) {
            uint8_t k;
            if (j < 0 || len >= m + n - 1)               /* never on an admissible path: a guard, not a case */
                goto done;
            pi[len] = i;
            pj[len] = j;
            len++;
            if (i == 0 && (j == 0 || fs))
                break;
            k = bp[(size_t)(i - r0) * (size_t)n + (size_t)j];
            if (k == BP_DIAG) {
                i--;
                j--;
            } else if (k == BP_UP) {
                i--;
            } else {
                j--;
            }
        }
    }
    for (i = 0; i < len / 2; i++) {                      /* (0, 0) first */
        int64_t x = pi[i], y = pj[i];
        pi[i] = pi[len - 1 - i];
        pj[i] = pj[len - 1 - i];
        pi[len - 1 - i] = x;
        pj[len - 1 - i] = y;
    }
    ret = len;
done:
    if (ck)
        for (c = 0; c < nb; c++)
            free(ck[c]);
    free(ck);
    free(bp);
    free(za);
    free(zb);
    free(p);
    free(q);
    return ret;
}
