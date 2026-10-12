# cython: language_level=3
"""Fast, bit-identical implementations of the hot loops in cutils.pyx.

Every public function here reproduces the corresponding cutils/main function
exactly (same float32/float64 rounding, same numpy tie orders, same quirks) and
returns None when one of its preconditions does not hold, so that the caller can
run the original implementation instead.  Build with -ffp-contract=off (see
setup.py): fused multiply-adds would change the last bit of some results.
"""
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt, pow
from libc.stdlib cimport malloc, free

np.import_array()

# The emulations below were verified against numpy 1.x semantics (value-based
# casting, scalar reprs); under numpy >= 2 the original code behaves differently
# (NEP 50), so the fast paths are disabled there.
NUMPY_OK = int(np.__version__.split('.')[0]) < 2

cdef long long INT_MIN_ = -2147483648
cdef long long INT_MAX_ = 2147483647


# numpy DOUBLE_pairwise_sum (numpy/core/src/umath/loops_utils.h.src), PW_BLOCKSIZE = 128.
# np.sum(a) == 0.0 + pairwise_sum(a) bit-for-bit (np.average uses it for both sums).
cdef double pairwise_sum(const double* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, n2
    cdef double res, r0, r1, r2, r3, r4, r5, r6, r7
    if n < 8:
        res = -0.0
        for i in range(n):
            res += a[i]
        return res
    elif n <= 128:
        r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3]
        r4 = a[4]; r5 = a[5]; r6 = a[6]; r7 = a[7]
        i = 8
        while i < n - (n % 8):
            r0 += a[i]; r1 += a[i+1]; r2 += a[i+2]; r3 += a[i+3]
            r4 += a[i+4]; r5 += a[i+5]; r6 += a[i+6]; r7 += a[i+7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2)


cdef inline bint _fits_c_int(np.ndarray a):
    # cutils assigns these np.int64 bins to C int variables, which raises
    # OverflowError outside the int range; the fast kernels must not see such values.
    if a.shape[0] == 0:
        return True
    return a.min() >= INT_MIN_ and a.max() <= INT_MAX_


# ---------------------------------------------------------------------------
# PASEF centroiding: exact replacement of cutils.centroid_pasef_scan
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Py_ssize_t _centroid_kernel(const np.int64_t[::1] mzf, const np.int64_t[::1] imf,
                                 const double[::1] mz, const double[::1] it, const double[::1] im,
                                 double hacc, double imacc, float pasefmini, int pasefminlh,
                                 double[::1] out_mz, double[::1] out_i, double[::1] out_im,
                                 Py_ssize_t[::1] tmp, double[::1] wbuf, double[::1] pbuf) noexcept nogil:
    # Same walk as cutils.centroid_pasef_scan.  The 'banned_idx' set of the original
    # can never match (every banned index is below the current cursor), so it is omitted.
    # Returns the number of output peaks, or -1 if a cluster has zero total weight
    # (np.average raises ZeroDivisionError there; the caller falls back to the original).
    cdef Py_ssize_t N = mz.shape[0], nout = 0, p = 0, j, k, ltmp
    cdef int mvi, ivi
    cdef float macc, iv
    cdef double s, scl, mzv, imv
    while p < N:
        macc = <float>(mz[p] * 1e-6 * hacc)
        mvi = <int>mzf[p]
        ivi = <int>imf[p]
        tmp[0] = p
        ltmp = 1
        j = p + 1
        while j < N:
            if <int>mzf[j] - mvi > 1:
                break
            elif fabs(mz[p] - mz[j]) <= <double>macc:
                if abs(ivi - <int>imf[j]) <= 1:
                    if fabs(im[p] - im[j]) <= imacc:
                        tmp[ltmp] = j
                        ltmp += 1
                        p = j
            j += 1
        if ltmp >= pasefminlh:
            if ltmp == 1:
                iv = <float>it[p]
                if iv >= pasefmini:
                    out_i[nout] = iv
                    out_mz[nout] = <float>mz[p]
                    out_im[nout] = <float>im[p]
                    nout += 1
            else:
                # builtin sum() of the intensities: sequential, in double (numpy 1.x promotes 0 + np.float32)
                s = 0.0
                for k in range(ltmp):
                    wbuf[k] = it[tmp[k]]
                    s = s + wbuf[k]
                iv = <float>s
                if iv >= pasefmini:
                    # np.average(x, weights=w) == (0.0 + pairwise_sum(x*w)) / (0.0 + pairwise_sum(w))
                    scl = 0.0 + pairwise_sum(&wbuf[0], ltmp)
                    if scl == 0.0:
                        return -1
                    for k in range(ltmp):
                        pbuf[k] = mz[tmp[k]] * wbuf[k]
                    mzv = (0.0 + pairwise_sum(&pbuf[0], ltmp)) / scl
                    for k in range(ltmp):
                        pbuf[k] = im[tmp[k]] * wbuf[k]
                    imv = (0.0 + pairwise_sum(&pbuf[0], ltmp)) / scl
                    out_i[nout] = iv
                    out_mz[nout] = <float>mzv
                    out_im[nout] = <float>imv
                    nout += 1
        p += 1
    return nout


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _builtin_max(const double[::1] a) noexcept nogil:
    # Python's max(iterable): keep the first item, replace when item > current
    cdef double best = a[0]
    cdef Py_ssize_t i
    for i in range(1, a.shape[0]):
        if a[i] > best:
            best = a[i]
    return best


def centroid_pasef_scan_fast(dict scandict, float mz_step, float hill_mz_accuracy, float ion_mobility_accuracy, float pasefmini, int pasefminlh):
    """Exact replacement of cutils.centroid_pasef_scan.

    Returns (mz, intensity, im) as float64 arrays (the values of the lists the
    original returns), or None when the input is outside what the fast kernel
    reproduces (non-float64 m/z or IM, intensity not float32/float64, empty frame,
    bins outside the C int range, NaN/inf, zero-weight clusters, numpy >= 2).
    """
    if not NUMPY_OK:
        return None
    mz_ar = scandict['m/z array']
    it_ar = scandict['intensity array']
    im_ar = scandict['mean inverse reduced ion mobility array']
    if not (isinstance(mz_ar, np.ndarray) and isinstance(it_ar, np.ndarray) and isinstance(im_ar, np.ndarray)):
        return None
    if mz_ar.ndim != 1 or it_ar.ndim != 1 or im_ar.ndim != 1:
        return None
    if mz_ar.dtype != np.float64 or im_ar.dtype != np.float64 or (it_ar.dtype != np.float32 and it_ar.dtype != np.float64):
        return None
    cdef Py_ssize_t N = mz_ar.shape[0], n
    if N == 0 or it_ar.shape[0] != N or im_ar.shape[0] != N:
        return None
    cdef float ion_mobility_step = _builtin_max(np.ascontiguousarray(im_ar)) * ion_mobility_accuracy
    with np.errstate(all='ignore'):
        imf = (im_ar / ion_mobility_step).astype(int)
        mzf = (mz_ar / mz_step).astype(int)
    if not (_fits_c_int(imf) and _fits_c_int(mzf)):
        return None
    # same np.argsort call (same dtype, default kind) as the original -> same tie order
    idx = np.argsort(mzf)
    mzf = np.ascontiguousarray(mzf[idx], dtype=np.int64)
    imf = np.ascontiguousarray(imf[idx], dtype=np.int64)
    mz = np.ascontiguousarray(mz_ar[idx])
    it = np.ascontiguousarray(it_ar[idx], dtype=np.float64)
    im = np.ascontiguousarray(im_ar[idx])
    out_mz = np.empty(N)
    out_i = np.empty(N)
    out_im = np.empty(N)
    tmp = np.empty(N, dtype=np.intp)
    wbuf = np.empty(N)
    pbuf = np.empty(N)
    cdef const np.int64_t[::1] v_mzf = mzf
    cdef const np.int64_t[::1] v_imf = imf
    cdef const double[::1] v_mz = mz
    cdef const double[::1] v_it = it
    cdef const double[::1] v_im = im
    cdef double[::1] v_omz = out_mz
    cdef double[::1] v_oi = out_i
    cdef double[::1] v_oim = out_im
    cdef double[::1] v_wb = wbuf
    cdef double[::1] v_pb = pbuf
    cdef Py_ssize_t[::1] v_tmp = tmp
    cdef double hacc = hill_mz_accuracy, imacc = ion_mobility_accuracy
    with nogil:
        n = _centroid_kernel(v_mzf, v_imf, v_mz, v_it, v_im, hacc, imacc, pasefmini, pasefminlh,
                             v_omz, v_oi, v_oim, v_tmp, v_wb, v_pb)
    if n < 0:
        return None
    return out_mz[:n].copy(), out_i[:n].copy(), out_im[:n].copy()


# ---------------------------------------------------------------------------
# Hill linking: exact replacement of cutils.detect_hills
# ---------------------------------------------------------------------------

cdef inline bint _float_array(object a):
    return isinstance(a, np.ndarray) and a.ndim == 1 and (a.dtype == np.float64 or a.dtype == np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def detect_hills_fast(list data_for_analyse_tmp, float hill_mass_accuracy, float mz_step, float paseftol, int md_correction_int=1, bint want_mass_diff=False):
    """Exact replacement of cutils.detect_hills (dia or not: pass the matching tolerance).

    Returns (hills_dict, total_mass_diff) where hills_dict holds int64 numpy arrays
    'hills_idx_array', 'orig_idx_array' and 'scan_idx_array' (the original returns
    the same values as Python lists, plus three per-peak lists nothing reads), and
    total_mass_diff is a list of floats (empty unless want_mass_diff).
    Returns None when the inputs are outside what this function reproduces.

    Semantics reproduced (see cutils.detect_hills):
    * peaks of a frame are visited in np.argsort(intensity)[::-1] order and the
      previous frame's m/z buckets list peaks in that order;
    * 'all_idx = prev_fast_dict[fm]; all_idx += ...' extends the previous frame's
      bucket lists IN PLACE.  Only the order of the last occurrence of each peak in
      a candidate list matters for the selection below, so buckets are kept
      de-duplicated in last-occurrence order (A += X -> (A minus X) ++ X);
    * the ion mobility test (list membership in the +-1 IM buckets) is
      equivalent to |IM bin(prev) - IM bin(cur)| <= 1;
    * float32 arithmetic of the cdef float variables, the last candidate with the
      maximal float32 intensity wins, and the previous peak with index 0 is never
      banned (best_idx_prev != 0 test).
    """
    if not NUMPY_OK or md_correction_int not in (1, 2, 3) or paseftol < 0:
        return None
    cdef bint use_im = paseftol > 0
    cdef Py_ssize_t nfr = len(data_for_analyse_tmp), f
    cdef long long kmin = 0, kmax = 0
    cdef bint have_k = False
    lens = np.empty(nfr, dtype=np.int64)
    # pass 1: validate and get the global m/z bin range
    with np.errstate(all='ignore'):
        for f in range(nfr):
            z = data_for_analyse_tmp[f]
            if not isinstance(z, dict):
                return None
            mz_a = z.get('m/z array')
            it_a = z.get('intensity array')
            if not (_float_array(mz_a) and _float_array(it_a)) or mz_a.shape[0] != it_a.shape[0]:
                return None
            if use_im:
                im_a = z.get('mean inverse reduced ion mobility array')
                if not _float_array(im_a) or im_a.shape[0] != mz_a.shape[0]:
                    return None
                fi_a = (im_a / paseftol).astype(int)
                if not _fits_c_int(fi_a):
                    return None
            lens[f] = mz_a.shape[0]
            if mz_a.shape[0]:
                fm_a = (mz_a / mz_step).astype(int)
                if not _fits_c_int(fm_a):
                    return None
                if not have_k:
                    kmin = fm_a.min(); kmax = fm_a.max(); have_k = True
                else:
                    kmin = min(kmin, <long long>fm_a.min()); kmax = max(kmax, <long long>fm_a.max())
    if kmax - kmin > 100000000:
        return None   # dense bin table would be too large; the dict-based original copes

    cdef Py_ssize_t N = lens.sum()
    off = np.zeros(nfr + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    cdef Py_ssize_t maxn = lens.max() if nfr else 0
    hills = np.arange(N, dtype=np.int64)
    cdef np.int64_t[::1] v_hills = hills
    tmd = np.empty(N if want_mass_diff else 1, dtype=np.float32)
    cdef float[::1] v_tmd = tmd
    cdef Py_ssize_t ntmd = 0

    cdef long long bmin = kmin - 3
    head = np.full(kmax - kmin + 7, -1, dtype=np.int64)
    cdef np.int64_t[::1] v_head = head
    bstart_a = np.empty(maxn + 1, dtype=np.int64)
    blen_a = np.empty(maxn + 1, dtype=np.int64)
    bkey_a = np.empty(maxn + 1, dtype=np.int64)
    mark_a = np.zeros(maxn + 1, dtype=np.int64)
    banned_a = np.zeros(maxn + 1, dtype=np.uint8)
    cdef np.int64_t[::1] bstart = bstart_a
    cdef np.int64_t[::1] blen = blen_a
    cdef np.int64_t[::1] bkey = bkey_a
    cdef np.int64_t[::1] mark = mark_a
    cdef np.uint8_t[::1] banned = banned_a
    cdef Py_ssize_t cap = 4 * maxn + 16
    pool_a = np.empty(cap, dtype=np.int64)
    cdef np.int64_t[::1] pool = pool_a

    cdef const double[::1] c_mz
    cdef const float[::1] c_it
    cdef const np.int64_t[::1] c_fm
    cdef const np.int64_t[::1] c_fi
    cdef const np.int64_t[::1] c_order
    cdef const double[::1] p_mz
    cdef const float[::1] p_it
    cdef const np.int64_t[::1] p_fi

    cdef Py_ssize_t n, j, q, la, lx, st, stx, w, w0, cst, cln, ncap, acc, i
    cdef Py_ssize_t nb_prev = 0, pool_used = 0, np_prev = 0
    cdef long long s0, sp = 0, stamp = 0
    cdef long long idx, fmv, fiv, b, b1, b2, b3, a, x, y, ext, ip, dfi, best_prev
    cdef int k
    cdef bint have_prev = False, found
    cdef float mzc, d, cmd, ci, best_int, best_md
    cdef float htol = hill_mass_accuracy
    cdef double mzc64, koef

    with np.errstate(all='ignore'):
        for f in range(nfr):
            z = data_for_analyse_tmp[f]
            mz_a = z['m/z array']
            it_a = z['intensity array']
            n = mz_a.shape[0]
            s0 = off[f]
            # identical call to the original: descending order of np.argsort(intensity)
            order_a = np.ascontiguousarray(np.argsort(it_a)[::-1], dtype=np.int64)
            fm_a = np.ascontiguousarray((mz_a / mz_step).astype(int), dtype=np.int64)
            if use_im:
                fi_a = np.ascontiguousarray((z['mean inverse reduced ion mobility array'] / paseftol).astype(int), dtype=np.int64)
            else:
                fi_a = fm_a
            mz64 = np.ascontiguousarray(mz_a, dtype=np.float64)
            it32 = np.ascontiguousarray(it_a, dtype=np.float32)
            c_mz = mz64
            c_it = it32
            c_fm = fm_a
            c_fi = fi_a
            c_order = order_a

            if have_prev and nb_prev > 0:
                for i in range(np_prev):
                    banned[i] = 0
                for j in range(n):
                    idx = c_order[j]
                    fmv = c_fm[idx]
                    fiv = c_fi[idx]
                    b1 = v_head[fmv - bmin]
                    b2 = v_head[fmv - 1 - bmin]
                    b3 = v_head[fmv + 1 - bmin]
                    if b1 < 0 and b2 < 0 and b3 < 0:
                        continue
                    if b1 >= 0:
                        a = b1; x = b2; y = b3
                    elif b2 >= 0:
                        a = b2; x = b3; y = -1
                    else:
                        a = b3; x = -1; y = -1
                    for k in range(2):
                        ext = x if k == 0 else y
                        if ext < 0:
                            continue
                        # A' = (A minus X) ++ X  (de-duplicated in-place list extension)
                        la = blen[a]; lx = blen[ext]
                        stamp += 1
                        stx = bstart[ext]
                        for q in range(lx):
                            mark[pool[stx + q]] = stamp
                        if pool_used + la + lx > cap:
                            ncap = max(2 * cap, pool_used + la + lx + 16)
                            pool2 = np.empty(ncap, dtype=np.int64)
                            pool2[:pool_used] = pool_a[:pool_used]
                            pool_a = pool2
                            pool = pool_a
                            cap = ncap
                        st = bstart[a]
                        if st + la == pool_used:
                            w = st            # bucket is at the end of the pool: compact in place
                        else:
                            w = pool_used
                        w0 = w
                        for q in range(la):
                            ip = pool[st + q]
                            if mark[ip] != stamp:
                                pool[w] = ip
                                w += 1
                        for q in range(lx):
                            pool[w] = pool[stx + q]
                            w += 1
                        bstart[a] = w0
                        blen[a] = w - w0
                        pool_used = w
                    cst = bstart[a]
                    cln = blen[a]

                    mzc = <float>c_mz[idx]
                    mzc64 = <double>mzc
                    if md_correction_int == 1:
                        koef = sqrt(600.0 / mzc64)
                    elif md_correction_int == 2:
                        koef = 1.0
                    else:
                        koef = 600.0 / mzc64
                    best_int = 0
                    best_prev = 0
                    best_md = 0
                    found = False
                    for q in range(cln):
                        ip = pool[cst + q]
                        if banned[ip]:
                            continue
                        if use_im:
                            dfi = p_fi[ip] - fiv
                            if dfi > 1 or dfi < -1:
                                continue
                        d = <float>((mzc64 - p_mz[ip]) / mzc64 * 1e6)
                        d = <float>(<double>d * koef)
                        cmd = fabs(d)
                        ci = p_it[ip]
                        if cmd <= htol and ci >= best_int:
                            best_int = ci
                            best_prev = ip
                            best_md = d
                            found = True
                    if found:
                        v_hills[s0 + idx] = v_hills[sp + best_prev]
                        if best_prev != 0:
                            banned[best_prev] = 1
                            if want_mass_diff:
                                v_tmd[ntmd] = best_md
                                ntmd += 1

            # buckets of the current frame (become the 'previous' buckets)
            for b in range(nb_prev):
                v_head[bkey[b] - bmin] = -1
            nb_prev = 0
            for j in range(n):
                idx = c_order[j]
                fmv = c_fm[idx]
                b = v_head[fmv - bmin]
                if b < 0:
                    b = nb_prev
                    nb_prev += 1
                    v_head[fmv - bmin] = b
                    bkey[b] = fmv
                    blen[b] = 0
                blen[b] += 1
            acc = 0
            for b in range(nb_prev):
                bstart[b] = acc
                acc += blen[b]
                blen[b] = 0
            if acc > cap:
                cap = 2 * acc
                pool_a = np.empty(cap, dtype=np.int64)
                pool = pool_a
            for j in range(n):
                idx = c_order[j]
                b = v_head[c_fm[idx] - bmin]
                pool[bstart[b] + blen[b]] = idx
                blen[b] += 1
            pool_used = acc
            p_mz = c_mz
            p_it = c_it
            p_fi = c_fi
            sp = s0
            np_prev = n
            have_prev = True

    frame_of_peak = np.repeat(np.arange(nfr, dtype=np.int64), lens)
    hills_dict = {
        'hills_idx_array': hills,
        'orig_idx_array': np.arange(N, dtype=np.int64) - off[:nfr][frame_of_peak],
        'scan_idx_array': frame_of_peak,
    }
    total_mass_diff = tmd[:ntmd].astype(np.float64).tolist() if want_mass_diff else []
    return hills_dict, total_mass_diff


# ---------------------------------------------------------------------------
# Hill splitting and per-hill aggregates: exact replacement of
# main.split_peaks_multi + cutils.split_peaks + cutils.process_hills
# (the Python orchestration lives in fastpath.py)
# ---------------------------------------------------------------------------

cdef inline double _seq_dot_c(const double* x, Py_ssize_t n, double c) noexcept nogil:
    # numpy uses BLAS ddot for the edge windows of np.convolve; for these short
    # windows it is a plain sequential loop (no FMA)
    cdef double s = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        s += x[i] * c
    return s


cdef Py_ssize_t _meanfilt7(const double* d, Py_ssize_t L, double* out, double* rd) noexcept nogil:
    """Bit-exact model of np.clip(np.convolve(d, np.ones(7)/7, mode='same'), 0, None)
    (cutils.meanfilt with window_width 3).  Returns the output length, which is
    max(L, 7): for L < 7 numpy swaps the operands and returns 7 values."""
    cdef double c = 1.0 / 7.0
    cdef double s
    cdef Py_ssize_t i, j, nl, nr, n, o, nout
    if L >= 7:
        for i in range(3):
            out[i] = _seq_dot_c(d, 4 + i, c)
        for i in range(L - 6):
            s = 0.0
            for j in range(7):
                s += d[i + j] * c
            out[3 + i] = s
        for i in range(3):
            out[L - 3 + i] = _seq_dot_c(d + L - 6 + i, 6 - i, c)
        nout = L
    else:
        for i in range(L):
            rd[i] = d[L - 1 - i]
        nl = L // 2
        nr = L - nl - 1
        n = L - nl
        o = 0
        for i in range(nl):
            out[o] = _seq_dot_c(rd + nl - i, n, c)
            o += 1
            n += 1
        for i in range(7 - L + 1):
            s = 0.0
            for j in range(L):
                s += c * rd[j]
            out[o] = s
            o += 1
        for i in range(nr):
            n -= 1
            out[o] = _seq_dot_c(rd, n, c)
            o += 1
        nout = 7
    for i in range(nout):
        if out[i] < 0.0:
            out[i] = 0.0
    return nout


cdef inline double _smax(const double* a, Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t n) noexcept nogil:
    # builtin max(a[lo:hi]) with Python slice clamping (slices are never empty here)
    cdef double m
    cdef Py_ssize_t i
    if hi > n:
        hi = n
    m = a[lo]
    for i in range(lo + 1, hi):
        if a[i] > m:
            m = a[i]
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def split_hills_kernel(const double[::1] I, const np.int64_t[::1] hill_off, int mlh, float hvf):
    """Valley detection of cutils.split_peaks on hills stored as CSR (intensities in
    (hill, scan) order).  mlh = int(max(2, minlh)).  Returns a uint8 'segment start'
    flag per peak: the new hill ids are np.cumsum(flags) (first-appearance relabelling
    in split_peaks_multi)."""
    cdef Py_ssize_t n = I.shape[0], nh = hill_off.shape[0] - 1, h, a, b, L, nsm, k, maxL = 0
    cdef Py_ssize_t idx, c_len, nmin, l_idx, end_idx, inc
    cdef double l_r, r_r, hvf_d = hvf
    cdef float mult_val, min_val
    starts_a = np.zeros(n, dtype=np.uint8)
    cdef np.uint8_t[::1] starts = starts_a
    for h in range(nh):
        L = hill_off[h + 1] - hill_off[h]
        if L > maxL:
            maxL = L
    cdef double* sm = <double*>malloc((maxL + 8) * sizeof(double))
    cdef double* rd = <double*>malloc((maxL + 8) * sizeof(double))
    cdef Py_ssize_t* min_idx_list = <Py_ssize_t*>malloc((maxL + 2) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* recheck = <Py_ssize_t*>malloc((maxL + 2) * sizeof(Py_ssize_t))
    if sm == NULL or rd == NULL or min_idx_list == NULL or recheck == NULL:
        free(sm); free(rd); free(min_idx_list); free(recheck)
        raise MemoryError()
    try:
        with nogil:
            for h in range(nh):
                a = hill_off[h]
                b = hill_off[h + 1]
                if b > a:
                    starts[a] = 1
                L = b - a
                if L < mlh * 2:
                    continue
                nsm = _meanfilt7(&I[a], L, sm, rd)
                c_len = L - mlh
                idx = mlh - 1
                nmin = 0
                min_val = 0
                l_idx = 0
                while idx <= c_len:
                    if nmin > 0 and idx >= min_idx_list[nmin - 1] + mlh - (min_idx_list[nmin - 1] - recheck[nmin - 1]):
                        l_idx = min_idx_list[nmin - 1]
                    l_r = _smax(sm, l_idx, idx, nsm) / sm[idx]
                    if l_r >= hvf_d:
                        r_r = _smax(sm, idx + 1, nsm, nsm) / sm[idx]
                        if r_r >= hvf_d:
                            mult_val = <float>(l_r * r_r)
                            inc = 1 if l_r > r_r else 0
                            if mlh <= idx + inc and idx + inc <= c_len:
                                if nmin == 0 or idx + inc >= min_idx_list[nmin - 1] + mlh:
                                    min_idx_list[nmin] = idx + inc
                                    recheck[nmin] = idx
                                    nmin += 1
                                    min_val = mult_val
                                elif mult_val > min_val:
                                    min_idx_list[nmin - 1] = idx + inc
                                    recheck[nmin - 1] = idx
                                    min_val = mult_val
                    idx += 1
                for k in range(nmin):
                    end_idx = min_idx_list[k + 1] if k + 1 < nmin else L
                    r_r = _smax(sm, recheck[k] + 1, end_idx + 1, nsm) / sm[recheck[k]]
                    if r_r >= hvf_d:
                        starts[a + min_idx_list[k]] = 1
    finally:
        free(sm); free(rd); free(min_idx_list); free(recheck)
    return starts_a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hill_aggregates(const double[::1] mz, const double[::1] I, const double[::1] im, const np.int64_t[::1] off, float mz_step, float paseftol, bint use_im):
    """Per-hill values of cutils.process_hills:
    hills_mz_median: float32 accumulation of mz*I and I in scan order (cdef float loop);
    hills_im_median: float32(np.average(im, weights=I)) = (0 + pairwise(im*I)) / (0 + pairwise(I));
    bin keys (int)(float32 value / float32 step).
    Returns (mz_med f32, im_med f32, mz_bin int64, im_bin int64), or None if a value
    would be non-finite or a bin would not fit a C int (the original raises there)."""
    cdef Py_ssize_t nh = off.shape[0] - 1, h, p, a, b, n = I.shape[0]
    cdef float accm, acci, m32, i32, mm, im32, q
    cdef double avg, scl
    cdef bint bad = False
    mz_med_a = np.empty(nh, dtype=np.float32)
    im_med_a = np.zeros(nh, dtype=np.float32)
    mz_bin_a = np.empty(nh, dtype=np.int64)
    im_bin_a = np.zeros(nh, dtype=np.int64)
    prod_a = np.empty(n if use_im else 1, dtype=np.float64)
    cdef float[::1] mz_med = mz_med_a
    cdef float[::1] im_med = im_med_a
    cdef np.int64_t[::1] mz_bin = mz_bin_a
    cdef np.int64_t[::1] im_bin = im_bin_a
    cdef double[::1] prod = prod_a
    with nogil:
        if use_im:
            for p in range(n):
                prod[p] = im[p] * I[p]
        for h in range(nh):
            a = off[h]
            b = off[h + 1]
            accm = 0
            acci = 0
            for p in range(a, b):
                m32 = <float>mz[p]
                i32 = <float>I[p]
                accm = accm + m32 * i32
                acci = acci + i32
            if acci == 0:
                bad = True
                break
            mm = accm / acci
            mz_med[h] = mm
            q = mm / mz_step
            if not (q > -2147483648.0 and q < 2147483648.0):
                bad = True
                break
            mz_bin[h] = <int>q
            if use_im:
                scl = 0.0 + pairwise_sum(&I[a], b - a)
                if scl == 0.0:
                    bad = True
                    break
                avg = (0.0 + pairwise_sum(&prod[a], b - a)) / scl
                im32 = <float>avg
                im_med[h] = im32
                q = im32 / paseftol
                if not (q > -2147483648.0 and q < 2147483648.0):
                    bad = True
                    break
                im_bin[h] = <int>q
    if bad:
        return None
    return mz_med_a, im_med_a, mz_bin_a, im_bin_a


# ---------------------------------------------------------------------------
# Isotope cluster search: exact replacement of cutils.get_initial_isotopes and
# cutils.checking_cos_correlation_for_carbon for hills coming from process_hills
# with contiguous scans.
# ---------------------------------------------------------------------------

import itertools
import math
import sys
from .cutils import checking_cos_correlation_for_carbon

# cos_correlation_new sums Python floats with builtin sum(); Python >= 3.12 uses
# compensated summation there, which the C emulation below does not reproduce.
CARBON_C_OK = sys.version_info[:2] < (3, 12)


cdef inline int _set_mask(int n) noexcept nogil:
    # CPython set growth policy (setobject.c): the table starts with 8 slots; after an
    # insert, if fill*5 >= mask*3 it is resized to the smallest power of 2 > used*4
    # (used*2 above 50000).
    cdef int mask = 7, fill = 0, minused, newsize
    while fill < n:
        fill += 1
        if fill * 5 >= mask * 3:
            minused = fill * 4 if fill <= 50000 else fill * 2
            newsize = 8
            while newsize <= minused:
                newsize <<= 1
            mask = newsize - 1
    return mask


def set_order_selfcheck():
    """Check the set-iteration-order model used by _cos_rt against this interpreter."""
    cdef int a, n, mask, m, v
    for n in range(1, 700, 3):
        for a in (0, 1, 5, 37, 128, 1000, 4093):
            s1 = set(np.arange(a, a + n, dtype=np.int64))
            s2 = set(np.arange(a - 3, a + n + 5, dtype=np.int64))
            real = [int(x) for x in s1.intersection(s2)]
            mask = _set_mask(n)
            m = (a + mask) & ~mask
            if m > a + n - 1 or m == a:
                emu = list(range(a, a + n))
            else:
                emu = list(range(m, a + n)) + list(range(a, m))
            if real != emu:
                return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float _cos_rt(int h1, int h2, const int[::1] first, const int[::1] last, const np.int64_t[::1] ioff, const double[::1] ival, const float[::1] sq) noexcept:
    # cutils.cos_correlation for contiguous hills: iterate the scan intersection in
    # CPython set-iteration order (ascending slot = scan & mask, a rotation of the
    # range), accumulate in float32 via double adds, divide in float32.
    cdef int a = first[h1] if first[h1] > first[h2] else first[h2]
    cdef int b = last[h1] if last[h1] < last[h2] else last[h2]
    cdef int n = b - a + 1
    cdef int mask, m, v
    cdef float top = 0
    cdef np.int64_t o1 = ioff[h1] - first[h1], o2 = ioff[h2] - first[h2]
    if n <= 0:
        return 0
    mask = _set_mask(n)
    m = (a + mask) & ~mask
    if m > b or m == a:
        for v in range(a, b + 1):
            top = <float>(<double>top + ival[o1 + v] * ival[o2 + v])
    else:
        for v in range(m, b + 1):
            top = <float>(<double>top + ival[o1 + v] * ival[o2 + v])
        for v in range(a, m):
            top = <float>(<double>top + ival[o1 + v] * ival[o2 + v])
    return top / (sq[h1] * sq[h2])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cos_new_c(double* th, int nth, double* ex, int nex, float* corr, float* expl) noexcept nogil:
    # cutils.cos_correlation_new with its C-float locals (py3.11 sequential sum())
    cdef float theor_total_sum, top, i1, i2, bottom, numb
    cdef double s, s1, s2
    cdef int k, suit_len
    s = 0.0
    for k in range(nth):
        s = s + th[k]
    theor_total_sum = <float>s
    suit_len = nth if nth < nex else nex
    top = 0
    for k in range(suit_len):
        i1 = <float>th[k]
        i2 = <float>ex[k]
        top = top + i1 * i2
    if not top:
        corr[0] = 0
        expl[0] = 0
        return
    s1 = 0.0
    for k in range(suit_len):
        numb = <float>th[k]
        s1 = s1 + <double>(numb * numb)
    s2 = 0.0
    for k in range(suit_len):
        numb = <float>ex[k]
        s2 = s2 + <double>(numb * numb)
    bottom = <float>(sqrt(s1) * sqrt(s2))
    s = 0.0
    for k in range(suit_len):
        s = s + th[k]
    expl[0] = <float>(s / <double>theor_total_sum)
    corr[0] = top / bottom


@cython.cdivision(True)
cdef void _carbon_c(double* th, int nth, double* ex, int nex, float thresh, float* best_cor_out, int* best_pos_out) noexcept nogil:
    # cutils.checking_cos_correlation_for_carbon
    cdef float averagineCorrelation, averagineExplained, best_cor
    cdef int best_pos, pos
    best_pos = 1
    best_cor = 0
    pos = nex
    while pos != 1:
        _cos_new_c(th, nth, ex, pos, &averagineCorrelation, &averagineExplained)
        if averagineExplained >= 0.5 and averagineCorrelation >= thresh:
            if averagineCorrelation > best_cor:
                best_cor = averagineCorrelation
                best_pos = pos
            break
        pos -= 1
    best_cor_out[0] = best_cor
    best_pos_out[0] = best_pos


def carbon_lists(list theoretical_list, list experimental_list, float thresh):
    """Drop-in for cutils.checking_cos_correlation_for_carbon (same values and types)."""
    cdef double th[64]
    cdef double ex[64]
    cdef int nth = len(theoretical_list), nex = len(experimental_list), k
    cdef float bc
    cdef int bp
    if not CARBON_C_OK or nth > 64 or nex > 64 or nex < 1:
        return checking_cos_correlation_for_carbon(theoretical_list, experimental_list, thresh)
    for k in range(nth):
        th[k] = theoretical_list[k]
    for k in range(nex):
        ex[k] = experimental_list[k]
    _carbon_c(th, nth, ex, nex, thresh, &bc, &bp)
    return bc, bp


@cython.boundscheck(False)
@cython.wraparound(False)
def _hill_stats(const np.int64_t[::1] ioff, const double[::1] ival):
    # sqrt: f32(math.sqrt(sum(v**2))) as get_and_calc_values_for_cos_corr (sequential, pow(v, 2));
    # apex: first strict maximum starting from 0, as get_and_calc_apex_intensity_and_scan
    cdef Py_ssize_t n = ioff.shape[0] - 1, i, j
    cdef double s, best
    sq = np.empty(n, dtype=np.float32)
    apex = np.empty(n, dtype=np.float64)
    cdef float[::1] sqv = sq
    cdef double[::1] apv = apex
    for i in range(n):
        s = 0.0
        best = 0
        for j in range(ioff[i], ioff[i + 1]):
            s = s + pow(ival[j], 2.0)
            if ival[j] > best:
                best = ival[j]
        sqv[i] = <float>sqrt(s)
        apv[i] = best
    return sq, apex


def build_isotope_index(csr, int T=32):
    """Search index over hills for get_initial_isotopes_fast.

    Hills with span <= T scans are grouped by their own m/z bin (the key that
    process_hills used for hills_mz_median_fast_dict) and sorted by first scan;
    longer hills are kept in a per-bin list.  Returns None if a hill does not have
    contiguous scans (the fast cosine relies on it)."""
    seg_off = csr['seg_off']
    n = len(seg_off) - 1
    first = np.ascontiguousarray(csr['scans'][seg_off[:-1]], dtype=np.int32)
    last = np.ascontiguousarray(csr['scans'][seg_off[1:] - 1], dtype=np.int32)
    length = last.astype(np.int64) - first + 1
    if not np.array_equal(length, np.diff(seg_off)):
        return None
    key = csr['mz_bin']
    kmin = int(key.min()) - 2
    nb = int(key.max()) - kmin + 3
    k0 = key - kmin
    short = length <= T
    sidx = np.flatnonzero(short)
    o = np.lexsort((sidx, first[sidx], k0[sidx]))
    sidx = np.ascontiguousarray(sidx[o], dtype=np.int32)
    soff = np.zeros(nb + 1, dtype=np.int64)
    np.cumsum(np.bincount(k0[sidx], minlength=nb), out=soff[1:])
    lidx = np.flatnonzero(~short)
    o = np.lexsort((lidx, k0[lidx]))
    lidx = np.ascontiguousarray(lidx[o], dtype=np.int32)
    loff = np.zeros(nb + 1, dtype=np.int64)
    np.cumsum(np.bincount(k0[lidx], minlength=nb), out=loff[1:])
    ival = np.ascontiguousarray(csr['I64'])
    ioff = np.ascontiguousarray(seg_off, dtype=np.int64)
    sq, apex = _hill_stats(ioff, ival)
    return dict(mz=np.ascontiguousarray(csr['mz_med'], dtype=np.float32),
                imkey=np.ascontiguousarray(csr['im_bin'], dtype=np.int32),
                first=first, last=last, kmin=kmin, nb=nb, T=T,
                sidx=sidx, sfirst=np.ascontiguousarray(first[sidx]), soff=soff, lidx=lidx, loff=loff,
                ioff=ioff, ival=ival, sq=sq, apex=apex)


cdef inline void _isort(int* a, int n) noexcept nogil:
    cdef int i, j, x
    for i in range(1, n):
        x = a[i]
        j = i - 1
        while j >= 0 and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def get_initial_isotopes_fast(dict hills_dict, float isotopes_mass_accuracy, list isotopes_list, dict a, int min_charge, int max_charge, float mz_step, float paseftol, float faims_val, float ivf, list sorted_idx_child_process, int md_correction_int, dict index):
    """cutils.get_initial_isotopes with the candidate scan of the whole-run m/z bin
    lists replaced by a lookup in build_isotope_index; candidates are emitted in
    ascending hill index order (the fast-dict list order), and the scan-set/dict
    cosine, apex and carbon cosine are computed from the CSR arrays with the same
    float32/float64 arithmetic.  Everything else is verbatim."""

    cdef list ready, charges, candidates, tmp_candidates
    cdef int idx_1, hill_idx_1, idx_2, hill_idx_2, im_to_check_fast, hill_scans_1_list_first, hill_scans_1_list_last, ch_cur, ch_iter
    cdef int charge, isotope_number, m_to_check_fast, max_pos, i_local_isotope
    cdef float im_mz_1, hill_mz_1, hill_mz_2, m_to_check, mass_diff_abs, cos_cor_RT
    cdef float local_minimum, hill_intensity_apex_1, hill_intensity_apex_2
    cdef double mass_diff_ppm
    cdef dict banned_charges, local_isotopes_dict
    cdef dict charge_ban_map

    cdef const float[::1] X_mz = index['mz']
    cdef const int[::1] X_imkey = index['imkey']
    cdef const int[::1] X_first = index['first']
    cdef const int[::1] X_last = index['last']
    cdef const int[::1] X_sidx = index['sidx']
    cdef const int[::1] X_sfirst = index['sfirst']
    cdef const np.int64_t[::1] X_soff = index['soff']
    cdef const int[::1] X_lidx = index['lidx']
    cdef const np.int64_t[::1] X_loff = index['loff']
    cdef int kmin = index['kmin'], nb = index['nb'], T = index['T']
    cdef const np.int64_t[::1] X_ioff = index['ioff']
    cdef const double[::1] X_ival = index['ival']
    cdef const float[::1] X_sq = index['sq']
    cdef const double[::1] X_apex = index['apex']
    cdef int b, bb, lo, hi, mid, j, ncand, ci, f1lim, cap
    cdef bint use_im = paseftol > 0
    buf_a = np.empty(max(16, X_mz.shape[0] + 1), dtype=np.int32)
    cdef int[::1] buf = buf_a
    cap = buf.shape[0]
    hills_idx_array_unique = hills_dict['hills_idx_array_unique']
    hills_lengths = hills_dict['hills_lengths']

    charge_ban_map = {1: [1, ]}
    for ch_cur in range(2, max_charge + 1):
        charge_ban_map[ch_cur] = [ch_iter for ch_iter in charge_ban_map.keys() if (ch_cur % ch_iter) == 0]

    ready = []

    charges = list(range(min_charge, max_charge + 1, 1)[::-1])

    for idx_1 in sorted_idx_child_process:
        hill_idx_1 = hills_idx_array_unique[idx_1]
        hill_mz_1 = X_mz[idx_1]

        if paseftol > 0:
            im_mz_1 = hills_dict['hills_im_median'][idx_1]
            im_to_check_fast = int(im_mz_1 / paseftol)

        banned_charges = dict()

        hill_scans_1_number = hills_lengths[idx_1]
        hill_scans_1_list_first = X_first[idx_1]
        hill_scans_1_list_last = X_last[idx_1]
        f1lim = hill_scans_1_list_first - T + 1

        mz_tol = isotopes_mass_accuracy * 1e-6 * hill_mz_1

        for charge in charges:

            candidates = []

            for isotope_number in isotopes_list[1:]:

                tmp_candidates = []

                m_to_check = hill_mz_1 + (1.00335 * isotope_number / charge)
                m_to_check_fast = int(m_to_check/mz_step)

                # candidates: hills in m/z bins key-1..key+1 that overlap in scans,
                # pass the IM bin test and the m/z tolerance
                ncand = 0
                for bb in range(m_to_check_fast - 1, m_to_check_fast + 2):
                    b = bb - kmin
                    if b < 0 or b >= nb:
                        continue
                    lo = X_soff[b]
                    hi = X_soff[b + 1]
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if X_sfirst[mid] <= hill_scans_1_list_last:
                            lo = mid + 1
                        else:
                            hi = mid
                    j = lo - 1
                    while j >= X_soff[b] and X_sfirst[j] >= f1lim:
                        idx_2 = X_sidx[j]
                        j -= 1
                        if X_last[idx_2] < hill_scans_1_list_first:
                            continue
                        if use_im and (X_imkey[idx_2] < im_to_check_fast - 1 or X_imkey[idx_2] > im_to_check_fast + 1):
                            continue
                        hill_mz_2 = X_mz[idx_2]
                        mass_diff_abs = hill_mz_2 - m_to_check
                        if abs(mass_diff_abs) <= mz_tol:
                            buf[ncand] = idx_2
                            ncand += 1
                    for j in range(X_loff[b], X_loff[b + 1]):
                        idx_2 = X_lidx[j]
                        if X_last[idx_2] < hill_scans_1_list_first or X_first[idx_2] > hill_scans_1_list_last:
                            continue
                        if use_im and (X_imkey[idx_2] < im_to_check_fast - 1 or X_imkey[idx_2] > im_to_check_fast + 1):
                            continue
                        hill_mz_2 = X_mz[idx_2]
                        mass_diff_abs = hill_mz_2 - m_to_check
                        if abs(mass_diff_abs) <= mz_tol:
                            buf[ncand] = idx_2
                            ncand += 1
                # ascending hill index = order of the hills_mz_median_fast_dict lists
                if ncand > 1:
                    _isort(&buf[0], ncand)

                for ci in range(ncand):
                    idx_2 = buf[ci]
                    hill_mz_2 = X_mz[idx_2]
                    mass_diff_abs = hill_mz_2 - m_to_check
                    # contiguous hills overlapping in scans => non-empty scan intersection
                    cos_cor_RT = _cos_rt(idx_1, idx_2, X_first, X_last, X_ioff, X_ival, X_sq)

                    if cos_cor_RT >= 0.6:

                        hill_idx_2 = hills_idx_array_unique[idx_2]

                        if md_correction_int == 1:
                            mass_diff_error_koeff = math.sqrt(600 / m_to_check)
                        elif md_correction_int == 2:
                            mass_diff_error_koeff = 1
                        elif md_correction_int == 3:
                            mass_diff_error_koeff = 600 / m_to_check

                        mass_diff_ppm = mass_diff_abs*1e6/m_to_check * mass_diff_error_koeff

                        local_isotopes_dict = {
                            'isotope_number': isotope_number,
                            'isotope_hill_idx': hill_idx_2,
                            'isotope_idx': idx_2,
                            'cos_cor': cos_cor_RT,
                            'mass_diff_ppm': mass_diff_ppm,
                        }

                        tmp_candidates.append(local_isotopes_dict)

                if len(tmp_candidates):
                    candidates.append(tmp_candidates)

                if len(candidates) < isotope_number:
                    break

            if len(candidates) >= banned_charges.get(charge, 1):

                neutral_mass = hill_mz_1 * charge

                tmp_intensity, max_pos = a[int(100 * (neutral_mass // 100))]
                if max_pos < 4:
                    max_pos = 4

                hill_intensity_apex_1 = X_apex[idx_1]

                all_theoretical_int = [
                    hill_intensity_apex_1 *
                    tmp_intensity[z] /
                    tmp_intensity[0] for z in isotopes_list]

                for iter_candidates in itertools.product(*candidates):

                    all_exp_intensity = [hill_intensity_apex_1, ]

                    i_local_isotope = 1

                    for local_isotopes_dict in iter_candidates:

                        idx_2 = local_isotopes_dict['isotope_idx']

                        hill_intensity_apex_2 = X_apex[idx_2]

                        if i_local_isotope > max_pos:
                            if i_local_isotope == max_pos + 1 or hill_intensity_apex_2 < local_minimum:
                                local_minimum = hill_intensity_apex_2
                                local_minimum_pos = i_local_isotope
                            if hill_intensity_apex_2 >= ivf * local_minimum:
                                all_exp_intensity = all_exp_intensity[:local_minimum_pos+1]
                                break

                        all_exp_intensity.append(hill_intensity_apex_2)
                        i_local_isotope += 1

                    cos_corr, number_of_passed_isotopes = carbon_lists(all_theoretical_int, all_exp_intensity, 0.6)
                    if neutral_mass >= 1500:
                        cos_corr2, number_of_passed_isotopes2 = carbon_lists(all_theoretical_int[1:], all_exp_intensity[:-1], 0.6)
                        if cos_corr2 >= cos_corr:
                            cos_corr = cos_corr2
                            number_of_passed_isotopes = number_of_passed_isotopes2

                    if cos_corr:

                        iter_candidates = iter_candidates[:number_of_passed_isotopes]

                        local_res_dict = {
                            'monoisotope hill idx': hill_idx_1,
                            'monoisotope idx': idx_1,
                            'cos_cor_isotopes': cos_corr,
                            'hill_mz_1': hill_mz_1,
                            'isotopes': iter_candidates,
                            'nIsotopes': number_of_passed_isotopes,
                            'nScans': hill_scans_1_number,
                            'charge': charge,
                            'FAIMS': faims_val,
                            'im': 0 if paseftol == 0 else im_mz_1,
                            'intensity_array_for_cos_corr': [all_theoretical_int[:number_of_passed_isotopes+1], all_exp_intensity[:number_of_passed_isotopes+1]],
                        }

                        ready.append(local_res_dict)

                        for ch_v in charge_ban_map[charge]:
                            banned_charges[ch_v] = max(number_of_passed_isotopes, banned_charges.get(ch_v, 1))
    return ready

