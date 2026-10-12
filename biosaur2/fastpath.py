"""Fast paths for the default feature detection pipeline (biosaur2.main).

Everything here produces results identical to the original code in main.py /
cutils.pyx.  Each function checks its preconditions and returns None (or False)
when they do not hold, so that the caller runs the original implementation.
"""
import gc
import sys
import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

try:
    from . import cfast
    FAST_OK = bool(cfast.NUMPY_OK)
except ImportError:  # extension not built
    cfast = None
    FAST_OK = False

# Python floats in the per-hill lists behave like the np.float64 scalars the
# original stores there, except for builtin sum() on Python >= 3.12 (compensated
# summation of exact floats).  There the lists hold numpy scalars like the original.
_NATIVE_FLOAT_LISTS = sys.version_info[:2] < (3, 12)

_set_order_ok = None


def set_order_ok():
    """The fast isotope cosine emulates CPython's set iteration order; check it once."""
    global _set_order_ok
    if _set_order_ok is None:
        try:
            _set_order_ok = bool(cfast.set_order_selfcheck())
        except Exception:
            _set_order_ok = False
        if not _set_order_ok:
            logger.debug('set iteration order check failed, isotope search fast path disabled')
    return _set_order_ok


# ---------------------------------------------------------------- memory helpers

from .utils import malloc_trim  # noqa: F401

class gc_paused:
    """Disable the cyclic garbage collector inside the block.

    The hill and feature structures are millions of small containers without
    reference cycles; collections over them cost about a minute per run and
    free nothing."""

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, *exc):
        if self.was_enabled:
            gc.enable()
        return False


# ---------------------------------------------------------------- split + process

def _frame_arrays_ok(data, key, allowed):
    dt = None
    for z in data:
        a = z.get(key)
        if not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype not in allowed:
            return None
        if dt is None:
            dt = a.dtype
        elif a.dtype != dt:
            return None
    return dt if dt is not None else np.dtype(np.float64)


def _gather(data, key, dtype, ss_order, frame_bounds, os_):
    """values[p] = data[ss[p]][key][os_[p]] for every hill peak p."""
    out = np.empty(len(os_), dtype=dtype)
    for f in range(len(frame_bounds) - 1):
        a, b = frame_bounds[f], frame_bounds[f + 1]
        if a == b:
            continue
        sel = ss_order[a:b]
        out[sel] = data[f][key][os_[sel]]
    return out


def split_process_fast(hills_dict, data, mz_step, paseftol, args):
    """main.split_peaks_multi followed by cutils.process_hills (nprocs is hard-coded
    to 1 in split_peaks_multi) on flat arrays.

    Returns a dict of CSR arrays describing the final hills (hill k = new id k+1,
    peaks in scan order), or None when the original functions must run instead:
    duplicate (hill, scan) pairs (the original's unstable float-key argsorts decide
    their order), no hill reaching minlh (the original crashes), unexpected dtypes,
    negative or non-finite intensities, or values the original would reject.
    """
    if not FAST_OK:
        return None
    if not (0 < len(data)):
        return None
    use_im = paseftol > 0
    float_types = (np.dtype(np.float64), np.dtype(np.float32))
    it_dt = _frame_arrays_ok(data, 'intensity array', float_types)
    mz_dt = _frame_arrays_ok(data, 'm/z array', float_types)
    if it_dt is None or mz_dt is None:
        return None
    if use_im:
        im_dt = _frame_arrays_ok(data, 'mean inverse reduced ion mobility array', float_types)
        # np.average(im, weights=I) runs in float64 only if one of them is float64
        if im_dt is None or (im_dt == np.float32 and it_dt == np.float32):
            return None

    h = np.asarray(hills_dict['hills_idx_array'], dtype=np.int64)
    s = np.asarray(hills_dict['scan_idx_array'], dtype=np.int64)
    o = np.asarray(hills_dict['orig_idx_array'], dtype=np.int64)
    if len(h) == 0 or h.min() < 0 or len(s) != len(h) or len(o) != len(h):
        return None
    if len(s) > 1 and np.any(np.diff(s) < 0):
        return None   # detect_hills output is frame-major; anything else: original path

    minlh = args['minlh']
    counts = np.bincount(h)
    keep = counts[h] >= minlh
    pos = np.flatnonzero(keep)
    if len(pos) == 0:
        return None
    # the original orders peaks with np.argsort(h + (s+1)/(max(s)+2)); for unique
    # (hill, scan) pairs that is a stable grouping of the frame-major input by hill
    perm = pos[np.argsort(h[pos], kind='stable')]
    hs = h[perm]
    ss = s[perm]
    os_ = o[perm]
    del h, s, o, counts, keep, pos, perm
    if len(hs) > 1 and np.any((hs[1:] == hs[:-1]) & (ss[1:] == ss[:-1])):
        return None
    hstart = np.flatnonzero(np.concatenate(([True], hs[1:] != hs[:-1])))
    hill_off = np.append(hstart, len(hs)).astype(np.int64)
    n_hills_presplit = len(hstart)
    del hs, hstart

    nfr = len(data)
    if ss.max() >= nfr:
        return None
    ss_order = np.argsort(ss, kind='stable')
    frame_bounds = np.zeros(nfr + 1, dtype=np.int64)
    np.cumsum(np.bincount(ss, minlength=nfr), out=frame_bounds[1:])
    lens = np.array([len(z['m/z array']) for z in data], dtype=np.int64)
    if np.any(os_ < 0) or np.any(os_ >= lens[ss]):
        return None

    I = _gather(data, 'intensity array', it_dt, ss_order, frame_bounds, os_)
    I64 = I.astype(np.float64, copy=False)
    if not np.all(np.isfinite(I64)) or np.any(I64 < 0):
        return None

    mlh = int(max(2, minlh))
    starts = cfast.split_hills_kernel(I64, hill_off, mlh, args['hvf'])
    new_id = np.cumsum(starts, dtype=np.int64)
    seg_off = np.append(np.flatnonzero(starts), len(starts)).astype(np.int64)
    del starts
    seg_len = np.diff(seg_off)
    if np.any(seg_len < minlh):
        return None   # cannot happen (segments are >= max(2, minlh)); keep the original's semantics if it does
    if use_im and it_dt == np.float32 and seg_len.max() >= 8192:
        return None   # np.average would sum the float32 weights in buffered float64 chunks

    mz = _gather(data, 'm/z array', mz_dt, ss_order, frame_bounds, os_)
    mz64 = mz.astype(np.float64, copy=False)
    if use_im:
        im64 = _gather(data, 'mean inverse reduced ion mobility array', im_dt, ss_order, frame_bounds, os_).astype(np.float64)
        if not np.all(np.isfinite(im64)):
            return None
    else:
        im64 = mz64
    if not np.all(np.isfinite(mz64)):
        return None
    agg = cfast.hill_aggregates(mz64, I64, im64, seg_off, mz_step, paseftol, use_im)
    if agg is None:
        return None
    mz_med, im_med, mz_bin, im_bin = agg
    del im64

    return dict(seg_off=seg_off, scans=ss, orig=os_, new_id=new_id, I=I, I64=I64, mz=mz,
                mz_med=mz_med, im_med=im_med, mz_bin=mz_bin, im_bin=im_bin, use_im=use_im,
                n_hills_presplit=n_hills_presplit)


class _CSRLists(object):
    """Read-only sequence of per-hill lists sliced out of a flat array on access
    (avoids keeping millions of Python objects alive).  Each access returns a new
    list with the element types of the original per-hill lists."""

    def __init__(self, values, off, native):
        self.values = values
        self.off = off
        self.native = native

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        a, b = self.off[i], self.off[i + 1]
        v = self.values[a:b]
        return v.tolist() if self.native else list(v)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def hills_dict_from_csr(csr, full, write_hills=True):
    """hills_dict as cutils.process_hills returns it.

    full=False: only what calc_peptide_features / process_hills_extra /
    get_initial_isotopes_fast read (per-hill lists are materialised lazily).
    full=True: every key with real lists, sets and the m/z / IM bin dicts, for the
    original get_initial_isotopes (identical contents and in-bin order)."""
    seg_off = csr['seg_off']
    N = len(seg_off) - 1
    native_I = _NATIVE_FLOAT_LISTS and csr['I'].dtype == np.float64
    native_mz = _NATIVE_FLOAT_LISTS and csr['mz'].dtype == np.float64
    hd = {}
    if full:
        hd['hills_idx_array'] = csr['new_id']
        hd['scan_idx_array'] = csr['scans']
        hd['orig_idx_array'] = csr['orig']
    hd['hills_idx_array_unique'] = list(range(1, N + 1))
    hd['hills_mz_median'] = csr['mz_med'].astype(np.float64).tolist()
    if csr['use_im']:
        hd['hills_im_median'] = csr['im_med'].astype(np.float64).tolist()
    hd['hills_lengths'] = np.diff(seg_off).tolist()
    if not full:
        hd['hills_scan_lists'] = _CSRLists(csr['scans'], seg_off, True)
        hd['hills_intensity_array'] = _CSRLists(csr['I'], seg_off, native_I)
        if write_hills:
            hd['tmp_mz_array'] = _CSRLists(csr['mz'], seg_off, native_mz)
    else:
        offs = seg_off.tolist()
        lo, hi = offs[:-1], offs[1:]
        sl = csr['scans'].tolist()
        hd['hills_scan_lists'] = [sl[a:b] for a, b in zip(lo, hi)]
        del sl
        hd['hills_scan_sets'] = [set(x) for x in hd['hills_scan_lists']]
        il = csr['I'].tolist() if native_I else list(csr['I'])
        hd['hills_intensity_array'] = [il[a:b] for a, b in zip(lo, hi)]
        del il
        ml = csr['mz'].tolist() if native_mz else list(csr['mz'])
        hd['tmp_mz_array'] = [ml[a:b] for a, b in zip(lo, hi)]
        del ml
        fd = defaultdict(list)
        first = csr['scans'][seg_off[:-1]].tolist()
        last = csr['scans'][seg_off[1:] - 1].tolist()
        for i1, b, f, l in zip(range(N), csr['mz_bin'].tolist(), first, last):
            tv = (i1, f, l)
            fd[b - 1].append(tv)
            fd[b].append(tv)
            fd[b + 1].append(tv)
        hd['hills_mz_median_fast_dict'] = fd
        if csr['use_im']:
            imd = defaultdict(set)
            im_bin = csr['im_bin']
            for u in np.unique(np.concatenate([im_bin - 1, im_bin, im_bin + 1])).tolist():
                imd[u] = set(np.flatnonzero(np.abs(im_bin - u) <= 1).tolist())
            hd['hills_im_median_fast_dict'] = imd
    hd['hills_idict'] = [None] * N
    hd['hill_sqrt_of_i'] = [None] * N
    hd['hills_intensity_apex'] = [None] * N
    hd['hills_scan_apex'] = [None] * N
    return hd


# ---------------------------------------------------------------- feature detection

def sorted_hill_order(hills_dict):
    # the original: Python's stable sort of hill indices by hills_mz_median
    return np.argsort(np.asarray(hills_dict['hills_mz_median'], dtype=np.float64), kind='stable').tolist()


def carbon(theoretical_list, experimental_list, thresh):
    if FAST_OK:
        return cfast.carbon_lists(theoretical_list, experimental_list, thresh)
    from .cutils import checking_cos_correlation_for_carbon
    return checking_cos_correlation_for_carbon(theoretical_list, experimental_list, thresh)


def filter_by_mass_error(ready, isotopes_mass_error_map):
    """First filter loop of process_features_iteration.  The original deletes
    rejected items in place; building the list of survivors in order is identical
    and linear."""
    out = []
    append = out.append
    for pep_feature in ready:
        tmp = []
        for cand in pep_feature['isotopes']:
            map_val = isotopes_mass_error_map[cand['isotope_number']]
            if abs(cand['mass_diff_ppm'] - map_val[0]) <= 5 * map_val[1]:
                tmp.append(cand)
            else:
                break
        tmp_n_isotopes = len(tmp)
        if tmp_n_isotopes:
            all_theoretical_int, all_exp_intensity = pep_feature['intensity_array_for_cos_corr']
            all_theoretical_int = all_theoretical_int[:tmp_n_isotopes+1]
            all_exp_intensity = all_exp_intensity[:tmp_n_isotopes+1]
            cos_corr, number_of_passed_isotopes = carbon(all_theoretical_int, all_exp_intensity, 0.6)
            if cos_corr:
                pep_feature['cos_cor_isotopes'] = cos_corr
                pep_feature['isotopes'] = tmp
                pep_feature['nIsotopes'] = tmp_n_isotopes + 1
                pep_feature['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]
                append(pep_feature)
    return out


def select_features(ready):
    """Second (greedy) loop of process_features_iteration.  In the original, cur_l
    is 0 at the top of every iteration, so 'del ready[cur_l]' pops the front of the
    list (quadratic).  Here a head index replaces the deletions and the re-sort
    sorts the remaining slice; the same items are mutated in the same order.
    The input list is emptied (like the original's deletions) so that rejected
    items are freed while the loop runs."""
    func_for_sort = lambda x: -x['nIsotopes']-x['cos_cor_isotopes']
    ready_final = []
    ready_set = set()
    items = ready
    ready = sorted(items, key=func_for_sort)
    items.clear()
    cur_isotopes = ready[0]['nIsotopes']
    head = 0
    n = len(ready)
    while head < n:
        pep_feature = ready[head]
        n_iso = pep_feature['nIsotopes']
        if n_iso < cur_isotopes:
            ready = sorted(ready[head:], key=func_for_sort)
            n = len(ready)
            head = 0
            cur_isotopes = n_iso
            pep_feature = ready[0]
        if pep_feature['monoisotope hill idx'] not in ready_set:
            if not any(cand['isotope_hill_idx'] in ready_set for cand in pep_feature['isotopes']):
                ready_final.append(pep_feature)
                ready_set.add(pep_feature['monoisotope hill idx'])
                for cand in pep_feature['isotopes']:
                    ready_set.add(cand['isotope_hill_idx'])
                ready[head] = None
                head += 1
            else:
                tmp = []
                for cand in pep_feature['isotopes']:
                    if cand['isotope_hill_idx'] not in ready_set:
                        tmp.append(cand)
                    else:
                        break
                tmp_n_isotopes = len(tmp)
                if tmp_n_isotopes:
                    all_theoretical_int, all_exp_intensity = pep_feature['intensity_array_for_cos_corr']
                    all_theoretical_int = all_theoretical_int[:tmp_n_isotopes+1]
                    all_exp_intensity = all_exp_intensity[:tmp_n_isotopes+1]
                    cos_corr, number_of_passed_isotopes = carbon(all_theoretical_int, all_exp_intensity, 0.6)
                    if cos_corr:
                        pep_feature['cos_cor_isotopes'] = cos_corr
                        pep_feature['isotopes'] = tmp
                        pep_feature['nIsotopes'] = tmp_n_isotopes + 1
                        pep_feature['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]
                        # the modified item is examined again (head unchanged)
                    else:
                        ready[head] = None
                        head += 1
                else:
                    ready[head] = None
                    head += 1
        else:
            ready[head] = None
            head += 1
    return ready_final, ready_set
