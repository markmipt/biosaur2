cimport cython
import numpy as np
cimport numpy as np
import itertools
import math
from collections import Counter, defaultdict
from copy import copy

np.import_array()

cdef dict charge_ban_map

charge_ban_map = {
    8: (1, 2, 4, ),
    7: (1, ),
    6: (1, 2, 3, ),
    5: (1, ),
    4: (1, 2, ),
    3: (1, ),
    2: (1, ),
    1: (1, ),
}


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def get_and_calc_values_for_cos_corr(dict hills_dict, int idx_1):

    hill_idict_1 = hills_dict['hills_idict'][idx_1]
    if hill_idict_1 is None:
        hill_idict_1 = dict()
        for scan_id_val, intensity_val in zip(hills_dict['hills_scan_lists'][idx_1], hills_dict['hills_intensity_array'][idx_1]):
            hill_idict_1[scan_id_val] = intensity_val
        hills_dict['hills_idict'][idx_1] = hill_idict_1

    hill_sqrt_of_i_1 = hills_dict['hill_sqrt_of_i'][idx_1]
    if hill_sqrt_of_i_1 is None:
        hill_sqrt_of_i_1 = math.sqrt(sum(v**2 for v in hill_idict_1.values()))
        hills_dict['hill_sqrt_of_i'][idx_1] = hill_sqrt_of_i_1

    return hills_dict, hill_idict_1, hill_sqrt_of_i_1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def get_and_calc_apex_intensity_and_scan(dict hills_dict, int idx_1):

    hill_intensity_apex_1 = hills_dict['hills_intensity_apex'][idx_1]
    hill_scan_apex_1 = hills_dict['hills_scan_apex'][idx_1]
    if hill_intensity_apex_1 is None:

        hill_intensity_apex_1 = 0

        for int_val, scan_val in zip(hills_dict['hills_intensity_array'][idx_1], hills_dict['hills_scan_lists'][idx_1]):
            if int_val > hill_intensity_apex_1:
                hill_intensity_apex_1 = int_val
                hill_scan_apex_1 = scan_val

        hills_dict['hills_intensity_apex'][idx_1] = hill_intensity_apex_1
        hills_dict['hills_scan_apex'][idx_1] = hill_scan_apex_1

    return hills_dict, hill_intensity_apex_1, hill_scan_apex_1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def cos_correlation(set hill_scans_1, dict hill_idict_1, float hill_sqrt_of_i_1, set hill_scans_2, dict hill_idict_2, float hill_sqrt_of_i_2):

    cdef int i
    cdef float top

    top = 0
    for i in  hill_scans_1.intersection(hill_scans_2):
        top += hill_idict_1.get(i, 0) * hill_idict_2.get(i, 0)
    
    return top / (hill_sqrt_of_i_1 * hill_sqrt_of_i_2)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def get_initial_isotopes(dict hills_dict, float isotopes_mass_accuracy, list isotopes_list, dict a, int min_charge, int max_charge, float mz_step, float paseftol, float faims_val, list sorted_idx_child_process):

    cdef list ready, charges, hill_scans_1_list, hill_scans_2_list, candidates, tmp_candidates
    cdef int idx_1, hill_idx_1, idx_2, hill_idx_2, im_to_check_fast, hill_scans_1_list_first, hill_scans_1_list_last
    cdef int hill_scans_2_list_first, hill_scans_2_list_last, charge, isotope_number, m_to_check_fast
    cdef float im_mz_1, hill_mz_1, im_mz_2, hill_mz_2, m_to_check, mass_diff_abs, cos_cor_RT
    cdef float hill_sqrt_of_i_1, hill_sqrt_of_i_2
    cdef double mass_diff_ppm
    cdef dict banned_charges, hill_idict_1, hill_idict_2, local_isotopes_dict
    cdef set hill_scans_1, hill_scans_2

    ready = []

    charges = list(range(min_charge, max_charge + 1, 1)[::-1])

    for idx_1 in sorted_idx_child_process:
        hill_idx_1 = hills_dict['hills_idx_array_unique'][idx_1]
        hill_mz_1 = hills_dict['hills_mz_median'][idx_1]

    #for (idx_1, hill_idx_1), hill_mz_1 in sorted(list(zip(list(enumerate(hills_dict['hills_idx_array_unique'])), hills_dict['hills_mz_median'])), key=lambda x: x[-1]):

        if paseftol > 0:
            im_mz_1 = hills_dict['hills_im_median'][idx_1]
            im_to_check_fast = int(im_mz_1 / paseftol)

        banned_charges = dict()

        hill_scans_1_number = hills_dict['hills_lengths'][idx_1]
        hill_scans_1 = hills_dict['hills_scan_sets'][idx_1]
        hill_scans_1_list = hills_dict['hills_scan_lists'][idx_1]
        hill_scans_1_list_first, hill_scans_1_list_last = hill_scans_1_list[0], hill_scans_1_list[-1]

        mz_tol = isotopes_mass_accuracy * 1e-6 * hill_mz_1

        for charge in charges:
            
            candidates = []

            for isotope_number in isotopes_list[1:]:

                tmp_candidates = []

                m_to_check = hill_mz_1 + (1.00335 * isotope_number / charge)
                m_to_check_fast = int(m_to_check/mz_step)

                for idx_2, hill_scans_2_list_first, hill_scans_2_list_last in hills_dict['hills_mz_median_fast_dict'][m_to_check_fast]:

                    if paseftol == 0 or idx_2 in hills_dict['hills_im_median_fast_dict'][im_to_check_fast]:

                        if not hill_scans_1_list_last < hill_scans_2_list_first and not hill_scans_2_list_last < hill_scans_1_list_first:
                            hill_mz_2 = hills_dict['hills_mz_median'][idx_2]
                            mass_diff_abs = hill_mz_2 - m_to_check

                            if abs(mass_diff_abs) <= mz_tol:
                                hill_scans_2 = hills_dict['hills_scan_sets'][idx_2]
                                if len(hill_scans_1.intersection(hill_scans_2)) >= 1:


                                    hills_dict, hill_idict_1, hill_sqrt_of_i_1 = get_and_calc_values_for_cos_corr(hills_dict, idx_1)
                                    hills_dict, hill_idict_2, hill_sqrt_of_i_2 = get_and_calc_values_for_cos_corr(hills_dict, idx_2)

                                    cos_cor_RT = cos_correlation(hill_scans_1, hill_idict_1, hill_sqrt_of_i_1, hill_scans_2, hill_idict_2, hill_sqrt_of_i_2)

                                    if cos_cor_RT >= 0.6:

                                        hill_idx_2 = hills_dict['hills_idx_array_unique'][idx_2]

                                        hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, idx_1)
                                        hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, idx_2)

                                        mass_diff_ppm = mass_diff_abs*1e6/m_to_check

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

                tmp_intensity = a[int(100 * (neutral_mass // 100))]

                _, hill_intensity_apex_1, hill_scan_apex_1 = get_and_calc_apex_intensity_and_scan(hills_dict, idx_1)
                mono_hills_scan_lists =  hills_dict['hills_scan_lists'][idx_1]
                mono_hills_intensity_list =  hills_dict['hills_intensity_array'][idx_1]

                all_theoretical_int = [
                    hill_intensity_apex_1 *
                    tmp_intensity[z] /
                    tmp_intensity[0] for z in isotopes_list]

                for iter_candidates in itertools.product(*candidates):

                    all_exp_intensity = [hill_intensity_apex_1, ]

                    for local_isotopes_dict in iter_candidates:

                        idx_2 = local_isotopes_dict['isotope_idx']

                        _, hill_intensity_apex_2, hill_scan_apex_2 = get_and_calc_apex_intensity_and_scan(hills_dict, idx_2)
                        
                        all_exp_intensity.append(hill_intensity_apex_2)
                    cos_corr, number_of_passed_isotopes = checking_cos_correlation_for_carbon(all_theoretical_int, all_exp_intensity, 0.6)

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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def cos_correlation_new(list theoretical_list, list experimental_list):
    cdef float theor_total_sum, top, i1, i2, bottom, numb, averagineExplained
    cdef int suit_len


    theor_total_sum = sum(theoretical_list)
    suit_len = min(len(theoretical_list), len(experimental_list))
    theoretical_list = theoretical_list[:suit_len]
    experimental_list = experimental_list[:suit_len]

    top = 0

    for i1, i2 in zip(theoretical_list, experimental_list):
        top += i1 * i2

    if not top:
        return 0, 0
    else:

        bottom = math.sqrt(sum([numb * numb for numb in theoretical_list])) * \
            math.sqrt(sum([numb * numb for numb in experimental_list]))

        averagineExplained = sum(theoretical_list) / theor_total_sum

        return top / bottom, averagineExplained

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def checking_cos_correlation_for_carbon(list theoretical_list, list experimental_list, float thresh):

    cdef float averagineCorrelation, averagineExplained, best_cor
    cdef int best_pos, pos

    best_pos = 1
    best_cor = 0

    pos = len(experimental_list)

    while pos != 1:

        averagineCorrelation, averagineExplained = cos_correlation_new(theoretical_list, experimental_list[:pos])

        if averagineExplained >= 0.5 and averagineCorrelation >= thresh:
            if averagineCorrelation > best_cor:
                best_cor = averagineCorrelation
                best_pos = pos

            break

        pos -= 1

    return best_cor, best_pos


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def get_fast_dict(np.ndarray mz_sorted, float mz_step, list basic_id_sorted):

    cdef dict fast_dict
    cdef list fast_array
    cdef int idx, fm

    fast_dict = dict()
    fast_array = list((mz_sorted/mz_step).astype(int))
    for idx, fm in zip(basic_id_sorted, fast_array):
        if fm not in fast_dict:
            fast_dict[fm] = [idx, ]
        else:
            fast_dict[fm].append(idx)
    return fast_dict, fast_array


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def meanfilt(list data, int window_width):
    cdef np.ndarray cumsum_vec, ma_vec_array
    cdef list ma_vec
    cumsum_vec = np.cumsum(data, dtype=float)
    cumsum_vec[window_width:] = cumsum_vec[window_width:] - cumsum_vec[:-window_width]
    ma_vec = data[:1] + list(cumsum_vec / window_width) + data[-1:]
    return ma_vec


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def split_peaks(dict hills_dict, list data_for_analyse_tmp, dict args, dict counter_hills_idx, list sorted_idx_child_process, np.ndarray sorted_idx_array_child_process, int nproc, int checked_id):

    cdef float hillValleyFactor, min_val, mult_val
    cdef int min_length_hill, idx_start, idx_end, cur_new_idx, idx_1, hill_idx, hill_length, c_len, l_idx
    cdef np.ndarray idx_sort, tmp_scans, tmp_orig_idx, new_index_list
    cdef list tmp_intensity, smothed_intensity, min_idx_list, recheck_r_r 

    new_index_list = copy(sorted_idx_array_child_process)

    hillValleyFactor = args['hvf']
    min_length_hill = args['minlh']
    min_length_hill = max(2, min_length_hill)

    idx_start = 0
    idx_end = 0

    cur_new_idx = checked_id + max(sorted_idx_child_process) + 1

    for hill_idx in sorted_idx_child_process:

        hill_length = counter_hills_idx[hill_idx]
        idx_end = idx_start + hill_length


        if hill_length >= min_length_hill * 2:


            tmp_scans = hills_dict['scan_idx_array'][checked_id+idx_start:checked_id+idx_end]
            tmp_orig_idx = hills_dict['orig_idx_array'][checked_id+idx_start:checked_id+idx_end]
            tmp_intensity = [data_for_analyse_tmp[scan_val][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]

            smothed_intensity = meanfilt(tmp_intensity, 3)
            c_len = hill_length - min_length_hill
            idx = int(min_length_hill) - 1
            min_idx_list = []
            min_val = 0
            l_idx = 0
            recheck_r_r = []


            while idx <= c_len:

                if len(min_idx_list) and idx >= min_idx_list[-1] + min_length_hill:
                    l_idx = min_idx_list[-1]

                l_r = max(smothed_intensity[l_idx:idx]) / float(smothed_intensity[idx])
                if l_r >= hillValleyFactor:
                    r_r = max(smothed_intensity[idx+1:]) / float(smothed_intensity[idx])
                    if r_r >= hillValleyFactor:
                        mult_val = l_r * r_r
                        include_factor = (1 if l_r > r_r else 0)
                        if (min_length_hill <= idx + include_factor <= c_len):
                            if not len(min_idx_list) or idx + include_factor >= min_idx_list[-1] + min_length_hill:
                                min_idx_list.append(idx + include_factor)
                                recheck_r_r.append(idx)
                                min_val = mult_val
                            elif mult_val > min_val:
                                min_idx_list[-1] = idx + include_factor
                                recheck_r_r[-1] = idx
                                min_val = mult_val
                idx += 1
            if len(min_idx_list):
                for min_idx, end_idx, recheck_idx in zip(min_idx_list, min_idx_list[1:] + [idx_start+hill_length, ], recheck_r_r):
                    r_r = max(smothed_intensity[recheck_idx+1:end_idx]) / float(smothed_intensity[recheck_idx])
                    if r_r >= hillValleyFactor:
                        new_index_list[idx_start+min_idx:idx_start+hill_length] = cur_new_idx
                        cur_new_idx += 1

        idx_start = idx_end

    return new_index_list




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def split_peaks_old(dict hills_dict, list data_for_analyse_tmp, dict args):

    cdef float hillValleyFactor, min_val, mult_val
    cdef int min_length_hill, idx_start, idx_end, cur_new_idx, idx_1, hill_idx, hill_length, c_len, l_idx
    cdef np.ndarray tmp_hill_length, idx_minl, idx_sort, tmp_scans, tmp_orig_idx
    cdef list tmp_intensity, smothed_intensity, min_idx_list, recheck_r_r

    hillValleyFactor = args['hvf']
    min_length_hill = args['minlh']
    min_length_hill = max(2, min_length_hill)


    hills_dict['orig_idx_array'] = np.array(hills_dict['orig_idx_array'])
    hills_dict['scan_idx_array'] = np.array(hills_dict['scan_idx_array'])
    hills_dict['hills_idx_array'] = np.array(hills_dict['hills_idx_array'])

    counter_hills_idx = Counter(hills_dict['hills_idx_array'])

    tmp_hill_length = np.array([counter_hills_idx[hill_idx] for hill_idx in hills_dict['hills_idx_array']])
    idx_minl = tmp_hill_length >= min_length_hill
    hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_minl]
    hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_minl]
    hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_minl]

    if len(hills_dict['orig_idx_array']):

        idx_sort = np.argsort(hills_dict['hills_idx_array'] + (hills_dict['scan_idx_array'] / (hills_dict['scan_idx_array'].max()+1)))
        hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_sort]
        hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_sort]
        hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_sort]

        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))

        idx_start = 0
        idx_end = 0

        cur_new_idx = max(hills_dict['hills_idx_array_unique']) + 1

        for idx_1, hill_idx in enumerate(hills_dict['hills_idx_array_unique']):

            hill_length = counter_hills_idx[hill_idx]
            idx_end = idx_start + hill_length

            if hill_length >= min_length_hill * 2:


                tmp_scans = hills_dict['scan_idx_array'][idx_start:idx_end]
                tmp_orig_idx = hills_dict['orig_idx_array'][idx_start:idx_end]
                tmp_intensity = [data_for_analyse_tmp[scan_val]['intensity array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]

                smothed_intensity = meanfilt(tmp_intensity, 3)
                c_len = hill_length - min_length_hill
                idx = int(min_length_hill) - 1
                min_idx_list = []
                min_val = 0
                l_idx = 0
                recheck_r_r = []

                while idx <= c_len:

                    if len(min_idx_list) and idx >= min_idx_list[-1] + min_length_hill:
                        l_idx = min_idx_list[-1]

                    l_r = max(smothed_intensity[l_idx:idx]) / float(smothed_intensity[idx])
                    if l_r >= hillValleyFactor:
                        r_r = max(smothed_intensity[idx+1:]) / float(smothed_intensity[idx])
                        if r_r >= hillValleyFactor:
                            mult_val = l_r * r_r
                            include_factor = (1 if l_r > r_r else 0)
                            if (min_length_hill <= idx + include_factor <= c_len):
                                if not len(min_idx_list) or idx + include_factor >= min_idx_list[-1] + min_length_hill:
                                    min_idx_list.append(idx + include_factor)
                                    recheck_r_r.append(idx)
                                    min_val = mult_val
                                elif mult_val > min_val:
                                    min_idx_list[-1] = idx + include_factor
                                    recheck_r_r[-1] = idx
                                    min_val = mult_val
                    idx += 1
                if len(min_idx_list):
                    for min_idx, end_idx, recheck_idx in zip(min_idx_list, min_idx_list[1:] + [idx_start+hill_length, ], recheck_r_r):
                        r_r = max(smothed_intensity[recheck_idx+1:end_idx]) / float(smothed_intensity[recheck_idx])
                        if r_r >= hillValleyFactor:
                            hills_dict['hills_idx_array'][idx_start+min_idx:idx_start+hill_length] = cur_new_idx
                            cur_new_idx += 1

            idx_start = idx_end

        hills_dict['hills_idx_array'] = list(hills_dict['hills_idx_array'])
        del hills_dict['hills_idx_array_unique']

    return hills_dict



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def detect_hills(list data_for_analyse_tmp, dict args, float mz_step, float paseftol, bint dia=False):

    cdef dict hills_dict, prev_fast_dict, z
    cdef list total_num_hills, total_mass_diff, spec_mean_mass_accuracy, basic_id_sorted, all_idx, all_idx_im, all_prevs
    cdef set banned_prev_idx_set, all_idx_im_set
    cdef float hill_mass_accuracy, best_intensity, mz_cur, cur_intensity, cur_mass_diff_with_sign, cur_mass_diff, best_mass_diff
    cdef int last_idx, prev_idx, spec_idx, len_mz, idx, fm, fi, best_idx_prev, idx_prev
    cdef np.ndarray idx_for_sort, mz_sorted, im_sorted
    cdef bint flag1, flag2, flag3, flag1_im, flag2_im, flag3_im

    hills_dict = {}

    if dia is False:
        hill_mass_accuracy = args['htol']
    else:
        hill_mass_accuracy = args['diahtol']

    hills_dict['hills_idx_array'] = []
    hills_dict['orig_idx_array'] = []
    hills_dict['scan_idx_array'] = []
    hills_dict['mzs_array'] = []
    hills_dict['intensity_array'] = []
    if paseftol > 0:
        hills_dict['im_array'] = []
        prev_fast_dict_im = dict()

    last_idx = -1
    prev_idx = -1
    prev_fast_dict = dict()

    total_num_hills = []
    total_mass_diff = []

    for spec_idx, z in enumerate(data_for_analyse_tmp):

        len_mz = len(z['m/z array'])

        hills_dict['hills_idx_array'].extend(list(range(last_idx+1, last_idx+1+len_mz, 1)))
        hills_dict['orig_idx_array'].extend(range(len_mz))
        hills_dict['scan_idx_array'].extend([spec_idx] * len_mz)

        idx_for_sort = np.argsort(z['intensity array'])[::-1]

        mz_sorted = z['m/z array'][idx_for_sort]
        basic_id_sorted = list(np.array(range(len_mz))[idx_for_sort])

        hills_dict['mzs_array'].extend(z['m/z array'])
        hills_dict['intensity_array'].extend(z['intensity array'])

        if paseftol > 0:
            im_sorted = z['mean inverse reduced ion mobility array'][idx_for_sort]
            hills_dict['im_array'].extend(z['mean inverse reduced ion mobility array'])

            fast_dict_im, fast_array_im = get_fast_dict(im_sorted, paseftol, basic_id_sorted)

        fast_dict, fast_array = get_fast_dict(mz_sorted, mz_step, basic_id_sorted)

        banned_prev_idx_set = set()

        for idx, fm, fi in zip(basic_id_sorted, fast_array, (fast_array if paseftol == 0 else fast_array_im)):

            flag1 = fm in prev_fast_dict
            flag2 = fm-1 in prev_fast_dict
            flag3 = fm+1 in prev_fast_dict

            if flag1 or flag2 or flag3:

                if flag1:
                    all_idx = prev_fast_dict[fm]
                    if flag2:
                        all_idx += prev_fast_dict[fm-1]
                    if flag3:
                        all_idx += prev_fast_dict[fm+1]
                elif flag2:
                    all_idx = prev_fast_dict[fm-1]
                    if flag3:
                        all_idx += prev_fast_dict[fm+1]
                elif flag3:
                    all_idx = prev_fast_dict[fm+1]

                if paseftol > 0:
                    flag1_im = fi in prev_fast_dict_im
                    flag2_im = fi-1 in prev_fast_dict_im
                    flag3_im = fi+1 in prev_fast_dict_im

                best_intensity = 0
                best_idx_prev = 0
                mz_cur = z['m/z array'][idx]


                #all_prevs = [[idx_prev, data_for_analyse_tmp[spec_idx-1]['intensity array'][idx_prev]] for idx_prev in all_idx if (idx_prev not in banned_prev_idx_set and (paseftol == 0 or idx_prev in all_idx_im_set))]
                all_prevs = [[idx_prev, data_for_analyse_tmp[spec_idx-1]['intensity array'][idx_prev]] for idx_prev in all_idx if (idx_prev not in banned_prev_idx_set and (paseftol == 0 or (flag2_im and idx_prev in prev_fast_dict_im[fi-1]) or (flag1_im and idx_prev in prev_fast_dict_im[fi]) or (flag3_im and idx_prev in prev_fast_dict_im[fi+1])))]
                for idx_prev, cur_intensity in all_prevs:
                    cur_mass_diff_with_sign = (mz_cur - data_for_analyse_tmp[spec_idx-1]['m/z array'][idx_prev]) / mz_cur * 1e6
                    cur_mass_diff = abs(cur_mass_diff_with_sign)
                    if cur_mass_diff <= hill_mass_accuracy and cur_intensity >= best_intensity:
                        best_mass_diff = cur_mass_diff
                        best_intensity = cur_intensity
                        best_idx_prev = idx_prev
                        hills_dict['hills_idx_array'][last_idx+1+idx] = hills_dict['hills_idx_array'][prev_idx+1+idx_prev]
                        
                if best_idx_prev != 0:
                    banned_prev_idx_set.add(best_idx_prev)


        prev_fast_dict = fast_dict
        if paseftol > 0:
            prev_fast_dict_im = fast_dict_im
        prev_idx = last_idx
        last_idx = last_idx+len_mz

    return hills_dict


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def process_hills(dict hills_dict, list data_for_analyse_tmp, float mz_step, float paseftol, dict args, bint dia=False):

    cdef dict 
    cdef float mz_median, i_sum_tmp, mz_val_tmp, i_val_tmp, im_median
    cdef list tmp_intensity, tmp_mz_array, tmp_im_array, tmp_scans_list, tmp_scans, tmp_orig_idx
    cdef set tmp_scans_set
    cdef tuple tmp_val
    cdef int min_length_hill, idx_start, idx_end, idx_1, hill_idx, hill_length, mz_median_int, im_median_int
    cdef np.ndarray tmp_hill_length, idx_minl


    counter_hills_idx = Counter(hills_dict['hills_idx_array'])
    if dia is False:
        min_length_hill = args['minlh']
    else:
        min_length_hill = args['diaminlh']

    hills_dict['orig_idx_array'] = np.array(hills_dict['orig_idx_array'])
    hills_dict['scan_idx_array'] = np.array(hills_dict['scan_idx_array'])
    hills_dict['hills_idx_array'] = np.array(hills_dict['hills_idx_array'])


    tmp_hill_length = np.array([counter_hills_idx[hill_idx] for hill_idx in hills_dict['hills_idx_array']])
    idx_minl = tmp_hill_length >= min_length_hill
    hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_minl]
    hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_minl]
    hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_minl]

    if len(hills_dict['hills_idx_array']):

        idx_sort = np.argsort(hills_dict['hills_idx_array'] + (hills_dict['scan_idx_array'] / (hills_dict['scan_idx_array'].max()+1)))
        hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_sort]
        hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_sort]
        hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_sort]

        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))
        hills_dict['hills_mz_median'] = []
        hills_dict['hills_mz_median_fast_dict'] = defaultdict(list)
        if paseftol > 0:
            hills_dict['hills_im_median'] = []
            hills_dict['hills_im_median_fast_dict'] = defaultdict(set)

        hills_dict['hills_intensity_array'] = []
        hills_dict['hills_scan_sets'] = []
        hills_dict['hills_scan_lists'] = []
        hills_dict['hills_lengths'] = []
        hills_dict['tmp_mz_array'] = []

        hills_dict['scan_idx_array'] = list(hills_dict['scan_idx_array'])
        hills_dict['orig_idx_array'] = list(hills_dict['orig_idx_array'])
        hills_dict['hills_idx_array'] = list(hills_dict['hills_idx_array'])

        idx_start = 0
        idx_end = 0


        for idx_1, hill_idx in enumerate(hills_dict['hills_idx_array_unique']):

            hill_length = counter_hills_idx[hill_idx]
            idx_end = idx_start + hill_length

            tmp_scans = hills_dict['scan_idx_array'][idx_start:idx_end]
            tmp_orig_idx = hills_dict['orig_idx_array'][idx_start:idx_end]

            tmp_intensity = [data_for_analyse_tmp[scan_val]['intensity array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
            tmp_mz_array = [data_for_analyse_tmp[scan_val]['m/z array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
            mz_median = 0
            i_sum_tmp = 0
            for mz_val_tmp, i_val_tmp in zip(tmp_mz_array, tmp_intensity):
                mz_median += mz_val_tmp * i_val_tmp
                i_sum_tmp += i_val_tmp
            mz_median = mz_median / i_sum_tmp
            if paseftol > 0:
                tmp_im_array = [data_for_analyse_tmp[scan_val]['mean inverse reduced ion mobility array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
                im_median = np.average(tmp_im_array, weights=tmp_intensity)
            tmp_scans_list = tmp_scans
            tmp_scans_set = set(tmp_scans)

            idx_start = idx_end

            hills_dict['hills_mz_median'].append(mz_median)

            mz_median_int = int(mz_median/mz_step)
            tmp_val = (idx_1, tmp_scans_list[0], tmp_scans_list[-1])
            hills_dict['hills_mz_median_fast_dict'][mz_median_int-1].append(tmp_val)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int].append(tmp_val)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int+1].append(tmp_val)

            if paseftol > 0:
                hills_dict['hills_im_median'].append(im_median)

                im_median_int = int(im_median/paseftol)
                hills_dict['hills_im_median_fast_dict'][im_median_int-1].add(idx_1)
                hills_dict['hills_im_median_fast_dict'][im_median_int].add(idx_1)
                hills_dict['hills_im_median_fast_dict'][im_median_int+1].add(idx_1)


            hills_dict['hills_intensity_array'].append(tmp_intensity)
            hills_dict['hills_scan_sets'].append(tmp_scans_set)
            hills_dict['hills_scan_lists'].append(tmp_scans_list)
            hills_dict['hills_lengths'].append(hill_length)
            hills_dict['tmp_mz_array'].append(tmp_mz_array)


        hills_dict['hills_idict'] = [None] * len(hills_dict['hills_idx_array_unique'])
        hills_dict['hill_sqrt_of_i'] = [None] * len(hills_dict['hills_idx_array_unique'])
        hills_dict['hills_intensity_apex'] = [None] * len(hills_dict['hills_idx_array_unique'])
        hills_dict['hills_scan_apex'] = [None] * len(hills_dict['hills_idx_array_unique'])

    return hills_dict