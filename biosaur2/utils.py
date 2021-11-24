from pyteomics import mzml
import numpy as np
from collections import defaultdict, Counter
from os import path
import math
from scipy.optimize import curve_fit

def noisygaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

def calibrate_mass(bwidth, mass_left, mass_right, true_md):

    bbins = np.arange(-mass_left, mass_right, bwidth)
    H1, b1 = np.histogram(true_md, bins=bbins)
    b1 = b1 + bwidth
    b1 = b1[:-1]

    popt, pcov = curve_fit(noisygaus, b1, H1, p0=[1, np.median(true_md), 1, 1])
    mass_shift, mass_sigma = popt[1], abs(popt[2])
    return mass_shift, mass_sigma, pcov[0][0]

def meanfilt(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] -
              cumsum_vec[:-window_width]) / window_width
    ma_vec = data[:1] + list(ma_vec) + data[-1:]
    return ma_vec

def split_peaks(hills_dict, data_for_analyse_tmp, args):

    hillValleyFactor = args['hvf']
    min_length_hill = args['minlh']


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
        min_length_hill = args['minlh']
        min_length_hill = max(2, min_length_hill)

        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))

        idx_start = 0
        idx_end = 0

        cur_new_idx = max(hills_dict['hills_idx_array_unique']) + 1

        for idx_1, hill_idx in enumerate(hills_dict['hills_idx_array_unique']):

            hill_length = counter_hills_idx[hill_idx]
            idx_end = idx_start + hill_length

            if hill_length >= min_length_hill * 2:# + 1:


                tmp_scans = hills_dict['scan_idx_array'][idx_start:idx_end]
                tmp_orig_idx = hills_dict['orig_idx_array'][idx_start:idx_end]
                tmp_intensity = [data_for_analyse_tmp[scan_val]['intensity array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
                tmp_mz_array = [data_for_analyse_tmp[scan_val]['m/z array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]

                smothed_intensity = meanfilt(tmp_intensity, 2)
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

        print(len(set(hills_dict['hills_idx_array'])), len(hills_dict['hills_idx_array_unique']), '!!!')

        hills_dict['hills_idx_array'] = list(hills_dict['hills_idx_array'])
        del hills_dict['hills_idx_array_unique']

    return hills_dict

def get_and_calc_values_for_cos_corr(hills_dict, idx_1, hill_length_1_morethan1):

    hill_idict_1 = hills_dict['hills_idict'][idx_1]
    if hill_idict_1 is None:
        hill_idict_1 = dict()
        if hill_length_1_morethan1:
            for scan_id_val, intensity_val in zip(hills_dict['hills_scan_lists'][idx_1], hills_dict['hills_intensity_array'][idx_1]):
                hill_idict_1[scan_id_val] = intensity_val
        else:
            hill_idict_1[hills_dict['hills_scan_lists'][idx_1]] = hills_dict['hills_intensity_array'][idx_1]
        hills_dict['hills_idict'][idx_1] = hill_idict_1

    hill_sqrt_of_i_1 = hills_dict['hill_sqrt_of_i'][idx_1]
    if hill_sqrt_of_i_1 is None:
        hill_sqrt_of_i_1 = math.sqrt(sum(v**2 for v in hill_idict_1.values()))
        hills_dict['hill_sqrt_of_i'][idx_1] = hill_sqrt_of_i_1

    return hills_dict, hill_idict_1, hill_sqrt_of_i_1

def get_and_calc_apex_intensity_and_scan(hills_dict, hill_length_1_morethan1, idx_1):

    hill_intensity_apex_1 = hills_dict['hills_intensity_apex'][idx_1]
    hill_scan_apex_1 = hills_dict['hills_scan_apex'][idx_1]
    if hill_intensity_apex_1 is None:

        if hill_length_1_morethan1:

            hill_intensity_apex_1 = 0

            for int_val, scan_val in zip(hills_dict['hills_intensity_array'][idx_1], hills_dict['hills_scan_lists'][idx_1]):
                if int_val > hill_intensity_apex_1:
                    hill_intensity_apex_1 = int_val
                    hill_scan_apex_1 = scan_val

        else:
            hill_intensity_apex_1 = hills_dict['hills_intensity_array'][idx_1]
            hill_scan_apex_1 = hills_dict['hills_scan_lists'][idx_1]

        hills_dict['hills_intensity_apex'][idx_1] = hill_intensity_apex_1
        hills_dict['hills_scan_apex'][idx_1] = hill_scan_apex_1

    return hills_dict, hill_intensity_apex_1, hill_scan_apex_1


def cos_correlation(hill_length_1, hill_scans_1, hill_idict_1, hill_sqrt_of_i_1, hill_length_2, hill_scans_2, hill_idict_2, hill_sqrt_of_i_2):

    inter_set = hill_scans_1.intersection(hill_scans_2)

    top = 0
    for i in inter_set:
        h1_val = hill_idict_1.get(i, 0)
        h2_val = hill_idict_2.get(i, 0)
        top += h1_val * h2_val
    
    
    bottom = hill_sqrt_of_i_1 * hill_sqrt_of_i_2
    return top / bottom


def cos_correlation_new(theoretical_list, experimental_list, shf):

    theor_total_sum = sum(theoretical_list)
    theoretical_list = theoretical_list[shf:]
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

def checking_cos_correlation_for_carbon(
        theoretical_list, experimental_list, thresh, allowed_shift=False):

    best_value = 0
    best_shift = 0
    best_pos = 1
    best_cor = 0

    exp_list_len = len(experimental_list)

    # for shf in [0, 1]:
    for shf in [0, ]:

        if allowed_shift is False or shf == allowed_shift:

            pos = int(exp_list_len)

            while pos != 1:

                averagineCorrelation, averagineExplained = cos_correlation_new(
                    theoretical_list, experimental_list[:pos], shf)

                if averagineExplained >= 0.5 and averagineCorrelation >= thresh:
                    tmp_val = averagineCorrelation
                    if tmp_val > best_value:
                        best_value = tmp_val
                        best_cor = averagineCorrelation
                        best_shift = shf
                        best_pos = pos

                    break

                pos -= 1

            if best_value:
                break

    return best_cor, best_pos, best_shift

def calc_peptide_features(hills_dict, peptide_features, negative_mode, faims_val, RT_dict, data_start_id):

    for pep_feature in peptide_features:

        pep_feature['mz'] = pep_feature['hill_mz_1']

        pep_feature['massCalib'] = pep_feature['mz'] * pep_feature['charge'] - 1.0072765 * pep_feature['charge'] * (-1 if negative_mode else 1) - pep_feature['shift'] * 1.00335

        pep_feature['rtApex'] = RT_dict[hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]+data_start_id]
        pep_feature['intensityApex'] = hills_dict['hills_intensity_apex'][pep_feature['monoisotope idx']]
        pep_feature['rtStart'] = 1
        pep_feature['rtEnd'] = 1

    return peptide_features


def write_output(peptide_features, args, write_header=True):

    input_mzml_path = args['file']

    if args['o']:
        output_file = args['o']
    else:
        output_file = path.splitext(input_mzml_path)[0]\
            + path.extsep + 'features.tsv'

    columns_for_output = [
        'massCalib',
        'rtApex',
        'intensityApex',
        'charge',
        'nIsotopes',
        'nScans',
        'mz',
        'shift',
        'rtStart',
        'rtEnd',
        'FAIMS',
        'im',
        'mono_hills_scan_lists',
        'mono_hills_intensity_list',
    ]

    if write_header:

        out_file = open(output_file, 'w')
        out_file.write('\t'.join(columns_for_output) + '\n')
        out_file.close()

    out_file = open(output_file, 'a')
    for pep_feature in peptide_features:
        out_file.write('\t'.join([str(pep_feature[col]) for col in columns_for_output]) + '\n')

    out_file.close()

def process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args, dia=False):

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

        # hills_dict['hills_idx_array_unique'] = sorted([hill_idx for hill_idx in set(hills_dict['hills_idx_array']) if counter_hills_idx[hill_idx] >= min_length_hill])
        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))
        hills_dict['hills_mz_median'] = []
        hills_dict['hills_mz_median_fast_dict'] = defaultdict(set)
        if paseftol is not False:
            hills_dict['hills_im_median'] = []
            hills_dict['hills_im_median_fast_dict'] = defaultdict(set)
            
        hills_dict['hills_intensity_array'] = []
        hills_dict['hills_scan_sets'] = []
        hills_dict['hills_scan_lists'] = []
        hills_dict['hills_lengths'] = []
        hills_dict['tmp_mz_array'] = []

        idx_start = 0
        idx_end = 0

        for idx_1, hill_idx in enumerate(hills_dict['hills_idx_array_unique']):

            hill_length = counter_hills_idx[hill_idx]
            idx_end = idx_start + hill_length

            if hill_length > 1:

                # idx_end = idx_start + hill_length

                # tmp_idx = hills_dict['hills_idx_array'] == hill_idx
                # tmp_idx2 = hills_dict['hills_idx_array'] == hill_idx

                # tmp_idx = idx_start

                tmp_scans = hills_dict['scan_idx_array'][idx_start:idx_end]
                tmp_orig_idx = hills_dict['orig_idx_array'][idx_start:idx_end]

                tmp_intensity = [data_for_analyse_tmp[scan_val]['intensity array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
                # mz_median = np.median([data_for_analyse_tmp[scan_val]['m/z array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)])
                tmp_mz_array = [data_for_analyse_tmp[scan_val]['m/z array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
                mz_median = np.average(tmp_mz_array, weights=tmp_intensity)
                # mz_median = np.median(tmp_mz_array)
                if paseftol is not False:
                    tmp_im_array = [data_for_analyse_tmp[scan_val]['mean inverse reduced ion mobility array'][orig_idx_val] for orig_idx_val, scan_val in zip(tmp_orig_idx, tmp_scans)]
                    im_median = np.average(tmp_im_array, weights=tmp_intensity)
                tmp_scans_list = tmp_scans
                tmp_scans_set = set(tmp_scans)

            else:
                tmp_mz_array = [hills_dict['mzs_array'][hill_idx], ]
                mz_median = hills_dict['mzs_array'][hill_idx]
                if paseftol is not False:
                    tmp_im_array = [hills_dict['im_array'][hill_idx], ]
                    im_median = hills_dict['im_array'][hill_idx]
                tmp_intensity = hills_dict['intensity_array'][hill_idx]
                # tmp_scans = set([hills_dict['scan_idx_array'][hill_idx], ])
                tmp_scans_set = hills_dict['scan_idx_array'][hill_idx]
                tmp_scans_list = tmp_scans_set

            idx_start = idx_end

            hills_dict['hills_mz_median'].append(mz_median)

            mz_median_int = int(mz_median/mz_step)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int-1].add(idx_1)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int].add(idx_1)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int+1].add(idx_1)

            if paseftol is not False:
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

def centroid_pasef_data(data_for_analyse_tmp, args, mz_step):

    cnt_ms1_scans = len(data_for_analyse_tmp)
    for spec_idx, z in enumerate(data_for_analyse_tmp):

        print('PASEF scans analysis: %d/%d' % (spec_idx+1, cnt_ms1_scans))
        print('number of m/z peaks in scan: %d' % (len(z['m/z array'])))

        if 'ignore_ion_mobility' not in z:

            mz_ar_new = []
            intensity_ar_new = []
            ion_mobility_ar_new = []

            mz_ar = z['m/z array']
            intensity_ar = z['intensity array']
            ion_mobility_ar = z['mean inverse reduced ion mobility array']

            ion_mobility_accuracy = args['paseftol']
            ion_mobility_step = max(ion_mobility_ar) * ion_mobility_accuracy

            ion_mobility_ar_fast = (ion_mobility_ar/ion_mobility_step).astype(int)
            mz_ar_fast = (mz_ar/mz_step).astype(int)

            max_mz_int = max(mz_ar_fast) * 2

            # print('HERE1')

            idx = np.argsort(mz_ar_fast)
            mz_ar_fast = mz_ar_fast[idx]
            ion_mobility_ar_fast = ion_mobility_ar_fast[idx]

            mz_ar = mz_ar[idx]
            intensity_ar = intensity_ar[idx]
            ion_mobility_ar = ion_mobility_ar[idx]

            idx_ar = list(range(len(mz_ar)))

            # print('HERE2')

            # for peak_idx in idx_ar:

            max_peak_idx = len(mz_ar)
            
            peak_idx = 0
            while peak_idx < max_peak_idx:

                mz_val_int = mz_ar_fast[peak_idx]
                ion_mob_val_int = ion_mobility_ar_fast[peak_idx]

                tmp = [peak_idx, ]

                peak_idx_2 = peak_idx + 1

                while peak_idx_2 < max_peak_idx:
                    mz_val_int_2 = mz_ar_fast[peak_idx_2]
                    if mz_val_int_2 - mz_val_int > 1:
                        break
                    else:
                        ion_mob_val_int_2 = ion_mobility_ar_fast[peak_idx_2]
                        if abs(ion_mob_val_int - ion_mob_val_int_2) <= 1:
                            tmp.append(peak_idx_2)
                            peak_idx = peak_idx_2
                    peak_idx_2 += 1

                # if ion_mob_val_int != -1:

                #     mask1 = ion_mobility_ar_fast == ion_mob_val_int-1
                #     mask2 = ion_mobility_ar_fast == ion_mob_val_int
                #     mask3 = ion_mobility_ar_fast == ion_mob_val_int+1

                #     idx1 = mask1 + mask2 + mask3

                #     mask1 = mz_ar_fast == mz_val_int-1
                #     mask2 = mz_ar_fast == mz_val_int
                #     mask3 = mz_ar_fast == mz_val_int+1

                #     idx2 = mask1 + mask2 + mask3

                #     idx3 = idx1 * idx2

                #     all_intensity = intensity_ar[idx3]
                all_intensity = [intensity_ar[p_id] for p_id in tmp]
                i_val_new = sum(all_intensity)

                if i_val_new >= args['pasefmini']:

                    all_mz = [mz_ar[p_id] for p_id in tmp]
                    all_ion_mob = [ion_mobility_ar[p_id] for p_id in tmp]

                    mz_val_new = np.average(all_mz, weights=all_intensity)
                    ion_mob_new = np.average(all_ion_mob, weights=all_intensity)

                    intensity_ar_new.append(i_val_new)
                    mz_ar_new.append(mz_val_new)
                    ion_mobility_ar_new.append(ion_mob_new)

                peak_idx += 1
            
            data_for_analyse_tmp[spec_idx]['m/z array'] = np.array(mz_ar_new)
            data_for_analyse_tmp[spec_idx]['intensity array'] = np.array(intensity_ar_new)
            data_for_analyse_tmp[spec_idx]['mean inverse reduced ion mobility array'] = np.array(ion_mobility_ar_new)

        print('number of m/z peaks in scan after centroiding: %d' % (len(data_for_analyse_tmp[spec_idx]['m/z array'])))
        print('\n')

    data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'] > 0)]
    print('Number of MS1 scans after combining ion mobility peaks: ', len(data_for_analyse_tmp))

            # fast_dict = defaultdict(set)
            # for peak_idx, (mz_val_int, ion_mob_val_int) in enumerate(zip(mz_ar_fast, ion_mobility_ar_fast)):

            #     fast_dict[(mz_val_int-1, ion_mob_val_int)].add(peak_idx)
            #     fast_dict[(mz_val_int, ion_mob_val_int)].add(peak_idx)
            #     fast_dict[(mz_val_int+1, ion_mob_val_int)].add(peak_idx)

            #     fast_dict[(mz_val_int-1, ion_mob_val_int-1)].add(peak_idx)
            #     fast_dict[(mz_val_int, ion_mob_val_int-1)].add(peak_idx)
            #     fast_dict[(mz_val_int+1, ion_mob_val_int-1)].add(peak_idx)

            #     fast_dict[(mz_val_int-1, ion_mob_val_int+1)].add(peak_idx)
            #     fast_dict[(mz_val_int, ion_mob_val_int+1)].add(peak_idx)
            #     fast_dict[(mz_val_int+1, ion_mob_val_int+1)].add(peak_idx)


    #         print('HERE2')

    #         hill_length = []
    #         peak_idx_array = []
    #         for peak_idx, (mz_val_int, ion_mob_val_int) in enumerate(zip(mz_ar_fast, ion_mobility_ar_fast)):
    #             hill_length.append(len(fast_dict[(mz_val_int, ion_mob_val_int)]))
    #             peak_idx_array.append(peak_idx)
    #         peak_idx_array = np.array(peak_idx_array)
        

    #         print('HERE3')

    #         added_idx = set()
    #         idx_sort = np.argsort(hill_length)[::-1]
    #         for peak_idx in peak_idx_array[idx_sort]:
    #             if peak_idx not in added_idx:
    #                 mz_val_int = mz_ar_fast[peak_idx]
    #                 ion_mob_val_int = ion_mobility_ar_fast[peak_idx]
    #                 all_idx = set([p_id for p_id in fast_dict[(mz_val_int, ion_mob_val_int)] if p_id not in added_idx])
    #                 if len(all_idx):
    #                     added_idx.update(all_idx)
            
    #                     all_intensity = [intensity_ar[p_id] for p_id in all_idx]
    #                     i_val_new = sum(all_intensity)

    #                     if i_val_new >= args['pasefmini']:

    #                         all_mz = [mz_ar[p_id] for p_id in all_idx]
    #                         all_ion_mob = [ion_mobility_ar[p_id] for p_id in all_idx]

    #                         mz_val_new = np.average(all_mz, weights=all_intensity)
    #                         ion_mob_new = np.average(all_ion_mob, weights=all_intensity)

    #                         intensity_ar_new.append(i_val_new)
    #                         mz_ar_new.append(mz_val_new)
    #                         ion_mobility_ar_new.append(ion_mob_new)
            
    #         data_for_analyse_tmp[spec_idx]['m/z array'] = np.array(mz_ar_new)
    #         data_for_analyse_tmp[spec_idx]['intensity array'] = np.array(intensity_ar_new)
    #         data_for_analyse_tmp[spec_idx]['mean inverse reduced ion mobility array'] = np.array(ion_mobility_ar_new)

    # data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'] > 0)]
    # print('Number of MS1 scans after combining ion mobility peaks: ', len(data_for_analyse_tmp))
        
    return data_for_analyse_tmp

def detect_hills(data_for_analyse_tmp, args, mz_step, paseftol, dia=False):

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
    if paseftol is not False:
        hills_dict['im_array'] = []
        prev_fast_dict_im = dict()

    # hills_idx_array = []
    # orig_idx_array = []
    # scan_idx_array = []
    # mzs_array = []
    # intensity_array = []

    last_idx = -1
    prev_idx = -1
    prev_fast_dict = dict()
    # prev_median_error = hill_mass_accuracy / 5
    # mass_shift = 0

    for spec_idx, z in enumerate(data_for_analyse_tmp):

        # active_masses_array = []

        len_mz = len(z['m/z array'])

        hills_dict['hills_idx_array'].extend(list(range(last_idx+1, last_idx+1+len_mz, 1)))
        hills_dict['orig_idx_array'].extend(range(len_mz))
        hills_dict['scan_idx_array'].extend([spec_idx] * len_mz)

        idx_for_sort = np.argsort(z['intensity array'])[::-1]

        mz_sorted = z['m/z array'][idx_for_sort]
        basic_id_sorted = np.array(range(len_mz))[idx_for_sort]

        hills_dict['mzs_array'].extend(z['m/z array'])
        hills_dict['intensity_array'].extend(z['intensity array'])

        if paseftol is not False:
            im_sorted = z['mean inverse reduced ion mobility array'][idx_for_sort]
            hills_dict['im_array'].extend(z['mean inverse reduced ion mobility array'])

            fast_dict_im = defaultdict(set)
            fast_array_im = (im_sorted/paseftol).astype(int)
            for idx, fm in zip(basic_id_sorted, fast_array_im):
                fast_dict_im[fm-1].add(idx)
                fast_dict_im[fm+1].add(idx)
                fast_dict_im[fm].add(idx)

        
        fast_dict = defaultdict(set)
        fast_array = (mz_sorted/mz_step).astype(int)
        for idx, fm in zip(basic_id_sorted, fast_array):
            fast_dict[fm-1].add(idx)
            fast_dict[fm+1].add(idx)
            fast_dict[fm].add(idx)

        banned_prev_idx_set = set()

        for idx, fm, fi in zip(basic_id_sorted, fast_array, (fast_array if paseftol is False else fast_array_im)):
            if fm in prev_fast_dict:

                # best_active_mass_diff = 1e6
                # best_active_flag = False

                best_mass_diff = 1e6
                best_intensity = 0
                best_idx_prev = False
                mz_cur = z['m/z array'][idx]
                for idx_prev in prev_fast_dict[fm]:
                    if idx_prev not in banned_prev_idx_set and (paseftol is False or idx_prev in prev_fast_dict_im[fi]):
                        cur_mass_diff_with_sign = (mz_cur - data_for_analyse_tmp[spec_idx-1]['m/z array'][idx_prev]) / mz_cur * 1e6
                        cur_mass_diff = abs(cur_mass_diff_with_sign)
                        cur_intensity = data_for_analyse_tmp[spec_idx-1]['intensity array'][idx_prev]
                        # if cur_mass_diff <= hill_mass_accuracy and cur_mass_diff <= best_mass_diff:
                        if cur_mass_diff <= hill_mass_accuracy and cur_intensity >= best_intensity:
                        # if cur_mass_diff <= (prev_median_error * 5) and cur_intensity >= best_intensity:
                        # if abs(cur_mass_diff_with_sign - mass_shift) <= (prev_median_error * 5) and cur_intensity >= best_intensity:
                            # best_active_flag = True
                            best_mass_diff = cur_mass_diff
                            # best_active_mass_diff = cur_mass_diff_with_sign
                            best_intensity = cur_intensity
                            best_idx_prev = idx_prev
                            hills_dict['hills_idx_array'][last_idx+1+idx] = hills_dict['hills_idx_array'][prev_idx+1+idx_prev]
                if best_idx_prev is not False:
                    banned_prev_idx_set.add(best_idx_prev)
                # if best_active_flag:
                #     active_masses_array.append(best_active_mass_diff)


                    # cur_mass_diff = (mz_cur - data_for_analyse_tmp[spec_idx-1]['m/z array'][idx_prev]) / mz_cur * 1e6
                    # # cur_mass_diff_norm = abs(cur_mass_diff) * (1 if cur_mass_diff < 0 else 2)
                    # if -hill_mass_accuracy - 2.5 <= cur_mass_diff <= hill_mass_accuracy and abs(cur_mass_diff) <= best_mass_diff:
                    #     best_mass_diff = abs(cur_mass_diff)
                    #     # hills_dict['hills_idx_array'][last_idx+1+idx] = prev_idx + 1 + idx_prev
                    #     hills_dict['hills_idx_array'][last_idx+1+idx] = hills_dict['hills_idx_array'][prev_idx+1+idx_prev]



        prev_fast_dict = fast_dict
        if paseftol is not False:
            prev_fast_dict_im = fast_dict_im
        prev_idx = last_idx
        last_idx = last_idx+len_mz

        # if len(active_masses_array) > 100:
        #     # prev_median_error = np.median(active_masses_array)

        #     true_md = np.array(active_masses_array)

        #     mass_left = -min(true_md)
        #     mass_right = max(true_md)


        #     try:
        #         mass_shift, mass_sigma, covvalue = calibrate_mass(0.05, mass_left, mass_right, true_md)
        #     except:
        #         mass_shift = 0
        #         mass_sigma = np.median(active_masses_array)
        #     #     mass_shift, mass_sigma, covvalue = calibrate_mass(0.25, mass_left, mass_right, true_md)
        #     # if np.isinf(covvalue):
        #     #     mass_shift = 0
        #     #     mass_sigma = np.median(active_masses_array)
        #         # mass_shift, mass_sigma, covvalue = calibrate_mass(0.05, mass_left, mass_right, true_md)

        #     print(mass_shift, mass_sigma, ', median ppm', len(active_masses_array))
        #     prev_median_error = mass_sigma
        #     # print(prev_median_error, ', median ppm')
        #     # break

    print(last_idx)
    print(len(hills_dict['hills_idx_array']))
    print(len(set(hills_dict['hills_idx_array'])))

    return hills_dict#hills_idx_array, orig_idx_array, scan_idx_array, mzs_array, intensity_array


def process_mzml(args):

    input_mzml_path = args['file']
    min_intensity = args['mini']
    min_mz = args['minmz']
    max_mz = args['maxmz']

    skipped = 0
    data_for_analyse = []

    cnt = 0

    for z in mzml.read(input_mzml_path):
        if z['ms level'] == 1:

            if 'mean inverse reduced ion mobility array' not in z:
                z['ignore_ion_mobility'] = True
                z['mean inverse reduced ion mobility array'] = np.zeros(len(z['m/z array']))

            idx = z['intensity array'] >= min_intensity
            z['intensity array'] = z['intensity array'][idx]
            z['m/z array'] = z['m/z array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] >= min_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] <= max_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = np.argsort(z['m/z array'])
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            cnt += 1

            # if len(data_for_analyse) > 50:
            #     break

            if len(z['m/z array']):
                data_for_analyse.append(z)
            else:
                skipped += 1


    print('Number of MS1 scans: ' + str(len(data_for_analyse)))
    print('Number of skipped MS1 scans: ' + str(skipped))

    return data_for_analyse



def process_mzml_dia(args):

    input_mzml_path = args['file']
    # min_intensity = args['mini']
    # min_mz = args['minmz']
    # max_mz = args['maxmz']
    min_intensity = 0
    min_mz = 1
    max_mz = 1e6

    skipped = 0
    data_for_analyse = []

    cnt = 0

    for z in mzml.read(input_mzml_path):
        if z['ms level'] == 2:

            if 'mean inverse reduced ion mobility array' not in z:
                z['ignore_ion_mobility'] = True
                z['mean inverse reduced ion mobility array'] = np.zeros(len(z['m/z array']))

            idx = z['intensity array'] >= min_intensity
            z['intensity array'] = z['intensity array'][idx]
            z['m/z array'] = z['m/z array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] >= min_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] <= max_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = np.argsort(z['m/z array'])
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            cnt += 1

            # if len(data_for_analyse) > 5000:
            #     break

            if len(z['m/z array']):
                data_for_analyse.append(z)
            else:
                skipped += 1


    print('Number of MS2 scans: ' + str(len(data_for_analyse)))
    print('Number of skipped MS2 scans: ' + str(skipped))

    return data_for_analyse