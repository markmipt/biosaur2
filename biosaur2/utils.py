from pyteomics import mzml
import numpy as np
from collections import defaultdict, Counter
from os import path
import math
from scipy.optimize import curve_fit
from .cutils import get_fast_dict

class MS1OnlyMzML(mzml.MzML): 
     _default_iter_path = '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]' 
     _use_index = False 
     _iterative = False

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

def calc_peptide_features(hills_dict, peptide_features, negative_mode, faims_val, RT_dict, data_start_id):

    for pep_feature in peptide_features:

        pep_feature['mz'] = pep_feature['hill_mz_1']
        pep_feature['nScans'] = hills_dict['hills_lengths'][pep_feature['monoisotope idx']]

        pep_feature['massCalib'] = pep_feature['mz'] * pep_feature['charge'] - 1.0072765 * pep_feature['charge'] * (-1 if negative_mode else 1)

        pep_feature['scanApex'] = hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]
        pep_feature['rtApex'] = RT_dict[hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]+data_start_id]
        pep_feature['intensityApex'] = hills_dict['hills_intensity_apex'][pep_feature['monoisotope idx']]
        pep_feature['rtStart'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][0]+data_start_id]
        pep_feature['rtEnd'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][-1]+data_start_id]

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
        'rtStart',
        'rtEnd',
        'FAIMS',
        'im',
        'mono_hills_scan_lists',
        'mono_hills_intensity_list',
        'scanApex',
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
        hills_dict['hills_mz_median_fast_dict'] = defaultdict(list)
        if paseftol is not False:
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
                # if hill_length > 2:
                #     mz_median = np.average(tmp_mz_array, weights=tmp_intensity)
                # else:
                mz_median = 0
                i_sum_tmp = 0
                for mz_val_tmp, i_val_tmp in zip(tmp_mz_array, tmp_intensity):
                    mz_median += mz_val_tmp * i_val_tmp
                    i_sum_tmp += i_val_tmp
                mz_median = mz_median / i_sum_tmp
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
            tmp_val = (idx_1, tmp_scans_list[0], tmp_scans_list[-1])
            hills_dict['hills_mz_median_fast_dict'][mz_median_int-1].append(tmp_val)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int].append(tmp_val)
            hills_dict['hills_mz_median_fast_dict'][mz_median_int+1].append(tmp_val)

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

    total_num_hills = []
    total_mass_diff = []

    for spec_idx, z in enumerate(data_for_analyse_tmp):

        spec_mean_mass_accuracy = []

        # active_masses_array = []

        len_mz = len(z['m/z array'])

        hills_dict['hills_idx_array'].extend(list(range(last_idx+1, last_idx+1+len_mz, 1)))
        hills_dict['orig_idx_array'].extend(range(len_mz))
        hills_dict['scan_idx_array'].extend([spec_idx] * len_mz)

        idx_for_sort = np.argsort(z['intensity array'])[::-1]

        mz_sorted = z['m/z array'][idx_for_sort]
        basic_id_sorted = list(np.array(range(len_mz))[idx_for_sort])

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

        fast_dict, fast_array = get_fast_dict(mz_sorted, mz_step, basic_id_sorted)

        banned_prev_idx_set = set()

        for idx, fm, fi in zip(basic_id_sorted, fast_array, (fast_array if paseftol is False else fast_array_im)):

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

                # all_idx = prev_fast_dict.get(fm-1, []) + prev_fast_dict.get(fm, []) + prev_fast_dict.get(fm+1, [])

                # best_active_mass_diff = 1e6
                # best_active_flag = False

                best_mass_diff = 1e6
                best_intensity = 0
                best_idx_prev = False
                mz_cur = z['m/z array'][idx]

                all_prevs = [[idx_prev, data_for_analyse_tmp[spec_idx-1]['intensity array'][idx_prev]] for idx_prev in all_idx if idx_prev not in banned_prev_idx_set and (paseftol is False or idx_prev in prev_fast_dict_im[fi])]
                # print(all_prevs[:2])
                # all_prevs = sorted(all_prevs, key=lambda x: -x[1])
                # print(all_prevs[:2])
                # print('\n')
                # for idx_prev in prev_fast_dict[fm]:
                #     if idx_prev not in banned_prev_idx_set and (paseftol is False or idx_prev in prev_fast_dict_im[fi]):
                for idx_prev, cur_intensity in all_prevs:
                    cur_mass_diff_with_sign = (mz_cur - data_for_analyse_tmp[spec_idx-1]['m/z array'][idx_prev]) / mz_cur * 1e6
                    cur_mass_diff = abs(cur_mass_diff_with_sign)
                    # cur_intensity = data_for_analyse_tmp[spec_idx-1]['intensity array'][idx_prev]
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
                        # break
                if best_idx_prev is not False:
                    banned_prev_idx_set.add(best_idx_prev)
                    spec_mean_mass_accuracy.append(best_mass_diff)
                # if best_active_flag:
                #     active_masses_array.append(best_active_mass_diff)


                    # cur_mass_diff = (mz_cur - data_for_analyse_tmp[spec_idx-1]['m/z array'][idx_prev]) / mz_cur * 1e6
                    # # cur_mass_diff_norm = abs(cur_mass_diff) * (1 if cur_mass_diff < 0 else 2)
                    # if -hill_mass_accuracy - 2.5 <= cur_mass_diff <= hill_mass_accuracy and abs(cur_mass_diff) <= best_mass_diff:
                    #     best_mass_diff = abs(cur_mass_diff)
                    #     # hills_dict['hills_idx_array'][last_idx+1+idx] = prev_idx + 1 + idx_prev
                    #     hills_dict['hills_idx_array'][last_idx+1+idx] = hills_dict['hills_idx_array'][prev_idx+1+idx_prev]

        # print(np.mean(spec_mean_mass_accuracy), len(spec_mean_mass_accuracy))
        total_num_hills.append(len(spec_mean_mass_accuracy))
        if len(spec_mean_mass_accuracy) == 0:
            total_mass_diff.append(0)
        else:
            total_mass_diff.append(np.mean(spec_mean_mass_accuracy))

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

    for z in MS1OnlyMzML(source=input_mzml_path):
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

    if len(data_for_analyse) == 0:
        raise Exception('no MS1 scans in input file')

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