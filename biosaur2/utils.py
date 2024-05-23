from pyteomics import mzml
import numpy as np
from collections import defaultdict, Counter
from os import path
import math
from scipy.optimize import curve_fit
import logging
logger = logging.getLogger(__name__)
from .cutils import get_fast_dict, get_and_calc_apex_intensity_and_scan, centroid_pasef_scan

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


def process_hills_extra(hills_dict, RT_dict, faims_val, data_start_id):

    hills_features = []
    for idx_1 in range(len(hills_dict['hills_idx_array_unique'])):
        hill_feature = {}
        hills_dict, hill_intensity_apex_1, hill_scan_apex_1 = get_and_calc_apex_intensity_and_scan(hills_dict, idx_1)
        hill_feature['mz'] = hills_dict['hills_mz_median'][idx_1]
        hill_feature['nScans'] = hills_dict['hills_lengths'][idx_1]
        hill_feature['rtApex'] = RT_dict[hill_scan_apex_1+data_start_id]
        hill_feature['intensityApex'] = hill_intensity_apex_1
        hill_feature['intensitySum'] = sum(hills_dict['hills_intensity_array'][idx_1])
        hill_feature['rtStart'] = RT_dict[hills_dict['hills_scan_lists'][idx_1][0]+data_start_id]
        hill_feature['rtEnd'] = RT_dict[hills_dict['hills_scan_lists'][idx_1][-1]+data_start_id]
        hill_feature['FAIMS'] = faims_val
        if 'hills_im_median' in hills_dict:
            hill_feature['im'] = hills_dict['hills_im_median'][idx_1]
        else:
            hill_feature['im'] = 0
        hill_feature['hill_idx'] = hills_dict['hills_idx_array_unique'][idx_1]
        hill_feature['hills_scan_lists'] = hills_dict['hills_scan_lists'][idx_1]
        hill_feature['hills_intensity_list'] = hills_dict['hills_intensity_array'][idx_1]
        hill_feature['hills_mz_array'] = hills_dict['tmp_mz_array'][idx_1]
        hills_features.append(hill_feature)

    return hills_dict, hills_features


def calc_peptide_features(hills_dict, peptide_features, negative_mode, faims_val, RT_dict, data_start_id, isotopes_for_intensity):

    for pep_feature in peptide_features:

        pep_feature['mz'] = pep_feature['hill_mz_1']
        pep_feature['isoerror'] = pep_feature['isotopes'][0]['mass_diff_ppm']
        pep_feature['isoerror2'] = pep_feature['isotopes'][1]['mass_diff_ppm'] if len(pep_feature['isotopes']) > 1 else -100
        pep_feature['nScans'] = hills_dict['hills_lengths'][pep_feature['monoisotope idx']]

        pep_feature['massCalib'] = pep_feature['mz'] * pep_feature['charge'] - 1.0072765 * pep_feature['charge'] * (-1 if negative_mode else 1)

        hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, pep_feature['monoisotope idx'])
        pep_feature['intensityApex'] = hills_dict['hills_intensity_apex'][pep_feature['monoisotope idx']]
        pep_feature['intensitySum'] = sum(hills_dict['hills_intensity_array'][pep_feature['monoisotope idx']])

        if isotopes_for_intensity != 0:
            idx_cur = 0
            for cand in pep_feature['isotopes']:
                idx_cur += 1
                if idx_cur == isotopes_for_intensity + 1:
                    break
                else:
                    iso_idx = cand['isotope_idx']
                    hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, iso_idx)
                    pep_feature['intensityApex'] += hills_dict['hills_intensity_apex'][iso_idx]
                    pep_feature['intensitySum'] += sum(hills_dict['hills_intensity_array'][iso_idx])
                

        pep_feature['scanApex'] = hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]
        pep_feature['rtApex'] = RT_dict[hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]+data_start_id]
        pep_feature['rtStart'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][0]+data_start_id]
        pep_feature['rtEnd'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][-1]+data_start_id]
        pep_feature['mono_hills_scan_lists'] = hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']]
        pep_feature['mono_hills_intensity_list'] =  hills_dict['hills_intensity_array'][pep_feature['monoisotope idx']]

    return peptide_features


def write_output(peptide_features, args, write_header=True, hills=False):

    input_mzml_path = args['file']

    if args['o']:
        output_file = args['o'] if not hills else (path.splitext(args['o'])[0]\
            + path.extsep + 'hills.tsv')
    else:
        output_file = path.splitext(input_mzml_path)[0]\
            + path.extsep + ('features.tsv' if not hills else 'hills.tsv')

    if hills:

        columns_for_output = [
            'rtApex',
            'intensityApex',
            'intensitySum',
            'nScans',
            'mz',
            'rtStart',
            'rtEnd',
            'FAIMS',
            'im',
        ]
        if args['write_extra_details']:
            columns_for_output += ['hill_idx', 'hills_scan_lists', 'hills_intensity_list', 'hills_mz_array']
    else:
        columns_for_output = [
            'massCalib',
            'rtApex',
            'intensityApex',
            'intensitySum',
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
            'isoerror',
            'isoerror2',
        ]
        if args['write_extra_details']:
            columns_for_output += ['isoerror','isotopes','intensity_array_for_cos_corr','monoisotope hill idx','monoisotope idx']

    if write_header:

        out_file = open(output_file, 'w')
        out_file.write('\t'.join(columns_for_output) + '\n')
        out_file.close()

    out_file = open(output_file, 'a')
    for pep_feature in peptide_features:
        out_file.write('\t'.join([str(pep_feature[col]) for col in columns_for_output]) + '\n')

    out_file.close()


def centroid_pasef_data(data_for_analyse_tmp, args, mz_step):

    cnt_ms1_scans = len(data_for_analyse_tmp)

    ion_mobility_accuracy = args['paseftol']
    hill_mz_accuracy = args['htol']
    pasefmini = args['pasefmini']
    pasefminlh = args['pasefminlh']
    for spec_idx, z in enumerate(data_for_analyse_tmp):

        logger.debug('PASEF scans analysis: %d/%d', spec_idx+1, cnt_ms1_scans)
        logger.debug('number of m/z peaks in scan: %d', len(z['m/z array']))

        if 'ignore_ion_mobility' not in z:

            # mz_ar_new = []
            # intensity_ar_new = []
            # ion_mobility_ar_new = []

            # mz_ar = z['m/z array']
            # intensity_ar = z['intensity array']
            # ion_mobility_ar = z['mean inverse reduced ion mobility array']

            # ion_mobility_step = max(ion_mobility_ar) * ion_mobility_accuracy

            # ion_mobility_ar_fast = (ion_mobility_ar/ion_mobility_step).astype(int)
            # mz_ar_fast = (mz_ar/mz_step).astype(int)

            # idx = np.argsort(mz_ar_fast)
            # mz_ar_fast = mz_ar_fast[idx]
            # ion_mobility_ar_fast = ion_mobility_ar_fast[idx]

            # mz_ar = mz_ar[idx]
            # intensity_ar = intensity_ar[idx]
            # ion_mobility_ar = ion_mobility_ar[idx]

            # max_peak_idx = len(mz_ar)

            # banned_idx = set()

            # peak_idx = 0
            # while peak_idx < max_peak_idx:

            #     if peak_idx not in banned_idx:

            #         mass_accuracy_cur = mz_ar[peak_idx] * 1e-6 * hill_mz_accuracy

            #         mz_val_int = mz_ar_fast[peak_idx]
            #         ion_mob_val_int = ion_mobility_ar_fast[peak_idx]

            #         tmp = [peak_idx, ]

            #         peak_idx_2 = peak_idx + 1

            #         while peak_idx_2 < max_peak_idx:


            #             if peak_idx_2 not in banned_idx:

            #                 mz_val_int_2 = mz_ar_fast[peak_idx_2]
            #                 if mz_val_int_2 - mz_val_int > 1:
            #                     break
            #                 elif abs(mz_ar[peak_idx]-mz_ar[peak_idx_2]) <= mass_accuracy_cur:
            #                     ion_mob_val_int_2 = ion_mobility_ar_fast[peak_idx_2]
            #                     if abs(ion_mob_val_int - ion_mob_val_int_2) <= 1:
            #                         if abs(ion_mobility_ar[peak_idx] - ion_mobility_ar[peak_idx_2]) <= ion_mobility_accuracy:
            #                             tmp.append(peak_idx_2)
            #                             peak_idx = peak_idx_2
            #             peak_idx_2 += 1

            #     all_intensity = [intensity_ar[p_id] for p_id in tmp]
            #     i_val_new = sum(all_intensity)

            #     if i_val_new >= pasefmini and len(all_intensity) >= pasefminlh:

            #         all_mz = [mz_ar[p_id] for p_id in tmp]
            #         all_ion_mob = [ion_mobility_ar[p_id] for p_id in tmp]

            #         mz_val_new = np.average(all_mz, weights=all_intensity)
            #         ion_mob_new = np.average(all_ion_mob, weights=all_intensity)

            #         intensity_ar_new.append(i_val_new)
            #         mz_ar_new.append(mz_val_new)
            #         ion_mobility_ar_new.append(ion_mob_new)

            #         banned_idx.update(tmp)

            #     peak_idx += 1

            mz_ar_new, intensity_ar_new, ion_mobility_ar_new = centroid_pasef_scan(z, mz_step, hill_mz_accuracy, ion_mobility_accuracy, pasefmini, pasefminlh)

            data_for_analyse_tmp[spec_idx]['m/z array'] = np.array(mz_ar_new)
            data_for_analyse_tmp[spec_idx]['intensity array'] = np.array(intensity_ar_new)
            data_for_analyse_tmp[spec_idx]['mean inverse reduced ion mobility array'] = np.array(ion_mobility_ar_new)

        logger.debug('number of m/z peaks in scan after centroiding: %d', len(data_for_analyse_tmp[spec_idx]['m/z array']))

    data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'] > 0)]
    logger.info('Number of MS1 scans after combining ion mobility peaks: %d', len(data_for_analyse_tmp))

    return data_for_analyse_tmp

def process_profile(data_for_analyse_tmp):

    data_for_analyse_tmp_out = []

    for z in data_for_analyse_tmp:

        best_mz = 0
        best_int = 0
        best_im = 0
        prev_mz = False
        prev_int = False

        threshold = 0.05

        ar1 = []
        ar2 = []
        ar3 = []
        for mzv, intv, imv in zip(z['m/z array'], z['intensity array'], z['mean inverse reduced ion mobility array']):
            if prev_mz is False:
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif mzv - prev_mz > threshold:
                ar1.append(best_mz)
                ar2.append(best_int)
                ar3.append(best_im)
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif best_int > prev_int and intv > prev_int:
                ar1.append(best_mz)
                ar2.append(best_int)
                ar3.append(best_im)
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif intv > best_int:
                best_mz = mzv
                best_int = intv
                best_im = imv
            prev_mz = mzv
            prev_int = intv

        ar1.append(best_mz)
        ar2.append(best_int)
        ar3.append(best_im)

        z['m/z array'] = np.array(ar1)
        z['intensity array'] = np.array(ar2)
        z['mean inverse reduced ion mobility array'] = np.array(ar3)

        data_for_analyse_tmp_out.append(z)
    return data_for_analyse_tmp_out



def process_tof(data_for_analyse_tmp):

            # print(len(z['m/z array']))
    universal_dict = {}
    cnt = 0


    temp_i = defaultdict(list)
    for z in data_for_analyse_tmp:
        cnt += 1
        fast_set = z['m/z array'] // 50

        if cnt <= 25:



            for l in set(fast_set):

                if l not in universal_dict:

                    idxt = fast_set == l
                    true_i = np.log10(z['intensity array'])[idxt]
                    temp_i[l].extend(true_i)

                    if len(temp_i[l]) > 150:

                        temp_i[l] = np.array(temp_i[l])
                        i_left = temp_i[l].min()
                        i_right = temp_i[l].max()

                        i_shift, i_sigma, covvalue = calibrate_mass(0.05, i_left, i_right, temp_i[l])
                        # median_val = 
                        print(i_shift, i_sigma, covvalue)
                        universal_dict[l] = 10**(i_shift + 2 * i_sigma)#10**(np.median(true_i[idxt]) * 2)
            

    cnt = 0

    for z in data_for_analyse_tmp:

        fast_set = z['m/z array'] // 50
        while cnt <= 50:

            cnt += 1

            temp_i = []

            for l in set(fast_set):
                idxt = fast_set == l
                true_i = np.log10(z['intensity array'])[idxt]
                temp_i.extend(true_i)

                if len(true_i) > 150:

                    i_left = true_i.min()
                    i_right = true_i.max()

                    i_shift, i_sigma, covvalue = calibrate_mass(0.05, i_left, i_right, true_i)
                    # median_val = 
                    print(i_shift, i_sigma, covvalue)
                    universal_dict[l] = 10**(i_shift + 3 * i_sigma)#10**(np.median(true_i[idxt]) * 2)
            

            
        thresholds = [universal_dict.get(zz, 150) for zz in list(fast_set)]
        idxt2 = z['intensity array'] <= thresholds
        z['intensity array'][idxt2] = -1


        idx = z['intensity array'] > 0
        z['intensity array'] = z['intensity array'][idx]
        z['m/z array'] = z['m/z array'][idx]
        z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]



        cnt += 1

        data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'])]

    return data_for_analyse_tmp


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

            if 'raw ion mobility array' in z:
                z['mean inverse reduced ion mobility array'] = z['raw ion mobility array']

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


    logger.info('Number of MS1 scans: %d', len(data_for_analyse))
    logger.info('Number of skipped MS1 scans: %d', skipped)

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


    logger.info('Number of MS2 scans: %d', len(data_for_analyse))
    logger.info('Number of skipped MS2 scans: %d', skipped)

    return data_for_analyse
