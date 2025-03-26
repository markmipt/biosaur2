from . import utils
import numpy as np
import pandas as pd
from scipy.stats import binom
import itertools
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from .cutils import get_initial_isotopes, checking_cos_correlation_for_carbon, split_peaks, split_peaks_old, detect_hills, process_hills
from multiprocessing import Queue, Process, cpu_count
from collections import Counter, defaultdict
import os

def process_features_iteration(hills_dict, faims_val, mz_step, paseftol, RT_dict, data_start_id, write_header, args):
    isotopes_mass_accuracy = args['itol']


    isotopes_list = list(range(10))
    averagine_mass = 111.1254
    averagine_C = 4.9384
    a = dict()

    for i in range(0, 20000, 100):
        int_arr = binom.pmf(
            isotopes_list,
            round(float(i) / averagine_mass * averagine_C),
            0.0107
        )
        max_pos = np.argmax(int_arr)
        int_arr_norm = int_arr / int_arr.sum()
        a[i] = (int_arr_norm, max_pos)

    min_charge = args['cmin']
    max_charge = args['cmax']
    ivf = args['ivf']

    n_procs = args['nprocs']


    if n_procs == 1:
        qout = []
        ready = []
        procs = []

        sorted_idx_full = [idx_1 for (idx_1, hill_idx_1), hill_mz_1 in sorted(list(zip(list(enumerate(hills_dict['hills_idx_array_unique'])), hills_dict['hills_mz_median'])), key=lambda x: x[-1])]
        sorted_idx_child_process = sorted_idx_full

        qout = get_initial_isotopes_python(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val, ivf, list(sorted_idx_child_process), qout, win_sys=True)

        ready.extend(qout)

    else:
        qout = Queue()
        ready = []
        procs = []

        sorted_idx_full = [idx_1 for (idx_1, hill_idx_1), hill_mz_1 in sorted(list(zip(list(enumerate(hills_dict['hills_idx_array_unique'])), hills_dict['hills_mz_median'])), key=lambda x: x[-1])]
        len_full = len(sorted_idx_full)
        step = int(len_full / n_procs)
        for i in range(n_procs):
            sorted_idx_child_process = sorted_idx_full[i*step:i*step+step]

            p = Process(
                target=get_initial_isotopes_python,
                args=(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val, ivf, list(sorted_idx_child_process), qout))
            p.start()
            procs.append(p)

        for _ in range(n_procs):
            for ready_child_process in iter(qout.get, None):
                ready.extend(ready_child_process)
        for p in procs:
            p.join()


    logger.info('Number of potential isotope clusters: %d', len(ready))


    if args['ignore_iso_calib']:
        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = [0, args['itol']]
    else:

        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = []

        for i in range(9):
            tmp = []
            for pf in ready:
                isotopes = pf['isotopes']
                scans = pf['nScans']
                if len(isotopes) >= i + 1 and scans >= 3:
                    tmp.append(isotopes[i]['mass_diff_ppm'])
            isotopes_mass_error_map[i+1] = tmp

        for ic in range(1, 10, 1):
            if ic <= 3:

                if len(isotopes_mass_error_map[ic]) >= 1000:

                    try:

                        true_md = np.array(isotopes_mass_error_map[ic])

                        mass_left = -min(isotopes_mass_error_map[ic])
                        mass_right = max(isotopes_mass_error_map[ic])


                        mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.05, mass_left, mass_right, true_md)
                        if abs(mass_shift) >= max(mass_left, mass_right):
                            mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.25, mass_left, mass_right, true_md)
                        if np.isinf(covvalue):
                            mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.05, mass_left, mass_right, true_md)

                        isotopes_mass_error_map[ic] = [mass_shift, mass_sigma]

                    except:
                        isotopes_mass_error_map[ic] = [0, 10]

                else:
                    if ic -1 in isotopes_mass_error_map:
                        isotopes_mass_error_map[ic] = deepcopy(isotopes_mass_error_map[ic-1])
                        isotopes_mass_error_map[ic][0] += isotopes_mass_error_map[ic-1][0] - isotopes_mass_error_map.get(ic-2, [0, ])[0]
                        isotopes_mass_error_map[ic][1] *= isotopes_mass_error_map[ic-1][1] / isotopes_mass_error_map.get(ic-2, isotopes_mass_error_map[ic-1])[1]

                    else:
                        isotopes_mass_error_map[ic] = [0, 10]

            else:
                isotopes_mass_error_map[ic] = deepcopy(isotopes_mass_error_map[ic-1])
                isotopes_mass_error_map[ic][0] += isotopes_mass_error_map[ic-1][0] - isotopes_mass_error_map.get(ic-2, [0, ])[0]
                isotopes_mass_error_map[ic][1] *= isotopes_mass_error_map[ic-1][1] / isotopes_mass_error_map.get(ic-2, isotopes_mass_error_map[ic-1])[1]

    logger.info('Average mass shift between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][0])
    logger.info('Average mass std between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][1])

    logger.debug(isotopes_mass_error_map)

    max_l = len(ready)
    cur_l = 0

    while cur_l < max_l:
        pep_feature = ready[cur_l]

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
            cos_corr, number_of_passed_isotopes = checking_cos_correlation_for_carbon(all_theoretical_int, all_exp_intensity, 0.6)
            if cos_corr:

                ready[cur_l]['cos_cor_isotopes'] = cos_corr
                ready[cur_l]['isotopes'] = tmp
                ready[cur_l]['nIsotopes'] = tmp_n_isotopes + 1
                ready[cur_l]['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]


            else:
                del ready[cur_l]
                max_l -= 1
                cur_l -= 1


        else:
            del ready[cur_l]
            max_l -= 1
            cur_l -= 1

        cur_l += 1

    logger.info('Number of potential isotope clusters after smart mass accuracy for isotopes: %d', len(ready))

    max_l = len(ready)
    cur_l = 0

    func_for_sort = lambda x: -x['nIsotopes']-x['cos_cor_isotopes']

    ready_final = []
    ready_set = set()
    ready = sorted(ready, key=func_for_sort)
    cur_isotopes = ready[0]['nIsotopes']


    while cur_l < max_l:
        pep_feature = ready[cur_l]
        n_iso = pep_feature['nIsotopes']
        if n_iso < cur_isotopes:
            ready = sorted(ready, key=func_for_sort)
            cur_isotopes = n_iso
            cur_l = 0
            pep_feature = ready[cur_l]

        if pep_feature['monoisotope hill idx'] not in ready_set:
            if not any(cand['isotope_hill_idx'] in ready_set for cand in pep_feature['isotopes']):
                ready_final.append(pep_feature)
                ready_set.add(pep_feature['monoisotope hill idx'])
                for cand in pep_feature['isotopes']:
                    ready_set.add(cand['isotope_hill_idx'])
                del ready[cur_l]
                max_l -= 1
                cur_l -= 1

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
                    cos_corr, number_of_passed_isotopes = checking_cos_correlation_for_carbon(all_theoretical_int, all_exp_intensity, 0.6)
                    if cos_corr:
                        ready[cur_l]['cos_cor_isotopes'] = cos_corr
                        ready[cur_l]['isotopes'] = tmp
                        ready[cur_l]['nIsotopes'] = tmp_n_isotopes + 1
                        ready[cur_l]['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]
                        cur_l -= 1
                    else:
                        del ready[cur_l]
                        max_l -= 1
                        cur_l -= 1


                else:
                    del ready[cur_l]
                    max_l -= 1
                    cur_l -= 1
        else:
            del ready[cur_l]
            max_l -= 1
            cur_l -= 1

        cur_l += 1

    logger.info('Number of detected isotope clusters: %d', len(ready_final))


    negative_mode = args['nm']
    isotopes_for_intensity = args['iuse']
    peptide_features = utils.calc_peptide_features(hills_dict, ready_final, negative_mode, faims_val, RT_dict, data_start_id, isotopes_for_intensity)

    utils.write_output(peptide_features, args, write_header)

    return ready_set

def split_peaks_python(qout, hills_dict, data_for_analyse_tmp, args, counter_hills_idx, sorted_idx_child_process, sorted_idx_array_child_process, i, checked_id, win_sys=False):

    new_index_list = split_peaks(hills_dict, data_for_analyse_tmp, args, counter_hills_idx, sorted_idx_child_process, sorted_idx_array_child_process, i, checked_id)
    if win_sys:
        return (i, list(new_index_list))
    else:
        qout.put((i, list(new_index_list)))
        qout.put(None)

def split_peaks_multi(hills_dict, data_for_analyse_tmp, hvf, args):

    hillValleyFactor = hvf
    min_length_hill = args['minlh']

    hills_dict['orig_idx_array'] = np.array(hills_dict['orig_idx_array'])
    hills_dict['scan_idx_array'] = np.array(hills_dict['scan_idx_array'])
    hills_dict['hills_idx_array'] = np.array(hills_dict['hills_idx_array'])

    counter_hills_idx = Counter(hills_dict['hills_idx_array'])
    counter_hills_idx_2 = dict()
    for k,v in counter_hills_idx.items():
        counter_hills_idx_2[k] = v
    counter_hills_idx = counter_hills_idx_2

    tmp_hill_length = np.array([counter_hills_idx[hill_idx] for hill_idx in hills_dict['hills_idx_array']])
    idx_minl = tmp_hill_length >= min_length_hill
    hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_minl]
    hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_minl]
    hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_minl]

    all_sets = []
    all_sorted_idx = []

    if len(hills_dict['orig_idx_array']):

        idx_sort = np.argsort(hills_dict['hills_idx_array'] + ((hills_dict['scan_idx_array'] + 1) / (hills_dict['scan_idx_array'].max()+2)))
        hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_sort]
        hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_sort]
        hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_sort]

        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))

        data_for_analyse_tmp_intensity = [z['intensity array'] for z in data_for_analyse_tmp]

        n_procs = args['nprocs']



        if n_procs == 1:
            qout = []
            procs = []
            new_idx_res = dict()
            checked_id = 0

            sorted_idx_child_process = list(hills_dict['hills_idx_array_unique'])
            idx_unique_set = set(sorted_idx_child_process)
            local_idx = np.array([z in idx_unique_set for z in list(hills_dict['hills_idx_array'])])
            sorted_idx_array_child_process = hills_dict['hills_idx_array'][local_idx]
            sorted_idx_child_process = sorted(list(idx_unique_set))
            i = 0

            qout = split_peaks_python(qout, hills_dict, data_for_analyse_tmp_intensity, args, counter_hills_idx, sorted_idx_child_process, sorted_idx_array_child_process, i, checked_id, win_sys=True)

            new_idx_res[qout[0]] = qout[1]

        else:

            qout = Queue()
            new_idx_res = dict()
            procs = []
            ar2 = []
            len_full = len(hills_dict['hills_idx_array_unique'])
            if len_full <= 1000 * n_procs:
                n_procs = 1
            step = int(len_full / n_procs) + 1
            checked_id = 0
            for i in range(n_procs):
                sorted_idx_child_process = list(hills_dict['hills_idx_array_unique'][i*step:i*step+step])
                idx_unique_set = set(sorted_idx_child_process)
                all_sets.append(idx_unique_set)
                local_idx = np.array([z in idx_unique_set for z in list(hills_dict['hills_idx_array'])])
                sorted_idx_array_child_process = hills_dict['hills_idx_array'][local_idx]
                ar2.append(sorted_idx_array_child_process)
                sorted_idx_child_process = sorted(list(idx_unique_set))

                all_sorted_idx.append(local_idx)
                p = Process(
                    target=split_peaks_python,
                    args=(qout, hills_dict, data_for_analyse_tmp_intensity, args, counter_hills_idx, sorted_idx_child_process, sorted_idx_array_child_process, i, checked_id))
                checked_id += len(sorted_idx_array_child_process)
                    # args=(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val, list(sorted_idx_child_process), qout))
                p.start()
                procs.append(p)

            for _ in range(n_procs):
                for ready_child_process in iter(qout.get, None):
                    new_idx_res[ready_child_process[0]] = ready_child_process[1]
            for p in procs:
                p.join()

    final_idx_array = []
    last_id = 1
    for i in range(n_procs):
        added_idx_map = {}
        for ii, idx_val in enumerate(new_idx_res[i]):
            if idx_val not in added_idx_map:
                added_idx_map[idx_val] = int(last_id)
                last_id += 1
            final_idx_array.append(added_idx_map[idx_val])

    # hills_dict['scan_idx_array'] = np.concatenate([hills_dict['scan_idx_array'][all_sorted_idx[i]] for i in range(n_procs)])
    # hills_dict['orig_idx_array'] = np.concatenate([hills_dict['orig_idx_array'][all_sorted_idx[i]] for i in range(n_procs)])

    hills_dict['hills_idx_array'] = list(final_idx_array)
    del hills_dict['hills_idx_array_unique']

    return hills_dict

def get_initial_isotopes_python(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val, ivf, sorted_idx_child_process, qout, win_sys=False):

    ready_local = get_initial_isotopes(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val, ivf, sorted_idx_child_process)
    if win_sys:
        return ready_local
    else:
        qout.put(ready_local)
        qout.put(None)

def process_file(args):

    input_file_path = args['file']

    if input_file_path.lower().endswith('.mzml'):

        write_header = True

        data_for_analyse = utils.process_mzml(args)

        #Process faims

        faims_set = set([z.get('FAIMS compensation voltage', 0) for z in data_for_analyse])
        if any(z for z in faims_set):
            logger.info('Detected FAIMS values: %s', faims_set)

        data_start_id = 0
        data_cur_id = 0
        RT_dict = dict()

        for faims_val in faims_set:

            if len(faims_set) > 1:
                logger.info('Spectra analysis for CV = %.3f', faims_val)

            data_for_analyse_tmp = []
            for z in data_for_analyse:
                if not faims_val or z['FAIMS compensation voltage'] == faims_val:
                    data_for_analyse_tmp.append(z)
                    RT_dict[data_cur_id] = float(z['scanList']['scan'][0]['scan start time'])
                    data_cur_id += 1
                    


            hill_mass_accuracy = args['htol']
            max_mz_value = 0
            for z in data_for_analyse_tmp:
                max_mz_value = max(max_mz_value, z['m/z array'].max())

            mz_step = hill_mass_accuracy * 1e-6 * max_mz_value

            #Process TOF
            if args['tof']:
                data_for_analyse_tmp = utils.process_tof(data_for_analyse_tmp)

            #Process profile
            if args['profile']:
                data_for_analyse_tmp = utils.process_profile(data_for_analyse_tmp)

            #Process ion mobility

            if all('ignore_ion_mobility' not in z for z in data_for_analyse_tmp):
                utils.centroid_pasef_data(data_for_analyse_tmp, args, mz_step)
            else:
                args['paseftol'] = 0

            paseftol = args['paseftol']

            if args['use_hill_calib']:

                l_data = len(data_for_analyse_tmp)

                if l_data <= 1000:

                    hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol)

                else:

                    hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp[int(l_data/2)-500:int(l_data/2)+500], args, mz_step, paseftol)

                true_md = np.array(total_mass_diff)

                mass_left = -min(true_md)
                mass_right = max(true_md)
                mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.05, mass_left, mass_right, true_md)
                if abs(mass_shift) >= max(mass_left, mass_right):
                    mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.25, mass_left, mass_right, true_md)
                if np.isinf(covvalue):
                    mass_shift, mass_sigma, covvalue = utils.calibrate_mass(0.05, mass_left, mass_right, true_md)

                args['htol'] = min(args['htol'], 5 * mass_sigma)

                logger.info('Automatically optimized htol parameter: %.3f ppm', args['htol'])

            hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol)



            logger.info('Detected number of hills before splitting: %d', len(set(hills_dict['hills_idx_array'])))

            hills_dict = split_peaks_multi(hills_dict, data_for_analyse_tmp, args['hvf'], args)
            logger.info('Starting hills processing')
            hills_dict = process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args)

            logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array'])))
            if args['write_hills']:
                hills_dict, hills_features = utils.process_hills_extra(hills_dict, RT_dict, faims_val, data_start_id, mz_step, paseftol)
                utils.write_output(hills_features, args, write_header, hills=True)



            ready_set = process_features_iteration(hills_dict, faims_val, mz_step, paseftol, RT_dict, data_start_id, write_header, args)

            write_header = False

            data_start_id += len(data_for_analyse_tmp)



    elif input_file_path.lower().endswith('.hills.tsv'):
        hills_features = pd.read_table(input_file_path)
        RT_dict = False
        write_header = True
        data_start_id = 0

        if np.any(hills_features['FAIMS']):
            paseftol = 0
        
        else:
            if np.any(hills_features['im']):
                paseftol = args['paseftol']
            else:
                paseftol = 0


        if paseftol == 0:
            faims_set = set(hills_features['FAIMS'])
        else:
            faims_set = set([0, ])

        if any(z for z in faims_set):
            logger.info('Detected FAIMS values: %s', faims_set)

        for faims_val in faims_set:

            if len(faims_set) > 1:
                logger.info('Spectra analysis for CV = %.3f', faims_val)

            if paseftol == 0:
                hills_features_local = hills_features[hills_features['FAIMS'] == faims_val]
            else:
                hills_features_local = hills_features

            hill_mass_accuracy = args['htol']
            hills_dict, mz_step = utils.get_hills_dict_from_hills_features(hills_features_local, hill_mass_accuracy, paseftol)


            logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array_unique'])))


            ready_set = process_features_iteration(hills_dict, faims_val, mz_step, paseftol, RT_dict, data_start_id, write_header, args)


            write_header = False
