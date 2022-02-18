from . import utils
import numpy as np
from scipy.stats import binom
import itertools
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from .cutils import get_initial_isotopes, checking_cos_correlation_for_carbon, split_peaks

def process_file(args):

    data_for_analyse = utils.process_mzml(args)
    write_header = True

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

        #Process ion mobility

        if all('ignore_ion_mobility' not in z for z in data_for_analyse_tmp):
            utils.centroid_pasef_data(data_for_analyse_tmp, args, mz_step)
        else:
            args['paseftol'] = False

        paseftol = args['paseftol']

        hills_dict = utils.detect_hills(data_for_analyse_tmp, args, mz_step, paseftol)

        hills_dict = split_peaks(hills_dict, data_for_analyse_tmp, args)

        hills_dict = utils.process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args)

        logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array'])))

        isotopes_mass_accuracy = args['itol']


        isotopes_list = list(range(10))
        averagine_mass = 111.1254
        averagine_C = 4.9384
        a = dict()

        for i in range(100, 20000, 100):
            int_arr = binom.pmf(
                isotopes_list,
                float(i) /
                averagine_mass *
                averagine_C,
                0.0107)
            int_arr_norm = int_arr / int_arr.sum()
            a[i] = int_arr_norm

        min_charge = args['cmin']
        max_charge = args['cmax']


        ready = get_initial_isotopes(hills_dict, isotopes_mass_accuracy, isotopes_list, a, min_charge, max_charge, mz_step, paseftol, faims_val)

        logger.info('Number of potential isotope clusters: %d', len(ready))

        negative_mode = args['nm']

        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = []

        for i in range(9):
            tmp = []
            for pf in ready:
                isotopes = pf['isotopes']
                if len(isotopes) >= i + 1:
                    tmp.append(isotopes[i]['mass_diff_ppm'])
            isotopes_mass_error_map[i+1] = tmp

        for ic in range(1, 10, 1):
            if ic == 1:

                if len(isotopes_mass_error_map[ic]) >= 100:

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
                    isotopes_mass_error_map[ic] = [0, 10]

            else:
                isotopes_mass_error_map[ic] = deepcopy(isotopes_mass_error_map[ic-1])
                isotopes_mass_error_map[ic][0] = isotopes_mass_error_map[ic][0] - 0.45

        logger.info('Average mass shift between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][0])
        logger.info('Average mass std between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][1])


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


        cnt_mark = 0

        while cur_l < max_l:
            cnt_mark += 1
            pep_feature = ready[cur_l]
            n_iso = pep_feature['nIsotopes']
            if n_iso < cur_isotopes:
                ready = sorted(ready, key=func_for_sort)
                cur_isotopes = n_iso
                cur_l = 0

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


        peptide_features = utils.calc_peptide_features(hills_dict, ready_final, negative_mode, faims_val, RT_dict, data_start_id)

        utils.write_output(peptide_features, args, write_header)

        write_header = False

        data_start_id += len(data_for_analyse_tmp)
