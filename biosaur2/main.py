from . import utils
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import binom
from scipy.stats import scoreatpercentile
import itertools
from copy import deepcopy

def process_file(args):

    data_for_analyse = utils.process_mzml(args)
    write_header = True

    #Process faims
    faims_set = set([z.get('FAIMS compensation voltage', None) for z in data_for_analyse])
    if any(z is not None for z in faims_set):
        print('Detected FAIMS values: %s' % str(faims_set))

    data_start_id = 0
    data_cur_id = 0
    RT_dict = dict()

    for faims_val in faims_set:

        if len(faims_set) > 1:
            print('Spectra analysis for CV = %.3f' % (faims_val, ))

        data_for_analyse_tmp = []
        for z in data_for_analyse:
            if faims_val is None or z['FAIMS compensation voltage'] == faims_val:
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

        hills_dict = utils.split_peaks(hills_dict, data_for_analyse_tmp, args)

        hills_dict = utils.process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args)

        print('Detected number of hills: %d' % (len(set(hills_dict['hills_idx_array'])), ))

        isotopes_mass_accuracy = args['itol']

        ready = []
        averagine_mass = 111.1254
        averagine_C = 4.9384
        isotopes_list = list(range(10))
        prec_masses = []

        a = dict()

        for i in range(100, 20000, 100):
            int_arr = binom.pmf(
                isotopes_list,
                float(i) /
                averagine_mass *
                averagine_C,
                0.0107)
            prec_masses.append(i)
            int_arr_norm = int_arr / int_arr.sum()
            a[i] = int_arr_norm
        
        min_charge = args['cmin']
        max_charge = args['cmax']
        charges = list(range(min_charge, max_charge + 1, 1)[::-1])

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

        isotope_series = dict()
        for charge in charges:
            isotope_series[charge] = dict()

        for idx_1, hill_idx_1 in enumerate(hills_dict['hills_idx_array_unique']):

            if paseftol is not False:
                im_mz_1 = hills_dict['hills_im_median'][idx_1]
                im_to_check_fast = int(hills_dict['hills_im_median'][idx_1] / paseftol)

            banned_charges = dict()

            # Monoisotope candidate m/z value
            hill_mz_1 = hills_dict['hills_mz_median'][idx_1]

            # Monoisotope candidate scans
            hill_scans_1 = hills_dict['hills_scan_sets'][idx_1]
            hill_length_1 = hills_dict['hills_lengths'][idx_1]
            hill_length_1_morethan1 = hill_length_1 > 1

            mz_tol = isotopes_mass_accuracy * 1e-6 * hill_mz_1

            for charge in charges:
                
                candidates = []

                series_1 = idx_1

                k = i
                ks = i
                for isotope_number in isotopes_list[1:]:

                    tmp_candidates = []

                    m_to_check = hill_mz_1 + (1.00335 * isotope_number / charge)
                    m_to_check_fast = int(m_to_check/mz_step)

                    for idx_2 in hills_dict['hills_mz_median_fast_dict'][m_to_check_fast]:

                        if paseftol is False or idx_2 in hills_dict['hills_im_median_fast_dict'][im_to_check_fast]:

                            series_2 = idx_2
                            
                            if isotope_number > isotope_series[charge].get((series_1, series_2), 0):

                                hill_scans_2 = hills_dict['hills_scan_sets'][idx_2]
                                hill_length_2 = hills_dict['hills_lengths'][idx_2]
                                hill_length_2_morethan1 = hill_length_2 > 1

                                if len(hill_scans_1.intersection(hill_scans_2)) >= 2:

                                    hill_mz_2 = hills_dict['hills_mz_median'][idx_2]
                                    mass_diff_abs = hill_mz_2 - m_to_check

                                    if abs(mass_diff_abs) <= mz_tol:

                                        if (hill_length_1_morethan1 and hill_length_2_morethan1):

                                            hills_dict, hill_idict_1, hill_sqrt_of_i_1 = utils.get_and_calc_values_for_cos_corr(hills_dict, idx_1, hill_length_1_morethan1)
                                            hills_dict, hill_idict_2, hill_sqrt_of_i_2 = utils.get_and_calc_values_for_cos_corr(hills_dict, idx_2, hill_length_2_morethan1)

                                            cos_cor_RT = utils.cos_correlation(hill_length_1, hill_scans_1, hill_idict_1, hill_sqrt_of_i_1, hill_length_2, hill_scans_2, hill_idict_2, hill_sqrt_of_i_2)
                                        else:
                                            cos_cor_RT = 1.0

                                        if cos_cor_RT >= 0.6:

                                            hill_idx_2 = hills_dict['hills_idx_array_unique'][idx_2]

                                            hills_dict, _, _ = utils.get_and_calc_apex_intensity_and_scan(hills_dict, hill_length_1_morethan1, idx_1)
                                            hills_dict, _, _ = utils.get_and_calc_apex_intensity_and_scan(hills_dict, hill_length_2_morethan1, idx_2)

                                            local_isotopes_dict = {
                                                'isotope_number': isotope_number,
                                                'isotope_hill_idx': hill_idx_2,
                                                'isotope_idx': idx_2,
                                                'cos_cor': cos_cor_RT,
                                                'mass_diff_ppm': mass_diff_abs/m_to_check*1e6
                                            }

                                            tmp_candidates.append(local_isotopes_dict)

                                            isotope_series[charge][(series_1, series_2)] = isotope_number
                                            series_1 = idx_2


                    if len(tmp_candidates):
                        candidates.append(tmp_candidates)

                    if len(candidates) < isotope_number:
                        break



                if len(candidates) >= banned_charges.get(charge, 1):

                    neutral_mass = hill_mz_1 * charge

                    tmp_intensity = a[int(100 * (neutral_mass // 100))]

                    _, hill_intensity_apex_1, hill_scan_apex_1 = utils.get_and_calc_apex_intensity_and_scan(hills_dict, 0, idx_1)
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

                            _, hill_intensity_apex_2, hill_scan_apex_2 = utils.get_and_calc_apex_intensity_and_scan(hills_dict, 0, idx_2)
                            
                            all_exp_intensity.append(hill_intensity_apex_2)
                        (
                            cos_corr,
                            number_of_passed_isotopes,
                            shift) = utils.checking_cos_correlation_for_carbon(
                            all_theoretical_int, all_exp_intensity, 0.6)

                        if cos_corr:

                            iter_candidates = iter_candidates[:number_of_passed_isotopes]

                            local_res_dict = {
                                'monoisotope hill idx': hill_idx_1,
                                'monoisotope idx': idx_1,
                                'cos_cor_isotopes': cos_corr,
                                'hill_mz_1': hill_mz_1,
                                'isotopes': iter_candidates,
                                'nIsotopes': number_of_passed_isotopes,
                                'nScans': hill_length_1, 
                                'charge': charge,
                                'FAIMS': faims_val,
                                'shift': shift,
                                'im': 0 if paseftol is False else im_mz_1,
                                'intensity_array_for_cos_corr': [all_theoretical_int[:number_of_passed_isotopes+1], all_exp_intensity[:number_of_passed_isotopes+1]],
                                'mono_hills_scan_lists': list(mono_hills_scan_lists),
                                'mono_hills_intensity_list': list(mono_hills_intensity_list),
                            }

                            ready.append(local_res_dict)

                            for ch_v in charge_ban_map[charge]:
                                banned_charges[ch_v] = max(number_of_passed_isotopes, banned_charges.get(ch_v, 1))

        negative_mode = args['nm']

        print('Number of potential isotope clusters: ', len(ready))


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

        print('Average mass shift between monoisotopic and first 13C isotope: %.3f ppm' % (isotopes_mass_error_map[1][0]))
        print('Average mass std between monoisotopic and first 13C isotope: %.3f ppm' % (isotopes_mass_error_map[1][1]))


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
                (cos_corr,
                        number_of_passed_isotopes,
                        _) = utils.checking_cos_correlation_for_carbon(
                        all_theoretical_int, all_exp_intensity, 0.6, allowed_shift=pep_feature['shift'])
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

        print('Number of potential isotope clusters after smart mass accuracy for isotopes: ', len(ready))

        max_l = len(ready)
        cur_l = 0

        func_for_sort = lambda x: -x['nIsotopes']-x['cos_cor_isotopes']+x['shift']*1e6

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
                        (cos_corr,
                                number_of_passed_isotopes,
                                _) = utils.checking_cos_correlation_for_carbon(
                                all_theoretical_int, all_exp_intensity, 0.6, allowed_shift=pep_feature['shift'])
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

        print('Number of detected isotope clusters: ', len(ready_final))


        # import pickle
        # pickle.dump(ready_final, open('/home/mark/ready_final.pickle', 'wb'))
        
        peptide_features = utils.calc_peptide_features(hills_dict, ready_final, negative_mode, faims_val, RT_dict, data_start_id)

        utils.write_output(peptide_features, args, write_header)

        write_header = False



        data_start_id += len(data_for_analyse_tmp)

        # break



