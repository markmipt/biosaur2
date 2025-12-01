from . import utils
from . import main
import itertools
from os import path
import pandas as pd
import ast
import math
import logging
from .cutils import get_initial_isotopes, checking_cos_correlation_for_carbon, split_peaks, split_peaks_old, detect_hills, process_hills, get_and_calc_values_for_cos_corr, cos_correlation, get_and_calc_apex_intensity_and_scan
import numpy as np

logger = logging.getLogger(__name__)


def process_file(args):

    input_mzml_path = args['file']
    basename_mzml = path.basename(input_mzml_path)

    if args['o']:
        output_file = args['o']
    else:
        output_file = path.splitext(input_mzml_path)[0]\
            + path.extsep + 'features.tsv'

    df1_features = pd.read_table(output_file)
    # df1_features['mono_hills_scan_lists'] = df1_features['mono_hills_scan_lists'].apply(lambda x: ast.literal_eval(','.join(x.split(' '))))
    # df1_features['mono_hills_scan_lists'] = df1_features['mono_hills_scan_lists'].apply(lambda x: x.replace('[', '').replace(']', '').split(' '))#ast.literal_eval(','.join(x.split(' '))))
    df1_features['mono_hills_scan_lists'] = df1_features['mono_hills_scan_lists'].apply(lambda x: ast.literal_eval(x))
    df1_features['mono_hills_scan_sets'] = df1_features['mono_hills_scan_lists'].apply(lambda x: set(x))
    df1_features['mono_hills_intensity_list'] = df1_features['mono_hills_intensity_list'].apply(lambda x: ast.literal_eval(x))

    def calc_idict(x):
        hill_idict_1 = dict()
        for scan_id_val, intensity_val in zip(x['mono_hills_scan_lists'], x['mono_hills_intensity_list']):
            hill_idict_1[scan_id_val] = intensity_val
        return hill_idict_1

    def calc_sqrt_of_i(x):
        hill_sqrt_of_i_1 = math.sqrt(sum(v**2 for v in x['hill_idict_1'].values()))
        return hill_sqrt_of_i_1

    df1_features['hill_idict_1'] = df1_features.apply(calc_idict, axis=1)
    df1_features['hill_sqrt_of_i_1'] = df1_features.apply(calc_sqrt_of_i, axis=1)

    df1_features = df1_features.sort_values(by='rtEnd').reset_index(drop=True)

    logger.debug(df1_features.head())


    if args['mgf']:
        outmgf_name = args['mgf']
    else:
        outmgf_name = path.splitext(input_mzml_path)[0]\
            + path.extsep + 'mgf'

    outmgf = open(outmgf_name, 'w')
    t_i = 1


    data_for_analyse, ms1_count, ms2_count = utils.process_mzml_dia(args)

    isolation_target_func = lambda x: x['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z']
    isolation_window_func = lambda x: x['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset']

    #Process faims
    faims_set = set([z.get('FAIMS compensation voltage', None) for z in data_for_analyse])
    if any(z is not None for z in faims_set):
        logger.info('Detected FAIMS values: %s', faims_set)

    #Process windows
    windows_set = set([isolation_target_func(z) for z in data_for_analyse])
    logger.info('Detected windows values: %s', windows_set)

    ms1_to_ms2_in_cycle_koef = int(ms1_count / (ms2_count / len(windows_set)))
    logging.info('MS1 to MS2 scans ratio per full cycle: %d', ms1_to_ms2_in_cycle_koef)

    RT_dict = dict()

    for (faims_val, window_val) in itertools.product(faims_set, windows_set):

        isolation_window = False
        data_cur_id = 0

        data_for_analyse_tmp = []
        for z in data_for_analyse:
            if faims_val is None or z['FAIMS compensation voltage'] == faims_val:
                if isolation_target_func(z) == window_val:
                    data_for_analyse_tmp.append(z)
                    RT_dict[data_cur_id] = float(z['scanList']['scan'][0]['scan start time'])
                    data_cur_id += 1

                    if isolation_window is False:
                        isolation_window = isolation_window_func(z)

        hill_mass_accuracy = args['diahtol']
        max_mz_value = 0
        for z in data_for_analyse_tmp:
            max_mz_value = max(max_mz_value, z['m/z array'].max())

        mz_step = hill_mass_accuracy * 1e-6 * max_mz_value

        #Process ion mobility

        if all('ignore_ion_mobility' not in z for z in data_for_analyse_tmp):
            logger.debug('%d %d', sum(('ignore_ion_mobility' in z for z in data_for_analyse_tmp)), len(data_for_analyse_tmp))
            utils.centroid_pasef_data(data_for_analyse_tmp, args, mz_step)
        else:
            args['paseftol'] = False

        paseftol = args['paseftol']
        diadynrange = args['diadynrange']

        hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol, dia=True)

        hills_dict = main.split_peaks_multi(hills_dict, data_for_analyse_tmp, args['hvf'], args)

        mz_step = isolation_window

        hills_dict = process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args, dia=True)

        num_hills = len(set(hills_dict['hills_idx_array']))

        logger.info('All data converted to %d hills...', num_hills)


        all_hills_idx = list(range(len(hills_dict['hills_idx_array_unique'])))
        # for idx_2 in all_hills_idx:
        #     hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, idx_2)
        # idx_sorted_by_intensity = np.argsort(hills_dict['hills_intensity_apex'])[::-1]
        idx_sorted_by_rt_start = np.argsort([hills_dict['hills_scan_lists'][idx_2][0] for idx_2 in all_hills_idx])
        last_scans_ms2 = np.array([hills_dict['hills_scan_lists'][idx_2][-1] for idx_2 in all_hills_idx])[idx_sorted_by_rt_start]

        diaminlh = int(args['diaminlh'])

        logger.info('All Apexes are calculated')

        if num_hills:

            df1_features_in_window = df1_features[df1_features['mz'].apply(lambda x: window_val - isolation_window <= x <= window_val + isolation_window)]


            for ms1_mz, ms1_I, ms1_RT, ms1_ch, hill_scans_1, hill_scans_1_list, ms1_intensities, hill_idict_1, hill_sqrt_of_i_1 in df1_features_in_window[['mz', 'intensityApex', 'rtApex','charge', 'mono_hills_scan_sets', 'mono_hills_scan_lists', 'mono_hills_intensity_list', 'hill_idict_1', 'hill_sqrt_of_i_1']].values:



                # if window_val - isolation_window <= ms1_mz <= window_val + isolation_window:

                if ms1_to_ms2_in_cycle_koef != 1:
                    hill_scans_1 = set(z/ms1_to_ms2_in_cycle_koef for z in hill_scans_1 if z % ms1_to_ms2_in_cycle_koef == 0)
                    hill_scans_1_list = list(hill_scans_1)
                    hill_idict_1_tmp = dict()
                    for z in hill_scans_1:
                        hill_idict_1_tmp[z] = hill_idict_1[z*ms1_to_ms2_in_cycle_koef]
                    hill_idict_1 = hill_idict_1_tmp
                    hill_sqrt_of_i_1 = math.sqrt(sum(v**2 for v in hill_idict_1.values()))

                hill_scans_1_list_first, hill_scans_1_list_last = hill_scans_1_list[0], hill_scans_1_list[-1]

                tmp_candidates = []
                intensity_threshold = 0

                # start_pos = 0
                # for start_pos in last_scans_ms2:

                # for idx_2, hill_idx_2 in enumerate(hills_dict['hills_idx_array_unique']):
                # for idx_2 in idx_sorted_by_intensity:
                for idx_2 in idx_sorted_by_rt_start[last_scans_ms2 >= hill_scans_1_list_first]:

                    hill_scans_2_list = hills_dict['hills_scan_lists'][idx_2]
                    hill_scans_2_list_first, hill_scans_2_list_last = hill_scans_2_list[0], hill_scans_2_list[-1]

                    # if hill_scans_2_list_last < hill_scans_1_list_first:
                    #     pass

                    if hill_scans_1_list_last < hill_scans_2_list_first:
                        break
                # for idx_2 in hills_dict['hills_mz_median_fast_dict'][m_to_check_fast]:

                    # if paseftol is False or idx_2 in hills_dict['hills_im_median_fast_dict'][im_to_check_fast]:

                        # series_2 = idx_2

                    # if diadynrange and intensity_threshold:
                    #     if hills_dict['hills_intensity_apex'][idx_2] < intensity_threshold:
                    #         break

                    # if not hill_scans_1_list_last < hill_scans_2_list_first and not hill_scans_2_list_last < hill_scans_1_list_first:
                    hill_scans_2 = hills_dict['hills_scan_sets'][idx_2]
                    if len(hill_scans_1.intersection(hill_scans_2)) >= diaminlh:


                        hills_dict, hill_idict_2, hill_sqrt_of_i_2 = get_and_calc_values_for_cos_corr(hills_dict, idx_2)


                        # cos_cor_RT = 1.0
                        cos_cor_RT = cos_correlation(hill_scans_1, hill_idict_1, hill_sqrt_of_i_1, hill_scans_2, hill_idict_2, hill_sqrt_of_i_2)


                        if cos_cor_RT >= 0.6:

                            hill_mz_2 = hills_dict['hills_mz_median'][idx_2]

                            # hill_idx_2 = hills_dict['hills_idx_array_unique'][idx_2]

                            hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, idx_2)

                            local_isotopes_dict = {
                                'm/z': hill_mz_2,
                                'intensity': hills_dict['hills_intensity_apex'][idx_2],
                            }

                            # if diadynrange and not intensity_threshold:
                            #     intensity_threshold = hills_dict['hills_intensity_apex'][idx_2] / diadynrange

                            tmp_candidates.append(local_isotopes_dict)

                if len(tmp_candidates) >= args['min_ms2_peaks']:
                    outmgf.write('BEGIN IONS\n')
                    outmgf.write('TITLE=%s.%d.%d.%d\n' % (basename_mzml, t_i, t_i, ms1_ch))
                    outmgf.write('RTINSECONDS=%f\n' % (ms1_RT * 60, ))
                    outmgf.write('PEPMASS=%f %f\n' % (ms1_mz, ms1_I))
                    outmgf.write('CHARGE=%d+\n' % (ms1_ch, ))
                    outmgf.write('MS1Intensity=%f\n' % (ms1_I, ))
                    for local_fragment in tmp_candidates:
                        outmgf.write('%f %f\n' % (local_fragment['m/z'], local_fragment['intensity']))
                    outmgf.write('END IONS\n\n')
                    t_i += 1


        logger.info('Chunk ready')