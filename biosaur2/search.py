from . import main, main_dia
import argparse
from copy import deepcopy
import logging
import os

def run():
    parser = argparse.ArgumentParser(
        description='A feature detection LC-MS1 spectra',
        epilog='''

    Example usage
    -------------
    $ biosaur2 input.mzML
    -------------
    ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('files', help='input mzML files', nargs='+')
    parser.add_argument('-mini', help='min intensity', default=1, type=float)
    parser.add_argument('-minmz', help='min mz', default=350, type=float)
    parser.add_argument('-maxmz', help='max mz', default=1500, type=float)
    parser.add_argument('-pasefmini', help='min intensity after combining hills in PASEF analysis', default=100, type=float)
    parser.add_argument('-htol', help='mass accuracy for hills in ppm', default=8, type=float)
    parser.add_argument('-itol', help='mass accuracy for isotopes in ppm', default=8, type=float)
    parser.add_argument('-paseftol', help='ion mobility accuracy for hills', default=0.05, type=float)
    parser.add_argument('-nm', help='negative mode. 1-true, 0-false', default=0, type=int)
    parser.add_argument('-o', help='path to output features file', default='')
    parser.add_argument('-hvf', help='Threshold to split hills into multiple if local minimum intensity multiplied by hvf is less than both surrounding local maximums', default=1.3, type=float)
    # parser.add_argument('-fdr', help='protein fdr filter in %%', default=1.0, type=float)
    # parser.add_argument('-i', help='minimum number of isotopes', default=2, type=int)
    parser.add_argument('-minlh', help='minimum length for hill', default=2, type=int)
    parser.add_argument('-pasefminlh', help='minimum length for pasef hill', default=1, type=int)
    # parser.add_argument('-ts', help='Two-stage RT training: 0 - turn off, 1 - turn one, 2 - turn on and use additive model in the first stage (Default)', default=2, type=int)
    # parser.add_argument('-sc', help='minimum number of scans for peptide feature', default=3, type=int)
    # parser.add_argument('-lmin', help='min length of peptides', default=7, type=int)
    # parser.add_argument('-lmax', help='max length of peptides', default=30, type=int)
    # parser.add_argument('-e', help='cleavage rule in quotes!. X!Tandem style for cleavage rules', default='[RK]|{P}')
    # parser.add_argument('-mc', help='number of missed cleavages', default=0, type=int)
    parser.add_argument('-cmin', help='min charge', default=1, type=int)
    parser.add_argument('-cmax', help='max charge', default=6, type=int)
    parser.add_argument('-nprocs', help='number of processes', default=4, type=int)
    # parser.add_argument('-fmods', help='fixed modifications. in mass1@aminoacid1,mass2@aminoacid2 format', default='57.021464@C')
    # parser.add_argument('-ad', help='add decoy', default=0, type=int)
    # parser.add_argument('-ml', help='use machine learning for PFMs', default=1, type=int)
    # parser.add_argument('-prefix', help='decoy prefix', default='DECOY_')
    # parser.add_argument('-nproc',   help='number of processes', default=1, type=int)
    # parser.add_argument('-elude', help='path to elude binary file. If empty, the built-in additive model will be used for RT prediction', default='')
    # parser.add_argument('-deeplc', help='path to deeplc', default='')
    # parser.add_argument('-deeplc_library', help='path to deeplc library', default='')
    # parser.add_argument('-pl', help='path to list of peptides for RT calibration', default='')
    parser.add_argument('-dia',  help='create mgf file for DIA MS/MS. Experimental', action='store_true')
    parser.add_argument('-diahtol', help='mass accuracy for DIA hills in ppm', default=25, type=float)
    parser.add_argument('-diaminlh', help='minimum length for dia hill', default=1, type=int)
    parser.add_argument('-mgf', help='path to output mgf file', default='')
    parser.add_argument('-debug', help='log debugging information', action='store_true')
    parser.add_argument('-tof', help='smart tof processing. Experimental', action='store_true')
    parser.add_argument('-profile', help='profile processing. Experimental', action='store_true')
    parser.add_argument('-write_hills', help='write tsv file with detected hills', action='store_true')
    # parser.add_argument('-diaitol', help='mass accuracy for DIA isotopes in ppm', default=25, type=float)
    args = vars(parser.parse_args())
    logging.basicConfig(format='%(levelname)9s: %(asctime)s %(message)s',
            datefmt='[%H:%M:%S]', level=[logging.INFO, logging.DEBUG][args['debug']])
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.debug('Starting with args: %s', args)

    if os.name == 'nt':
        # logger.info('Turning off multiprocessing for Windows system')
        args['nprocs'] = 1

    for filename in args['files']:
        logger.info('Starting file: %s', filename)
        if 1:
            args['file'] = filename
            main.process_file(deepcopy(args))
            logger.info('Feature detection is finished for file: %s', filename)
            if args['dia']:
                main_dia.process_file(deepcopy(args))
        
        # except Exception as e:
        #     logger.error(e)
        #     logger.error('Feature detection failed for file: %s', filename)

if __name__ == '__main__':
    run()
