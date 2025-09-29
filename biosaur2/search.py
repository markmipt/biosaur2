from . import main, main_dia
import argparse
from copy import deepcopy
import logging
import os


def run():
    parser = argparse.ArgumentParser(
        description="A feature detection LC-MS1 spectra",
        epilog="""

    Example usage
    -------------
    $ biosaur2 input.mzML
    -------------
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "files", help="input mzML or hills (Experimental) files", nargs="+"
    )
    parser.add_argument("-mini", help="min intensity", default=1, type=float)
    parser.add_argument("-minmz", help="min mz", default=350, type=float)
    parser.add_argument("-maxmz", help="max mz", default=1500, type=float)
    parser.add_argument(
        "-pasefmini",
        help="min intensity after combining hills in PASEF analysis",
        default=100,
        type=float,
    )
    parser.add_argument(
        "-htol", help="mass accuracy for hills in ppm", default=8, type=float
    )
    parser.add_argument(
        "-itol", help="mass accuracy for isotopes in ppm", default=8, type=float
    )
    parser.add_argument(
        "-ignore_iso_calib",
        help="Turn off accurate isotope error estimation",
        action="store_true",
    )
    parser.add_argument(
        "-use_hill_calib",
        help="Experimental. Turn on accurate hills error estimation",
        action="store_true",
    )
    parser.add_argument(
        "-paseftol", help="ion mobility accuracy for hills", default=0.05, type=float
    )
    parser.add_argument(
        "-nm", help="negative mode. 1-true, 0-false", default=0, type=int
    )
    parser.add_argument("-o", help="path to output features file", default="")
    parser.add_argument(
        "-iuse",
        help="Number of isotopes used for intensity calucation. 0 - only mono, -1 - use all, 1 - use mono and first isotope, etc.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-hvf",
        help="Threshold to split hills into multiple if local minimum intensity multiplied by hvf is less than both surrounding local maximums",
        default=1.3,
        type=float,
    )
    parser.add_argument(
        "-ivf",
        help="Threshold to split isotope pattern into multiple features if local minimum intensity multiplied by ivf is less right local maximum",
        default=5.0,
        type=float,
    )
    parser.add_argument("-minlh", help="minimum length for hill", default=2, type=int)
    parser.add_argument(
        "-pasefminlh", help="minimum length for pasef hill", default=1, type=int
    )
    parser.add_argument("-cmin", help="min charge", default=1, type=int)
    parser.add_argument("-cmax", help="max charge", default=6, type=int)
    parser.add_argument("-nprocs", help="number of processes", default=4, type=int)
    parser.add_argument(
        "-dia", help="create mgf file for DIA MS/MS. Experimental", action="store_true"
    )
    parser.add_argument(
        "-diahtol", help="mass accuracy for DIA hills in ppm", default=25, type=float
    )
    parser.add_argument(
        "-diaminlh", help="minimum length for dia hill", default=1, type=int
    )
    parser.add_argument("-diadynrange", help="diadynrange", default=1000, type=int)
    parser.add_argument("-min_ms2_peaks", help="min_ms2_peaks", default=5, type=int)
    parser.add_argument("-mgf", help="path to output mgf file", default="")
    parser.add_argument("-debug", help="log debugging information", action="store_true")
    parser.add_argument(
        "-tof", help="smart tof processing. Experimental", action="store_true"
    )
    parser.add_argument(
        "-profile", help="profile processing. Experimental", action="store_true"
    )
    parser.add_argument(
        "-write_hills", help="write tsv file with detected hills", action="store_true"
    )
    parser.add_argument(
        "-write_extra_details",
        help="write extra details for features",
        action="store_true",
    )
    parser.add_argument(
        "-combine_every",
        help="combine every n ms1 scans, useful for e.g. gas phase fractionation data",
        default=1,
        type=int,
    )
    args = vars(parser.parse_args())
    logging.basicConfig(
        format="%(levelname)9s: %(asctime)s %(message)s",
        datefmt="[%H:%M:%S]",
        level=[logging.INFO, logging.DEBUG][args["debug"]],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.debug("Starting with args: %s", args)

    if os.name == "nt":
        # logger.info('Turning off multiprocessing for Windows system')
        args["nprocs"] = 1

    for filename in args["files"]:
        logger.info("Starting file: %s", filename)
        if 1:
            args["file"] = filename
            main.process_file(deepcopy(args))
            logger.info("Feature detection is finished for file: %s", filename)
            if args["dia"]:
                main_dia.process_file(deepcopy(args))

        # except Exception as e:
        #     logger.error(e)
        #     logger.error('Feature detection failed for file: %s', filename)


if __name__ == "__main__":
    run()
