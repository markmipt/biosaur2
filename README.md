biosaur2 - A feature detection LC-MS1 spectra. This project is a rewriten version of Biosaur software (https://github.com/abdrakhimov1/Biosaur).
-----------------------------------------------------------------------

The centroided mzML file is required for of the script.

Algorithm can be run with following command:

    biosaur2 path_to_MZML

The script output contains tsv table with peptide features.

All available arguments can be shown with command "biosaur2 -h".

The default parameter minlh (the minimal number of consecutive scans for peptide feature) is 1 and this value is optimimal for ultra-short LC gradients (a few minutes). For the longer LC gradients, this value can be increased for reducing of feature detection time and removing noise isotopic clusters.

For TOF data please add "-tof" argument.

For PASEF data please convert mzML file using msconvert and '--combineIonMobilitySpectra --filter "msLevel 1" ' options. Do not use option --filter "scanSumming"! The latter is often required for MS/MS data analysis but breaks MS1 feature detection. 

For negative mode data please add "-nm" argument.

Citing biosaur2
-------------------
Abdrakhimov, et al. Biosaur: An open-source Python software for liquid chromatography-mass spectrometry peptide feature detection with ion mobility support. https://doi.org/10.1002/rcm.9045

Installation
-------------
Using the pip:

    pip install biosaur2
    

Links
-----

- GitHub repo & issue tracker: https://github.com/markmipt/biosaur2
- Mailing list: markmipt@gmail.com
