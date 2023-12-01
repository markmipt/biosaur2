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
    
Available parameters
-------------
-minlh: Minimum number of MS1 scans for peaks extracted from the mzML file. Optimal usually is in 1-3 range for 5-15 min LC gradients and 5-10 for 60-180 min gradients. Default = 2

-mini : Minimal intensity threshold for peaks extracted from the mzML file. Default = 1

-minmz : Minimal m/z value for peaks extracted from the mzML file. Default = 350

-maxmz : Maximal m/z value for peaks extracted from the mzML file. Default = 1500

-htol : Mass accuracy in ppm to combine peaks into hills between scans. Default = 8 ppm

-itol : Mass accuracy in ppm for isotopic hills. Default = 8 ppm

-ignore_iso_calib : Turn off accurate isotope error estimation if added as the parameter. Input "itol" value will be used instead of gaussian fitting of mass errors and systematic shifts for every isotope number.  

-o : Path to output feature files. Default is the name of the input mzML file with added “.features.tsv” mask stored in the folder of the original mzML file

-hvf: Threshold to split hills into multiple if local minimum intensity multiplied by hvf is less than both surrounding local maximums. All peaks after splitting must have at least max(2, minlh) MS1 scans. Default = 1.3

-ivf: Threshold to split isotope pattern into multiple features if local minimum intensity multiplied by ivf is less right local maximum. Local minimum position should be higher than max(4rd isotope, isotope position with maximum intensity according to averagine model). Default = 5.0

-nm : Negative mode. 1-true, 0-false. Affect only neutral mass column calculated in the output features table.  Default = 0

-cmin: Minimum allowed charge for isotopic clusters. Default = 1

-cmax: Maximal allowed charge for isotopic clusters. Default = 6

-nprocs: Number of processes used by biosau2. Automatically set to 1 for Windows system due to multiprocessing issues. Default = 4

-write_hills: Add hills output if added as the parameter

-paseminlh: For TIMS-TOF data. Minimum number of ion mobility values for m/z peaks to be kept in the analysis. Default = 1

-paseftol: For TIMS-TOF data. Ion mobility tolerance used to combine close peaks into a single one. Default = 0.05

-pasefmini: For TIMS-TOF data. Minimal intensity threshold for peaks after combining peaks with close m/z (itol option) and ion mobility (paseftol option) values. Default = 100

-tof: Experimental. If added as the parameter, biosaur2 estimates noise intensity distribution across m/z range and automatically calculates intensity cutoffs for different m/z value ranges. This is an alternative way to reduce noise to the "-mini" option which is a fixed intensity threshold for all m/z values. Can be usefull for TOF data

    

Links
-----

- GitHub repo & issue tracker: https://github.com/markmipt/biosaur2
- Mailing list: markmipt@gmail.com
