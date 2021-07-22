# LDMX8GeVScripts
LDMX Background Rejection Analysis at 8 GeV.

Maintained by Erik Wallin. 

## Eval
Evaluate the BDT score of ldmx-sw 1.7.0, v9-geometry data, vetoing only on ECal features. Note that you will have to provide your paths to the ldmx-sw shared libraries and header files. 

The evaluation macro takes the following arguments:
- TString Infilename
- TString Outfilename
- bool noise, whether more noise should be added (removes artificial noise ECal hits).
- double noiseFactor, increases the width of gaussian noise by this factor.
- bool completeOutput, when true all BDT variables will be written to the output file. Used to create training data for the BDT.
- bool reconstruct, when true it will ignore BDT variables in the input file and recalculates them again. 

## TODO
Add scripts to check for duplicate events.

