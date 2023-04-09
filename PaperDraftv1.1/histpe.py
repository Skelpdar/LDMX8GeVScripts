import numpy as np
import sklearn.metrics as metrics
import sys

import uproot4 as uproot

bkgfile = uproot.open(sys.argv[1])
print("Loading score")
bkg = bkgfile["BDTree/maxPE"].array(library="np")
print("Loaded bkg")

trig = bkgfile["BDTree/frontEcal"].array(library="np")
bkg = bkg[trig < 3160]

bkg = np.clip(bkg,-100,99.5)

s = np.sum(bkg)

bins = np.linspace(0,500,500+1)
hist = np.histogram(bkg,bins=bins)
#hist[0] = hist[0]/s
np.save(sys.argv[2], hist, allow_pickle=True)
print(hist)
