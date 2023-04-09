import numpy as np
import sklearn.metrics as metrics
import sys

import uproot4 as uproot

#filename = "../March2023/results/bkgEval.root"
#filename = "../March2023/ecalmumu/ecalmumu.root"
#filename = "../March2023/targetgammamumu/targetgammamumu.root"
filename = "../March2023/results/targetPN.root"

#filename = "../March2023/signalEval/sigEvalAugust0.001.root"
#filename = "../March2023/signalEval/sigEvalAugust0.01.root"
#filename = "../March2023/signalEval/sigEvalAugust0.1.root"
#filename = "../March2023/signalEval/sigEvalAugust1.0.root"
print(filename)
bkgfile = uproot.open(filename)

print("Loading score")
frontEcal = bkgfile["BDTree/frontEcal"].array(library="np")
summedEcal = bkgfile["BDTree/summedDet"].array(library="np")
bkg = bkgfile["BDTree/Score"].array(library="np")
track = bkgfile["BDTree/passesTrackVeto"].array(library="np")
weight = bkgfile["BDTree/eventWeight"].array(library="np")
hcal = bkgfile["BDTree/maxPE"].array(library="np")

print("Loaded bkg")

i = 0
n = 0
m = 0
l = 0
p = 0

eventsum = 0
for k in range(0,len(bkg)):
    eventsum += weight[k]*550
print("Total weight")
print(eventsum)

print("Bkg events:")
print(len(bkg))
for k in range(0,len(bkg)):
    if k % 1e6 == 0:
        print(k)
    if(frontEcal[k] < 3160):
        i += weight[k]*550
        if(summedEcal[k] < 3160):
            p += weight[k]*550
            if(track[k]):
                n += weight[k]*550
                #if(bkg[k] > 0.9998927116394043):
                #August cut
                if(bkg[k] > 0.99814772605896):
                    m += weight[k]*550
                    #m += 1
                    if(hcal[k] < 8):
                        l += weight[k]*550

print("trig: " + str(i))
print("summed: " + str(p))
print("track veto:" + str(n))
print("0.99995 BDT: " + str(m))
print("hcal: " + str(l))
