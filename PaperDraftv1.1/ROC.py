import numpy as np
import sklearn.metrics as metrics
import sys

import uproot4 as uproot

trig = float(sys.argv[4])

#bkgfile = uproot.open("../March2022/results/bkgEval.root")
bkgfile = uproot.open(sys.argv[1])
print("Loading score")
bkg = bkgfile["BDTree/Score"].array(library="np")[10000000:20000000]
bkgtrig = bkgfile["BDTree/frontEcal"].array(library="np")[10000000:20000000]
bkg = bkg[bkgtrig < trig]
print("Loaded bkg")

#sigfile = uproot.open("../March2022/results/sigEval1.0.root")
sigfile = uproot.open(sys.argv[2])
print("Loading score")
sig = sigfile["BDTree/Score"].array(library="np")
sigtrig = sigfile["BDTree/frontEcal"].array(library="np")
sig = sig[sigtrig < trig]
print("Loaded signal")

data = np.zeros((2,len(sig)+len(bkg)))

for k in range(0,len(sig)):
    data[1][k] = sig[k]
    data[0][k] = 1

for k in range(0,len(bkg)):
    data[1][len(sig)+k] = bkg[k]

label = data[0]
pred = data[1]

print(label)
print(pred)

###Adjust this x value if necessary depending on how certain the model should be on signal predictions.
###predLabel is the predicted label, either signal or background, 1 should be signal and 0 bakcground, but test that this is true by using a set containing only one of signal or background.
#predLabel = np.asarray([1 if x >0.5 else 0 for x in pred])
#accuracy = metrics.accuracy_score(label,predLabel)
#print("Accuracy: ",accuracy)
fpr, tpr, threshold = metrics.roc_curve(label, pred)
#print("Threshold: " + str(tpr))
for k in range(0,len(tpr)):
    #if tpr[k] >= 0.7:
    #    print("70% signal threshold: " + str(threshold[k]))
    #    break
    if tpr[k] >= 0.85:
        print("85% signal threshold: " + str(threshold[k]))
        np.save("threshold85",threshold[k])
        break

roc_auc = metrics.auc(fpr, tpr)
print("AUC: " + str(roc_auc))

print(fpr)
print("TPR")
for x in tpr:
    print(x)
print("threshold")
for x in threshold:
    print(x)

np.save(sys.argv[3]+"_fpr",np.array(fpr))
np.save(sys.argv[3]+"_tpr",np.array(tpr))
