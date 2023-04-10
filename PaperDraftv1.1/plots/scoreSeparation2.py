import numpy as np
#from math import floor
#import random
#from optparse import OptionParser
#import math as math
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as metrics

import uproot

#plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"svg.fonttype": 'none'})
plt.rcParams['svg.fonttype'] = 'none'

###Save the trained BDT models and their associated .npy files of predictions and truth values for signal and background in a subdirectory of the directory that this file (bdtROC.py) is saved in.
###Then, I checked the accuracies and AUCs of all of the models by running the following command in the directory where the .npy files are:
### find ./ -type f -name "*.npy" -exec python ../bdtROC.py --file "{}" \; &> termout.txt
###This will output a text file called termout.txt that contains columns with the accuracies, AUCs, and names of all trained models in the directory.
###The most accurate or highest AUC can then be found by sorting according to the desired value.
###The ROC curves themselves can also be plotted and saved here.

###This will take the desired .npy as input
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--file', dest='file', default='bdttrain/signal_bdttrain_tree.root', help='name of signal file')
parser.add_option('--bkg', dest='bkg')
parser.add_option('--dir', dest='dir')

(options, args) = parser.parse_args()

fig=plt.figure()

#bkgfile = uproot.open("bkgEval.root")
#bkg = bkgfile["BDTree/Score"].array()[:int(1e7)]

bkg = np.load("march2023hist_bkg.npy",allow_pickle=True)[0]
bkg = bkg/np.sum(bkg)
sig1_0 = np.load("march2023hist_1.0.npy",allow_pickle=True)[0]
sig1_0 = sig1_0/np.sum(sig1_0)
sig0_1 = np.load("march2023hist_0.1.npy",allow_pickle=True)[0]
sig0_1 = sig0_1/np.sum(sig0_1)
sig0_01 = np.load("march2023hist_0.01.npy",allow_pickle=True)[0]
sig0_01 = sig0_01/np.sum(sig0_01)
sig0_001 = np.load("march2023hist_0.001.npy",allow_pickle=True)[0]
sig0_001 = sig0_001/np.sum(sig0_001)
edges = np.load("march2023hist_bkg.npy",allow_pickle=True)[1]
print(bkg)

#sig10file = uproot.open("sigEval1.0.root")
#sig10 = sig10file["BDTree/Score"].array()
#sig01file = uproot.open("sigEval0.1.root")
#sig01 = sig01file["BDTree/Score"].array()
#sig001file = uproot.open("sigEval0.01.root")
#sig001 = sig01file["BDTree/Score"].array()
#sig0001file = uproot.open("sigEval0.001.root")
#sig0001 = sig0001file["BDTree/Score"].array()
plt.yscale("log")
bins = np.linspace(0,1,100)
bins[0] = -0.05
bins[-1] = 1.05
#plt.hist(bkg,bins=bins, histtype="step", linewidth=3, color="#990000")
#plt.bar(bins[:-1], bkg/np.sum(bkg)*99, linewidth=3, color="#990000", width=1/99)
left,right = edges[:-1],edges[1:]
X = np.array([left,right]).T.flatten()
Y = np.array([bkg,bkg]).T.flatten()
plt.plot(X,Y, linewidth=3,color="#990000")

plt.plot([-9999],[-9999],label=r"{\small Photo-nuclear}", linewidth=3, color="#990000")

#plt.hist(sig10,bins=bins, density=True, histtype="step",  linewidth=2, color="#006600", linestyle="-")
Y = np.array([sig1_0,sig1_0]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#006600")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1000\$ MeV}", linewidth=2, color="#006600", linestyle="-")

#plt.hist(sig01,bins=bins, density=True, histtype="step",  linewidth=2, color="#6600cc", linestyle="--")
Y = np.array([sig0_1,sig0_1]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#6600cc", linestyle="--")

plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 100\$ MeV}", linewidth=2, color="#6600cc", linestyle="--")

#plt.hist(sig001,bins=bins, density=True, histtype="step",  linewidth=2, color="#ff9900",linestyle="-.")
Y = np.array([sig0_01,sig0_01]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#ff9900", linestyle="-.")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 10\$ MeV}", linewidth=2, color="#ff9900",linestyle="-.")

#plt.hist(sig0001,bins=bins, density=True, histtype="step",  linewidth=2, color="#0066cc", linestyle=":")
Y = np.array([sig0_001,sig0_001]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#0066cc", linestyle=":")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1\$ MeV}", linewidth=2, color="#0066cc", linestyle=":")
plt.ylabel("{\small \\textbf{Event Fraction}}", labelpad=-20)
plt.xlabel("{\small \\textbf{BDT Discriminator Value}}", labelpad=10)
plt.xlim([0,1])
axs = fig.get_axes()
plt.yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
axs[0].set_yticklabels(["\$10^{0}\$","\$10^{-1}\$","\$10^{-2}\$","\$10^{-3}\$","\$10^{-4}\$","\$10^{-5}\$"])
plt.ylim(3e-6,1)
plt.legend(frameon=False,borderaxespad=3, labelspacing=1.5)
plt.title("{\small \\textbf{\\textit{LDMX}} {\\textit{Simulation}} Internal}",loc="right")

axs[0].tick_params(axis="x",direction="in", top=True, bottom=True, pad=8)
axs[0].tick_params(axis="y",which="both",direction="in", left=True, right=True)

plt.savefig("ScoreSeparation.svg", format="svg")

import subprocess
subprocess.run(["inkscape", "-D", "-z", "--file=ScoreSeparation.svg", "--export-pdf=ScoreSeparation.pdf", "--export-latex"])
subprocess.run(["sed","-i",'s/ScoreSeparation.pdf/figures\/8GeV\/ScoreSeparation.pdf/g',"ScoreSeparation.pdf_tex"])

plt.show()
