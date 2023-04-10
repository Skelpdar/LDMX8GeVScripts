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

bkg = np.load("march2023hist_bkg_pe.npy",allow_pickle=True)[0]
bkg = bkg/np.sum(bkg)
sig1_0 = np.load("march2023hist_1.0_pe.npy",allow_pickle=True)[0]
sig1_0 = sig1_0/np.sum(sig1_0)
sig0_1 = np.load("march2023hist_0.1_pe.npy",allow_pickle=True)[0]
sig0_1 = sig0_1/np.sum(sig0_1)
sig0_01 = np.load("march2023hist_0.01_pe.npy",allow_pickle=True)[0]
sig0_01 = sig0_01/np.sum(sig0_01)
sig0_001 = np.load("march2023hist_0.001_pe.npy",allow_pickle=True)[0]
sig0_001 = sig0_001/np.sum(sig0_001)
edges = np.load("march2023hist_bkg_pe.npy",allow_pickle=True)[1]
print(bkg)

bkg4 = np.load("march2023hist_bkg_pe_4GeV.npy",allow_pickle=True)[0]
bkg4 = bkg4/np.sum(bkg4)
sig1_04 = np.load("march2023hist_1.0_pe_4GeV.npy",allow_pickle=True)[0]
sig1_04 = sig1_04/np.sum(sig1_04)
sig0_14 = np.load("march2023hist_0.1_pe_4GeV.npy",allow_pickle=True)[0]
sig0_14 = sig0_14/np.sum(sig0_14)
sig0_014 = np.load("march2023hist_0.01_pe_4GeV.npy",allow_pickle=True)[0]
sig0_014 = sig0_014/np.sum(sig0_014)
sig0_0014 = np.load("march2023hist_0.001_pe_4GeV.npy",allow_pickle=True)[0]
sig0_0014 = sig0_0014/np.sum(sig0_0014)
edges4 = np.load("march2023hist_bkg_pe_4GeV.npy",allow_pickle=True)[1]

#sig10file = uproot.open("sigEval1.0.root")
#sig10 = sig10file["BDTree/Score"].array()
#sig01file = uproot.open("sigEval0.1.root")
#sig01 = sig01file["BDTree/Score"].array()
#sig001file = uproot.open("sigEval0.01.root")
#sig001 = sig01file["BDTree/Score"].array()
#sig0001file = uproot.open("sigEval0.001.root")
#sig0001 = sig0001file["BDTree/Score"].array()
plt.yscale("log")
bins = np.linspace(0,500,501)
bins[0] = -0.5
bins[-1] = 500.5
#plt.hist(bkg,bins=bins, histtype="step", linewidth=3, color="#990000")
#plt.bar(bins[:-1], bkg/np.sum(bkg)*99, linewidth=3, color="#990000", width=1/99)
left,right = edges[:-1],edges[1:]
X = np.array([left,right]).T.flatten()
Y = np.array([bkg,bkg]).T.flatten()

plt.vlines(8,1e-7,1000, linestyle=":",color="black", label="{\small HCal veto cut}")

plt.plot(X,Y, linewidth=3,color="#990000", linestyle="solid")

X = np.array([left,right]).T.flatten()
Y = np.array([bkg4,bkg4]).T.flatten()
plt.plot(X,Y, linewidth=3,color="#990000", linestyle="dashed")

plt.plot([-9999],[-9999],label=r"{\small Photo-nuclear, 8 GeV}", linewidth=3, color="#990000")

#plt.hist(sig10,bins=bins, density=True, histtype="step",  linewidth=2, color="#006600", linestyle="-")
Y = np.array([sig1_0,sig1_0]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#006600", linestyle="solid")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1000\$ MeV, 8 GeV}", linewidth=2, color="#006600", linestyle="solid")

#plt.hist(sig01,bins=bins, density=True, histtype="step",  linewidth=2, color="#6600cc", linestyle="--")
Y = np.array([sig0_1,sig0_1]).T.flatten()
#plt.plot(X,Y, linewidth=2,color="#6600cc", linestyle="--")

#plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 100\$ MeV}", linewidth=2, color="#6600cc", linestyle="--")

#plt.hist(sig001,bins=bins, density=True, histtype="step",  linewidth=2, color="#ff9900",linestyle="-.")
Y = np.array([sig0_01,sig0_01]).T.flatten()
#plt.plot(X,Y, linewidth=2,color="#ff9900", linestyle="-.")
#plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 10\$ MeV}", linewidth=2, color="#ff9900",linestyle="-.")

#plt.hist(sig0001,bins=bins, density=True, histtype="step",  linewidth=2, color="#0066cc", linestyle=":")
Y = np.array([sig0_001,sig0_001]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#0066cc", linestyle="solid")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1\$ MeV, 8 GeV}", linewidth=2, color="#0066cc", linestyle="solid")

#4 GeV

plt.plot([-9999],[-9999],label=r"{\small Photo-nuclear 4 GeV}", linewidth=3, color="#990000", linestyle="dashed")

#plt.hist(sig10,bins=bins, density=True, histtype="step",  linewidth=2, color="#006600", linestyle="-")
Y = np.array([sig1_04,sig1_04]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#006600", linestyle="dashed")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1000\$ MeV, 4 GeV}", linewidth=2, color="#006600", linestyle="dashed")

#plt.hist(sig01,bins=bins, density=True, histtype="step",  linewidth=2, color="#6600cc", linestyle="--")
Y = np.array([sig0_14,sig0_14]).T.flatten()
#plt.plot(X,Y, linewidth=2,color="#6600cc", linestyle="--")
#plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 100\$ MeV, 4GeV}", linewidth=2, color="#6600cc", linestyle="--")

#plt.hist(sig001,bins=bins, density=True, histtype="step",  linewidth=2, color="#ff9900",linestyle="-.")
Y = np.array([sig0_014,sig0_014]).T.flatten()
#plt.plot(X,Y, linewidth=2,color="#ff9900", linestyle="-.")
#plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 10\$ MeV, 4 GeV}", linewidth=2, color="#ff9900",linestyle="-.")

#plt.hist(sig0001,bins=bins, density=True, histtype="step",  linewidth=2, color="#0066cc", linestyle=":")
Y = np.array([sig0_0014,sig0_0014]).T.flatten()
plt.plot(X,Y, linewidth=2,color="#0066cc", linestyle="dashed")
plt.plot([-9999],[-9999],label=r"{\small \$m_{A'} = 1\$ MeV, 4 GeV}", linewidth=2, color="#0066cc", linestyle="dashed")

plt.ylabel("{\small \\textbf{Event Fraction}}", labelpad=-20)
plt.xlabel("{\small \\textbf{Maximum Photo-electrons in an HCal Module}}", labelpad=10)


plt.xlim([0,100])
axs = fig.get_axes()
plt.yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
plt.xticks([0,8,20,40,60,80,100])
axs[0].set_yticklabels(["\$10^{0}\$","\$10^{-1}\$","\$10^{-2}\$", "\$10^{-3}\$", "\$10^{-4}\$","\$10^{-5}\$","\$10^{-6}\$"])
plt.ylim(5e-6,1)
plt.legend(frameon=False,borderaxespad=1)
plt.title("{\small \\textbf{\\textit{LDMX}} {\\textit{Simulation}} Internal}",loc="right")

axs[0].tick_params(axis="x",direction="in", top=True, bottom=True, pad=8)
axs[0].tick_params(axis="y",which="both",direction="in", left=True, right=True)

plt.savefig("peSeparationBoth.svg", format="svg")

import subprocess
subprocess.run(["inkscape", "-D", "-z", "--file=peSeparationBoth.svg", "--export-pdf=peSeparationBoth.pdf", "--export-latex"])
subprocess.run(["sed","-i",'s/peSeparationBoth.pdf/figures\/8GeV\/peSeparationBoth.pdf/g',"peSeparationBoth.pdf_tex"])

#plt.show()
