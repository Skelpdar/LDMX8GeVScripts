import uproot4 as uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

from scipy.stats import gaussian_kde

plt.rcParams.update({"svg.fonttype": 'none', "xtick.labelsize": 8})
plt.rcParams['svg.fonttype'] = 'none'

sigfile = uproot.open("sigAugust1.0.root")

sigscore = sigfile["BDTree/Score"].array(library="np")
sigmaxPE = sigfile["BDTree/maxPE"].array(library="np")
#sigtrack = sigfile["BDTree/passesTrackVeto"].array(library="np")

#sigmaxPE = sigmaxPE[sigscore > 0.998]
#sigscore = sigscore[sigscore > 0.998]
sigmaxPE = sigmaxPE[sigscore > 0.995]
sigscore = sigscore[sigscore > 0.995]

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3, sharey=False, figsize=(8,4), gridspec_kw={'width_ratios':[1,1,0.07]})
#fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3, sharey=False, figsize=(8,5))

bkgscore = []
bkgmaxPE = []
for array in uproot.iterate("ecalmumuAugust.root:BDTree", ["Score", "maxPE"], cut="(Score > 0.995) &  (maxPE < 500)"):
    for i in array["Score"]:
        bkgscore.append(i)
    for i in array["maxPE"]:
        if(i) == -1:
            bkgmaxPE.append(i)
        else:
            bkgmaxPE.append(i)
            


bkgscore2 = []
bkgmaxPE2 = []
for array in uproot.iterate("bkgEval.root:BDTree", ["Score", "maxPE"], cut="(Score > 0.995) &  (maxPE < 500)"):
    for i in array["Score"]:
        bkgscore2.append(i)
    for i in array["maxPE"]:
        if(i) == -1:
            bkgmaxPE2.append(i)
        else:
            bkgmaxPE2.append(i)

binsx = np.linspace(0,500,500)
binsy = np.linspace(0.995,1,25)

bins = [binsx,binsy]

#h = ax.hist2d(sigmaxPE,sigscore, bins=[300,100], norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap("jet"))
h = ax1.hist2d(sigmaxPE,sigscore, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap("jet"), rasterized=True)
h2 = ax2.hist2d(sigmaxPE,sigscore, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap("jet"), rasterized=True)

ax1.hlines(0.99814772605896,0,600,label="{\small BDT Cut}", color="black")
ax2.hlines(0.99814772605896,0,600,label="{\small BDT Cut}", color="black")
ax1.plot(bkgmaxPE, bkgscore, linewidth=0, marker='o', markersize=3.5, color="white", label="{\small ECal Muon Conversion}", markeredgecolor="black")
ax2.plot(bkgmaxPE2, bkgscore2, linewidth=0, marker='^', markersize=3.5, color="white", label="ECal PN", markeredgecolor="black")
ax1.plot([-10], [-10], linewidth=0, marker='^', markersize=3.5, color="white", label="{\small ECal PN}", markeredgecolor="black")

ax1.tick_params(axis="x",direction="in", top=True, bottom=True)
ax1.tick_params(axis="y",which="both",direction="in", left=True, right=True)
ax2.tick_params(axis="x",direction="in", top=True, bottom=True)
ax2.tick_params(axis="y",which="both",direction="in", left=True, right=True)

axins = ax1.inset_axes([0.5,0.25,0.47,0.47])
axins2 = ax2.inset_axes([0.5,0.25,0.47,0.47])

axins.hist2d(sigmaxPE,sigscore, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap("jet"))
axins2.hist2d(sigmaxPE,sigscore, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap("jet"))
#axins.hlines(0.99814772605896,0,600,label="BDT Cut", color="black")
axins.plot(bkgmaxPE, bkgscore, linewidth=0, marker='o', markersize=5, color="white", markeredgecolor="black")
axins2.plot(bkgmaxPE2, bkgscore2, linewidth=0, marker='^', markersize=5, color="white", markeredgecolor="black", label="{\small ECal PN}")

x1, x2, y1, y2 = 0, 8, 0.99814772605896, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1,y2)
#axins.set_yticks([0,8])
#axins.set_xticklabels('')
axins.set_yticklabels('')
axins.set_xticks([0,4,8])
axins.set_xticklabels(["0","4","8"])
#axins.set_xticks([0,4,8])
#axins.set_xticklabels(["0","4","8"])
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1,y2)
#axins.set_yticks([0,8])
#axins.set_xticklabels('')
axins2.set_yticklabels('')
axins2.set_xticks([])
axins2.set_xticklabels([])

axins.tick_params(axis="x",direction="in", top=True, bottom=True, pad=8)
axins.tick_params(axis="y",which="both",direction="in", left=False, right=False)
axins2.tick_params(axis="x",direction="in", top=True, bottom=True, pad=8)
axins2.tick_params(axis="y",which="both",direction="in", left=False, right=False)

ax1.indicate_inset_zoom(axins, edgecolor="red", linestyle='--', linewidth=1.5, alpha=0.7)
ax2.indicate_inset_zoom(axins2, edgecolor="red", linestyle='--', linewidth=1.5, alpha=0.7)

plt.title("{\small \\textbf{\\textit{LDMX}} {\\textit{Simulation}} Internal}", loc="right")

ax1.set_xlim((0,500))
ax2.set_xlim((0,500))
ax1.set_xlabel("{\small \\textbf{Maximum Photo-electrons}}")
ax2.set_xlabel("{\small \\textbf{Maximum Photo-electrons}}")
ax1.set_ylabel("{\small \\textbf{BDT Discriminator Value}}")
ax1.set_yticks([0.995,0.99814772605896,1.0])
ax2.set_yticks([])
ax1.set_yticklabels(["0.995","0.998...","1.0"])

#plt.colorbar()
leg = ax1.legend(loc=3, frameon=True, edgecolor=(0,0,0,1), fancybox=False, borderpad=0.6, handletextpad=0.5,fontsize=8)
leg.get_frame().set_boxstyle(rounding_size=0)

#plt.tight_layout(w_pad=0)

#fig.subplots_adjust(right=1)
#cbar_ax = fig.add_axes()

cbar = fig.colorbar(h[3], label="{\small \\textbf{Signal Rate (Arbitrary Units)}}", ticks=[1e0,1e1,1e2,1e3,1e4,1e5], cax=ax3)
cbar.ax.set_yticklabels(["\$10^{0}\$","\$10^{1}\$","\$10^{2}\$","\$10^{3}\$","\$10^{4}\$","\$10^{5}\$"])
cbar.set_label(label="{\small \\textbf{Signal Rate (Arbitrary Units)}}", labelpad=-25)

plt.tight_layout(w_pad=-16)

plt.savefig("scorevsPEdouble.png")

plt.savefig("scorevsPEdouble.svg", format="svg", dpi=300)
import subprocess
subprocess.run(["inkscape", "-D", "-z", "--file=scorevsPEdouble.svg", "--export-pdf=scorevsPEdouble.pdf", "--export-latex"])

subprocess.run(["sed","-i",'s/scorevsPEdouble.pdf/figures\/8GeV\/scorevsPEdouble.pdf/g',"scorevsPEdouble.pdf_tex"])

plt.show()
