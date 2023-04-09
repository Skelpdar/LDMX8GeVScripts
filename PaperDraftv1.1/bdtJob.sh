#!/bin/bash

#SBATCH -t 00:1440:00
#SBATCH -A LU2021-2-100
#SBATCH --mem-per-cpu=10000

#SBATCH -N 1
#SBATCH --tasks-per-node=1

source /ldmx/libs/ldmx-sw-install/bin/ldmx-setup-env.sh

cd bdtjobs/

python bdtMakerSeg2.py --tree_number 1600 --eta 0.039 --depth 14 --bkg_file train/bkgBig.root --sig_file train/signal.root --out_name august19
