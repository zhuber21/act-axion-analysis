###################################
# collect_assess_detection_npy_files.py
#
# Short python script to take the individual npy files 
# of amplitudes at the full range of frequencies 
# for each simulated null test split and combine them into a single
# output npy file. The index of the array is the sim number.
####################################

import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_to_files", help=
    "The absolute path to where the npy files are",type=str)
args = parser.parse_args()

path_to_files = args.path_to_files

# Group all outputs from split one
sim_paths = sorted(glob.glob(path_to_files+'*sim*_amplitudes.npy'))
first_fname = sim_paths[0].split('/')[-1]
output_tag = first_fname[:-24] 

all_sim_array = []

for i in range(len(sim_paths)):
    all_sim_array.append(np.load(sim_paths[i], allow_pickle=True))

# Assumes every file has the same number of sampled freqs (which they should)
all_sim_array = np.array(all_sim_array)

output_fname = path_to_files + output_tag + '_all_sims_amplitudes.npy'
np.save(output_fname, all_sim_array)

