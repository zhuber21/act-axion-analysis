###################################
# collect_npy_files.py
#
# Short python script to take the individual npy files
# for each depth-1 map and combine them into a single
# output npy file with the depth-1 map names as dictionary keys.
####################################

import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_to_files", help=
    "The absolute path to where the npy files are",type=str)
args = parser.parse_args()

path_to_files = args.path_to_files

paths = sorted(glob.glob(path_to_files+'*.npy'))
fnames = [path.split('/')[-1] for path in paths]

# Strips '_results.npy' off the end
# and the depth-1 name off the front.
# All that should be left is the tag I supplied at runtime.
output_tag = fnames[0][27:-12] 

input_dict = {}

for i in range(len(paths)):
    # Reconstructing the full map name to match serial output dictionary
    map_name = fnames[i][:27]+'map.fits'
    input_dict[map_name] = np.load(paths[i], allow_pickle=True).item()

#print(input_dict)
output_fname = path_to_files + 'angle_calc_results_' + output_tag + '.npy'
np.save(output_fname, input_dict)
