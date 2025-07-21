"""
    calc_null_test_ptes.py
    Written by ZBH, 7/21/2025

    Gathers code for analyzing the outputs of get_amp_null_test.py
    into a script (had previously been in a Jupyter notebook).

    Generates PTEs for each null test by calculating the T_null
    variable (eq. 5.15 in my disseration) for the real data and for
    all the sims, then calculates how many sims have a T_null that
    exceeds the real data. Saves the outputs to a text file and a PNG
    of the histogram of PTEs for all tested frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt

output_file_path = '/global/homes/z/zbh5/null_test_outputs/null_test_results_temporal_Feb2019_20250716/'
output_file_tag = 'temporal_Feb2019_20250716'

# Loading results files
sim_amps_split_one = np.load(output_file_path+'temporal_Feb2019_20250716_all_sims_split_one.npy',allow_pickle=True)
sim_amps_split_two = np.load(output_file_path+'temporal_Feb2019_20250716_all_sims_split_two.npy',allow_pickle=True)
real_amps_split_one = np.load(output_file_path+'temporal_Feb2019_20250716_real_split_one.npy',allow_pickle=True)
real_amps_split_two = np.load(output_file_path+'temporal_Feb2019_20250716_real_split_two.npy',allow_pickle=True)

n_sims = sim_amps_split_one.shape[0]
n_freqs = sim_amps_split_one.shape[1]

# Constructing T_null
real_diff = real_amps_split_one - real_amps_split_two
sims_diff = sim_amps_split_one - sim_amps_split_two
norm_factor = np.std(sims_diff,axis=0)**2 # Normalizing by std of sims for each freq

tnull_real = real_diff**2 / norm_factor
tnull_sims = np.empty(sims_diff.shape)
for i in range(n_sims):
    tnull_sims[i] = sims_diff[i]**2 / norm_factor

# Calculating PTE for each freq - checking how many sims exceed the real value
ptes = np.empty(n_freqs)

for i in range(n_freqs):
    ptes[i] = np.where(tnull_real[i] < tnull_sims[:,i])[0].size / n_sims

# Writing small output text file
with open(output_file_path+output_file_tag+'_pte_results.txt','w') as f:
    f.write(f"Number of freqs tested: {n_freqs}\n")
    f.write(f"Number of sims generated for PTE calculation: {n_sims}\n")
    f.write(f"Min PTE: {np.min(ptes)}; Max PTE: {np.max(ptes)}")

# Generate output histogram
plt.hist(ptes,bins=14,edgecolor='black',range=(0.0,1.0),
        label=f"Min PTE: {np.min(ptes)}; Max PTE: {np.max(ptes)}")
plt.title(f"PTEs {output_file_tag} {n_sims} Sims")
plt.grid()
plt.legend()
plt.savefig(output_file_path+output_file_tag+'_pte_hist.png', dpi=300)
plt.close()
