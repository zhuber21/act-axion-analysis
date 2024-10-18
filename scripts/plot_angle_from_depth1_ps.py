# plot_angle_from_depth1_ps.py
# Script to plot the output results of get_angle_from_depth1_ps.py
# without rerunning the whole calculation - just loads output data products
# Does not reload maps or beams for plotting
# Check the options, output_dir_root, and data_timestamp before running!

import numpy as np
import os
import sys
import axion_osc_analysis_depth1_ps as aoa

# Deciding which plots to make
# This script does not reload maps, so there is no plot_maps option
# Also omitting beam and filtering tfunc plotting for now
# Currently will set these manually instead of reloading/altering config file or making these command line options
plot_all_spectra = False
plot_summary_spectra = False
#plot_likelihood = False            # Would need to reconstruct likelihood for each one and cycle through all maps
plot_angle_hist = True
# Should also add option for plotting timestreams

# Loading data from run
output_dir_root = "/home/zbh5/act_analysis/act_axion_analysis/results/"
if not os.path.exists(output_dir_root): # Make sure root path is right
    print("Output directory does not exist! Exiting.")
    sys.exit()
data_timestamp = '1727394152' # The 10-digit timestamp associated with a run of get_angle_from_depth1_ps.py - MAY NEED CHANGED EACH TIME!
output_dir_path = output_dir_root + "/angle_calc_" + data_timestamp + '/'
data_path = output_dir_path + 'angle_calc_' + data_timestamp + '_spectra.npy'
data_dict = np.load(data_path,allow_pickle=True).item()

if plot_all_spectra:
    print("Saving plots for all spectra")
    aoa.plot_spectra_individually(output_dir_path, data_dict)

if plot_summary_spectra:
    print("Saving spectra summary plots")
    aoa.plot_spectra_summary(output_dir_path, data_dict)

#if plot_likelihood:
#    pass # for future implementation if needed

if plot_angle_hist:
    maps = np.array(list(data_dict.keys()))
    angles = np.zeros(len(maps))
    for i in range(len(maps)):
        angles[i] = data_dict[maps[i]]['meas_angle']
    # Extracting angles from 
    print("Plotting histogram of angles")
    aoa.plot_angle_hist(output_dir_path, angles, maps)
