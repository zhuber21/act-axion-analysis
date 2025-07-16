"""
    get_amp_null_test.py
    Written by ZBH, 7/14/2025

    Performs cuts on f150 and f090 timestreams, combines results
    into one timestream, splits it with indices from supplied npy
    files, and calculates best fit amplitude for many frequencies
    for each split for the real data and many sims.
    
    Currently hardcoded to use the angles and errorbars from the FWHM method
"""

import numpy as np
from scipy import stats
import yaml
import argparse
import time
import os
import logging
from mpi4py import MPI
from act_axion_analysis import timestream_analysis as ta

# Setting up MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("config_file", help=
    "The name of the YAML config file",type=str)
parser.add_argument("output_dir_tag", help=
    "Tag to put in output directory/file names. \
    E.g. null_test_results_<output_dir_tag> will be output directory",type=str)
args = parser.parse_args()

#######################################################################################
# Loading all preliminaries 

# Load in the YAML file
yaml_name = args.config_file

with open(yaml_name, 'r') as file:
    config = yaml.safe_load(file)

# Output file path
output_tag = args.output_dir_tag
output_dir_root = config['output_dir_root']
if not os.path.exists(output_dir_root): # Make sure root path is right
    print("Output directory does not exist! Exiting.")
    raise OSError(f"Directory not found: {output_dir_root}")
output_dir_path = output_dir_root + "null_test_results_" + output_tag + '/'
if not os.path.exists(output_dir_path): # Make new folder for this run - should be unique
    # Ignoring race condition where one process makes the directory before a handful of others
    # As long as one of them makes it at basically the same synchonicity as the others, all should be well
    try:
        os.makedirs(output_dir_path)
    except FileExistsError:
        pass

# Setting up logger - making a separate one for each process in output_dir/log/
logger = logging.getLogger(__name__)
if not os.path.exists(output_dir_path + 'log/'):
    # Ignoring race condition where one process makes the directory before a handful of others
    try:
        os.makedirs(output_dir_path + 'log/')
    except FileExistsError:
        pass
log_filename = output_dir_path+'log/process{:02d}_run.log'.format(rank)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', style='{',
                    format='{asctime} {levelname} {filename}:{lineno}: {message}',
                    handlers=[logging.FileHandler(filename=log_filename)]
                    )
logger.info(f"Using config file: {yaml_name}")

# Setting cut variables from the config file
errorbar_cut_threshold = config['errorbar_cut_threshold']
peak_cut_threshold = config['peak_cut_threshold']
fwhm_cut_threshold = config['fwhm_cut_threshold']
duration_cut_threshold  = config['duration_cut_threshold']

# Setting frequency, amplitude, and phase variables from the config file
num_days_max_freq_period = config['num_days_max_freq_period']
min_freq_period_factor = config['min_freq_period_factor']
freq_oversample_factor = config['freq_oversample_factor']
min_amplitude = config['min_amplitude']
max_amplitude = config['max_amplitude']
amplitude_step = config['amplitude_step']
min_phase_deg = config['min_phase_deg']
max_phase_deg = config['max_phase_deg']
phase_step_deg = config['phase_step_deg']

# Number of sims
n_sims = config['n_sims']

# Setting input file variables
f150_result_file = config['f150_result_file']
f090_result_file = config['f090_result_file'] 
split_one_indices_file = config['split_one_indices']
split_two_indices_file = config['split_two_indices']

# Checking that input files exist
if not os.path.exists(f150_result_file): 
    logger.error("Cannot find f150 results npy file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {f150_result_file}")
if not os.path.exists(f090_result_file): 
    logger.error("Cannot find f090 results npy file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {f090_result_file}")
if not os.path.exists(split_one_indices_file): 
    logger.error("Cannot find split one indices npy file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {split_one_indices_file}")
if not os.path.exists(split_two_indices_file): 
    logger.error("Cannot find split two indices npy file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {split_two_indices_file}")

# Load indices for each split - calculated separately in one of the generate_*_indices.py scripts (one per test)
# Don't need the .item() at the end for these since they are arrays instead of dicts
split_one_idxs = np.load(split_one_indices_file,allow_pickle=True)
split_two_idxs = np.load(split_two_indices_file,allow_pickle=True)

# Load results from depth-1 angle and errorbar calculation
f150_spectra_dict = np.load(f150_result_file,allow_pickle=True).item()
f090_spectra_dict = np.load(f090_result_file,allow_pickle=True).item()

maps_f150 = np.array(list(f150_spectra_dict.keys()))
maps_f090 = np.array(list(f090_spectra_dict.keys()))
n_maps_f150 = len(maps_f150); n_maps_f090 = len(maps_f090); n_maps = n_maps_f150 + n_maps_f090

# Extract relevant variables
cut_flags_f150 = np.zeros(n_maps_f150)
angles_deg_skewnorm_f150 = np.zeros(n_maps_f150); errbars_deg_skewnorm_f150 = np.zeros(n_maps_f150); fwhms_skewnorm_f150 = np.zeros(n_maps_f150)
initial_timestamps_f150 = np.zeros(n_maps_f150); final_timestamps_f150 = np.zeros(n_maps_f150)

for i in range(n_maps_f150):
    cut_flags_f150[i] = f150_spectra_dict[maps_f150[i]]['map_cut']
    angles_deg_skewnorm_f150[i] = f150_spectra_dict[maps_f150[i]]['meas_angle_skewnorm-method']
    errbars_deg_skewnorm_f150[i] = f150_spectra_dict[maps_f150[i]]['meas_errbar_skewnorm-method']
    fwhms_skewnorm_f150[i] = f150_spectra_dict[maps_f150[i]]['fwhm_skewnorm-method']
    initial_timestamps_f150[i] = f150_spectra_dict[maps_f150[i]]['initial_timestamp']
    final_timestamps_f150[i] = f150_spectra_dict[maps_f150[i]]['final_timestamp']
    
cut_flags_f090 = np.zeros(n_maps_f090)
angles_deg_skewnorm_f090 = np.zeros(n_maps_f090); errbars_deg_skewnorm_f090 = np.zeros(n_maps_f090); fwhms_skewnorm_f090 = np.zeros(n_maps_f090)
initial_timestamps_f090 = np.zeros(n_maps_f090); final_timestamps_f090 = np.zeros(n_maps_f090)

for i in range(n_maps_f090):
    cut_flags_f090[i] = f090_spectra_dict[maps_f090[i]]['map_cut']
    angles_deg_skewnorm_f090[i] = f090_spectra_dict[maps_f090[i]]['meas_angle_skewnorm-method']
    errbars_deg_skewnorm_f090[i] = f090_spectra_dict[maps_f090[i]]['meas_errbar_skewnorm-method']
    fwhms_skewnorm_f090[i] = f090_spectra_dict[maps_f090[i]]['fwhm_skewnorm-method']
    initial_timestamps_f090[i] = f090_spectra_dict[maps_f090[i]]['initial_timestamp']
    final_timestamps_f090[i] = f090_spectra_dict[maps_f090[i]]['final_timestamp']

# Apply cuts
durations_f150 = final_timestamps_f150 - initial_timestamps_f150
durations_f090 = final_timestamps_f090 - initial_timestamps_f090

# Doing basic cuts - f150
logger.info(f"Total number of f150 maps: {n_maps_f150}")
# Identifying maps that weren't cut by the galaxy mask
cut_good_f150 = np.where(cut_flags_f150 != 1)[0]
logger.info(f"# of f150 maps passing galaxy mask cut: {len(cut_good_f150)}")# Checking for failed fits
errorbar_good_f150 = np.where(np.abs(errbars_deg_skewnorm_f150) > errorbar_cut_threshold)[0]
logger.info(f"# of f150 maps passing failed fit cut: {len(errorbar_good_f150)}")
# Eliminating pathological likelihoods that peak at boundaries
peak_good_f150 = np.where(np.abs(angles_deg_skewnorm_f150) < peak_cut_threshold)[0]
logger.info(f"# of f150 maps passing peak value cut: {len(peak_good_f150)}")
# Doing a cut based on FWHM of skewnormal distribution to eliminate bad fits
fwhm_good_f150 = np.where(np.abs(fwhms_skewnorm_f150) < fwhm_cut_threshold)[0]
logger.info(f"# of f150 maps passing FWHM cut: {len(fwhm_good_f150)}")
# Doing a cut based on map duration to eliminate super long maps
duration_good_f150 = np.where(durations_f150 < duration_cut_threshold)[0] # Cut all maps longer than 14 hours
logger.info(f"# of f150 maps passing duration cut: {len(duration_good_f150)}")
set_pass_all_cuts_f150 = np.intersect1d(duration_good_f150,np.intersect1d(fwhm_good_f150, np.intersect1d(np.intersect1d(cut_good_f150, errorbar_good_f150), peak_good_f150)))
logger.info(f"# of f150 maps passing all five cuts: {len(set_pass_all_cuts_f150)}")
logger.info(f"Percentage of f150 maps passing all five cuts: {len(set_pass_all_cuts_f150)/n_maps_f150}")
logger.info(f"Percentage of f150 maps not cut by galaxy mask passing the other four cuts: {len(set_pass_all_cuts_f150)/len(cut_good_f150)}")

# Doing basic cuts - f090
logger.info(f"Total number of f090 maps: {n_maps_f090}")
# Identifying maps that weren't cut by the galaxy mask
cut_good_f090 = np.where(cut_flags_f090 != 1)[0]
logger.info(f"# of f090 maps passing galaxy mask cut: {len(cut_good_f090)}")
# Checking for failed fits
errorbar_good_f090 = np.where(np.abs(errbars_deg_skewnorm_f090) > errorbar_cut_threshold)[0]
logger.info(f"# of f090 maps passing failed fit cut: {len(errorbar_good_f090)}")
# Eliminating pathological likelihoods that peak at boundaries
peak_good_f090 = np.where(np.abs(angles_deg_skewnorm_f090) < peak_cut_threshold)[0]
logger.info(f"# of f090 maps passing peak value cut: {len(peak_good_f090)}")
# Doing a cut based on FWHM of skewnormal distribution to eliminate bad fits
fwhm_good_f090 = np.where(np.abs(fwhms_skewnorm_f090) < fwhm_cut_threshold)[0]
logger.info(f"# of f090 maps passing FWHM cut: {len(fwhm_good_f090)}")
# Doing a cut based on map duration to eliminate super long maps
duration_good_f090 = np.where(durations_f090 < duration_cut_threshold)[0] # Cut all maps longer than 14 hours
logger.info(f"# of f090 maps passing duration cut: {len(duration_good_f090)}")
set_pass_all_cuts_f090 = np.intersect1d(duration_good_f090,np.intersect1d(fwhm_good_f090, np.intersect1d(np.intersect1d(cut_good_f090, errorbar_good_f090), peak_good_f090)))
logger.info(f"# of f090 maps passing all five cuts: {len(set_pass_all_cuts_f090)}")
logger.info(f"Percentage of f090 maps passing all five cuts: {len(set_pass_all_cuts_f090)/n_maps_f090}")
logger.info(f"Percentage of f090 maps not cut by galaxy mask passing the other four cuts: {len(set_pass_all_cuts_f090)/len(cut_good_f090)}")
logger.info(f"# of all maps passing all five cuts: {len(set_pass_all_cuts_f090)+len(set_pass_all_cuts_f150)}")
logger.info(f"Percentage of all maps passing all five cuts: {(len(set_pass_all_cuts_f090)+len(set_pass_all_cuts_f150))/n_maps}")

# Generate combined timestream of maps that pass cuts
combined_angles = np.concatenate((angles_deg_skewnorm_f150[set_pass_all_cuts_f150], angles_deg_skewnorm_f090[set_pass_all_cuts_f090]))
combined_errorbars = np.concatenate((errbars_deg_skewnorm_f150[set_pass_all_cuts_f150], errbars_deg_skewnorm_f090[set_pass_all_cuts_f090]))
combined_initial_time = np.concatenate((initial_timestamps_f150[set_pass_all_cuts_f150], initial_timestamps_f090[set_pass_all_cuts_f090]))
combined_final_time = np.concatenate((final_timestamps_f150[set_pass_all_cuts_f150], final_timestamps_f090[set_pass_all_cuts_f090]))
combined_mean_time = (combined_final_time + combined_initial_time) / 2.0
combined_obs_duration = combined_final_time - combined_initial_time
logger.info(f"Size of combined arrays: {combined_angles.shape}, {combined_errorbars.shape}, {combined_initial_time.shape}, {combined_final_time.shape}")

# Now sorting them in time
sort_indices = np.argsort(combined_mean_time)
sorted_combined_angles = combined_angles[sort_indices]
sorted_combined_errorbars = combined_errorbars[sort_indices]
sorted_combined_mean_time = combined_mean_time[sort_indices]
sorted_combined_obs_duration = combined_obs_duration[sort_indices]

# Set up frequencies to sample
logger.info("Setting up frequencies to sample")
# Setting up realistic test frequency range
hz_to_invdays = 24*60*60
freq_to_nat_units_mass = 6.58e-16 # s/ev^-1 - looked this up
# Minimum frequency set by a/total_time where a is some factor set in config file
total_time = np.round(sorted_combined_mean_time[-1]-sorted_combined_mean_time[0],4)
logger.info(f"Total time (s): {total_time}. Total time (d): {total_time/hz_to_invdays}. \
      Equivalent frequency (Hz): {1/total_time:.4e}. Equivalent frequency (inverse days): {hz_to_invdays/total_time:.4e}")
min_freq = min_freq_period_factor/total_time 
# Frequency spacing set by oversampling factor in config file - 5 matches PB
freq_spacing = min_freq / freq_oversample_factor
# Maximum frequency set by wanting to be a bit above the normal map duration and sampling gap
# Since both around 5-10 hours, 24 hours is a safe period for the max frequency.
# Number of days used here is set in the config file
max_freq = 1/(num_days_max_freq_period*24*3600)
input_fs = np.arange(min_freq, max_freq + freq_spacing, freq_spacing)
num_freqs = input_fs.size
logger.info(f"Min freq (Hz): {min_freq:.4e}, max freq (Hz): {max_freq:.4e}, freq spacing (Hz): {freq_spacing:.4e}, num freqs: {num_freqs}")
logger.info(f"Min freq (inv days): {min_freq*hz_to_invdays:.4e}, max freq (inv days): {max_freq*hz_to_invdays:.4e}, freq spacing (inv days): {freq_spacing*hz_to_invdays:.4e}, num freqs: {num_freqs}")
logger.info(f"Min mass (eV): {min_freq*freq_to_nat_units_mass:.4e}, max mass (eV): {max_freq*freq_to_nat_units_mass:.4e}, mass spacing (eV): {freq_spacing*freq_to_nat_units_mass:.4e}, num masses: {num_freqs}")

# Set up amplitude and phase grid to sample
logger.info("Setting up amplitude and phase grid to sample")
As = np.arange(min_amplitude, max_amplitude + amplitude_step, amplitude_step)
logger.info(f"Total number of amplitude points: {As.size}")
min_phase = np.deg2rad(min_phase_deg)
max_phase = np.deg2rad(max_phase_deg)
phase_step = np.deg2rad(phase_step_deg)
phases = np.arange(min_phase, max_phase, phase_step) # range of [0, 2*pi)
logger.info(f"Total number of phase points: {phases.size}")
logger.info(f"Total number of grid points: {As.size*phases.size}")

#######################################################################################################
# Defining single process function for main loop
def sim_process(input_fs, As, phases, split_one_idxs, split_two_idxs, angles, errorbars, durations, mean_times,
                output_dir_path, output_tag, sim_num):
    """Generates a simulated timestream from the real data, then splits it and calculates
       best fit amplitudes for each split. Saves output array of best fit amp at each freq for
       each split to a npy file for further processing."""

    # Generate simulated angles
    sim_timestream = stats.norm.rvs(loc=angles,scale=errorbars)

    # Run sim timestream (with real errorbars and times) for each split
    split_one_amps = ta.calc_marg_amp(input_fs, As, phases, 
                                sim_timestream[split_one_idxs],errorbars[split_one_idxs],
                                durations[split_one_idxs],mean_times[split_one_idxs])
    
    split_two_amps = ta.calc_marg_amp(input_fs, As, phases, 
                                sim_timestream[split_two_idxs],errorbars[split_two_idxs],
                                durations[split_two_idxs],mean_times[split_two_idxs])

    # Save outputs
    output_fname1 = output_dir_path + output_tag + f'_sim{sim_num:05}_split_one.npy'
    np.save(output_fname1, split_one_amps)
    output_fname2 = output_dir_path + output_tag + f'_sim{sim_num:05}_split_two.npy'
    np.save(output_fname2, split_two_amps)

#######################################################################################################

# Applying split - for rank 0, start by doing real timestream. For all other processes, go to doing sims right away.
if rank==0:
    # Dump all config info to YAML only once
    # All results are in separate npy files generated in process()
    config_output_dict = config
    config_output_name = output_dir_path + 'null_test_config_' + output_tag + ".yaml"
    with open(config_output_name, 'w') as file:
        yaml.dump(config_output_dict, file)

    # Run real timestream - returns an array of amps for all freqs per split
    logger.info("Running real timestream split one on process 0")
    try:
        split_one_amps = ta.calc_marg_amp(input_fs, As, phases, 
                                        sorted_combined_angles[split_one_idxs],sorted_combined_errorbars[split_one_idxs],
                                        sorted_combined_obs_duration[split_one_idxs],sorted_combined_mean_time[split_one_idxs])
        results_output_fname1 = output_dir_path + output_tag + '_real_split_one.npy'
        np.save(results_output_fname1, split_one_amps)
    except Exception as e:
        logger.error(f"Real timestream first split failed with error {e}")
    logger.info("Running real timestream split two on process 0")
    try:
        split_two_amps = ta.calc_marg_amp(input_fs, As, phases, 
                                        sorted_combined_angles[split_two_idxs],sorted_combined_errorbars[split_two_idxs],
                                        sorted_combined_obs_duration[split_two_idxs],sorted_combined_mean_time[split_two_idxs])
        results_output_fname2 = output_dir_path + output_tag + '_real_split_two.npy'
        np.save(results_output_fname2, split_two_amps)
    except Exception as e:
        logger.error(f"Real timestream second split failed with error {e}")
else:
    # This loop distributes some of the maps to each process
    # Only uses processes 1 and up to avoid running both the real timestream and a sim on process 0
    for i in range(rank-1, n_sims, size-1):
        logger.info(f"Processing sim {i} on process {rank}")
        try:
            sim_process(input_fs, As, phases, split_one_idxs, split_two_idxs, 
                        sorted_combined_angles, sorted_combined_errorbars, 
                        sorted_combined_obs_duration, sorted_combined_mean_time,
                        output_dir_path, output_tag, i)
        except Exception as e:
            logger.error(f"Sim {i} failed on process {rank} with error {e}")

logger.info(f"Finished running get_amp_null_test.py. Output is in: {output_dir_path}")
stop_time = time.time()
duration = stop_time-start_time
logger.info("Script took {:1.3f} seconds".format(duration))

