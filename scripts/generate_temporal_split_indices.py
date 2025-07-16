"""
    generate_temporal_split_indices.py
    Written by ZBH, 7/14/2025

    This script performs a set of cuts on the depth-1 angle
    results, splits the resulting timestream into two 
    roughly even parts in time, and saves the indices of
    the maps in each of the splits into npy files for use
    in the get_amp_null_test.py script.

    The purpose of putting this code and the settings used
    for doing the cuts and splits into this small script
    is so that they are easily saved and reproducible.
    The same settings for cuts should be used in
    get_amp_null_test.py when using the outputs of this script.
"""
import numpy as np
import logging

# Setting cut parameters
errorbar_cut_threshold = 0.1
peak_cut_threshold = 30.0
fwhm_cut_threshold = 50.0
duration_cut_threshold = 14*3600

# Setting split parameters
# March 1st, 2020 at UTC 00:00:00 is 1583038800 - around PA7 installation
# February 1st, 2019 at UTC 00:00:00 is 1548997200 - PS paper temporal null
time_of_split = 1583038800 
output_dir_path = "/pscratch/sd/z/zbh5/null_tests/"
split_one_fname = "temporal_split_indices_one.npy"
split_two_fname = "temporal_split_indices_two.npy"

# Set up log
logger = logging.getLogger(__name__)
log_filename = output_dir_path+'generate_temporal_split_run.log'
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', style='{',
                    format='{asctime} {levelname} {filename}:{lineno}: {message}',
                    handlers=[logging.FileHandler(filename=log_filename)]
                    )

# Loading angle calc results
combined_f150_spectra_dict = np.load('/pscratch/sd/z/zbh5/results/combined_angle_calc_results_f150.npy',allow_pickle=True).item()
combined_f090_spectra_dict = np.load('/pscratch/sd/z/zbh5/results/combined_angle_calc_results_f090.npy',allow_pickle=True).item()

maps_f150 = np.array(list(combined_f150_spectra_dict.keys()))
maps_f090 = np.array(list(combined_f090_spectra_dict.keys()))
n_maps_f150 = len(maps_f150); n_maps_f090 = len(maps_f090); n_maps = n_maps_f150 + n_maps_f090

# Pulling out various values for convenience
cut_flags_f150 = np.zeros(n_maps_f150)
angles_deg_skewnorm_f150 = np.zeros(n_maps_f150); errbars_deg_skewnorm_f150 = np.zeros(n_maps_f150); fwhms_skewnorm_f150 = np.zeros(n_maps_f150)
initial_timestamps_f150 = np.zeros(n_maps_f150); final_timestamps_f150 = np.zeros(n_maps_f150)

for i in range(n_maps_f150):
    cut_flags_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['map_cut']
    angles_deg_skewnorm_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['meas_angle_skewnorm-method']
    errbars_deg_skewnorm_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['meas_errbar_skewnorm-method']
    fwhms_skewnorm_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['fwhm_skewnorm-method']
    initial_timestamps_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['initial_timestamp']
    final_timestamps_f150[i] = combined_f150_spectra_dict[maps_f150[i]]['final_timestamp']
    
cut_flags_f090 = np.zeros(n_maps_f090)
angles_deg_skewnorm_f090 = np.zeros(n_maps_f090); errbars_deg_skewnorm_f090 = np.zeros(n_maps_f090); fwhms_skewnorm_f090 = np.zeros(n_maps_f090)
initial_timestamps_f090 = np.zeros(n_maps_f090); final_timestamps_f090 = np.zeros(n_maps_f090)

for i in range(n_maps_f090):
    cut_flags_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['map_cut']
    angles_deg_skewnorm_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['meas_angle_skewnorm-method']
    errbars_deg_skewnorm_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['meas_errbar_skewnorm-method']
    fwhms_skewnorm_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['fwhm_skewnorm-method']
    initial_timestamps_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['initial_timestamp']
    final_timestamps_f090[i] = combined_f090_spectra_dict[maps_f090[i]]['final_timestamp']

durations_f150 = final_timestamps_f150 - initial_timestamps_f150
durations_f090 = final_timestamps_f090 - initial_timestamps_f090

# Doing basic cuts - f150
logger.info("Total number of f150 maps: ", n_maps_f150)
# Identifying maps that weren't cut by the galaxy mask
cut_good_f150 = np.where(cut_flags_f150 != 1)[0]
logger.info("# of f150 maps passing galaxy mask cut: ", len(cut_good_f150))# Checking for failed fits
errorbar_good_f150 = np.where(np.abs(errbars_deg_skewnorm_f150) > errorbar_cut_threshold)[0]
logger.info("# of f150 maps passing failed fit cut: ", len(errorbar_good_f150))
# Eliminating pathological likelihoods that peak at boundaries
peak_good_f150 = np.where(np.abs(angles_deg_skewnorm_f150) < peak_cut_threshold)[0]
logger.info("# of f150 maps passing peak value cut: ", len(peak_good_f150))
# Doing a cut based on FWHM of skewnormal distribution to eliminate bad fits
fwhm_good_f150 = np.where(np.abs(fwhms_skewnorm_f150) < fwhm_cut_threshold)[0]
logger.info("# of f150 maps passing FWHM cut: ", len(fwhm_good_f150))
# Doing a cut based on map duration to eliminate super long maps
duration_good_f150 = np.where(durations_f150 < duration_cut_threshold)[0] # Cut all maps longer than 14 hours
logger.info("# of f150 maps passing duration cut: ", len(duration_good_f150))
set_pass_all_cuts_f150 = np.intersect1d(duration_good_f150,np.intersect1d(fwhm_good_f150, np.intersect1d(np.intersect1d(cut_good_f150, errorbar_good_f150), peak_good_f150)))
logger.info("# of f150 maps passing all five cuts: ", len(set_pass_all_cuts_f150))
logger.info(f"Percentage of f150 maps passing all five cuts: {len(set_pass_all_cuts_f150)/n_maps_f150}")
logger.info(f"Percentage of f150 maps not cut by galaxy mask passing the other four cuts: {len(set_pass_all_cuts_f150)/len(cut_good_f150)}")

# Doing basic cuts - f090
logger.info("Total number of f090 maps: ", n_maps_f090)
# Identifying maps that weren't cut by the galaxy mask
cut_good_f090 = np.where(cut_flags_f090 != 1)[0]
logger.info("# of f090 maps passing galaxy mask cut: ", len(cut_good_f090))
# Checking for failed fits
errorbar_good_f090 = np.where(np.abs(errbars_deg_skewnorm_f090) > errorbar_cut_threshold)[0]
logger.info("# of f090 maps passing failed fit cut: ", len(errorbar_good_f090))
# Eliminating pathological likelihoods that peak at boundaries
peak_good_f090 = np.where(np.abs(angles_deg_skewnorm_f090) < peak_cut_threshold)[0]
logger.info("# of f090 maps passing peak value cut: ", len(peak_good_f090))
# Doing a cut based on FWHM of skewnormal distribution to eliminate bad fits
fwhm_good_f090 = np.where(np.abs(fwhms_skewnorm_f090) < fwhm_cut_threshold)[0]
logger.info("# of f090 maps passing FWHM cut: ", len(fwhm_good_f090))
# Doing a cut based on map duration to eliminate super long maps
duration_good_f090 = np.where(durations_f090 < duration_cut_threshold)[0] # Cut all maps longer than 14 hours
logger.info("# of f090 maps passing duration cut: ", len(duration_good_f090))
set_pass_all_cuts_f090 = np.intersect1d(duration_good_f090,np.intersect1d(fwhm_good_f090, np.intersect1d(np.intersect1d(cut_good_f090, errorbar_good_f090), peak_good_f090)))
logger.info("# of f090 maps passing all five cuts: ", len(set_pass_all_cuts_f090))
logger.info(f"Percentage of f090 maps passing all five cuts: {len(set_pass_all_cuts_f090)/n_maps_f090}")
logger.info(f"Percentage of f090 maps not cut by galaxy mask passing the other four cuts: {len(set_pass_all_cuts_f090)/len(cut_good_f090)}")
logger.info("# of all maps passing all five cuts: ", len(set_pass_all_cuts_f090)+len(set_pass_all_cuts_f150))
logger.info(f"Percentage of all maps passing all five cuts: {(len(set_pass_all_cuts_f090)+len(set_pass_all_cuts_f150))/n_maps}")

# Combining the f090 and f150 data together and sorting by time
combined_angles = np.concatenate((angles_deg_skewnorm_f150[set_pass_all_cuts_f150], angles_deg_skewnorm_f090[set_pass_all_cuts_f090]))
combined_errorbars = np.concatenate((errbars_deg_skewnorm_f150[set_pass_all_cuts_f150], errbars_deg_skewnorm_f090[set_pass_all_cuts_f090]))
combined_initial_time = np.concatenate((initial_timestamps_f150[set_pass_all_cuts_f150], initial_timestamps_f090[set_pass_all_cuts_f090]))
combined_final_time = np.concatenate((final_timestamps_f150[set_pass_all_cuts_f150], final_timestamps_f090[set_pass_all_cuts_f090]))
combined_mean_time = (combined_final_time + combined_initial_time) / 2.0
combined_obs_duration = combined_final_time - combined_initial_time
# Now sorting them in time
sort_indices = np.argsort(combined_mean_time)
sorted_combined_angles = combined_angles[sort_indices]
sorted_combined_errorbars = combined_errorbars[sort_indices]
sorted_combined_mean_time = combined_mean_time[sort_indices]
sorted_combined_obs_duration = combined_obs_duration[sort_indices]

# Identifying split indices and saving to npy files
logger.info(f"Splitting around the timestamp: {time_of_split}")
split_one_indices = np.where(sorted_combined_mean_time < time_of_split)[0]
split_two_indices = np.where(sorted_combined_mean_time > time_of_split)[0]
# Checking that the number of maps and the inverse variance from the timestreams
# are roughly equal for the two splits
logger.info(f"Number of maps in split one: {split_one_indices.shape}")
logger.info(f"Number of maps in split two: {split_two_indices.shape}")
split_one_total_inverse_var = np.sum(1.0/combined_errorbars[split_one_indices]**2)
logger.info(f"Total inverse variance from errorbars in split one: {split_one_total_inverse_var}")
split_two_total_inverse_var = np.sum(1.0/combined_errorbars[split_two_indices]**2)
logger.info(f"Total inverse variance from errorbars in split two: {split_two_total_inverse_var}")

logger.info(f"Split one indices saved to {output_dir_path + split_one_fname}")
np.save(output_dir_path + split_one_fname, split_one_indices)
logger.info(f"Split two indices saved to {output_dir_path + split_two_fname}")
np.save(output_dir_path + split_two_fname, split_two_indices)
