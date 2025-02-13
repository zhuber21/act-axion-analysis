import numpy as np
import yaml
import argparse
import time
import logging
import os
from pixell import enmap
from tqdm import tqdm
import act_axion_analysis as aaa
from mpi4py import MPI

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
    E.g. angle_calc_<output_dir_tag> will be output directory",type=str)
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
output_dir_path = output_dir_root + "angle_calc_" + output_tag + '/'
if not os.path.exists(output_dir_path): # Make new folder for this run - should be unique
    os.makedirs(output_dir_path)

# Setting up logger - making a separate one for each process in output_dir/log/
logger = logging.getLogger(__name__)
os.makedirs(output_dir_path + 'log/')
log_filename = output_dir_path+'log/process{:02d}_run.log'.format(rank)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', style='{',
                    format='{asctime} {levelname} {filename}:{lineno}: {message}',
                    handlers=[logging.FileHandler(filename=log_filename)]
                    )
logger.info("Using config file: " + str(yaml_name))

# Setting common variables set in the config file
freq = config['freq'] # The frequency being tested this run
logger.info("Analyzing frequency " + freq)
kx_cut = config['kx_cut']
ky_cut = config['ky_cut']
unpixwin = config['unpixwin']
filter_radius = config['filter_radius']
angle_min_deg = config['angle_min_deg']
angle_max_deg = config['angle_max_deg']
num_pts = config['num_pts']
use_curvefit = config['use_curvefit']
use_ivar_weight = config['use_ivar_weight']
cross_calibrate = config['cross_calibrate']

# Check that paths exist to needed files
camb_file = config['theory_curves_path']
ref_path = config['ref_path']
#ref_beam_path = config['ref_beam_path'] # Will add back in if there is a separate beam for the reference map instead of averaging other beams
ref_ivar_path = config['ref_ivar_path']
pa4_beam_path = config['pa4_beam_path']
pa5_beam_path = config['pa5_beam_path']
pa6_beam_path = config['pa6_beam_path']
galaxy_mask_path = config['galaxy_mask_path']
obs_list_path = config['obs_path_stem']
obs_list = config['obs_list']
if not os.path.exists(camb_file): 
    logger.error("Cannot find CAMB file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {camb_file}")
if not os.path.exists(ref_path): 
    logger.error("Cannot find reference map file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {ref_path}")
#if not os.path.exists(ref_beam_path): 
#    logger.error("Cannot find beam file! Check config. Exiting.")
#    raise FileNotFoundError(f"File not found: {ref_beam_path}")
if not os.path.exists(ref_ivar_path): 
    logger.error("Cannot find ref map ivar file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {ref_ivar_path}")
if freq=='f150' or freq=='f220':
    if not os.path.exists(pa4_beam_path): 
        logger.error("Cannot find pa4 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa4_beam_path}")
if freq=='f090' or freq=='f150':
    if not os.path.exists(pa5_beam_path): 
        logger.error("Cannot find pa5 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa5_beam_path}")
    if not os.path.exists(pa6_beam_path): 
        logger.error("Cannot find pa6 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa6_beam_path}")
if not os.path.exists(galaxy_mask_path):
    logger.error("Cannot find galaxy mask file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {galaxy_mask_path}")
if not os.path.exists(obs_list): 
    logger.error("Cannot find observation list! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {obs_list}")
if obs_list[-3:] == 'txt':
    logger.info("Using list of observations at: " + str(obs_list))
else:
    logger.error("Please enter a valid text file in the obs_list field in the YAML file. Exiting.")
    raise ValueError("Please enter a valid text file in the obs_list field in the YAML file.")
if cross_calibrate:
    y_min = config['y_min']
    y_max = config['y_max']
    cal_num_pts = config['cal_num_pts']
    cal_use_curvefit = config['cal_use_curvefit']
    cal_bin_size = config['cal_bin_size']
    cal_lmin = config['cal_lmin']
    cal_lmax = config['cal_lmax']
    cal_map1_path = config['cal_map1_path']
    cal_ivar1_path = config['cal_ivar1_path']
    cal_map2_path = config['cal_map2_path']
    cal_ivar2_path = config['cal_ivar2_path']
    if not os.path.exists(cal_map1_path): 
        logger.error("Cannot find calibration map 1 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_map1_path}")
    if not os.path.exists(cal_ivar1_path): 
        logger.error("Cannot find calibration ivar 1 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_ivar1_path}")
    if not os.path.exists(cal_map2_path): 
        logger.error("Cannot find calibration map 2 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_map2_path}")
    if not os.path.exists(cal_ivar2_path): 
        logger.error("Cannot find calibration ivar 2 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_ivar2_path}")
    # Setting up bins for calibration
    cal_bins = np.arange(cal_lmin, cal_lmax, cal_bin_size)

# Setting bins
if config['bin_settings'] == "regular":
    bin_size = config['bin_size']
    lmin = config['lmin']
    lmax = config['lmax']
    bins = np.arange(lmin, lmax, bin_size)
    centers = (bins[1:] + bins[:-1])/2.0
elif config['bin_settings'] == "DR4":
    # Load bins from ACT DR4
    start_index = config['start_index']
    stop_index = config['stop_index']
    act_dr4_spectra_root = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
    full_bins, full_centers = np.loadtxt(act_dr4_spectra_root+'/resources/BIN_ACTPOL_50_4_SC_low_ell',
                                        usecols=(0,2), unpack=True)
    bins = full_bins[start_index:stop_index]
    centers = full_centers[start_index:stop_index-1]
else:
    logger.error("Please use valid bin_settings! Options are 'regular' and 'DR4'. Exiting.")
    raise ValueError("Please use valid bin_settings! Options are 'regular' and 'DR4'.")
logger.info("Finished loading bins")

# Setting plotting settings
plot_maps = config['plot_maps']
plot_likelihood = config['plot_likelihood']
plot_beam = config['plot_beam']
plot_tfunc = config['plot_tfunc'] 

# Load CAMB EE and BB spectrum (BB just for plotting)
logger.info("Starting to load CAMB spectra")
ell_camb,DlEE_camb,DlBB_camb = np.loadtxt(camb_file, usecols=(0,2,3), unpack=True)
# Note that ell runs from 2 to 5400
arr_len = ell_camb.size + 2
ell = np.zeros(arr_len)
ell[1] = 1.0
ell[2:] = ell_camb
ClEE = np.zeros(arr_len)
ClBB = np.zeros(arr_len)
ClEE[2:] = DlEE_camb * 2 * np.pi / (ell_camb*(ell_camb+1.0))
ClBB[2:] = DlBB_camb * 2 * np.pi / (ell_camb*(ell_camb+1.0))
digitized = np.digitize(ell, bins, right=True)
CAMB_ClEE_binned = np.bincount(digitized, ClEE.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
CAMB_ClBB_binned = np.bincount(digitized, ClBB.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
logger.info("Finished loading CAMB spectra")

# Loading all beams
logger.info("Starting to load beams")
if freq=='f090':
    # Only pa5 and pa6 at f090
    logger.info("Using pa5 beam " + str(pa5_beam_path))
    logger.info("Using pa6 beam " + str(pa6_beam_path))
    pa4_beam = []
    pa5_beam = aaa.load_and_bin_beam(pa5_beam_path,bins)
    pa6_beam = aaa.load_and_bin_beam(pa6_beam_path,bins)
    # For now, average these beams to get coadd/ref beam
    ref_beam = (pa5_beam+pa6_beam)/2.0
    if plot_beam:
        pa5_beam_name = os.path.split(pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa5_beam_name, centers, pa5_beam)
        pa6_beam_name = os.path.split(pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa6_beam_name, centers, pa6_beam)
        ref_beam_name = "f090_coadd_avg_beam"
        aaa.plot_beam(output_dir_path, ref_beam_name, centers, ref_beam)
elif freq=='f150':
    logger.info("Using pa4 beam " + str(pa4_beam_path))
    logger.info("Using pa5 beam " + str(pa5_beam_path))
    logger.info("Using pa6 beam " + str(pa6_beam_path))
    pa4_beam = aaa.load_and_bin_beam(pa4_beam_path,bins)
    pa5_beam = aaa.load_and_bin_beam(pa5_beam_path,bins)
    pa6_beam = aaa.load_and_bin_beam(pa6_beam_path,bins)
    # For now, average these beams to get coadd/ref beam
    ref_beam = (pa4_beam+pa5_beam+pa6_beam)/3.0
    if plot_beam:
        pa4_beam_name = os.path.split(pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa4_beam_name, centers, pa4_beam)
        pa5_beam_name = os.path.split(pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa5_beam_name, centers, pa5_beam)
        pa6_beam_name = os.path.split(pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa6_beam_name, centers, pa6_beam)
        ref_beam_name = "f150_coadd_avg_beam"
        aaa.plot_beam(output_dir_path, ref_beam_name, centers, ref_beam)
elif freq=='f220':
    logger.info("Using pa4 beam " + str(pa4_beam_path))
    pa4_beam = aaa.load_and_bin_beam(pa4_beam_path,bins)
    pa5_beam = []
    pa6_beam = []
    # only pa4 at f220
    ref_beam = pa4_beam
    if plot_beam:
        pa4_beam_name = os.path.split(pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aaa.plot_beam(output_dir_path, pa4_beam_name, centers, pa4_beam)
        ref_beam_name = "f220_coadd_avg_beam"
        aaa.plot_beam(output_dir_path, ref_beam_name, centers, ref_beam)
logger.info("Finished loading beams")

# Calculate filtering transfer function once since filtering is same for all maps
tfunc = aaa.get_tfunc(kx_cut, ky_cut, bins)
if plot_tfunc:
    aaa.plot_tfunc(output_dir_path, kx_cut, ky_cut, centers, tfunc)

# Loading in reference maps
logger.info("Starting to load ref map")
ref_maps, ref_ivar = aaa.load_ref_map(ref_path,ref_ivar_path)
logger.info("Finished loading ref map")

# Loading in galaxy mask
logger.info("Starting to load galaxy mask")
galaxy_mask = enmap.read_map(galaxy_mask_path)
logger.info("Finished loading galaxy mask")

if cross_calibrate:
    # Loading in calibration ivar and maps for cross-correlation
    logger.info("Starting to load calibration maps for cross-correlation")
    # only loading in T maps and trimming immediately to galaxy mask's shape to save memory
    cal_T_map1_act_footprint = enmap.read_map(cal_map1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    cal_T_ivar1_act_footprint = enmap.read_map(cal_ivar1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))
    cal_T_map2_act_footprint = enmap.read_map(cal_map2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    cal_T_ivar2_act_footprint = enmap.read_map(cal_ivar2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))
    logger.info("Finished loading calibration maps for cross-correlation")

#######################################################################################################
# Defining single process function for main loop
def process(map_name, obs_list_path, logger, 
            ref_maps, ref_ivar, galaxy_mask,
            kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
            plot_maps, plot_likelihood, output_dir_path, 
            bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
            pa4_beam, pa5_beam, pa6_beam, ref_beam, tfunc, 
            num_pts, angle_min_deg, angle_max_deg, use_curvefit, 
            cross_calibrate, cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
            cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
            cal_bins, y_min, y_max, cal_num_pts, cal_use_curvefit):
    """Function to call inside each process"""
    map_path = obs_list_path + map_name
    # outputs will be 1 if the map is cut, a bunch of things needed
    # for time estimation and TT calibration if not
    output_dict, outputs = aaa.estimate_pol_angle(map_path, map_name, logger, ref_maps, ref_ivar, galaxy_mask,
                                                  kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                                                  plot_maps, plot_likelihood, output_dir_path, 
                                                  bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
                                                  pa4_beam, pa5_beam, pa6_beam, ref_beam, tfunc, 
                                                  num_pts, angle_min_deg, angle_max_deg, use_curvefit)

    fit_values = [output_dict['meas_angle'], output_dict['meas_errbar']]
    if output_dict['map_cut']==0:
        depth1_mask = outputs[0]
        depth1_mask_indices = outputs[1]
        depth1_ivar = outputs[2]
        depth1_T = outputs[3]
        w_depth1 = outputs[4]
        depth1_beam = outputs[5]
        w2_depth1 = output_dict['w2_depth1']
        bincount = output_dict['bincount']
        logger.info("Fit values: "+str(fit_values))
        # depth1_mask will be the doubly tapered one if ivar weighting is on, the first filtering one if not
        initial_timestamp, median_timestamp = aaa.calc_median_timestamp(map_path, depth1_mask)
        logger.info("Initial timestamp: "+str(initial_timestamp))
        logger.info("Median timestamp: "+str(median_timestamp))
        output_dict.update({'initial_timestamp': initial_timestamp, 'median_timestamp': median_timestamp})

        if cross_calibrate:
            # Calling a single function to do all the TT cross-calibration
            # This makes the code a bit harder to read but improves memory usage by allowing
            # intermediate maps to be cleaned up when function call ends.
            cal_output_dict = aaa.cross_calibrate(cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
                                                  cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
                                                  depth1_ivar, depth1_mask, depth1_mask_indices,
                                                  galaxy_mask, depth1_T, w_depth1, w2_depth1, bincount,
                                                  cal_bins, tfunc, kx_cut, ky_cut, unpixwin, filter_radius, 
                                                  use_ivar_weight, depth1_beam, pa5_beam, pa6_beam, y_min, 
                                                  y_max, cal_num_pts, cal_use_curvefit)
            # Printing out calibration factor and errorbar
            cal_fit_values = (cal_output_dict['cal_factor'], cal_output_dict['cal_factor_errbar'])
            logger.info("TT calibration fit values: "+str(cal_fit_values))
            # Adding calibration keys to final dictionary
            output_dict.update(cal_output_dict)
    else:
        if cross_calibrate:
            # To ensure all keys are present whether map is cut or not
            ell_len = len(output_dict['ell'])
            output_dict.update({{'T1xcal1T': np.zeros(ell_len), 'cal1Txcal2T': np.zeros(ell_len),
                                 'cal1Txcal1T': np.zeros(ell_len), 'cal2Txcal2T': np.zeros(ell_len), 
                                 'T1xT1': np.zeros(ell_len), 'T1xcal2T': np.zeros(ell_len),
                                 'cal_factor': -9999, 'cal_factor_errbar': -9999,
                                 'w2_depth1xcal1': -9999, 'w2_depth1xcal2': -9999,
                                 'w2_cal1xcal2': -9999, 'w2_cal1xcal1': -9999, 'w2_cal2xcal2': -9999,
                                 'w2w4_all_three': -9999, 'w2w4_depth1xcal1': -9999, 'w2w4_cal1xcal2': -9999}})

    # At the end, save the output_dict to a npy file - there will be one per map
    # Can be loaded with np.load(results_output_fname, allow_pickle=True).item()
    # If the mask cuts the whole map, this will be the dict with -9999 everywhere
    map_name_no_ending = map_name[:-8] # Removing 'map.fits'
    results_output_fname = output_dir_path + map_name_no_ending + output_tag + '_results.npy'
    np.save(results_output_fname, output_dict)

#######################################################################################################
# Run main sequence

with open(obs_list, 'r') as f:
    lines = f.read().splitlines()

if rank==0:
    # Dump all config info to YAML only once
    # All results are in separate npy files generated in process()
    maps = []
    for line in lines:
        maps.append(line)
    config_output_dict = config
    config_output_dict['list_of_maps'] = maps
    config_output_name = output_dir_path + 'angle_calc_config_' + output_tag + ".yaml"
    with open(config_output_name, 'w') as file:
        yaml.dump(config_output_dict, file)

# This loop distributes some of the maps to each process
for i in range(rank, len(lines), size):
    map_name = lines[i]
    maps.append(map_name)
    logger.info("Processing " + map_name + " on process " + str(rank))
    process(map_name, obs_list_path, logger, 
            ref_maps, ref_ivar, galaxy_mask,
            kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
            plot_maps, plot_likelihood, output_dir_path, 
            bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
            pa4_beam, pa5_beam, pa6_beam, ref_beam, tfunc, 
            num_pts, angle_min_deg, angle_max_deg, use_curvefit, 
            cross_calibrate, cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
            cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
            cal_bins, y_min, y_max, cal_num_pts, cal_use_curvefit)

logger.info("Finished running get_angle_from_depth1_ps.py. Output is in: " + str(output_dir_path))
stop_time = time.time()
duration = stop_time-start_time
logger.info("Script took {:1.3f} seconds".format(duration))
