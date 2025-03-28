import numpy as np
import yaml
import argparse
import time
import logging
import os
import sys
from pixell import enmap
from tqdm import tqdm
from act_axion_analysis import axion_osc_analysis_depth1_ps as aoa
from act_axion_analysis import axion_osc_plotting as aop

start_time = time.time()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("config_file", help=
    "The name of the YAML config file",type=str)
args = parser.parse_args()

# Load in the YAML file
yaml_name = args.config_file

with open(yaml_name, 'r') as file:
    config = yaml.safe_load(file)

# Output file path
output_time = str(int(round(time.time())))
output_dir_root = config['output_dir_root']
if not os.path.exists(output_dir_root): # Make sure root path is right
    print("Output directory does not exist! Exiting.")
    raise OSError(f"Directory not found: {output_dir_root}")
output_dir_path = output_dir_root + "angle_calc_" + output_time + '/'
if not os.path.exists(output_dir_path): # Make new folder for this run - should be unique
    os.makedirs(output_dir_path)

# Setting up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', style='{',
                    format='{asctime} {levelname} {filename}:{lineno}: {message}',
                    handlers=[logging.FileHandler(filename=output_dir_path+'run.log'),
                              logging.StreamHandler(sys.stdout)]
                    )
logger.info("Using config file: " + str(yaml_name))

# Setting common variables set in the config file
freq = config['freq'] # The frequency being tested this run
if freq not in ['f090', 'f150', 'f220']:
    logger.error(f"freq {freq} does not match one of the three acceptable options!")
else:
    logger.info(f"Analyzing frequency {freq}")
kx_cut = config['kx_cut']
ky_cut = config['ky_cut']
unpixwin = config['unpixwin']
filter_radius = config['filter_radius']
angle_min_deg = config['angle_min_deg']
angle_max_deg = config['angle_max_deg']
num_pts = config['num_pts']
fit_method = config['fit_method']
if fit_method not in ['all', 'fwhm', 'curvefit', 'skewnorm', 'moment']:
    logger.error(f"fit_method must be one of 'all', 'fwhm', 'curvefit', 'skewnorm', or 'moment'! You supplied {fit_method}. Exiting.")
    raise ValueError(f"fit_method must be one of 'all', 'fwhm', 'curvefit', 'skewnorm', or 'moment'! You supplied {fit_method}.")
use_ivar_weight = config['use_ivar_weight']
cross_calibrate = config['cross_calibrate']

# Setting parameters for TT calibration, even if cross_calibrate=False
# so that the process() function call does not crash.
y_min = config['y_min']
y_max = config['y_max']
cal_num_pts = config['cal_num_pts']
cal_fit_method = config['cal_fit_method']
if cal_fit_method not in ['fwhm', 'curvefit', 'skewnorm', 'moment']:
    logger.error(f"fit_method must be one of 'fwhm', 'curvefit', 'skewnorm', or 'moment'! You supplied {cal_fit_method}. Exiting.")
    raise ValueError(f"fit_method must be one of 'fwhm', 'curvefit', 'skewnorm', or 'moment'! You supplied {cal_fit_method}.")
cal_bin_size = config['cal_bin_size']
cal_lmin = config['cal_lmin']
cal_lmax = config['cal_lmax']
cal_map1_path = config['cal_map1_path']
cal_ivar1_path = config['cal_ivar1_path']
cal_beam1_path = config['cal_beam1_path']
cal_map2_path = config['cal_map2_path']
cal_ivar2_path = config['cal_ivar2_path']
cal_beam2_path = config['cal_beam2_path']

# Check that paths exist to needed files
camb_file = config['theory_curves_path']
ref_pa4_path = config['ref_pa4_path']
ref_pa4_ivar_path = config['ref_pa4_ivar_path']
ref_pa4_beam_path = config['ref_pa4_beam_path']
ref_pa5_path = config['ref_pa5_path']
ref_pa5_ivar_path = config['ref_pa5_ivar_path']
ref_pa5_beam_path = config['ref_pa5_beam_path']
ref_pa6_path = config['ref_pa6_path']
ref_pa6_ivar_path = config['ref_pa6_ivar_path']
ref_pa6_beam_path = config['ref_pa6_beam_path']

pa4_beam_path = config['pa4_beam_path']
pa5_beam_path = config['pa5_beam_path']
pa6_beam_path = config['pa6_beam_path']
galaxy_mask_path = config['galaxy_mask_path']
obs_list_path = config['obs_path_stem']
obs_list = config['obs_list']
if not os.path.exists(camb_file): 
    logger.error("Cannot find CAMB file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {camb_file}")
if freq=='f150' or freq=='f220':
    if not os.path.exists(ref_pa4_path): 
        logger.error("Cannot find pa4 reference map file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa4_path}")
    if not os.path.exists(ref_pa4_beam_path): 
        logger.error("Cannot find pa4 ref beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa4_beam_path}")
    if not os.path.exists(ref_pa4_ivar_path): 
        logger.error("Cannot find pa4 ref map ivar file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa4_ivar_path}")
    if not os.path.exists(pa4_beam_path): 
        logger.error("Cannot find pa4 depth-1 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa4_beam_path}")
if freq=='f090' or freq=='f150':
    if not os.path.exists(ref_pa5_path): 
        logger.error("Cannot find pa5 reference map file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa5_path}")
    if not os.path.exists(ref_pa5_beam_path): 
        logger.error("Cannot find pa5 ref beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa5_beam_path}")
    if not os.path.exists(ref_pa5_ivar_path): 
        logger.error("Cannot find pa5 ref map ivar file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa5_ivar_path}")
    if not os.path.exists(pa5_beam_path): 
        logger.error("Cannot find pa5 depth-1 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa5_beam_path}")
    if not os.path.exists(ref_pa6_path): 
        logger.error("Cannot find pa6 reference map file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa6_path}")
    if not os.path.exists(ref_pa6_beam_path): 
        logger.error("Cannot find pa6 ref beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa6_beam_path}")
    if not os.path.exists(ref_pa6_ivar_path): 
        logger.error("Cannot find pa6 ref map ivar file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {ref_pa6_ivar_path}")
    if not os.path.exists(pa6_beam_path): 
        logger.error("Cannot find pa6 depth-1 beam file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {pa6_beam_path}")
if not os.path.exists(galaxy_mask_path):
    logger.error("Cannot find galaxy mask file! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {galaxy_mask_path}")
if not os.path.exists(obs_list): 
    logger.error("Cannot find observation list! Check config. Exiting.")
    raise FileNotFoundError(f"File not found: {obs_list}")
if obs_list[-3:] == 'txt':
    logger.info(f"Using list of observations at: {obs_list}")
else:
    logger.error("Please enter a valid text file in the obs_list field in the YAML file. Exiting.")
    raise ValueError("Please enter a valid text file in the obs_list field in the YAML file.")
if cross_calibrate:
    if not os.path.exists(cal_map1_path): 
        logger.error("Cannot find calibration map 1 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_map1_path}")
    if not os.path.exists(cal_ivar1_path): 
        logger.error("Cannot find calibration ivar 1 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_ivar1_path}")
    if not os.path.exists(cal_beam1_path): 
        logger.error("Cannot find calibration beam 1 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_beam1_path}")
    if not os.path.exists(cal_map2_path): 
        logger.error("Cannot find calibration map 2 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_map2_path}")
    if not os.path.exists(cal_ivar2_path): 
        logger.error("Cannot find calibration ivar 2 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_ivar2_path}")
    if not os.path.exists(cal_beam2_path): 
        logger.error("Cannot find calibration beam 2 file! Check config. Exiting.")
        raise FileNotFoundError(f"File not found: {cal_beam2_path}")
# Setting up bins for calibration
cal_bins = np.arange(cal_lmin, cal_lmax, cal_bin_size)
cal_centers = (cal_bins[1:] + cal_bins[:-1])/2.0

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
# These three only exist in the serial code since they act on the
# full dictionary at the end
plot_all_spectra = config['plot_all_spectra']
plot_summary_spectra = config['plot_summary_spectra'] 
plot_angle_hist = config['plot_angle_hist']

# Load CAMB EE and BB spectrum (BB just for plotting)
logger.info("Starting to load CAMB spectra")
ell_camb,DlEE_camb,DlBB_camb = np.loadtxt(camb_file, usecols=(0,2,3), unpack=True)
if ell_camb[0]==2:
    # ell runs from 2 to lmax in older CAMB files
    arr_len = ell_camb.size + 2
    ell = np.zeros(arr_len)
    ell[1] = 1.0
    ell[2:] = ell_camb
    ClEE = np.zeros(arr_len)
    ClBB = np.zeros(arr_len)
    ClEE[2:] = DlEE_camb * 2 * np.pi / (ell_camb*(ell_camb+1.0))
    ClBB[2:] = DlBB_camb * 2 * np.pi / (ell_camb*(ell_camb+1.0))
else:
    # in newer camb outputs, ell starts at zero
    arr_len = ell_camb.size
    ell = ell_camb
    ClEE = np.zeros(arr_len)
    ClBB = np.zeros(arr_len)
    ClEE[2:] = DlEE_camb[2:] * 2 * np.pi / (ell_camb[2:]*(ell_camb[2:]+1.0))
    ClBB[2:] = DlBB_camb[2:] * 2 * np.pi / (ell_camb[2:]*(ell_camb[2:]+1.0))
digitized = np.digitize(ell, bins, right=True)
CAMB_ClEE_binned = np.bincount(digitized, ClEE.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
CAMB_ClBB_binned = np.bincount(digitized, ClBB.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
logger.info("Finished loading CAMB spectra")

# Loading all beams
logger.info("Starting to load beams")
if freq=='f090':
    # Only pa5 and pa6 at f090
    logger.info(f"Using pa5 beam {pa5_beam_path}")
    logger.info(f"Using ref pa5 beam {ref_pa5_beam_path}")
    logger.info(f"Using pa6 beam {pa6_beam_path}")
    logger.info(f"Using ref pa6 beam {ref_pa6_beam_path}")
    pa4_beam = []
    pa5_beam = aoa.load_and_bin_beam(pa5_beam_path,bins)
    pa6_beam = aoa.load_and_bin_beam(pa6_beam_path,bins)
    ref_pa4_beam = []
    ref_pa5_beam = aoa.load_and_bin_beam(ref_pa5_beam_path,bins)
    ref_pa6_beam = aoa.load_and_bin_beam(ref_pa6_beam_path,bins)
    if plot_beam:
        pa5_beam_name = os.path.split(pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa5_beam_name, centers, pa5_beam)
        pa6_beam_name = os.path.split(pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa6_beam_name, centers, pa6_beam)
        ref_pa5_beam_name = os.path.split(ref_pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa5_beam_name, centers, ref_pa5_beam)
        ref_pa6_beam_name = os.path.split(ref_pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa6_beam_name, centers, ref_pa6_beam)
elif freq=='f150':
    logger.info(f"Using pa4 beam {pa4_beam_path}")
    logger.info(f"Using ref pa4 beam {ref_pa4_beam_path}")
    logger.info(f"Using pa5 beam {pa5_beam_path}")
    logger.info(f"Using ref pa5 beam {ref_pa5_beam_path}")
    logger.info(f"Using pa6 beam {pa6_beam_path}")
    logger.info(f"Using ref pa6 beam {ref_pa6_beam_path}")
    pa4_beam = aoa.load_and_bin_beam(pa4_beam_path,bins)
    pa5_beam = aoa.load_and_bin_beam(pa5_beam_path,bins)
    pa6_beam = aoa.load_and_bin_beam(pa6_beam_path,bins)
    ref_pa4_beam = aoa.load_and_bin_beam(ref_pa4_beam_path,bins)
    ref_pa5_beam = aoa.load_and_bin_beam(ref_pa5_beam_path,bins)
    ref_pa6_beam = aoa.load_and_bin_beam(ref_pa6_beam_path,bins)
    if plot_beam:
        pa4_beam_name = os.path.split(pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa4_beam_name, centers, pa4_beam)
        pa5_beam_name = os.path.split(pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa5_beam_name, centers, pa5_beam)
        pa6_beam_name = os.path.split(pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa6_beam_name, centers, pa6_beam)
        ref_pa4_beam_name = os.path.split(ref_pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa4_beam_name, centers, ref_pa4_beam)
        ref_pa5_beam_name = os.path.split(ref_pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa5_beam_name, centers, ref_pa5_beam)
        ref_pa6_beam_name = os.path.split(ref_pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa6_beam_name, centers, ref_pa6_beam)
elif freq=='f220':
    logger.info(f"Using pa4 beam {pa4_beam_path}")
    logger.info(f"Using ref pa4 beam {ref_pa4_beam_path}")
    # Only pa4 at f220
    pa4_beam = aoa.load_and_bin_beam(pa4_beam_path,bins)
    pa5_beam = []
    pa6_beam = []
    ref_pa4_beam = aoa.load_and_bin_beam(ref_pa4_beam_path,bins)
    ref_pa5_beam = []
    ref_pa6_beam = []
    if plot_beam:
        pa4_beam_name = os.path.split(pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, pa4_beam_name, centers, pa4_beam)
        ref_pa4_beam_name = os.path.split(ref_pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
        aop.plot_beam(output_dir_path, ref_pa4_beam_name, centers, ref_pa4_beam)
logger.info("Finished loading beams")

# Calculate filtering transfer function once since filtering is same for all maps
tfunc = aoa.get_tfunc(kx_cut, ky_cut, bins)
if plot_tfunc:
    aop.plot_tfunc(output_dir_path, kx_cut, ky_cut, centers, tfunc)

# Loading in reference maps for all arrays depending on frequency
# If ref maps are the same for different arrays, only load once to save time and memory
if freq=='f090':
    logger.info(f"Using pa5 ref map {ref_pa5_path} and ivar {ref_pa5_ivar_path}")
    ref_pa5_maps, ref_pa5_ivar = aoa.load_ref_map(ref_pa5_path,ref_pa5_ivar_path)
    logger.info(f"Using pa6 ref map {ref_pa6_path} and ivar {ref_pa6_ivar_path}")
    if ref_pa6_path == ref_pa5_path and ref_pa6_ivar_path == ref_pa5_ivar_path:
        ref_pa6_maps, ref_pa6_ivar = ref_pa5_maps, ref_pa5_ivar
    else:
        ref_pa6_maps, ref_pa6_ivar = aoa.load_ref_map(ref_pa6_path,ref_pa6_ivar_path)
elif freq=='f150':
    logger.info(f"Using pa4 ref map {ref_pa4_path} and ivar {ref_pa4_ivar_path}")
    ref_pa4_maps, ref_pa4_ivar = aoa.load_ref_map(ref_pa4_path,ref_pa4_ivar_path)
    logger.info(f"Using pa5 ref map {ref_pa5_path} and ivar {ref_pa5_ivar_path}")
    if ref_pa5_path == ref_pa4_path and ref_pa5_ivar_path == ref_pa4_ivar_path:
        ref_pa5_maps, ref_pa5_ivar = ref_pa4_maps, ref_pa4_ivar
    else:
        ref_pa5_maps, ref_pa5_ivar = aoa.load_ref_map(ref_pa5_path,ref_pa5_ivar_path)
    logger.info(f"Using pa6 ref map {ref_pa6_path} and ivar {ref_pa6_ivar_path}")
    if ref_pa6_path == ref_pa4_path and ref_pa6_ivar_path == ref_pa4_ivar_path:
        ref_pa6_maps, ref_pa6_ivar = ref_pa4_maps, ref_pa4_ivar
    elif ref_pa6_path == ref_pa5_path and ref_pa6_ivar_path == ref_pa5_ivar_path:
        ref_pa6_maps, ref_pa6_ivar = ref_pa5_maps, ref_pa5_ivar
    else:
        ref_pa6_maps, ref_pa6_ivar = aoa.load_ref_map(ref_pa6_path,ref_pa6_ivar_path)
elif freq=='f220':
    logger.info(f"Using pa4 ref map {ref_pa4_path} and ivar {ref_pa4_ivar_path}")
    ref_pa4_maps, ref_pa4_ivar = aoa.load_ref_map(ref_pa4_path,ref_pa4_ivar_path)

# Loading in galaxy mask
logger.info("Starting to load galaxy mask")
galaxy_mask = enmap.read_map(galaxy_mask_path)
logger.info("Finished loading galaxy mask")

if cross_calibrate:
    # Loading in calibration ivar and maps for cross-correlation
    logger.info("Starting to load calibration maps and beams for cross-correlation")

    # Check if the calibration map path is the same as the relevant reference map
    # Will just assign the variable to save time if they are the same
    # cal1 must be from pa5 and cal2 must be from pa6
    if cal_map1_path == ref_pa5_path:
        cal_T_map1_act_footprint = ref_pa5_maps[0]
        cal_T_ivar1_act_footprint = ref_pa5_ivar*2.0 # To get back to T noise
    else:
        # Check that it is a pa5 map
        if 'pa5' in cal_map1_path.split('_'):
            # only loading in T maps and trimming immediately to galaxy mask's shape to save memory (legacy from using Planck maps)
            cal_T_map1_act_footprint = enmap.read_map(cal_map1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
            cal_T_ivar1_act_footprint = enmap.read_map(cal_ivar1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))
        else:
            raise ValueError('cal_map1_path must be a pa5 coadd to avoid noise bias!')
    # Still load beam again in case cal_bins is different from bins
    cal_T_beam1 = aoa.load_and_bin_beam(cal_beam1_path,cal_bins)
    if cal_map2_path == ref_pa6_path:
        cal_T_map2_act_footprint = ref_pa6_maps[0]
        cal_T_ivar2_act_footprint = ref_pa6_ivar*2.0 # To get back to T noise
    else: 
        # Check that it is a pa6 map
        if 'pa6' in cal_map2_path.split('_'):   
            cal_T_map2_act_footprint = enmap.read_map(cal_map2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
            cal_T_ivar2_act_footprint = enmap.read_map(cal_ivar2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))
        else:
            raise ValueError('cal_map2_path must be a pa6 coadd to avoid noise bias!')
    cal_T_beam2 = aoa.load_and_bin_beam(cal_beam2_path,cal_bins)
    logger.info("Finished loading calibration maps and beams for cross-correlation")
else:
    cal_T_map1_act_footprint = None
    cal_T_ivar1_act_footprint = None
    cal_T_beam1 = []
    cal_T_map2_act_footprint = None
    cal_T_ivar2_act_footprint = None
    cal_T_beam2 = []

#######################################################################################################
# Defining single process function for main loop
def process(map_name, obs_list_path, logger, 
            ref_maps, ref_ivar, galaxy_mask,
            kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
            plot_maps, plot_likelihood, output_dir_path, 
            bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
            depth1_beam, cal_T_beam1, cal_T_beam2, ref_beam, tfunc, 
            num_pts, angle_min_deg, angle_max_deg, fit_method, 
            cross_calibrate, cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
            cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
            cal_bins, cal_centers, y_min, y_max, cal_num_pts, cal_fit_method):
    """Function to call for each map inside each process"""
    map_path = obs_list_path + map_name
    # outputs will be 1 if the map is cut, a bunch of things needed
    # for time estimation and TT calibration if not
    output_dict, outputs = aoa.estimate_pol_angle(map_path, map_name, logger, ref_maps, ref_ivar, galaxy_mask,
                                                  kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                                                  plot_maps, plot_likelihood, output_dir_path, 
                                                  bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
                                                  depth1_beam, ref_beam, tfunc, 
                                                  num_pts, angle_min_deg, angle_max_deg, fit_method)

    if fit_method == 'skewnorm' or fit_method == 'all':
        fit_values = [output_dict['meas_angle_skewnorm-method'], output_dict['meas_errbar_skewnorm-method']]
    elif fit_method=='fwhm':
        fit_values = [output_dict['meas_angle_fwhm-method'], output_dict['meas_errbar_fwhm-method']]
    elif fit_method=='curvefit':
        fit_values = [output_dict['meas_angle_gauss-curvefit'], output_dict['meas_errbar_gauss-curvefit']]
    else:
        fit_values = [output_dict['meas_angle_gauss-moment'], output_dict['meas_errbar_gauss-moment']]
    if output_dict['map_cut']==0:
        depth1_mask = outputs[0]
        depth1_mask_indices = outputs[1]
        depth1_ivar = outputs[2]
        depth1_T = outputs[3]
        w_depth1 = outputs[4]
        w2_depth1 = output_dict['w2_depth1']
        logger.info(f"Fit values: {fit_values}")
        # depth1_mask will be the doubly tapered one if ivar weighting is on, the first filtering one if not
        name_timestamp, median_timestamp, initial_timestamp, final_timestamp  = aoa.calc_timestamps(map_path, depth1_mask)
        logger.info(f"Name timestamp: {name_timestamp}; Median timestamp: {median_timestamp}; Initial timestamp: {initial_timestamp}; Final timestamp: {final_timestamp}")
        output_dict.update({'name_timestamp': name_timestamp, 'median_timestamp': median_timestamp,
                            'initial_timestamp': initial_timestamp,'final_timestamp': final_timestamp})

        if cross_calibrate:
            # Calling a single function to do all the TT cross-calibration
            # This makes the code a bit harder to read but improves memory usage by allowing
            # intermediate maps to be cleaned up when function call ends.
            map_array = map_name.split('_')[2]
            cal_output_dict = aoa.cross_calibrate(map_array, cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
                                                  cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
                                                  depth1_ivar, depth1_mask, depth1_mask_indices,
                                                  galaxy_mask, depth1_T, w_depth1, w2_depth1, cal_centers,
                                                  cal_bins, tfunc, kx_cut, ky_cut, unpixwin, filter_radius, 
                                                  use_ivar_weight, depth1_beam, cal_T_beam1, cal_T_beam2, y_min, 
                                                  y_max, cal_num_pts, cal_fit_method)
            # Printing out calibration factor and errorbar
            cal_fit_values = (cal_output_dict['cal_factor'], cal_output_dict['cal_factor_errbar'])
            logger.info(f"TT calibration fit values: {cal_fit_values}")
            # Adding calibration keys to final dictionary
            output_dict.update(cal_output_dict)
    else:
        if cross_calibrate:
            # To ensure all keys are present whether map is cut or not
            cal_ell_len = len(cal_centers)
            output_dict.update({'cal_ell': cal_centers, 'T1xpa5T': np.zeros(cal_ell_len), 'pa5Txpa6T': np.zeros(cal_ell_len),
                                 'pa5Txpa5T': np.zeros(cal_ell_len), 'pa6Txpa6T': np.zeros(cal_ell_len), 
                                 'T1xT1': np.zeros(cal_ell_len), 'T1xpa6T': np.zeros(cal_ell_len), 'cal_bincount': np.zeros(cal_ell_len),
                                 'cal_factor': -9999, 'cal_factor_errbar': -9999,
                                 'w2_depth1xpa5': -9999, 'w2_depth1xpa6': -9999,
                                 'w2_pa5xpa6': -9999, 'w2_pa5xpa5': -9999, 'w2_pa6xpa6': -9999,
                                 'w2w4_556d': -9999, 'w2w4_665d': -9999, 'w2w4_depth1xpa5': -9999, 
                                 'w2w4_pa5xpa6': -9999, 'w2w4_depth1xpa6': -9999})

    return output_dict

#######################################################################################################
# Run main sequence

with open(obs_list, 'r') as f:
    maps = f.read().splitlines()

angle_estimates = []
results_output = {}

for map_name in tqdm(maps):
    logger.info(f"Processing {map_name}")

    try:
        map_array = map_name.split('_')[2]
        if freq=='f090':
            if map_array == 'pa5':
                depth1_beam = pa5_beam
                ref_maps = ref_pa5_maps
                ref_ivar = ref_pa5_ivar
                ref_beam = ref_pa5_beam
            elif map_array == 'pa6':
                depth1_beam = pa6_beam
                ref_maps = ref_pa6_maps
                ref_ivar = ref_pa6_ivar
                ref_beam = ref_pa6_beam
            else:
                logger.error(f"Map array {map_array} must be 'pa5' or 'pa6' at {freq}!")
                raise ValueError(f"Map array {map_array} must be 'pa5' or 'pa6' at {freq}!")
        elif freq=='f150':
            if map_array == 'pa4':
                depth1_beam = pa4_beam
                ref_maps = ref_pa4_maps
                ref_ivar = ref_pa4_ivar
                ref_beam = ref_pa4_beam
            elif map_array == 'pa5':
                depth1_beam = pa5_beam
                ref_maps = ref_pa5_maps
                ref_ivar = ref_pa5_ivar
                ref_beam = ref_pa5_beam
            elif map_array == 'pa6':
                depth1_beam = pa6_beam
                ref_maps = ref_pa6_maps
                ref_ivar = ref_pa6_ivar
                ref_beam = ref_pa6_beam
            else:
                logger.error(f"Map array {map_array} must be 'pa4', 'pa5' or 'pa6' at {freq}!")
                raise ValueError(f"Map array {map_array} must be 'pa4', 'pa5' or 'pa6' at {freq}!")
        elif freq=='f220':
            if map_array == 'pa4':
                depth1_beam = pa4_beam
                ref_maps = ref_pa4_maps
                ref_ivar = ref_pa4_ivar
                ref_beam = ref_pa4_beam
            else:
                logger.error(f"Map array {map_array} must be 'pa4' at {freq}!")
                raise ValueError(f"Map array {map_array} must be 'pa4' at {freq}!")

        output_dict = process(map_name, obs_list_path, logger, 
                            ref_maps, ref_ivar, galaxy_mask,
                            kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                            plot_maps, plot_likelihood, output_dir_path, 
                            bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
                            depth1_beam, cal_T_beam1, cal_T_beam2, ref_beam, tfunc, 
                            num_pts, angle_min_deg, angle_max_deg, fit_method, 
                            cross_calibrate, cal_T_map1_act_footprint, cal_T_map2_act_footprint, 
                            cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
                            cal_bins, cal_centers, y_min, y_max, cal_num_pts, cal_fit_method)
        # At the end, save the output_dict to a npy file - there will be one per map
        # This ensures at least some of the results are saved if the script crashes
        # with memory issues (historically, a scourge of this project...).
        # Can be loaded with np.load(results_output_fname, allow_pickle=True).item()
        # If the mask cuts the whole map, this will be the dict with -9999 everywhere
        map_name_no_ending = map_name[:-8] # Removing 'map.fits'
        results_output_fname = output_dir_path + map_name_no_ending + output_time + '_results.npy'
        np.save(results_output_fname, output_dict)        

        # Assign the output_dict to the line in results_output so that one npy file
        # can be made as long as the script doesn't crash from memory usage
        results_output[map_name] = output_dict
        fit_values = [output_dict['meas_angle'], output_dict['meas_errbar']]
        angle_estimates.append(fit_values) 
    except Exception as e:
        logger.error(f"Map {map_name} failed with error {e}")
        # Ensuring that plotting code still works for all maps
        ell_len = len(centers)
        output_dict = {'ell': centers, 'E1xB2': np.zeros(ell_len), 'E2xB1': np.zeros(ell_len), 
                       'E1xE1': np.zeros(ell_len), 'B2xB2': np.zeros(ell_len), 'E2xE2': np.zeros(ell_len),
                       'B1xB1': np.zeros(ell_len), 'E1xE2': np.zeros(ell_len), 'B1xB2': np.zeros(ell_len),
                       'E1xB1': np.zeros(ell_len), 'E2xB2': np.zeros(ell_len), 'bincount': np.zeros(ell_len),
                       'estimator': np.zeros(ell_len), 'covariance': np.zeros(ell_len),
                       'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                       'w2_depth1': -9999, 'w2_cross': -9999, 'w2_ref': -9999, 'fsky': -9999,
                       'w2w4_depth1': -9999, 'w2w4_cross': -9999, 'w2w4_ref': -9999,
                       'meas_angle': -9999, 'meas_errbar': -9999,
                       'initial_timestamp': -9999, 'median_timestamp': -9999, 
                       'ivar_sum': -9999, 'residual_mean': -9999, 
                       'residual_sum': -9999, 'map_cut': 1}
        if cross_calibrate:
            # To ensure all keys are present whether map is cut or not
            cal_ell_len = len(cal_centers)
            output_dict.update({'cal_ell': cal_centers, 'T1xpa5T': np.zeros(cal_ell_len), 'pa5Txpa6T': np.zeros(cal_ell_len),
                                 'pa5Txpa5T': np.zeros(cal_ell_len), 'pa6Txpa6T': np.zeros(cal_ell_len), 
                                 'T1xT1': np.zeros(cal_ell_len), 'T1xpa6T': np.zeros(cal_ell_len), 'cal_bincount': np.zeros(cal_ell_len),
                                 'cal_factor': -9999, 'cal_factor_errbar': -9999,
                                 'w2_depth1xpa5': -9999, 'w2_depth1xpa6': -9999,
                                 'w2_pa5xpa6': -9999, 'w2_pa5xpa5': -9999, 'w2_pa6xpa6': -9999,
                                 'w2w4_556d': -9999, 'w2w4_665d': -9999, 'w2w4_depth1xpa5': -9999, 
                                 'w2w4_pa5xpa6': -9999, 'w2w4_depth1xpa6': -9999})
        map_name_no_ending = map_name[:-8] # Removing 'map.fits'
        results_output_fname = output_dir_path + map_name_no_ending + output_time + '_results.npy'
        np.save(results_output_fname, output_dict) 
        results_output[map_name] = output_dict
        angle_estimates.append([-9999,-9999])        

# Plot summary plots if desired
if plot_all_spectra:
    logger.info("Beginning to save plots for all spectra. This could take a while.")
    aop.plot_spectra_individually(output_dir_path, results_output)
    logger.info("Finished saving plots for all spectra.")
if plot_summary_spectra:
    logger.info("Beginning to save summary spectra plots.")
    aop.plot_spectra_summary(output_dir_path, results_output)
    logger.info("Finished saving summary spectra plots.")
if plot_angle_hist:
    logger.info("Plotting histogram of angles")
    aop.plot_angle_hist(output_dir_path, np.array(angle_estimates)[:,0], maps)

# Saving full results to a numpy file
# Can be loaded with np.load(results_output_fname, allow_pickle=True).item()
results_output_fname = output_dir_path + 'angle_calc_' + output_time + '_results.npy'
np.save(results_output_fname, results_output)

# Dump all config info to YAML
config_output_dict = config
config_output_dict['list_of_maps'] = maps
output_dict['results_output_fname'] = results_output_fname
config_output_name = output_dir_path + 'angle_calc_config_' + output_time + ".yaml"
with open(config_output_name, 'w') as file:
    yaml.dump(config_output_dict, file)
logger.info("Finished running get_angle_from_depth1_ps.py. Output is in: " + str(output_dir_path))
stop_time = time.time()
duration = stop_time-start_time
logger.info("Script took {:1.3f} seconds".format(duration))
