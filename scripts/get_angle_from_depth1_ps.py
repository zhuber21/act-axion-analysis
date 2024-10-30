import numpy as np
import yaml
import argparse
import time
import glob
import os
import sys
from pixell import enmap
from tqdm import tqdm
import healpy as hp
import axion_osc_analysis_depth1_ps as aoa

start_time = time.time()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("config_file", help=
    "The name of the YAML config file",type=str)
args = parser.parse_args()

# Load in the YAML file
yaml_name = args.config_file
print("Using config file: " + str(yaml_name))

with open(yaml_name, 'r') as file:
    config = yaml.safe_load(file)

# Setting common variables set in the config file
kx_cut = config['kx_cut']
ky_cut = config['ky_cut']
unpixwin = config['unpixwin']
filter_radius = config['filter_radius']
angle_min_deg = config['angle_min_deg']
angle_max_deg = config['angle_max_deg']
num_pts = config['num_pts']
use_curvefit = config['use_curvefit']
use_ivar_weight = config['use_ivar_weight']
cross_planck = config['cross_planck']

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
    print("Cannot find CAMB file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(ref_path): 
    print("Cannot find reference map file! Check config. Exiting.")
    sys.exit()
#if not os.path.exists(ref_beam_path): 
#    print("Cannot find beam file! Check config. Exiting.")
#    sys.exit()
if not os.path.exists(ref_ivar_path): 
    print("Cannot find ref map ivar file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(pa4_beam_path): 
    print("Cannot find pa4 beam file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(pa5_beam_path): 
    print("Cannot find pa5 beam file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(pa6_beam_path): 
    print("Cannot find pa6 beam file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(galaxy_mask_path):
    print("Cannot find galaxy mask file! Check config. Exiting.")
    sys.exit()
if not os.path.exists(obs_list): 
    print("Cannot find observation list! Check config. Exiting.")
    sys.exit()
if obs_list[-3:] == 'txt':
    print("Using list of observations at: " + str(obs_list))
else:
    print("Please enter a valid text file in the obs_list field in the YAML file. Exiting.")
    sys.exit()
if cross_planck:
    planck_split1_path = config['planck_split1_path']
    planck_ivar1_path = config['planck_ivar1_path']
    planck_split2_path = config['planck_split2_path']
    planck_ivar2_path = config['planck_ivar2_path']
    if not os.path.exists(planck_split1_path): 
        print("Cannot find Planck split 1 map file! Check config. Exiting.")
        sys.exit()
    if not os.path.exists(planck_ivar1_path): 
        print("Cannot find Planck split 1 ivar file! Check config. Exiting.")
        sys.exit()
    if not os.path.exists(planck_split2_path): 
        print("Cannot find Planck split 2 map file! Check config. Exiting.")
        sys.exit()
    if not os.path.exists(planck_ivar2_path): 
        print("Cannot find Planck split 2 ivar file! Check config. Exiting.")
        sys.exit()


# Output file path
output_time = str(int(round(time.time())))
output_dir_root = config['output_dir_root']
if not os.path.exists(output_dir_root): # Make sure root path is right
    print("Output directory does not exist! Exiting.")
    sys.exit()
output_dir_path = output_dir_root + "/angle_calc_" + output_time + '/'
if not os.path.exists(output_dir_path): # Make new folder for this run - should be unique
    os.makedirs(output_dir_path)

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
    print("Please use valid bin_settings! Options are 'regular' and 'DR4'. Exiting.")
    sys.exit()
print("Finished loading bins")

# Setting plotting settings
plot_maps = config['plot_maps']
plot_all_spectra = config['plot_all_spectra']
plot_summary_spectra = config['plot_summary_spectra']
plot_likelihood = config['plot_likelihood']
plot_beam = config['plot_beam']
plot_tfunc = config['plot_tfunc'] 
plot_angle_hist = config['plot_angle_hist']

# Load CAMB EE and BB spectrum (BB just for plotting)
print("Starting to load CAMB spectra")
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
print("Finished loading CAMB spectra")

# Loading in reference maps
print("Starting to load ref map")
ref_maps, ref_ivar = aoa.load_ref_map(ref_path,ref_ivar_path)
print("Finished loading ref map")

# Loading all beams
print("Starting to load beams")
pa4_beam = aoa.load_and_bin_beam(pa4_beam_path,bins)
pa5_beam = aoa.load_and_bin_beam(pa5_beam_path,bins)
pa6_beam = aoa.load_and_bin_beam(pa6_beam_path,bins)
# For now, average these beams to get coadd/ref beam
ref_beam = (pa4_beam+pa5_beam+pa6_beam)/3.0
if plot_beam:
    pa4_beam_name = os.path.split(pa4_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
    aoa.plot_beam(output_dir_path, pa4_beam_name, centers, pa4_beam)
    pa5_beam_name = os.path.split(pa5_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
    aoa.plot_beam(output_dir_path, pa5_beam_name, centers, pa5_beam)
    pa6_beam_name = os.path.split(pa6_beam_path)[1][:-4] # Extracting file name from path and dropping '.txt'
    aoa.plot_beam(output_dir_path, pa6_beam_name, centers, pa6_beam)
    ref_beam_name = "coadd_avg_beam"
    aoa.plot_beam(output_dir_path, ref_beam_name, centers, ref_beam)
print("Finished loading beams")

# Loading in galaxy mask
print("Starting to load galaxy mask")
galaxy_mask = enmap.read_map(galaxy_mask_path)
print("Finished loading galaxy mask")

if cross_planck:
    # Loading in Planck ivar and splits for cross-correlation
    print("Starting to load Planck splits for cross-correlation")
    # only loading in T maps and trimming immediately to galaxy mask's shape to save memory
    planck_T_split1_act_footprint = enmap.read_map(planck_split1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    planck_T_ivar1_act_footprint = enmap.read_map(planck_ivar1_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    planck_T_split2_act_footprint = enmap.read_map(planck_split2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    planck_T_ivar2_act_footprint = enmap.read_map(planck_ivar2_path, geometry=(galaxy_mask.shape,galaxy_mask.wcs))[0]
    # Generating a Gaussian beam for the Planck maps as first-order correction
    fwhm_planck_arcmin = 7.22
    planck_beam = hp.sphtfunc.gauss_beam(np.deg2rad(fwhm_planck_arcmin/60.0),lmax=lmax)
    planck_beam_norm = planck_beam[1:] / np.max(planck_beam[1:])
    digitized = np.digitize(range(1,lmax+1), bins, right=True)
    planck_beam_binned = np.bincount(digitized, planck_beam_norm.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
    print("Finished loading Planck splits for cross-correlation")

maps = []
angle_estimates = []
spectra_output = {}

# Calculate filtering transfer function once since filtering is same for all maps
tfunc = aoa.get_tfunc(kx_cut, ky_cut, bins)
if plot_tfunc:
    aoa.plot_tfunc(output_dir_path, kx_cut, ky_cut, centers, tfunc)

with open(obs_list) as f:
    lines = f.read().splitlines()

for line in tqdm(lines):
    print(line)
    map_path = obs_list_path + line

    outputs = aoa.load_and_filter_depth1(map_path, ref_maps, galaxy_mask, 
                                            kx_cut, ky_cut, unpixwin, 
                                            filter_radius=filter_radius,plot_maps=plot_maps,
                                            output_dir=output_dir_path)
    # If the full map has been cut by the galaxy mask, it return error code 1 instead of the regular outputs
    if outputs == 1:
        # saves output flag so we can see it is cut
        print("Map " + line + " was completely cut by galaxy mask.")
        ell_len = len(centers)
        spectra_output[line] = {'ell': centers, 'E1xB2': np.zeros(ell_len), 'E2xB1': np.zeros(ell_len), 
                                'E1xE1': np.zeros(ell_len), 'B2xB2': np.zeros(ell_len), 'E2xE2': np.zeros(ell_len),
                                'B1xB1': np.zeros(ell_len), 'E1xE2': np.zeros(ell_len), 'B1xB2': np.zeros(ell_len),
                                'E1xB1': np.zeros(ell_len), 'E2xB2': np.zeros(ell_len), 'binned_nu': np.zeros(ell_len),
                                'estimator': np.zeros(ell_len), 'covariance': np.zeros(ell_len),
                                'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                                'w2_depth1': -9999, 'w2_cross': -9999, 'w2_ref': -9999,
                                'meas_angle': -9999, 'meas_errbar': -9999, 
                                'ivar_sum': -9999, 'map_cut': 1}
        angle_estimates.append((-9999,-9999))
        maps.append(line)
        if plot_likelihood:
            # Make empty likelihood plot for web viewer
            angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
            map_name = os.path.split(line)[1][:-9] # removing "_map.fits"
            aoa.plot_likelihood(output_dir_path, map_name, angles_deg, np.zeros(num_pts), (-9999,-9999))
    else: 
        # Otherwise do everything you would normally do
        depth1_TEB = outputs[0]
        depth1_ivar = outputs[1]
        depth1_footprint = outputs[2]
        ref_TEB = outputs[3]
        ivar_sum = outputs[4]

        if cross_planck:
            # Moving trimming, ivar weighting, and Fourier transform to function
            # to avoid multiplying extra maps in memory
            planck_split1_fourier, planck_split2_fourier, w2_planck = aoa.planck_trim_and_fourier_transform(planck_T_split1_act_footprint,planck_T_ivar1_act_footprint,
                                                  planck_T_split2_act_footprint,planck_T_ivar2_act_footprint,
                                                  depth1_TEB[0].shape,depth1_TEB[0].wcs,depth1_footprint,use_ivar_weight)

        if use_ivar_weight:
            # Ivar weighting for depth-1 map
            depth1_TEB = aoa.apply_ivar_weighting(depth1_TEB, depth1_ivar, depth1_footprint)

            # Ivar weighting for reference map - already filtered and trimmed from ref_TEB above
            ref_map_trimmed_ivar = enmap.extract(ref_ivar,depth1_TEB[0].shape,depth1_TEB[0].wcs)
            ref_TEB = aoa.apply_ivar_weighting(ref_TEB, ref_map_trimmed_ivar, depth1_footprint)

            # Calculating approx correction for loss of power due to tapering for spectra for depth-1
            w_depth1 = depth1_ivar*depth1_footprint
            # Calculating approx correction for loss of power due to tapering for spectra for ref
            w_ref = ref_map_trimmed_ivar*depth1_footprint    
        else:
            # No ivar weighting
            w_depth1 = depth1_footprint # use this if using flat weighting
            w_ref = depth1_footprint # use this if using flat weighting 

        depth1_E = depth1_TEB[1]
        depth1_B = depth1_TEB[2]
        ref_E = ref_TEB[1]
        ref_B = ref_TEB[2]

        # Calculating w2 factors - all the same if not using ivar weighting, but different if using it
        w2_depth1 = np.mean(w_depth1**2)
        w2_cross = np.mean(w_depth1*w_ref)
        w2_ref = np.mean(w_ref**2)

        # Selecting the correct beam
        map_array = line.split('_')[2]
        if map_array == 'pa4':
            depth1_beam = pa4_beam
        elif map_array == 'pa5':
            depth1_beam = pa5_beam
        elif map_array == 'pa6':
            depth1_beam = pa6_beam
        else:
            print("Map " + line + " not in standard format for array beam selection! Choosing averaged beam.")
            depth1_beam = ref_beam

        # Calculate spectra
        # Spectra for estimator
        binned_E1xB2, bincount = aoa.spectrum_from_maps(depth1_E, ref_B, b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_E2xB1, _ = aoa.spectrum_from_maps(depth1_B, ref_E, b_ell_bin_1=ref_beam, b_ell_bin_2=depth1_beam, w2=w2_cross, bins=bins)
        # Spectra for covariance
        binned_E1xE1, _ = aoa.spectrum_from_maps(depth1_E, depth1_E, b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_B2xB2, _ = aoa.spectrum_from_maps(ref_B, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)
        binned_E2xE2, _ = aoa.spectrum_from_maps(ref_E, ref_E, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)
        binned_B1xB1, _ = aoa.spectrum_from_maps(depth1_B, depth1_B, b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_E1xE2, _ = aoa.spectrum_from_maps(depth1_E, ref_E, b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_B1xB2, _ = aoa.spectrum_from_maps(depth1_B, ref_B, b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_E1xB1, _ = aoa.spectrum_from_maps(depth1_E, depth1_B, b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_E2xB2, _ = aoa.spectrum_from_maps(ref_E, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)    
        # Accounting for transfer function
        binned_E1xB2 /= tfunc
        binned_E2xB1 /= tfunc
        binned_E1xE1 /= tfunc
        binned_B2xB2 /= tfunc
        binned_E2xE2 /= tfunc
        binned_B1xB1 /= tfunc
        binned_E1xE2 /= tfunc
        binned_B1xB2 /= tfunc
        binned_E1xB1 /= tfunc
        binned_E2xB2 /= tfunc
        # Accounting for modes lost to the mask and filtering - always uses w2 without ivar, regardless of ivar weighting
        binned_nu = bincount*np.mean(depth1_footprint**2)*tfunc
        
        if cross_planck:
            # Depth-1 T cross ACT ref T
            binned_T1xT2, _ = aoa.spectrum_from_maps(depth1_TEB[0], ref_TEB[0], b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
            binned_T1xT2 /= tfunc
            # Planck split 1 T cross Planck split 2 T
            # No tfunc correction because no filtering for these maps at the moment
            binned_P1TxP2T, _ = aoa.spectrum_from_maps(planck_split1_fourier, planck_split2_fourier, b_ell_bin_1=planck_beam_binned, 
                                                       b_ell_bin_2=planck_beam_binned, w2=w2_planck, bins=bins)
            # Taking ratio and averaging to get rough calibration factor
            cal_factor = np.mean(binned_T1xT2/binned_P1TxP2T)

        # Calculate estimator and covariance
        estimator = binned_E1xB2-binned_E2xB1
        covariance = ((1/binned_nu)*((binned_E1xE1*binned_B2xB2+binned_E1xB2**2)
                                    +(binned_E2xE2*binned_B1xB1+binned_E2xB1**2)
                                    -2*(binned_E1xE2*binned_B1xB2+binned_E1xB1*binned_E2xB2)))

        fit_values = aoa.sample_likelihood_and_fit(estimator,covariance,CAMB_ClEE_binned,num_pts=num_pts,
                                                angle_min_deg=angle_min_deg, angle_max_deg=angle_max_deg,
                                                use_curvefit=use_curvefit,plot_like=plot_likelihood,
                                                output_dir=output_dir_path,map_fname=line)

        print(fit_values)
        angle_estimates.append(fit_values)
        maps.append(line)
        if cross_planck:
            spectra_output[line] = {'ell': centers, 'E1xB2': binned_E1xB2, 'E2xB1': binned_E2xB1, 
                                    'E1xE1': binned_E1xE1, 'B2xB2': binned_B2xB2, 'E2xE2': binned_E2xE2,
                                    'B1xB1': binned_B1xB1, 'E1xE2': binned_E1xE2, 'B1xB2': binned_B1xB2,
                                    'E1xB1': binned_E1xB1, 'E2xB2': binned_E2xB2, 'binned_nu': binned_nu,
                                    'estimator': estimator, 'covariance': covariance,
                                    'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                                    'w2_depth1': w2_depth1, 'w2_cross': w2_cross, 'w2_ref': w2_ref,
                                    'meas_angle': fit_values[0], 'meas_errbar': fit_values[1], 
                                    'ivar_sum': ivar_sum, 'map_cut': 0,
                                    'T1xT2': binned_T1xT2, 'P1TxP2T': binned_P1TxP2T,
                                    'cal_factor': cal_factor, 'w2_planck': w2_planck}
        else:
            spectra_output[line] = {'ell': centers, 'E1xB2': binned_E1xB2, 'E2xB1': binned_E2xB1, 
                                    'E1xE1': binned_E1xE1, 'B2xB2': binned_B2xB2, 'E2xE2': binned_E2xE2,
                                    'B1xB1': binned_B1xB1, 'E1xE2': binned_E1xE2, 'B1xB2': binned_B1xB2,
                                    'E1xB1': binned_E1xB1, 'E2xB2': binned_E2xB2, 'binned_nu': binned_nu,
                                    'estimator': estimator, 'covariance': covariance,
                                    'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                                    'w2_depth1': w2_depth1, 'w2_cross': w2_cross, 'w2_ref': w2_ref,
                                    'meas_angle': fit_values[0], 'meas_errbar': fit_values[1], 
                                    'ivar_sum': ivar_sum, 'map_cut': 0}            

# Converting rho estimates to float from np.float64 for readability in yaml
angle_estimates_float = [[float(v),float(w)] for (v,w) in angle_estimates]

# Saving spectra to a numpy file
# Can be loaded with np.load(spectra_output_fname, allow_pickle=True).item()
spectra_output_fname = output_dir_path + 'angle_calc_' + output_time + '_spectra.npy'
np.save(spectra_output_fname, spectra_output)

if plot_all_spectra:
    print("Beginning to save plots for all spectra. This could take a while.")
    aoa.plot_spectra_individually(output_dir_path, spectra_output)
    print("Finished saving plots for all spectra.")
if plot_summary_spectra:
    print("Beginning to save summary spectra plots.")
    aoa.plot_spectra_summary(output_dir_path, spectra_output)
    print("Finished saving summary spectra plots.")
if plot_angle_hist:
    print("Plotting histogram of angles")
    aoa.plot_angle_hist(output_dir_path, np.array(angle_estimates)[:,0], maps)

# Dump all inputs and outputs to a YAML log
output_dict = config
output_dict['angle_estimates'] = angle_estimates_float
output_dict['list_of_maps'] = maps
output_dict['spectra_output_fname'] = spectra_output_fname

output_name = output_dir_path + 'angle_calc_' + output_time + ".yaml"
with open(output_name, 'w') as file:
    yaml.dump(output_dict, file)
print("Finished running get_angle_from_depth1_ps.py. Output is in: " + str(output_name))
stop_time = time.time()
duration = stop_time-start_time
print("Script took {:1.3f} seconds".format(duration))