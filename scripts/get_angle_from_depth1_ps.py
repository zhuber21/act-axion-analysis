import numpy as np
import yaml
import argparse
import time
import glob
from pixell import enmap
from tqdm import tqdm
import axion_osc_analysis_depth1_ps as aoa

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

# Setting commonly used variables set in the config file
kx_cut = config['kx_cut']
ky_cut = config['ky_cut']
#bin_size = config['bin_size'] # bins will be set by the ACT DR4 bins below
lmax = config['lmax']
start_index = config['start_index']
stop_index = config['stop_index']
unpixwin = config['unpixwin']
angle_min_deg = config['angle_min_deg']
angle_max_deg = config['angle_max_deg']
num_pts = config['num_pts']
camb_file = config['theory_curves_path']
ref_path = config['ref_path']
fname_ref_beam = config['ref_beam_path']
obs_list_path = config['obs_path_stem']

# Load bins from ACT DR4
act_dr4_spectra_root = '/home/zbh5/act_analysis/scripts/dr4_tests/act_dr4_spectra/'
full_bins, full_centers = np.loadtxt(act_dr4_spectra_root+'BIN_ACTPOL_50_4_SC_low_ell', usecols=(0,2), unpack=True)
bins, centers = full_bins[:stop_index], full_centers[:stop_index-1]
print("Finished	loading	ACT bins")

# Load CAMB EE spectrum
ell,DlEE = np.loadtxt(camb_file, usecols=(0,2), unpack=True)
# Note that ell runs from 2 to 5400
ClEE = DlEE * 2 * np.pi / (ell*(ell+1.))
max_ell = int(bins[-1])+2
digitized = np.digitize(ell[:max_ell], bins, right=True)
CAMB_ClEE_binned = np.bincount(digitized, ClEE[:max_ell].reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
print("Finished	loading	CAMB spectra")

# Cycle through all lines in the provided .txt file with paths to all the
# splits you want to calculate rho for, and calculate rho
obs_list_path = config['obs_path_stem']

if config['obs_list'][-3:] == 'txt':
    obs_list = config['obs_list']
    print("Using list of observations at: " + str(obs_list))
else:
    print("Please enter a valid text file containing the splits in the obs_list field in the YAML file.")

#Loading in reference maps
ref_maps, ref_ivar, ref_beam = aoa.load_ref_map_and_beam(ref_path,fname_ref_beam,lmax,bins)
print("Finished	loading	ref map	and beam")

maps = []
angle_estimates = []
spectra_output = {}

with open(obs_list) as f:
    lines = f.read().splitlines()

for line in tqdm(lines):
    print(line)
    map_path = obs_list_path + line
    depth1_TEB, depth1_ivar, depth1_footprint, ref_TEB = aoa.load_and_filter_depth1(map_path, ref_maps, kx_cut, ky_cut, unpixwin)
    # Ivar weighting for depth-1 map
    depth1_maps_realspace = enmap.harm2map(depth1_TEB, normalize = "phys")
    depth1_maps_ivar = enmap.zeros((3,) + depth1_maps_realspace[0].shape, wcs=depth1_maps_realspace[0].wcs)
    depth1_maps_ivar[0] = depth1_maps_realspace[0]*2.0*depth1_ivar*depth1_footprint # Weighting by the original temperature ivar for T
    depth1_maps_ivar[1] = depth1_maps_realspace[1]*depth1_ivar*depth1_footprint
    depth1_maps_ivar[2] = depth1_maps_realspace[2]*depth1_ivar*depth1_footprint
    # Converting back to harmonic space
    depth1_TEB = enmap.map2harm(depth1_maps_ivar, normalize = "phys")
    depth1_E = depth1_TEB[1]
    depth1_B = depth1_TEB[2]
    # Calculating approx correction for loss of power due to tapering for spectra for depth-1
    w_depth1 = depth1_ivar*depth1_footprint
    #w_depth1 = depth1_footprint # use this if using flat weighting
    # Ivar weighting for reference map - already filtered and trimmed from ref_TEB above
    ref_map_trimmed_ivar = enmap.extract(ref_ivar,depth1_TEB[0].shape,depth1_TEB[0].wcs)
    ref_maps_realspace = enmap.harm2map(ref_TEB, normalize = "phys")
    ref_maps_ivar = enmap.zeros((3,) + ref_maps_realspace[0].shape, wcs=ref_maps_realspace[0].wcs)
    ref_maps_ivar[0] = ref_maps_realspace[0]*2.0*ref_map_trimmed_ivar*depth1_footprint # Weighting by the original temperature ivar for T
    ref_maps_ivar[1] = ref_maps_realspace[1]*ref_map_trimmed_ivar*depth1_footprint
    ref_maps_ivar[2] = ref_maps_realspace[2]*ref_map_trimmed_ivar*depth1_footprint
    # Converting back to harmonic space
    ref_TEB = enmap.map2harm(ref_maps_ivar, normalize = "phys")
    ref_E = ref_TEB[1]
    ref_B = ref_TEB[2]
    # Calculating approx correction for loss of power due to tapering for spectra for ref
    w_ref = ref_map_trimmed_ivar*depth1_footprint
    #w_ref = depth1_footprint # use this if using flat weighting   

    tfunc = aoa.get_tfunc(kx_cut, ky_cut, int(bins[-1])+2, bins)

    # At some point need to implement a beam correction per depth-1 map if possible

    # Calculate spectra
    ell_b = centers
    # Spectra for estimator
    binned_E1xB2, bincount = aoa.spectrum_from_maps(depth1_E, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1*w_ref), bins=bins)
    binned_E2xB1, _ = aoa.spectrum_from_maps(depth1_B, ref_E, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1*w_ref), bins=bins)
    # Spectra for covariance
    binned_E1xE1, _ = aoa.spectrum_from_maps(depth1_E, depth1_E, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1**2), bins=bins)
    binned_B2xB2, _ = aoa.spectrum_from_maps(ref_B, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_ref**2), bins=bins)
    binned_E2xE2, _ = aoa.spectrum_from_maps(ref_E, ref_E, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_ref**2), bins=bins)
    binned_B1xB1, _ = aoa.spectrum_from_maps(depth1_B, depth1_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1**2), bins=bins)
    binned_E1xE2, _ = aoa.spectrum_from_maps(depth1_E, ref_E, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1*w_ref), bins=bins)
    binned_B1xB2, _ = aoa.spectrum_from_maps(depth1_B, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1*w_ref), bins=bins)
    binned_E1xB1, _ = aoa.spectrum_from_maps(depth1_E, depth1_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_depth1**2), bins=bins)
    binned_E2xB2, _ = aoa.spectrum_from_maps(ref_E, ref_B, b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=np.mean(w_ref**2), bins=bins)    
    # Accounting for transfer function
    binned_E1xB2 /= tfunc**2
    binned_E2xB1 /= tfunc**2
    binned_E1xE1 /= tfunc**2
    binned_B2xB2 /= tfunc**2
    binned_E2xE2 /= tfunc**2
    binned_B1xB1 /= tfunc**2
    binned_E1xE2 /= tfunc**2
    binned_B1xB2 /= tfunc**2
    binned_E1xB1 /= tfunc**2
    binned_E2xB2 /= tfunc**2
    binned_nu = bincount*np.mean(depth1_footprint**2) 
    
    # Calculate estimator and covariance
    estimator = binned_E1xB2[start_index:]-binned_E2xB1[start_index:]
    covariance = ((1/binned_nu[start_index:])*((binned_E1xE1[start_index:]*binned_B2xB2[start_index:]+binned_E1xB2[start_index:]**2)
                                +(binned_E2xE2[start_index:]*binned_B1xB1[start_index:]+binned_E2xB1[start_index:]**2)
                                -2*(binned_E1xE2[start_index:]*binned_B1xB2[start_index:]+binned_E1xB1[start_index:]*binned_E2xB2[start_index:])))

    fit_values = aoa.sample_likelihood_and_fit(estimator,covariance,CAMB_ClEE_binned[start_index:])

    print(fit_values)
    angle_estimates.append(fit_values)
    maps.append(line)
    spectra_output[line] = {'ell': centers, 'E1xB2': binned_E1xB2, 'E2xB1': binned_E2xB1, 
                            'E1xE1': binned_E1xE1, 'B2xB2': binned_B2xB2, 'E2xE2': binned_E2xE2,
                            'B1xB1': binned_B1xB1, 'E1xE2': binned_E1xE2, 'B1xB2': binned_B1xB2,
                            'E1xB1': binned_E1xB1, 'E2xB2': binned_E2xB2}

# Converting rho estimates to float from np.float64 for readability in yaml
angle_estimates_float = [[float(v),float(w)] for (v,w) in angle_estimates]

# Saving spectra to a numpy file
# Can be loaded with np.load(spectra_output_fname, allow_pickle=True).item()
output_time = str(int(round(time.time())))
spectra_output_fname = 'angle_calc_' + output_time + '_spectra.npy'
np.save(spectra_output_fname, spectra_output)

# Dump all inputs and outputs to a YAML log
output_dict = config
output_dict['angle_estimates'] = angle_estimates_float
output_dict['list_of_maps'] = maps
output_dict['spectra_output_fname'] = spectra_output_fname

output_name = 'angle_calc_' + output_time + ".yaml"
with open(output_name, 'w') as file:
    yaml.dump(output_dict, file)
print("Finished running get_angle_from_depth1_ps.py. Output is in: " + str(output_name))


