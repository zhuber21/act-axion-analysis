###################################
# validate_config_paths.py
#
# Short python script to load the config file
# and check that all files for a run are
# present at listed paths.
# Expected to be run manually before
# submitting a slurm job.
####################################

import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help=
    "The path to the YAML config file for the run",type=str)
args = parser.parse_args()

# Load in the YAML file
yaml_name = args.config_path
with open(yaml_name, 'r') as file:
    config = yaml.safe_load(file)

# Check that paths exist to needed files
freq = config['freq']
cross_calibrate = config['cross_calibrate']
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
output_dir_root = config['output_dir_root']
if not os.path.exists(output_dir_root): # Make sure root path is right
    print("Output directory does not exist!")
if not os.path.exists(camb_file): 
    print("Cannot find CAMB file! Check config.")
if not os.path.exists(ref_path): 
    print("Cannot find reference map file! Check config.")
#if not os.path.exists(ref_beam_path): 
#    print("Cannot find beam file! Check config. ")
if not os.path.exists(ref_ivar_path): 
    print("Cannot find ref map ivar file! Check config.")
if freq=='f150' or freq=='f220':
    if not os.path.exists(pa4_beam_path): 
        print("Cannot find pa4 beam file! Check config.")
if freq=='f090' or freq=='f150':
    if not os.path.exists(pa5_beam_path): 
        print("Cannot find pa5 beam file! Check config.")
    if not os.path.exists(pa6_beam_path): 
        print("Cannot find pa6 beam file! Check config.")
if not os.path.exists(galaxy_mask_path):
    print("Cannot find galaxy mask file! Check config.")
if not os.path.exists(obs_list): 
    print("Cannot find observation list! Check config.")
if obs_list[-3:] != 'txt':
    print("Please enter a valid text file in the obs_list field in the YAML file.")
if cross_calibrate:
    cal_map1_path = config['cal_map1_path']
    cal_ivar1_path = config['cal_ivar1_path']
    cal_map2_path = config['cal_map2_path']
    cal_ivar2_path = config['cal_ivar2_path']
    if not os.path.exists(cal_map1_path): 
        print("Cannot find calibration map 1 file! Check config.")
    if not os.path.exists(cal_ivar1_path): 
        print("Cannot find calibration ivar 1 file! Check config.")
    if not os.path.exists(cal_map2_path): 
        print("Cannot find calibration map 2 file! Check config.")
    if not os.path.exists(cal_ivar2_path): 
        print("Cannot find calibration ivar 2 file! Check config.")


# Checking observation list for all depth-1 map, ivar, time, and info files
obs_list_path = config['obs_path_stem']
if not os.path.exists(obs_list_path):
    print("Cannot find path to depth-1 maps! Check config.")
    print("Finished validate_config_paths.py. Unable to check depth-1 maps.")
else:
    with open(obs_list) as f:
        lines = f.read().splitlines()

    missing_files = 0
    for line in lines:
        line_root = line[:-8]
        if not os.path.exists(obs_list_path+line):
            print("Missing "+line)
            missing_files += 4
        else:
            if not os.path.exists(obs_list_path+line_root+'ivar.fits'):
                print("Missing "+line_root+'ivar.fits')
                missing_files += 1
            if not os.path.exists(obs_list_path+line_root+'time.fits'):
                print("Missing "+line_root+'time.fits')
                missing_files += 1
            if not os.path.exists(obs_list_path+line_root+'info.hdf'):
                print("Missing "+line_root+'info.hdf')
                missing_files += 1
    print("Finished validate_config_paths.py. Missing " + str(missing_files) + " depth-1 files.")
