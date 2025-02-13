act-axion-analysis
==============================

Overview
--------

This repo contains code to study time-dependent polarization rotations that could be induced in CMB 
polarization measurements made by the Atacama Cosmology Telescope by axion-like particles. 

The main analysis functions live in act_axion_analysis itself. The scripts folder contains the main 
script for running the code - get_angle_from_depth1_ps.py. As the name suggests, this script is tuned
to obtain a rotation angle relative to a reference map (usually a season-long or longer coadd map) by
using a power spectrum based estimator. It can also calculate a calibration factor of the TT spectrum
relative to two coadd maps. All of the various settings and paths to needed maps are set in the configuration
file, dr6_depth1_ps_config.yaml. See the configuration file section below for more information.

For running on Perlmutter at NERSC, there is also a shell script, run_alp_analysis.sh, that sets up the 
working environment, generates a slurm shell script to distribute the code across many nodes and processes,
and then runs the slurm shell script with sbatch. Before using, make sure to change the RUN_TAG environment
variable, the maximum walltime, the number of nodes, the number of processes per node, and any paths. 

The resources folder contains a handful of files that can be used for the CAMB theory spectra and the ACT
DR4 bins, if desired.

There are two main branches of this code: the main branch and the serial branch. The main branch uses mpi4py 
to perform parallel operations (both across nodes and multiple processes per node) for a computing cluster.
It has mostly been used with the Perlmutter computer at NERSC. The serial branch is an older version of the code
that runs all maps serially. This was the main code used for local development. 

The serial branch also contains a notebooks folder with various Jupyter notebooks that were used for testing and
debugging aspects of the code. The notebooks themselves contain some documentation, but these provide a sort of 
history of this project including notebooks with simulations that test parts of the code for accuracy. Many of the
notebooks, as a result, use older or developmental parts of the code, and they may be difficult to interpret
without additional notes or careful study. 


Dependencies
------------
The following packages are required to use this code. I usually install them from conda-forge
in a conda environment, and the version numbers in parentheses represent the versions against
which the code has been tested most extensively (as of February 2025).

* Python (3.12.8)
* pixell (0.27.2)
* numpy (1.26.4 - has not been tested yet for numpy 2.0+!)
* matplotlib (3.10.0)
* scipy (1.15.1)
* pyyaml (6.0.2)
* tqdm (4.67.1)
* mpi4py (4.0.2)

Other versions may work (especially slightly older versions than these), but no guarantees.

Installing all of these from conda-forge does have the downside that the pixell v0.27.2
wheel does not allow the Intel MKL backend for numpy to be installed (nor numpy 2.0+). The
ease of installation and ensuring that everything works together is enough that I have not
done more to get around these constraints.

Installation
------------
Currently must download from GitHub directly via ```git clone```.
Once downloaded, it can be installed with 
.. code-block:: console
		
   $ pip install . --user


Configuration File
------------------
As of February 2025, here are the available settings in the YAML config file and information about best
practices for setting them. The paths, in particular, assume that I am running the code for ACT DR6 data,
but they could be placed with appropriately similar maps, beams, etc. for future analyses with ACT/SO/etc.

* freq - the frequency of maps being run. Options are 'f090', 'f150', and 'f220' (though production runs on NERSC only occured for f090 and f150)

* Filtering parameters

  * kx_cut - cutoff in x Fourier modes (default 90)
  * ky_cut - cutoff in x Fourier modes (default 50)
  * unpixwin - boolean about whether to remove pixel window (default True)

* Apodization parameter

  * filter_radius - the apodization radius in degrees (default 0.5, though this is applied twice if using ivar weighting)

* Likelihood fitting settings

  * angle_min_deg - the minimum angle for the likelihood fitting (default -50.0)
  * angle_max_deg - the maximum angle for the likelihood fitting (default 50.0)
  * num_pts - the number of points between angle_min_deg and angle_max_deg at which the likelihood is evaluated (default 200000) 
  * use_curvefit - whether to use scipy curvefit to fit cal likelihoods (default True - better to use the full fit here since low S/N maps deviate from Gaussianity)

* Calibration factor likelihood fitting settings

  * y_min - the minimum calibration factor for the likelihood fitting (default -1.0 - allows us to catch low values with large errorbars)
  * y_max - the maximum calibration factor for the likelihood fitting (default 2.0)
  * cal_num_pts - the number of points between y_min and y_max at which the likelihood is evaluated (default 50000)
  * cal_use_curvefit - whether to use scipy curvefit to fit cal likelihoods (default False - actually often better to use Gaussian moments method here since these are all nice Gaussians)

* Calibration factor binning settings

  * cal_bin_size - bin width (in ell) of the bins for the TT calibration (usually 200)
  * cal_lmin - minimum ell for the TT calibration (usually 1000)
  * cal_lmax - maximum ell for the TT calibration (usually 2001 - this ensures that we get the bin ending at 2000)

* Angle estimator binning settings

  * bin_settings - options are "regular" and "DR4" to use even bins of "bin_size" or to use the ACT DR4 bins, respectively (default "regular" - the DR4 option is a legacy test)
  * bin_size - (used with "regular") bin width (in ell) of the bins for the angle estimation (usually 400)
  * lmin - (used with "regular") minimum ell for the angle estimation (usually 1000)
  * lmax - (used with "regular") maximum ell for the angle estimation (usually 3001)
  * start_index: 11  # Used with "DR4" - refers to index in DR4 bin file
  * stop_index: 47   # Used with "DR4" - refers to index in DR4 bin file

* Power spectra analysis settings

  * use_ivar_weight - boolean setting whether or not to use inverse variance weighting for calculating spectra (default True)
  * cross_calibrate - boolean setting whether or not to do TT calibration (usually True, but will not affect angle estimation and will speed things up to set to False)

* Output options - all the plotting booleans are generally False on NERSC, but were very helpful for debugging during local testing. There are additional options for the serial branch.

  * output_dir_root - path to the directory to which output files are saved (npy files with results, config YAML, any plots)
  * plot_maps - boolean for whether to save plots of trimmed maps and masks in analysis
  * plot_likelihood - boolean for whether to save plots of angle estimation likelihood
  * plot_beam - boolean for whether to save plots of binned beams
  * plot_tfunc - boolean for whether to save plot of binned filtering transfer function

* Paths

  * theory_curves_path - the path to a CAMB .dat file containing the best-fit LCDM cosmology spectra
  * ref_path - path to the reference map for the angle estimation (usually a full ACT DR6 coadd)
  * ref_ivar_path - path to the reference map inverse variance (ivar) map for the angle estimation
  * pa4_beam_path - path to beam tform file for ACT DR6 pa6 (e.g. coadd_pa4_f150_night_beam_tform_jitter_cmb.txt")
  * pa5_beam_path - path to beam tform file for ACT DR6 pa6 (e.g. coadd_pa5_f150_night_beam_tform_jitter_cmb.txt")
  * pa6_beam_path - path to beam tform file for ACT DR6 pa6 (e.g. coadd_pa6_f150_night_beam_tform_jitter_cmb.txt")
  * galaxy_mask_path - path to the galaxy mask (usually using the ACT 70% galaxy mask)
  * cal_map1_path - path to the map for the first calibration coadd

    * It is assumed that this map is a pa5 coadd - the beam is hardcoded in get_angle_from_depth1_ps.py to use pa5_beam_path

  * cal_ivar1_path - path to the ivar map for the first calibration coadd
  * cal_map2_path - path to the map for the second calibration coadd

    * It is assumed that this map is a pa6 coadd - the beam is hardcoded in get_angle_from_depth1_ps.py to use pa6_beam_path

  * cal_ivar2_path - path to the ivar map for the second calibration coadd
  * obs_list - a .txt file containing the names of all of the maps to run
  * obs_path_stem - the path to the directory containing all of the depth-1 maps

