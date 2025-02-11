import numpy as np
from pixell import enmap, enplot, bunch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy
from scipy import optimize as op

##########################################################
# Functions for loading and manipulating maps and other data products

def load_and_bin_beam(fname,bins):
    """
        For a given fname containing an ACT beam, loads in the ell and b_ell
        of the beam. Normalizes the beam and bins it into the same binning as
        the power spectra.
    """
    beam_ell, beam_tform = np.loadtxt(fname,usecols=(0,1),unpack=True)
    beam_ell_norm = beam_ell[1:] # The first index of the beam corresponds to the integrated solid angle
    beam_tform_norm = beam_tform[1:]/np.max(beam_tform[1:])

    digitized = np.digitize(beam_ell_norm, bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    beam_tform_norm_binned = np.bincount(digitized, beam_tform_norm.reshape(-1))[1:-1]/bincount
    return beam_tform_norm_binned
    
def load_ref_map(fname_ref,fname_ref_ivar):
    """
        Loads in the full reference map (T,Q,U). 
        Needs to be done once. Also loads in ref map ivar.
    """
    maps = enmap.read_map(fname_ref)
    ivar = 0.5*enmap.read_map(fname_ref_ivar) # 0.5 for polarization noise

    return maps, ivar

def apply_kspace_filter(maps, kx_cut, ky_cut, unpixwin):
    """
        Takes in a set of T/Q/U maps that already have a taper applied, apply
        a k-space filter to remove ground pickup, and returns the E and B maps

        Adjusted later to work for filtering a single T map too.
    """
    singleobs_TEB = enmap.map2harm(maps, normalize = "phys")

    if unpixwin:  # remove pixel window in Fourier space
        if len(singleobs_TEB.shape) == 3: # if there is a third index for multiple maps
            for i in range(len(singleobs_TEB)):
                wy, wx = enmap.calc_window(singleobs_TEB[i].shape)
                singleobs_TEB[i] /= wy[:, np.newaxis]
                singleobs_TEB[i] /= wx[np.newaxis, :]
        else:
            wy, wx = enmap.calc_window(singleobs_TEB.shape)
            singleobs_TEB /= wy[:, np.newaxis]
            singleobs_TEB /= wx[np.newaxis, :]

    ly, lx = singleobs_TEB.lmap()
    kfilter_x = np.abs(lx) >= kx_cut
    kfilter_y = np.abs(ly) >= ky_cut
    filtered_TEB = singleobs_TEB * kfilter_x * kfilter_y

    return filtered_TEB

def load_depth1_with_T(depth1_path,plot=False):
    """
        This function takes the path to a given depth-1 map, loads them
        and the ivar map, and returns the Q, U, and ivar maps as well as the
        shape and wcs of the maps for use in cutting out the right coadd area. 
    """
    
    depth1_maps = enmap.read_map(depth1_path)
    depth1_shape, depth1_wcs = depth1_maps[1].shape, depth1_maps[1].wcs
    
    ivar_path = depth1_path[:-8] + "ivar.fits"
    depth1_ivar = 0.5*enmap.read_map(ivar_path) # 0.5 for polarization noise
    
    if plot:
        eshow(depth1_maps[0], **keys_eshow)
        eshow(depth1_maps[1], **keys_eshow)
        eshow(depth1_maps[2], **keys_eshow)
        eshow(depth1_ivar, **keys_eshow)
    
    return depth1_maps, depth1_ivar, depth1_shape, depth1_wcs

def trim_ref_with_T(ref_maps,shape,wcs,plot=False):
    """
        Trims the full reference map down to a smaller size to match a given
        shape and wcs (from a particular depth-1, for example).
    """
    ref_maps_cut = enmap.extract(ref_maps,shape,wcs)

    if plot:
        eshow(ref_maps_cut[0], **keys_eshow)
        eshow(ref_maps_cut[1], **keys_eshow)
        eshow(ref_maps_cut[2], **keys_eshow)
    
    return ref_maps_cut

def get_distance(input_mask):
    """
    Copied from nawrapper to remove dependency, Feb 2025
    
    Construct a map of the distance to the nearest zero pixel in the input.

    Parameters
    ----------
    input_mask : enmap
        The input mask

    Returns
    -------
    dist : enmap
        This map is the same size as the `input_mask`. Each pixel of this map
        contains the distance to the nearest zero pixel at the corresponding
        location in the input_mask.

    """
    pixSize_arcmin = np.sqrt(input_mask.pixsize() * (60 * 180 / np.pi) ** 2)
    dist = scipy.ndimage.distance_transform_edt(np.asarray(input_mask))
    dist *= pixSize_arcmin / 60
    return dist


def apod_C2(input_mask, radius):
    """
    Copied from nawrapper to remove dependency, Feb 2025

    Apodizes an input mask over a radius in degrees.

    A sharp mask will cause complicated mode coupling and ringing. One solution
    is to smooth out the sharp edges. This function applies the C2 apodisation
    as defined in 0903.2350_.

    .. _0903.2350: https://arxiv.org/abs/0903.2350

    Parameters
    ----------
    input_mask: enmap
        The input mask (must have all pixel values be non-negative).
    radius: float
        Apodization radius in degrees.

    Returns
    -------
    result : enmap
        The apodized mask.

    """
    if radius == 0:
        return input_mask
    else:
        dist = get_distance(input_mask)
        id = np.where(dist > radius)
        win = dist / radius - np.sin(2 * np.pi * dist / radius) / (2 * np.pi)
        win[id] = 1

    return enmap.ndmap(win, input_mask.wcs)


def make_tapered_mask(map_to_mask,filter_radius=0.5,plot=False):
    """
        Makes a mask for a given map based on where the ivar map is nonzero.
        Also apodizes the mask and gets the indices of where the apodized
        mask is not equal to one (everything tapered or outside the mask)
        in order to set all points but those to zero after filtering.

        The default radius is 0.5 deg - we apply this twice though when using ivar
        weighting since we throw away the first taper post-filtering and make a new
        one post-ivar weighting, so the total radius is still around 1.0 deg as it
        originally was when tapering once.
    """
    footprint = 1*map_to_mask.astype(bool)
    mask = apod_C2(footprint,filter_radius)
    
    # Getting points to set to zero after filtering used for 
    # calculating ivar_sum in the non-tapered region and removing taper for second mask
    indices = np.nonzero(mask != 1)
    
    if plot:
        eshow(mask, **keys_eshow)
    
    return mask, indices

def calc_ivar_sum(tapered_mask, indices, ivar):
    """Moving the ivar sum calculation to another function to reduce memory overhead"""
    final_mask_without_taper = tapered_mask.copy() # So as to not alter the tapered mask
    final_mask_without_taper[indices] = 0.0
    ivar_sum = np.sum(final_mask_without_taper*ivar)
    return ivar_sum

def taper_mask_first_time(depth1_ivar, galaxy_mask, shape, wcs, filter_radius):
    """Moving generating the first mask for filtering to another function to reduce memory overhead"""
    galaxy_mask_cut = enmap.extract(galaxy_mask,shape,wcs)
    depth1_filtering_mask, depth1_indices = make_tapered_mask(depth1_ivar*galaxy_mask_cut,filter_radius=filter_radius)
    return depth1_filtering_mask, depth1_indices

def taper_mask_second_time(first_mask, first_indices, filter_radius):
    """Moving the second tapering of mask to make mask for ivar weighting to another function to reduce memory overhead"""
    first_mask_copy = first_mask.copy()      # ensuring no weirdness happens to first mask
    first_mask_copy[first_indices] = 0.0     # setting pixels in taper and outside to zero
    second_mask_tapered, second_indices = make_tapered_mask(first_mask_copy,filter_radius=filter_radius)
    return second_mask_tapered, second_indices

def load_and_filter_depth1(fname, ref_maps, galaxy_mask, kx_cut, ky_cut, unpixwin, filter_radius=1.0,plot_maps=False,output_dir=None,use_ivar_weight=False):
    """Loads depth-1 TQU, trims reference map to same size as depth-1, apodizes and filters depth-1 and coadd.
       Returns filtered depth-1 TEB, depth-1 ivar, depth-1 mask, filtered reference map TEB, and sum of the 
       inverse variance inside the non-tapered region of the ivar mask for noise/hits cuts."""
    
    depth1_maps, depth1_ivar, shape, wcs = load_depth1_with_T(fname)
    ref_cut = trim_ref_with_T(ref_maps,shape,wcs)
       
    # Apodize depth-1 and apply galaxy mask
    depth1_filtering_mask, depth1_indices = taper_mask_first_time(depth1_ivar, galaxy_mask, shape, wcs, filter_radius)
    test_mask = depth1_filtering_mask

    # This is clunky but should prevent edge case crashes if galaxy mask plus tapering eliminate whole map 
    if use_ivar_weight:
        if len(depth1_filtering_mask[depth1_filtering_mask==1.0]) > 0:
            # Set first mask's tapered region to zero and retaper to get mask for ivar weighting
            depth1_ivar_mask_tapered, depth1_ivar_indices = taper_mask_second_time(depth1_filtering_mask, depth1_indices, filter_radius)
            test_mask = depth1_ivar_mask_tapered

    # Checking if the galaxy mask and tapering eliminated the whole map
    if len(test_mask[test_mask==1.0]) > 0:
        # Filter depth-1
        filtered_depth1_TEB = apply_kspace_filter(depth1_maps*depth1_filtering_mask, kx_cut, ky_cut, unpixwin=unpixwin)
            
        # Apodize and filter coadd
        ref_cut_TEB = apply_kspace_filter(ref_cut*depth1_filtering_mask, kx_cut, ky_cut, unpixwin=unpixwin)

        # Calculating the sum of the ivar inside the second mask (w/o tapered part) to test if it is a good metric for data cuts
        # There might be more Pythonic ways to do this, but this works w/o changing the real mask
        if use_ivar_weight:
            ivar_sum = calc_ivar_sum(depth1_ivar_mask_tapered, depth1_ivar_indices, depth1_ivar)
            if plot_maps:
                # Plotting using the doubly tapered mask for the ivar weighting
                map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
                plot_T_maps(output_dir, map_fname, depth1_ivar_mask_tapered*depth1_maps[0], **keys_ewrite_T)
                plot_QU_maps(output_dir, map_fname, [depth1_ivar_mask_tapered*depth1_maps[1], depth1_ivar_mask_tapered*depth1_maps[2]], **keys_ewrite_QU)
                plot_T_ref_maps(output_dir, map_fname, depth1_ivar_mask_tapered*ref_cut[0], **keys_ewrite_ref_T)
                plot_QU_ref_maps(output_dir, map_fname, [depth1_ivar_mask_tapered*ref_cut[1], depth1_ivar_mask_tapered*ref_cut[2]],**keys_ewrite_ref_QU)
                plot_mask(output_dir, map_fname, depth1_ivar_mask_tapered, **keys_ewrite_mask)
                plot_EB_filtered_maps(output_dir, map_fname, filtered_depth1_TEB, depth1_ivar_mask_tapered, **keys_ewrite_EB)
            return filtered_depth1_TEB, depth1_ivar, depth1_ivar_mask_tapered, depth1_ivar_indices, ref_cut_TEB, ivar_sum
        else:
            ivar_sum = calc_ivar_sum(depth1_filtering_mask, depth1_ivar_indices, depth1_ivar)
            if plot_maps:
                # Plotting using the first (filtering) mask
                map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
                plot_T_maps(output_dir, map_fname, depth1_filtering_mask*depth1_maps[0], **keys_ewrite_T)
                plot_QU_maps(output_dir, map_fname, [depth1_filtering_mask*depth1_maps[1], depth1_filtering_mask*depth1_maps[2]], **keys_ewrite_QU)
                plot_T_ref_maps(output_dir, map_fname, depth1_filtering_mask*ref_cut[0], **keys_ewrite_ref_T)
                plot_QU_ref_maps(output_dir, map_fname, [depth1_filtering_mask*ref_cut[1], depth1_filtering_mask*ref_cut[2]],**keys_ewrite_ref_QU)
                plot_mask(output_dir, map_fname, depth1_filtering_mask, **keys_ewrite_mask)
                plot_EB_filtered_maps(output_dir, map_fname, filtered_depth1_TEB, depth1_filtering_mask, **keys_ewrite_EB)                
            return filtered_depth1_TEB, depth1_ivar, depth1_filtering_mask, depth1_indices, ref_cut_TEB, ivar_sum
    else:
        if plot_maps:
            # All will show nothing except the mask, but still want the empty plots for web viewer code
            map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
            # Deliberately plotting depth-1 maps without mask so I can see what they originally looked like,
            # but plotting everything else with mask to show it was completely cut
            plot_T_maps(output_dir, map_fname, depth1_maps[0], **keys_ewrite_T)
            plot_QU_maps(output_dir, map_fname, [depth1_maps[1], depth1_maps[2]], **keys_ewrite_QU)
            plot_T_ref_maps(output_dir, map_fname, test_mask*ref_cut[0], **keys_ewrite_ref_T)
            plot_QU_ref_maps(output_dir, map_fname, [test_mask*ref_cut[1], test_mask*ref_cut[2]],**keys_ewrite_ref_QU)
            plot_mask(output_dir, map_fname, test_mask, **keys_ewrite_mask)
            plot_EB_filtered_maps(output_dir, map_fname, depth1_maps, test_mask, cut_map=True, **keys_ewrite_EB)
        return 1 # returning an error code if there is nothing left in the map

def normalize_ivar_mask(input_ivar, mask_indices):
    """Putting the normalization of the ivar mask into a function to reduce memory footprint"""
    ivar_inside_taper = input_ivar.copy() # to ensure no weirdness happens to original ivar
    ivar_inside_taper[mask_indices] = 0.0 # Setting all tapered points and beyond to zero so normalization doesn't happen to a tapered point

    # There are a small number of isolated outlier pixels in the ivar map. To prevent them from
    # having a outsized effect on the normalization, we normalize to the 95th percentile of the map.
    # Anything in the 95th-100th percentile is set to 1.0
    # For very small maps, sometimes numpy rounds np.percentile to 0.0 - catch these and normalize by np.max() in those cases
    if np.percentile(2.0*ivar_inside_taper, 95)!=0.0:
        norm_ivar_T_mask = 2.0*ivar_inside_taper / np.percentile(2.0*ivar_inside_taper, 95) # Weighting by the original temperature ivar for T
        norm_ivar_T_mask[norm_ivar_T_mask > 1.0] = 1.0 # Setting any outliers to 1.0
    else:
        norm_ivar_T_mask = 2.0*ivar_inside_taper / np.max(2.0*ivar_inside_taper)
    if np.percentile(ivar_inside_taper, 95)!=0.0:
        norm_ivar_QU_mask = ivar_inside_taper / np.percentile(ivar_inside_taper, 95)
        norm_ivar_QU_mask[norm_ivar_QU_mask > 1.0] = 1.0 # Setting any outliers to 1.0
    else:
        norm_ivar_QU_mask = ivar_inside_taper / np.max(ivar_inside_taper)

    # Now that things are normalized in the region we will take the PS on, set the region
    # outside of it to 1.0 so that it gets tapered and masked appropriately by the PS mask
    norm_ivar_T_mask[mask_indices] = 1.0
    norm_ivar_QU_mask[mask_indices] = 1.0
    
    return norm_ivar_T_mask, norm_ivar_QU_mask

def apply_ivar_weighting(input_kspace_TEB_maps, input_ivar, ivar_mask, mask_indices):
    """For a set of TEB Fourier space maps, converts back to real space, multiplies by
       the normalized inverse variance map and the tapered mask, and converts back to Fourier space
       for PS calculation.
       The normalized inverse variance map is only calculated inside the tapered region so that the
       structure of the ivar map doesn't make the taper non-smooth and so that we are ivar weighting
       only in the region where the PS will be calculated.   
    """
    norm_ivar_T_mask, norm_ivar_QU_mask = normalize_ivar_mask(input_ivar, mask_indices)

    # Converting Fourier space maps to realspace, then multiplying by normalized ivar
    # mask and tapered PS mask
    maps_realspace = enmap.harm2map(input_kspace_TEB_maps, normalize = "phys")
    maps_ivar_weight = enmap.zeros((3,) + maps_realspace[0].shape, wcs=maps_realspace[0].wcs)
    maps_ivar_weight[0] = maps_realspace[0]*norm_ivar_T_mask*ivar_mask 
    maps_ivar_weight[1] = maps_realspace[1]*norm_ivar_QU_mask*ivar_mask
    maps_ivar_weight[2] = maps_realspace[2]*norm_ivar_QU_mask*ivar_mask
    # Converting back to harmonic space - already multiplied by tapered Ivar_mask above
    output_kspace_TEB_maps = enmap.map2harm(maps_ivar_weight, normalize = "phys")
    return output_kspace_TEB_maps, norm_ivar_QU_mask*ivar_mask

def cal_trim_and_fourier_transform(cal_T,depth1_ivar,galaxy_mask,shape,wcs,depth1_mask,kx_cut,ky_cut,unpixwin,filter_radius,use_ivar_weight):
    """
        For a given ACT coadd used for the overall calibration factor, trim the map to the size of the
        depth-1 map, filter it as normal, and return Fourier-transformed map and weight
        mask for calculating w2 factor in absence of ivar weighting.
    """
    # Trimming calibration maps to the size of the depth-1 map
    cal_T_map_trimmed = enmap.extract(cal_T,shape,wcs)

    # Remaking the filtering mask here if using ivar weighting to save space in RAM
    # If not using ivar weighting, the input depth1_mask is already the filtering mask calculated earlier.
    if use_ivar_weight:
        depth1_filtering_mask, _ = taper_mask_first_time(depth1_ivar, galaxy_mask, shape, wcs, filter_radius)
    else:
        depth1_filtering_mask = depth1_mask

    # Filtering
    filtered_cal_map_trimmed_fourier = apply_kspace_filter(cal_T_map_trimmed*depth1_filtering_mask, kx_cut, ky_cut, unpixwin=unpixwin)
    w_cal = depth1_filtering_mask

    return filtered_cal_map_trimmed_fourier, w_cal

def cal_normalize_ivar_mask(input_ivar, mask_indices):
    """Moving normalization of ivar mask for TT calibration to function to save memory"""
    ivar_inside_taper = input_ivar.copy() 
    ivar_inside_taper[mask_indices] = 0.0

    if np.percentile(ivar_inside_taper, 95)!=0.0:
        # Don't need factor of 2 because these ivar maps were never divided by 2 when loaded
        norm_ivar_T_mask = ivar_inside_taper / np.percentile(ivar_inside_taper, 95)
        norm_ivar_T_mask[norm_ivar_T_mask > 1.0] = 1.0 # Setting any outliers to 1.0
    else:
        norm_ivar_T_mask = ivar_inside_taper / np.max(ivar_inside_taper)

    norm_ivar_T_mask[mask_indices] = 1.0
    return norm_ivar_T_mask

def cal_apply_ivar_weighting(filtered_cal_T_fourier, cal_T_ivar, shape, wcs, depth1_ivar_mask, ivar_mask_indices):
    """
        Applying inverse variance weighting for TT calibration. This function is only called
        if the ivar weighting option is True, so it doesn't check for that. This also means that
        depth1_ivar_mask and ivar_mask_indices are guaranteed to be the doubly-tapered mask
        calculated earlier for the depth-1 mask.

        Returns the ivar weighted T map in Fourier space and the ivar weighted w factor.
    """
    cal_T_ivar_trimmed = enmap.extract(cal_T_ivar,shape,wcs)
    # Ivar weighting and converting to Fourier space
    # Copying the ivar normalization prescription in apply_ivar_weighting() but only for T
    norm_ivar_T_mask = cal_normalize_ivar_mask(cal_T_ivar_trimmed,ivar_mask_indices)

    filtered_cal_map_trimmed_realspace = enmap.harm2map(filtered_cal_T_fourier, normalize = "phys")
    w_cal = norm_ivar_T_mask*depth1_ivar_mask
    cal_map_fourier = enmap.map2harm(filtered_cal_map_trimmed_realspace*w_cal, normalize = "phys")
    return cal_map_fourier, w_cal

def calc_median_timestamp(map_path, mask):
    """
        Using time.fits and info.hdf files that accompany each depth-1 map to calculate the median timestamp
        of the region that is used for doing the power spectrum calculation (just the region where the tapered mask is 1.0). 

        Returns both the initial timestamp of the file (info_file.t) and the median timestamp after adding the median offset of
        the power spectrum region.
    """
    info_file_path = map_path[:-8] + 'info.hdf'
    time_file_path = map_path[:-8] + 'time.fits'
    info_file = bunch.read(info_file_path)    # info_file.t should be the timestamp in the file name - first timestamp in file
    time_map = enmap.read_map(time_file_path) # this map gives the time offset from info_file.t
    initial_timestamp = info_file.t
    # Restricting time_map just to the area inside the taper
    masked_time_map = time_map*mask
    median_time_offset = np.median(masked_time_map[mask==1.0])
    median_timestamp = initial_timestamp + median_time_offset # May want to round this eventually if median is not an integer...
    return initial_timestamp, median_timestamp
##########################################################


##########################################################
# Functions for doing PS calculations and estimator

def spectrum_from_maps(map1, map2, b_ell_bin_1, b_ell_bin_2, w2, bins):
    """Function modified from the one in ACT DR4/5 NB7 for binning a power spectrum for two maps.
       This function does account for a window correction for the apodizing at this point.
       Also accounts for a beam correction using a beam defined by b_ell.
    """
    spectrum = np.real(map1*np.conj(map2))

    modlmap = map1.modlmap()

    # Bin the power spectrum
    digitized = np.digitize(np.ndarray.flatten(modlmap), bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    binned = np.bincount(digitized, spectrum.reshape(-1))[1:-1]/bincount

    # Dividing by an approx. correction for the loss of power from tapering
    binned /= w2
    # Dividing by correction for the beam of each map
    binned /= (b_ell_bin_1*b_ell_bin_2)

    return binned, bincount

def get_tfunc(kx, ky, bins, lmax=5000):
    """Calculating transfer function for filtering with cutoffs kx and ky.
       lmax just needs to be higher than the range we want to measure since
       the bins will appropriately handle excluding higher ells."""
    cut = (ky+kx)*4
    ell = np.arange(lmax)
    tfunc = np.zeros(lmax)
    tfunc[1:] = 1 - cut / (2*np.pi*ell[1:])

    digitized = np.digitize(ell, bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    tfunc_binned = np.bincount(digitized, tfunc.reshape(-1))[1:-1]/bincount
    return tfunc_binned
    
def estimator_likelihood(angle, estimator, covariance, ClEE):
    """For a given difference in angle between the depth-1 map (map 1) and the coadd (map 2),
       returns the value of the normalized likelihood for our estimator.
       
       ClEE is the theory EE spectrum from CAMB"""

    numerator = (estimator - ClEE*np.sin(2*angle))**2
    denominator = 2*covariance
    likelihood = np.exp(-np.sum(numerator/denominator))
    return likelihood

def cal_likelihood(y, cal1xcal2, cal1xdepth1, cal1xcal1, cal2xcal2, cal2xdepth1, depth1xdepth1,
                   nu_cal1_cal2, nu_cal1_depth1, nu_all_three):
    """
        Calculates the likelihood for TT calibration for an input
        value of the calibration factor, y.

        estimator = TT_cal1xcal2 - y*TT_cal1xdepth1

        The covariance has different mode count factors for each of the
        three covariance terms that are summed to get the full covariance.
        The y factor also multiplies the TT spectrum for cal1xdepth1 
        wherever it appears in the covariance to account for how the
        changing calibration factor affects the covariance.
    """
    numerator = (cal1xcal2 - y*cal1xdepth1)**2
    covariance = ((1/nu_cal1_cal2)*(cal1xcal1*cal2xcal2+cal1xcal2**2)
                 +(1/nu_cal1_depth1)*(depth1xdepth1*cal1xcal1+y**2*cal1xdepth1**2)
               -2*(1/nu_all_three)*(y*cal1xdepth1*cal1xcal2+cal1xcal1*cal2xdepth1))
    denominator = 2*covariance
    likelihood = np.exp(-np.sum(numerator/denominator))
    return likelihood

def gaussian(x,mean,sigma):
    """Normalized Gaussian for curve_fit"""
    amp = 1.0
    return amp*np.exp(-(x-mean)**2/(2*sigma**2))

def gaussian_fit_curvefit(angles,data,guess=[1.0*np.pi/180.0, 5.0*np.pi/180.0]):
    """
        Uses scipy.optimize.curve_fit() to fit a Gaussian to the likelihood to
        get the mean and standard deviation.

        Assumes everything is in radians when fitting angles.

        The default guess (mean 1 deg, std dev 5 deg) here is for the angle likelihood. 
        When using this function to fit the TT calibration likelihood, pass in a different guess.
    """
    popt, pcov = op.curve_fit(gaussian,angles,data,guess,maxfev=50000)
    mean = popt[0]
    std_dev = np.abs(popt[1])
    return mean, std_dev

def gaussian_fit_moment(angles,data):
    """
       Uses moments to quickly find mean and standard deviation of a Gaussian
       for the likelihood.

       Assumes everything is in radians when fitting angles.
    """
    mean = np.sum(angles*data)/np.sum(data)
    std_dev = np.sqrt(abs(np.sum((angles-mean)**2*data)/np.sum(data)))
    return mean, std_dev
    
def sample_likelihood_and_fit(estimator,covariance,theory_ClEE,angle_min_deg=-20.0,angle_max_deg=20.0,
                              num_pts=10000,use_curvefit=True,plot_like=False,output_dir=None,map_fname=None):
    """
       Samples likelihood for a range of angles and returns the best fit for the
       mean and std dev of the resulting Gaussian in degrees.  
       Has the option to do the fitting with scipy.optimize.curve_fit() (set use_curvefit=True)
       or a method using moments of the Gaussian. The moments method is faster but less
       accurate when the likelihood deviates from Gaussianity in any way.
    """
    if(angle_min_deg >= angle_max_deg): 
        raise ValueError("The min angle must be smaller than the max!")
    angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
    angles_rad = np.deg2rad(angles_deg)
    
    bin_sampled_likelihood = [estimator_likelihood(angle,estimator,covariance,theory_ClEE) for angle in angles_rad]
    norm_sampled_likelihood = bin_sampled_likelihood/np.max(bin_sampled_likelihood)
    
    if use_curvefit:
        # Using default guess starting values set in function
        fit_values = gaussian_fit_curvefit(angles_rad,norm_sampled_likelihood)
        fit_values_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]
        # Could add some flag or option to redo the fit for a given map if curve_fit() returns
        # a bad stddev value from failing to fit - for now I will leave it so I can see easily by the stddev that it fails
    else:
        fit_values = gaussian_fit_moment(angles_rad,norm_sampled_likelihood)
        fit_values_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]

    # Calculating mean and sum of abs of residual between likelihood and best fit Gaussian
    # within 1 sigma of mean value - two possible ways of identifying and cutting poor S/N or bad fits
    residual = norm_sampled_likelihood - gaussian(angles_rad,fit_values[0],fit_values[1])
    minus_sigma_idx = np.searchsorted(angles_rad, fit_values[0]-fit_values[1])
    plus_sigma_idx = np.searchsorted(angles_rad, fit_values[0]+fit_values[1])
    residual_mean = np.mean(residual[minus_sigma_idx:plus_sigma_idx])
    residual_sum = np.sum(np.abs(residual[minus_sigma_idx:plus_sigma_idx]))

    if plot_like:
        map_name = os.path.split(map_fname)[1][:-9] # removing "_map.fits"
        plot_likelihood(output_dir, map_name, angles_deg, norm_sampled_likelihood,fit_values_deg,residual)

    return fit_values_deg, residual_mean, residual_sum

def cal_sample_likelihood_and_fit(cal1xcal2, cal1xdepth1, cal1xcal1, cal2xcal2, cal2xdepth1, 
                                  depth1xdepth1, nu_cal1_cal2, nu_cal1_depth1, nu_all_three, 
                                  y_min=0.7, y_max=1.3, num_pts=50000, use_curvefit=True):
    """
        Samples the likelihood for the TT calibration at num_pts values of the calibration
        factor, y, between the values y_min and y_max.
    """
    if(y_min >= y_max): 
        raise ValueError("The min y value must be smaller than the max!")
    y_values = np.linspace(y_min,y_max,num=num_pts)

    cal_sampled_likelihood = [cal_likelihood(y, cal1xcal2, cal1xdepth1, cal1xcal1, 
                                             cal2xcal2, cal2xdepth1, depth1xdepth1,
                                             nu_cal1_cal2, nu_cal1_depth1, nu_all_three) for y in y_values]
    norm_sampled_likelihood = cal_sampled_likelihood/np.max(cal_sampled_likelihood)

    if use_curvefit:
        # The default guess starting values are for the angle likelihood, so pass new ones here
        guess = [1.0, 0.2] # Guessing a center value of 1.0 with std dev 0.2
        fit_values = gaussian_fit_curvefit(y_values,norm_sampled_likelihood, guess=guess)
    else:
        fit_values = gaussian_fit_moment(y_values,norm_sampled_likelihood)
    
    return fit_values

def cross_calibrate(cal_T_map1_act_footprint, cal_T_map2_act_footprint,
                    cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
                    depth1_ivar, depth1_mask, depth1_mask_indices,
                    galaxy_mask, depth1_T, w_depth1, w2_depth1, bincount, bins, tfunc,
                    kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                    depth1_beam, pa5_beam, pa6_beam, y_min, y_max, cal_num_pts, cal_use_curvefit):
    """
        High-level function to do the TT cross-calibration. 
        Allows intermediate maps to be cleaned up when function closes for better RAM usage.

        Takes in all the variables needed for the subfunctions cal_trim_and_fourier_transform(),
        cal_apply_ivar_weighting(), spectrum_from_maps(), and cal_sample_likelihood_and_fit().

        Returns the calibration factor and uncertainty, all of the spectra, and all w factors.
    """
    shape = depth1_T.shape
    wcs = depth1_T.wcs

    # Moving trimming, ivar weighting, filtering, and Fourier transform to function
    # to avoid multiplying extra maps in memory - doing each cal map separately for same reason
    # These window factors are already normalized inside the mask
    cal_map1_fourier, w_cal1 = cal_trim_and_fourier_transform(cal_T_map1_act_footprint, depth1_ivar, galaxy_mask,
                                            shape, wcs, depth1_mask, kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight)
    cal_map2_fourier, w_cal2 = cal_trim_and_fourier_transform(cal_T_map2_act_footprint, depth1_ivar, galaxy_mask,
                                            shape, wcs, depth1_mask, kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight)
    if use_ivar_weight:
        # Will overwrite the filtered maps and w factors above with ivar weighted versions
        cal_map1_fourier, w_cal1 = cal_apply_ivar_weighting(cal_map1_fourier, cal_T_ivar1_act_footprint,
                                                            shape, wcs, depth1_mask, depth1_mask_indices)
        cal_map2_fourier, w_cal2 = cal_apply_ivar_weighting(cal_map2_fourier, cal_T_ivar2_act_footprint,
                                                            shape, wcs, depth1_mask, depth1_mask_indices)

    # Calculate calibration spectra and factor from likelihood
    w2_depth1xcal1 = np.mean(w_depth1*w_cal1)
    w2_depth1xcal2 = np.mean(w_depth1*w_cal2)
    w2_cal1xcal2 = np.mean(w_cal1*w_cal2)
    w2_cal1xcal1 = np.mean(w_cal1*w_cal1)
    w2_cal2xcal2 = np.mean(w_cal2*w_cal2)
    w2w4_depth1xcal1 = w2_depth1xcal1**2 / np.mean(w_depth1**2 * w_cal1**2)
    w2w4_cal1xcal2 = w2_cal1xcal2**2 / np.mean(w_cal1**2 * w_cal2**2)
    w2w4_all_three = w2_cal1xcal2*w2_depth1xcal1 / np.mean(w_cal1*w_cal1*w_cal2*w_depth1)
    # Depth-1 T cross cal map 1 T (pa5 coadd)
    binned_T1xcal1T, _ = spectrum_from_maps(depth1_T,cal_map1_fourier,b_ell_bin_1=depth1_beam,b_ell_bin_2=pa5_beam,w2=w2_depth1xcal1,bins=bins)
    binned_T1xcal1T /= tfunc
    # cal map 1 T (pa5 coadd) cross cal map 2 T (pa6 coadd)
    binned_cal1Txcal2T, _ = spectrum_from_maps(cal_map1_fourier,cal_map2_fourier,b_ell_bin_1=pa5_beam,b_ell_bin_2=pa6_beam,w2=w2_cal1xcal2,bins=bins)
    binned_cal1Txcal2T /= tfunc
    # Getting spectra for the covariance
    # cal map 1 T (pa5 coadd) cross cal map 1 T (pa5 coadd)
    binned_cal1Txcal1T, _ = spectrum_from_maps(cal_map1_fourier,cal_map1_fourier,b_ell_bin_1=pa5_beam,b_ell_bin_2=pa5_beam,w2=w2_cal1xcal1,bins=bins)
    binned_cal1Txcal1T /= tfunc
    # cal map 2 T (pa6 coadd) cross cal map 2 T (pa6 coadd)
    binned_cal2Txcal2T, _ = spectrum_from_maps(cal_map2_fourier,cal_map2_fourier,b_ell_bin_1=pa6_beam,b_ell_bin_2=pa6_beam,w2=w2_cal2xcal2,bins=bins)
    binned_cal2Txcal2T /= tfunc
    # Depth-1 T cross depth-1 T
    binned_T1xT1, _ = spectrum_from_maps(depth1_T,depth1_T,b_ell_bin_1=depth1_beam,b_ell_bin_2=depth1_beam,w2=w2_depth1,bins=bins)
    binned_T1xT1 /= tfunc
    # Depth-1 T cross cal map 2 T (pa6 coadd)
    binned_T1xcal2T, _ = spectrum_from_maps(depth1_T,cal_map2_fourier,b_ell_bin_1=depth1_beam,b_ell_bin_2=pa6_beam,w2=w2_depth1xcal2,bins=bins)
    binned_T1xcal2T /= tfunc

    # Constructing covariance mode count factors
    cal_binned_nu_cal1_cal2 = bincount*w2w4_cal1xcal2*np.sqrt(tfunc)
    cal_binned_nu_cal1_depth1 = bincount*w2w4_depth1xcal1*np.sqrt(tfunc)
    cal_binned_nu_all_three = bincount*w2w4_all_three*np.sqrt(tfunc)

    # Evaluating likelihood and fitting Gaussian for best fit calibration factor
    cal_fit_values = cal_sample_likelihood_and_fit(binned_cal1Txcal2T, binned_T1xcal1T, binned_cal1Txcal1T,
                                                   binned_cal2Txcal2T, binned_T1xcal2T, binned_T1xT1,
                                                   cal_binned_nu_cal1_cal2, cal_binned_nu_cal1_depth1, cal_binned_nu_all_three,
                                                   y_min=y_min,y_max=y_max,num_pts=cal_num_pts,use_curvefit=cal_use_curvefit)
    # Returning all the variables that I want to store in the output
    return  (cal_fit_values, binned_T1xcal1T, binned_cal1Txcal2T, binned_cal1Txcal1T, 
             binned_cal2Txcal2T, binned_T1xT1, binned_T1xcal2T,
             w2_depth1xcal1, w2_depth1xcal2, w2_cal1xcal2, w2_cal1xcal1, 
             w2_cal2xcal2, w2w4_all_three, w2w4_depth1xcal1, w2w4_cal1xcal2)

#########################################################
#########################################################
#########################################################
# Functions for viewing maps if needed
keys_eshow = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "mask": 0}

def eshow(x,**kwargs): 
    ''' Function to plot the maps for debugging '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "auto")

##########################################################
# Functions for saving output plots

keys_ewrite_QU = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 2000, "mask": 0}
keys_ewrite_ref_QU = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 500, "mask": 0}
keys_ewrite_T = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 2500, "mask": 0}
keys_ewrite_ref_T = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 1000, "mask": 0}
keys_ewrite_EB = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 1000, "mask": 0}
keys_ewrite_mask = {"downgrade": 1, "ticks": 5, "colorbar": True, "font_size": 40, "mask": 0}

def plot_T_maps(output_dir, map_name, maps, **kwargs):
    """Plotting just the T depth-1 map to use different range than Q/U."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_T = map_name + "_mapT"
    try: # Trying to catch issues with font_size being too big for very small maps
        plots = enplot.get_plots(maps, **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        plots = enplot.get_plots(maps, **kwargs)
    enplot.write(save_dir+save_fname_T, plots)

def plot_QU_maps(output_dir, map_name, maps, **kwargs):
    """Plotting Q/U depth-1 maps. Assumes Q, then U in an array."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_Q = map_name + "_mapQ"
    save_fname_U = map_name + "_mapU"
    try: # Trying to catch issues with font_size being too big for very small maps
        map_Q = enplot.get_plots(maps[0], **kwargs)
        map_U = enplot.get_plots(maps[1], **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        map_Q = enplot.get_plots(maps[0], **kwargs)
        map_U = enplot.get_plots(maps[1], **kwargs)
    enplot.write(save_dir+save_fname_Q, map_Q)
    enplot.write(save_dir+save_fname_U, map_U)

def plot_T_ref_maps(output_dir, map_name, maps, **kwargs):
    """Plotting just the T refernce map in the depth-1 shape. 
       Assumes multiplying by footprint first."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_T = map_name + "_refmapT_wmask"
    try: # Trying to catch issues with font_size being too big for very small maps
        plots = enplot.get_plots(maps, **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        plots = enplot.get_plots(maps, **kwargs)
    enplot.write(save_dir+save_fname_T, plots)

def plot_QU_ref_maps(output_dir, map_name, maps, **kwargs):
    """Plotting Q/U reference maps in the depth-1 shape.
       Assumes Q, then U in an array. Also assumes multiplying by footprint first."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_Q = map_name + "_refmapQ_wmask"
    save_fname_U = map_name + "_refmapU_wmask"
    try: # Trying to catch issues with font_size being too big for very small maps
        map_Q = enplot.get_plots(maps[0], **kwargs)
        map_U = enplot.get_plots(maps[1], **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        map_Q = enplot.get_plots(maps[0], **kwargs)
        map_U = enplot.get_plots(maps[1], **kwargs)
    enplot.write(save_dir+save_fname_Q, map_Q)
    enplot.write(save_dir+save_fname_U, map_U)

def plot_EB_filtered_maps(output_dir, map_name, depth1_TEB, map_mask, cut_map=False, **kwargs):
    """Converts the filtered Fourier space E and B maps back to real space and plots them.
       Assumes you pass in TEB, but only saves EB. Also multiplies by mask again to ignore
       any leakage from Fourier transforming."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_E = map_name + "_filtered_E"
    save_fname_B = map_name + "_filtered_B"
    if cut_map:
        maps_realspace = depth1_TEB # if the map is cut, don't inverse FT since there's nothing there
    else:
        maps_realspace = enmap.harm2map(depth1_TEB, normalize = "phys") # Does this actually leave them as EB?
    try: # Trying to catch issues with font_size being too big for very small maps
        map_E = enplot.get_plots(map_mask*maps_realspace[1], **kwargs)
        map_B = enplot.get_plots(map_mask*maps_realspace[2], **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        map_E = enplot.get_plots(map_mask*maps_realspace[1], **kwargs)
        map_B = enplot.get_plots(map_mask*maps_realspace[2], **kwargs)
    enplot.write(save_dir+save_fname_E, map_E)
    enplot.write(save_dir+save_fname_B, map_B)

def plot_mask(output_dir, map_name, map_mask, **kwargs):
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_mask = map_name + "_mask"
    try: # Trying to catch issues with font_size being too big for very small maps
        plots = enplot.get_plots(map_mask, **kwargs)
    except ValueError:
        # Could add a logging statement about trying a smaller font size
        kwargs['font_size'] = int(kwargs['font_size']/2)
        plots = enplot.get_plots(map_mask, **kwargs)
    enplot.write(save_dir+save_fname_mask, plots)

def cl_to_dl(cl, ell):
    """Helper function to convert C_ell to D_ell"""
    return cl*ell*(ell+1.0) / (2.0 * np.pi)

def dl_to_cl(dl, ell):
    """Helper function to convert D_ell to C_ell"""
    return dl*2.0*np.pi / (ell*(ell+1.0))

def plot_spectra_individually(output_dir, spectra):
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    maps = list(spectra.keys())
    ell_b = spectra[maps[0]]['ell']
    CAMB_ClEE_binned = spectra[maps[0]]['CAMB_EE']
    CAMB_ClBB_binned = spectra[maps[0]]['CAMB_BB']

    for i in tqdm(range(len(maps))):
        if spectra[maps[i]]['map_cut'] == 1: 
            # Making adjustments for any maps that were completely cut by galaxy mask
            # Just make empty linear scale plots so the web viewer layout is correct
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E1xE1'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E1xE1}$")
            plt.xlabel(r"$\ell$")
            plt.title("E1xE1 " + maps[i][:-9])
            plt.grid()
            output_fname = save_dir + maps[i][:-9] + "_e1xe1_spectrum_withCAMBee.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E2xE2'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E2xE2}$")
            plt.xlabel(r"$\ell$") 
            plt.title("E2xE2 " + maps[i][:-9])
            plt.grid()
            output_fname = save_dir + maps[i][:-9] + "_e2xe2_spectrum_withCAMBee.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['B1xB1'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{B1xB1}$")
            plt.xlabel(r"$\ell$")
            plt.title("B1xB1 " + maps[i][:-9])
            plt.grid()
            output_fname = save_dir + maps[i][:-9] + "_b1xb1_spectrum_withCAMBbb.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['B2xB2'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{B2xB2}$")
            plt.xlabel(r"$\ell$") 
            plt.title("B2xB2 " + maps[i][:-9])
            plt.grid()
            output_fname = save_dir + maps[i][:-9] + "_b2xb2_spectrum_withCAMBbb.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E1xB2'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E1xB2}$")
            plt.xlabel(r"$\ell$")
            plt.title("E1xB2 " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_e1xb2_spectrum.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E2xB1'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E2xB1}$")
            plt.xlabel(r"$\ell$") 
            plt.title("E2xB1 " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_e2xb1_spectrum.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            d_ell_covariance = cl_to_dl(cl_to_dl(spectra[maps[i]]['covariance'],ell_b),ell_b)
            plt.plot(ell_b, d_ell_covariance, marker='.', alpha=1.0) # Two factors of C_ell to D_ell because made of squares of spectra
            plt.ylabel("Covariance")
            plt.xlabel(r"$\ell$")
            plt.title("Covariance " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_covariance.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['estimator'],ell_b), marker='.', alpha=1.0, label='Estimator')
            plt.ylabel(r"$D_{\ell}^{E1xB2} - D_{\ell}^{E2xB1}$")
            plt.xlabel(r"$\ell$")
            plt.title("Estimator " + maps[i][:-9])
            plt.legend()
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_estimator.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, spectra[maps[i]]['binned_nu'], marker='.', alpha=1.0) # Not a spectra, so no conversion to D_ell
            plt.ylabel("Effective modes per bin")
            plt.xlabel(r"$\ell$")
            plt.title("$\\nu_b$ " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_modesperbin.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
        else:
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E1xE1'],ell_b), marker='.', alpha=1.0)
            plt.semilogy(ell_b, cl_to_dl(CAMB_ClEE_binned,ell_b), 'r.--', label="CAMB EE")
            plt.ylabel(r"$D_{\ell}^{E1xE1}$")
            plt.xlabel(r"$\ell$")
            plt.title("E1xE1 " + maps[i][:-9])
            plt.grid()
            plt.legend()
            output_fname = save_dir + maps[i][:-9] + "_e1xe1_spectrum_withCAMBee.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E2xE2'],ell_b), marker='.', alpha=1.0)
            plt.semilogy(ell_b, cl_to_dl(CAMB_ClEE_binned,ell_b), 'r.--', label="CAMB EE")
            plt.ylabel(r"$D_{\ell}^{E2xE2}$")
            plt.xlabel(r"$\ell$") 
            plt.title("E2xE2 " + maps[i][:-9])
            plt.grid()
            plt.legend()
            output_fname = save_dir + maps[i][:-9] + "_e2xe2_spectrum_withCAMBee.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B1xB1'],ell_b), marker='.', alpha=1.0)
            plt.semilogy(ell_b, cl_to_dl(CAMB_ClBB_binned,ell_b), 'r.--', label="CAMB BB")
            plt.ylabel(r"$D_{\ell}^{B1xB1}$")
            plt.xlabel(r"$\ell$")
            plt.title("B1xB1 " + maps[i][:-9])
            plt.grid()
            plt.legend()
            output_fname = save_dir + maps[i][:-9] + "_b1xb1_spectrum_withCAMBbb.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B2xB2'],ell_b), marker='.', alpha=1.0)
            plt.semilogy(ell_b, cl_to_dl(CAMB_ClBB_binned,ell_b), 'r.--', label="CAMB BB")
            plt.ylabel(r"$D_{\ell}^{B2xB2}$")
            plt.xlabel(r"$\ell$") 
            plt.title("B2xB2 " + maps[i][:-9])
            plt.grid()
            plt.legend()
            output_fname = save_dir + maps[i][:-9] + "_b2xb2_spectrum_withCAMBbb.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E1xB2'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E1xB2}$")
            plt.xlabel(r"$\ell$")
            plt.title("E1xB2 " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_e1xb2_spectrum.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()
            
            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E2xB1'],ell_b), marker='.', alpha=1.0)
            plt.ylabel(r"$D_{\ell}^{E2xB1}$")
            plt.xlabel(r"$\ell$") 
            plt.title("E2xB1 " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_e2xb1_spectrum.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            d_ell_covariance = cl_to_dl(cl_to_dl(spectra[maps[i]]['covariance'],ell_b),ell_b)
            plt.plot(ell_b, d_ell_covariance, marker='.', alpha=1.0) # Two factors of C_ell to D_ell because made of squares of spectra
            plt.ylabel("Covariance")
            plt.xlabel(r"$\ell$")
            plt.title("Covariance " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_covariance.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            est_err = np.sqrt(d_ell_covariance)
            plt.errorbar(ell_b, cl_to_dl(spectra[maps[i]]['estimator'],ell_b), yerr=est_err, marker='.', alpha=1.0, label='Estimator')
            # Plotting the best fit angle and the theory curve over measured estimator
            angle_rad = np.deg2rad(spectra[maps[i]]['meas_angle'])
            angle_errbar_rad = np.deg2rad(spectra[maps[i]]['meas_errbar'])
            plt.plot(ell_b, 1.0*cl_to_dl(CAMB_ClEE_binned,ell_b)*np.sin(2*angle_rad), marker='.', color='red', label="Theory estimator for best fit angle")
            # Plotting 1 sigma shadow above and below theory curve
            plt.fill_between(ell_b, 1.0*cl_to_dl(CAMB_ClEE_binned,ell_b)*np.sin(2*(angle_rad+angle_errbar_rad)), 
                            1.0*cl_to_dl(CAMB_ClEE_binned,ell_b)*np.sin(2*(angle_rad-angle_errbar_rad)), alpha=0.3, color='red')
            plt.ylabel(r"$D_{\ell}^{E1xB2} - D_{\ell}^{E2xB1}$")
            plt.xlabel(r"$\ell$")
            plt.title("Estimator " + maps[i][:-9])
            plt.legend()
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_estimator.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

            fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
            plt.plot(ell_b, spectra[maps[i]]['binned_nu'], marker='.', alpha=1.0) # Not a spectra, so no conversion to D_ell
            plt.ylabel("Effective modes per bin")
            plt.xlabel(r"$\ell$")
            plt.title("$\\nu_b$ " + maps[i][:-9])
            plt.grid()
            plt.axhline(y=0,color='gray',linewidth=2)
            output_fname = save_dir + maps[i][:-9] + "_modesperbin.png"
            plt.savefig(output_fname, dpi=300)
            plt.close()

def plot_spectra_summary(output_dir, spectra):
    """Plots some summary plots for the EE and BB autospectra for all maps in a run."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    maps = list(spectra.keys())
    ell_b = spectra[maps[0]]['ell']
    # Plotting all the EE autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    output_path_ee = save_dir + "ee_autospectra_all_test_maps.png"
    plt.semilogy(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_EE'],ell_b), 'r--', label="CAMB EE")
    for i in range(len(maps)):
        if spectra[maps[i]]['map_cut'] == 1:
            continue
        else:
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E1xE1'],ell_b),alpha=0.3)
    plt.ylabel(r"$D_{\ell}^{EE}$")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.legend()
    plt.title("EE Autospectra for All Depth-1 Maps")
    plt.savefig(output_path_ee, dpi=300)
    plt.close()

    # Plotting all the reference map EE autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    output_path_ee_ref = save_dir + "ee_autospectra_all_ref_maps.png"
    plt.semilogy(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_EE'],ell_b), 'r--', label="CAMB EE")
    for i in range(len(maps)):
        if spectra[maps[i]]['map_cut'] == 1:
            continue
        else:
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E2xE2'],ell_b),alpha=0.3)
    plt.ylabel(r"$D_{\ell}^{EE}$")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.legend()
    plt.title("EE Ref Map Autospectra in All Depth-1 Footprints")
    plt.savefig(output_path_ee_ref, dpi=300)
    plt.close()

    # Plotting all the BB autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained') 
    output_path_bb = save_dir + "bb_autospectra_all_test_maps.png"
    plt.semilogy(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_BB'],ell_b), 'r--', label="CAMB BB")
    for i in range(len(maps)):
        if spectra[maps[i]]['map_cut'] == 1:
            continue
        else:
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B1xB1'],ell_b),alpha=0.3)
    plt.ylabel(r"$D_{\ell}^{BB}$")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.legend()
    plt.title("BB Autospectra for All Depth-1 Maps")
    plt.savefig(output_path_bb, dpi=300)
    plt.close()

    # Plotting all the reference map BB autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    output_path_bb_ref = save_dir + "bb_autospectra_all_ref_maps.png"
    plt.semilogy(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_BB'],ell_b), 'r--', label="CAMB BB")
    for i in range(len(maps)):
        if spectra[maps[i]]['map_cut'] == 1:
            continue
        else:
            plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B2xB2'],ell_b),alpha=0.3)
    plt.ylabel(r"$D_{\ell}^{BB}$")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.legend()
    plt.title("BB Ref Map Autospectra in All Depth-1 Footprints")
    plt.savefig(output_path_bb_ref, dpi=300)
    plt.close()

def plot_likelihood(output_dir, map_name, angles, likelihood, gauss_fits, residual):
    """Plotting likelihood for angles in degrees."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_likelihood = map_name + "_likelihood.png"
    mean = gauss_fits[0]
    stddev = gauss_fits[1]
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.plot(angles, likelihood, 'b.', label='Mean={:1.3f}\n$\\sigma$={:1.3f}'.format(mean,stddev))
    plt.axvline(mean,alpha=0.3,color='black')
    # Could also move label to plt.figtext, but legend will auto adjust for me
    plt.plot(angles, gaussian(angles,mean,stddev), 'r', label='Fit Gaussian')
    plt.plot(angles, residual, 'gray', label="Residual")
    plt.legend()
    plt.ylabel("Likelihood")
    plt.xlabel("Angles (deg)")
    plt.grid()
    plt.title("Likelihood " + map_name)
    plt.savefig(save_dir+save_fname_likelihood, dpi=300)
    plt.close()

def plot_beam(output_dir, beam_name, ell, beam):
    """Plots binned ACT beam profile for file beam_name."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_beam = beam_name + ".png"
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.plot(ell, beam, 'b.')
    plt.ylabel("Beam transfer function")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.title("Beam Profile: " + beam_name)
    plt.savefig(save_dir+save_fname_beam, dpi=300)
    plt.close()

def plot_tfunc(output_dir, kx, ky, ell, tfunc):
    """Plots the filtering binned transfer function for some kx_cut and ky_cut values."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    str_k = 'kx'+ str(kx) + '_ky' + str(ky)
    save_fname_tfunc = "tfunc_"+str_k+".png"
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.plot(ell, tfunc, 'b.')
    plt.ylabel("Filtering transfer function")
    plt.xlabel(r"$\ell$")
    plt.grid()
    plt.title("Filtering tfunc for kx,ky: " + str(kx) + ',' + str(ky))
    plt.savefig(save_dir+save_fname_tfunc, dpi=300)
    plt.close()

def plot_angle_hist(output_dir,angles, maps):
    """Plots histogram of all angles for a given set of maps."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_hist = "angle_hist.png"
    # Could make a more complicated histogram with statistics overplotted
    # Plotting general histogram with all angles except the cut maps
    num_maps = len(angles)
    num_cut_maps = len(angles[angles == -9999])
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.hist(angles[angles != -9999], bins=30,
             label=str(num_maps) + " total maps \n" + str(num_cut_maps) + " cut maps")
    plt.title("Histogram of measured angles")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(save_dir+save_fname_hist, dpi=300)
    plt.close()
    # Also breaking it out by array
    map_names_split = np.array([a.split('_') for a in maps])
    maps_array = map_names_split[:,2] # Getting which array each map is from
    range_arrays = (angles[angles != -9999].min(),angles[angles != -9999].max()) # Ensuring same bins as total histogram
    for array in np.unique(maps_array):
        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        save_fname_hist = "angle_hist_" + array + ".png"
        angles_subset = angles[maps_array==array]
        num_maps = len(angles_subset)
        num_cut_maps = len(angles_subset[angles_subset == -9999])
        plt.hist(angles_subset[angles_subset != -9999], bins=30, range=range_arrays, 
                 label=str(num_maps) + " total maps, array " + array + "\n" + str(num_cut_maps) + " cut maps")
        plt.title("Histogram of measured angles, array " + array)
        plt.xlabel("Angle (deg)")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(save_dir+save_fname_hist, dpi=300)
        plt.close()

    # Combining them together
    # Could consider changing to a multihist or step hist later
    save_fname_hist = "angle_hist_all_combined.png"
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.hist(angles[angles != -9999], bins=30,alpha=0.5,label='All arrays')
    for array in np.unique(maps_array):
        angles_subset = angles[maps_array==array]
        plt.hist(angles_subset[angles_subset != -9999], bins=30, range=range_arrays, label=array, alpha=0.5)
    plt.title("Histogram of measured angles by array")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(save_dir+save_fname_hist, dpi=300)
    plt.close()
