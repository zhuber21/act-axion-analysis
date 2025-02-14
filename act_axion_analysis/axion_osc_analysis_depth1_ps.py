import numpy as np
from pixell import enmap, bunch
import os
import scipy
from scipy import optimize as op
from act_axion_analysis import axion_osc_plotting as aop

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
        aop.eshow(depth1_maps[0], **aop.keys_eshow)
        aop.eshow(depth1_maps[1], **aop.keys_eshow)
        aop.eshow(depth1_maps[2], **aop.keys_eshow)
        aop.eshow(depth1_ivar, **aop.keys_eshow)
    
    return depth1_maps, depth1_ivar, depth1_shape, depth1_wcs

def trim_ref_with_T(ref_maps,shape,wcs,plot=False):
    """
        Trims the full reference map down to a smaller size to match a given
        shape and wcs (from a particular depth-1, for example).
    """
    ref_maps_cut = enmap.extract(ref_maps,shape,wcs)

    if plot:
        aop.eshow(ref_maps_cut[0], **aop.keys_eshow)
        aop.eshow(ref_maps_cut[1], **aop.keys_eshow)
        aop.eshow(ref_maps_cut[2], **aop.keys_eshow)
    
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
        aop.eshow(mask, **aop.keys_eshow)
    
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
                aop.plot_T_maps(output_dir, map_fname, depth1_ivar_mask_tapered*depth1_maps[0], **aop.keys_ewrite_T)
                aop.plot_QU_maps(output_dir, map_fname, [depth1_ivar_mask_tapered*depth1_maps[1], depth1_ivar_mask_tapered*depth1_maps[2]], **aop.keys_ewrite_QU)
                aop.plot_T_ref_maps(output_dir, map_fname, depth1_ivar_mask_tapered*ref_cut[0], **aop.keys_ewrite_ref_T)
                aop.plot_QU_ref_maps(output_dir, map_fname, [depth1_ivar_mask_tapered*ref_cut[1], depth1_ivar_mask_tapered*ref_cut[2]],**aop.keys_ewrite_ref_QU)
                aop.plot_mask(output_dir, map_fname, depth1_ivar_mask_tapered, **aop.keys_ewrite_mask)
                aop.plot_EB_filtered_maps(output_dir, map_fname, filtered_depth1_TEB, depth1_ivar_mask_tapered, **aop.keys_ewrite_EB)
            return filtered_depth1_TEB, depth1_ivar, depth1_ivar_mask_tapered, depth1_ivar_indices, ref_cut_TEB, ivar_sum
        else:
            ivar_sum = calc_ivar_sum(depth1_filtering_mask, depth1_ivar_indices, depth1_ivar)
            if plot_maps:
                # Plotting using the first (filtering) mask
                map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
                aop.plot_T_maps(output_dir, map_fname, depth1_filtering_mask*depth1_maps[0], **aop.keys_ewrite_T)
                aop.plot_QU_maps(output_dir, map_fname, [depth1_filtering_mask*depth1_maps[1], depth1_filtering_mask*depth1_maps[2]], **aop.keys_ewrite_QU)
                aop.plot_T_ref_maps(output_dir, map_fname, depth1_filtering_mask*ref_cut[0], **aop.keys_ewrite_ref_T)
                aop.plot_QU_ref_maps(output_dir, map_fname, [depth1_filtering_mask*ref_cut[1], depth1_filtering_mask*ref_cut[2]],**aop.keys_ewrite_ref_QU)
                aop.plot_mask(output_dir, map_fname, depth1_filtering_mask, **aop.keys_ewrite_mask)
                aop.plot_EB_filtered_maps(output_dir, map_fname, filtered_depth1_TEB, depth1_filtering_mask, **aop.keys_ewrite_EB)                
            return filtered_depth1_TEB, depth1_ivar, depth1_filtering_mask, depth1_indices, ref_cut_TEB, ivar_sum
    else:
        if plot_maps:
            # All will show nothing except the mask, but still want the empty plots for web viewer code
            map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
            # Deliberately plotting depth-1 maps without mask so I can see what they originally looked like,
            # but plotting everything else with mask to show it was completely cut
            aop.plot_T_maps(output_dir, map_fname, depth1_maps[0], **aop.keys_ewrite_T)
            aop.plot_QU_maps(output_dir, map_fname, [depth1_maps[1], depth1_maps[2]], **aop.keys_ewrite_QU)
            aop.plot_T_ref_maps(output_dir, map_fname, test_mask*ref_cut[0], **aop.keys_ewrite_ref_T)
            aop.plot_QU_ref_maps(output_dir, map_fname, [test_mask*ref_cut[1], test_mask*ref_cut[2]],**aop.keys_ewrite_ref_QU)
            aop.plot_mask(output_dir, map_fname, test_mask, **aop.keys_ewrite_mask)
            aop.plot_EB_filtered_maps(output_dir, map_fname, depth1_maps, test_mask, cut_map=True, **aop.keys_ewrite_EB)
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
        aop.plot_likelihood(output_dir, map_name, angles_deg, norm_sampled_likelihood,fit_values_deg,residual)

    return fit_values_deg, residual_mean, residual_sum

def estimate_pol_angle(map_path, line, logger, ref_maps, ref_ivar, galaxy_mask,
                       kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                       plot_maps, plot_like, output_dir_path, 
                       bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
                       pa4_beam, pa5_beam, pa6_beam, ref_beam, tfunc, 
                       num_pts, angle_min_deg, angle_max_deg, use_curvefit):
    """
        High-level function to do the polarization angle estimation for each depth-1 map. 
        Allows intermediate maps to be cleaned up when function closes for better RAM usage.
    """
    outputs = load_and_filter_depth1(map_path, ref_maps, galaxy_mask, 
                                     kx_cut, ky_cut, unpixwin, 
                                     filter_radius=filter_radius,plot_maps=plot_maps,
                                     output_dir=output_dir_path,use_ivar_weight=use_ivar_weight)
    
    # If the full map has been cut by the galaxy mask, it return error code 1 instead of the regular outputs
    if outputs == 1:
        # saves output flag so we can see it is cut
        logger.info(f"Map {line} was completely cut by galaxy mask.")
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
        if plot_like:
            # Make empty likelihood plot for web viewer
            angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
            map_name = os.path.split(line)[1][:-9] # removing "_map.fits"
            aop.plot_likelihood(output_dir_path, map_name, angles_deg, np.zeros(num_pts), (-9999,-9999), np.zeros(num_pts))
        return output_dict, 1
    else: 
        # Otherwise do everything you would normally do
        depth1_TEB = outputs[0]
        depth1_ivar = outputs[1]
        # depth1_mask will be the doubly tapered mask if use_ivar_weight=True, the filtering mask if False
        # Same with indices
        depth1_mask = outputs[2]
        depth1_mask_indices = outputs[3]
        ref_TEB = outputs[4]
        ivar_sum = outputs[5]

        if use_ivar_weight:
            # Ivar weighting for depth-1 map
            # Calculating approx correction for loss of power due to tapering for spectra for depth-1
            # w_depth1 combines normalized ivar and geometric factors in this mask - normalization done in aoa.apply_ivar_weighting()
            depth1_TEB, w_depth1 = apply_ivar_weighting(depth1_TEB, depth1_ivar, depth1_mask, depth1_mask_indices)

            # Ivar weighting for reference map - already filtered and trimmed from ref_TEB above
            # w_ref combines normalized ivar and geometric factors in this mask
            ref_map_trimmed_ivar = enmap.extract(ref_ivar,depth1_TEB[0].shape,depth1_TEB[0].wcs)
            ref_TEB, w_ref = apply_ivar_weighting(ref_TEB, ref_map_trimmed_ivar, depth1_mask, depth1_mask_indices)
        else:
            # No ivar weighting
            w_depth1 = depth1_mask # use this if using flat weighting since only one taper is applied in this case
            w_ref = depth1_mask    # use this if using flat weighting since only one taper is applied in this case

        # Calculating w2 factors - all the same if not using ivar weighting, but different if using it
        # Using w2 factors for spectra corrections
        w2_depth1 = np.mean(w_depth1**2)
        w2_cross = np.mean(w_depth1*w_ref)
        w2_ref = np.mean(w_ref**2)
        # Calculating w2w4 factors - using w2w4_cross for the mode correction
        w2w4_depth1 = np.mean(w_depth1**2)**2 / np.mean(w_depth1**4)
        w2w4_cross = np.mean(w_depth1*w_ref)**2 / np.mean(w_depth1**2 * w_ref**2)
        w2w4_ref = np.mean(w_ref**2)**2 / np.mean(w_ref**4)

        # Selecting the correct beam
        map_array = line.split('_')[2]
        if map_array == 'pa4':
            depth1_beam = pa4_beam
        elif map_array == 'pa5':
            depth1_beam = pa5_beam
        elif map_array == 'pa6':
            depth1_beam = pa6_beam
        else:
            logger.info(f"Map {line} not in standard format for array beam selection! Choosing averaged beam.")
            depth1_beam = ref_beam

        # Calculate spectra
        # Spectra for estimator
        binned_E1xB2, bincount = spectrum_from_maps(depth1_TEB[1], ref_TEB[2], b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_E2xB1, _ = spectrum_from_maps(depth1_TEB[2], ref_TEB[1], b_ell_bin_1=ref_beam, b_ell_bin_2=depth1_beam, w2=w2_cross, bins=bins)
        # Spectra for covariance
        binned_E1xE1, _ = spectrum_from_maps(depth1_TEB[1], depth1_TEB[1], b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_B2xB2, _ = spectrum_from_maps(ref_TEB[2], ref_TEB[2], b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)
        binned_E2xE2, _ = spectrum_from_maps(ref_TEB[1], ref_TEB[1], b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)
        binned_B1xB1, _ = spectrum_from_maps(depth1_TEB[2], depth1_TEB[2], b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_E1xE2, _ = spectrum_from_maps(depth1_TEB[1], ref_TEB[1], b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_B1xB2, _ = spectrum_from_maps(depth1_TEB[2], ref_TEB[2], b_ell_bin_1=depth1_beam, b_ell_bin_2=ref_beam, w2=w2_cross, bins=bins)
        binned_E1xB1, _ = spectrum_from_maps(depth1_TEB[1], depth1_TEB[2], b_ell_bin_1=depth1_beam, b_ell_bin_2=depth1_beam, w2=w2_depth1, bins=bins)
        binned_E2xB2, _ = spectrum_from_maps(ref_TEB[1], ref_TEB[2], b_ell_bin_1=ref_beam, b_ell_bin_2=ref_beam, w2=w2_ref, bins=bins)    
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
        # Accounting for modes lost to the mask and filtering - uses w2w4_cross because estimator is made of cross spectra
        # Though the spectra are corrected by w2, the number of modes is corrected by w2w4
        # The thing returned by get_tfunc() is the t_b^2 factor from Steve's PS paper, which is the right correction for each spectrum.
        # We only want t_b in the mode correction, though, as in the text after Eq. 1 of Steve's paper.
        binned_nu = bincount*w2w4_cross*np.sqrt(tfunc)
        fsky = depth1_mask.area()/(4.*np.pi) # For comparing binned_nu to theoretical number of modes

        # Calculate estimator and covariance
        estimator = binned_E1xB2-binned_E2xB1
        covariance = ((1/binned_nu)*((binned_E1xE1*binned_B2xB2+binned_E1xB2**2)
                                    +(binned_E2xE2*binned_B1xB1+binned_E2xB1**2)
                                    -2*(binned_E1xE2*binned_B1xB2+binned_E1xB1*binned_E2xB2)))

        fit_values, residual_mean, residual_sum = sample_likelihood_and_fit(estimator,covariance,CAMB_ClEE_binned,num_pts=num_pts,
                                                                            angle_min_deg=angle_min_deg, angle_max_deg=angle_max_deg,
                                                                            use_curvefit=use_curvefit,plot_like=plot_like,
                                                                            output_dir=output_dir_path,map_fname=line)

        output_dict = {'ell': centers, 'E1xB2': binned_E1xB2, 'E2xB1': binned_E2xB1, 
                        'E1xE1': binned_E1xE1, 'B2xB2': binned_B2xB2, 'E2xE2': binned_E2xE2,
                        'B1xB1': binned_B1xB1, 'E1xE2': binned_E1xE2, 'B1xB2': binned_B1xB2,
                        'E1xB1': binned_E1xB1, 'E2xB2': binned_E2xB2, 'bincount': bincount,
                        'estimator': estimator, 'covariance': covariance,
                        'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                        'w2_depth1': w2_depth1, 'w2_cross': w2_cross, 'w2_ref': w2_ref, 'fsky': fsky,
                        'w2w4_depth1': w2w4_depth1, 'w2w4_cross': w2w4_cross, 'w2w4_ref': w2w4_ref,
                        'meas_angle': fit_values[0], 'meas_errbar': fit_values[1],
                        'ivar_sum': ivar_sum, 'residual_mean': residual_mean, 
                        'residual_sum': residual_sum, 'map_cut': 0}
        return output_dict, (depth1_mask, depth1_mask_indices, depth1_ivar, depth1_TEB[0], w_depth1, depth1_beam)


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
    cal_output_dict = {'T1xcal1T': binned_T1xcal1T, 'cal1Txcal2T': binned_cal1Txcal2T,
                       'cal1Txcal1T': binned_cal1Txcal1T, 'cal2Txcal2T': binned_cal2Txcal2T, 
                       'T1xT1': binned_T1xT1, 'T1xcal2T': binned_T1xcal2T,
                       'cal_factor': cal_fit_values[0], 'cal_factor_errbar': cal_fit_values[1],
                       'w2_depth1xcal1': w2_depth1xcal1, 'w2_depth1xcal2': w2_depth1xcal2,
                       'w2_cal1xcal2': w2_cal1xcal2, 'w2_cal1xcal1': w2_cal1xcal1, 'w2_cal2xcal2': w2_cal2xcal2,
                       'w2w4_all_three': w2w4_all_three, 'w2w4_depth1xcal1': w2w4_depth1xcal1, 'w2w4_cal1xcal2': w2w4_cal1xcal2}

    return cal_output_dict
