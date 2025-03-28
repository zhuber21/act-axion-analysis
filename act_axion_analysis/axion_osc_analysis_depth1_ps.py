import numpy as np
from pixell import enmap, bunch
import os
import scipy
from scipy import optimize as op
from act_axion_analysis import axion_osc_plotting as aop
from act_axion_analysis import utils

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

def calc_timestamps(map_path, mask):
    """
        Using time.fits and info.hdf files that accompany each depth-1 map to calculate the median timestamp
        of the region that is used for doing the power spectrum calculation (just the region where the tapered mask is 1.0). 

        Returns the timestamp in the filename (info_file.t), the median timestamp after adding the median offset of
        the power spectrum region, and the initial and final timestamps from the PS region.
    """
    info_file_path = map_path[:-8] + 'info.hdf'
    time_file_path = map_path[:-8] + 'time.fits'
    info_file = bunch.read(info_file_path)    # info_file.t should be the timestamp in the file name - first timestamp in file
    time_map = enmap.read_map(time_file_path) # this map gives the time offset from info_file.t
    name_timestamp = info_file.t
    # Restricting time_map just to the area inside the taper
    masked_time_map = time_map*mask
    median_time_offset = np.median(masked_time_map[mask==1.0])
    # The precision of the time.fits files seems to be four decimal places
    # Need to round because adding them to name_timestamp makes them floats with higher precision for some reason
    median_timestamp = np.round(name_timestamp + median_time_offset,4)
    initial_timestamp = np.round(name_timestamp + np.min(masked_time_map[mask==1.0]),4)
    final_timestamp = np.round(name_timestamp + np.max(masked_time_map[mask==1.0]),4)
    return name_timestamp, median_timestamp, initial_timestamp, final_timestamp
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

        Cal1 and Cal2 here may refer to either the pa5 coadd or the pa6 coadd, depending on
        which array the depth-1 map is. We want TT_cal1xdepth1 to be unbiased, so we make
        Cal1 the opposite array to depth1 (e.g. cal1 is the pa6 coadd if the depth-1 map is
        from pa5 - for pa4 it doesn't matter)

        The covariance has different mode count factors for each of the
        three covariance terms that are summed to get the full covariance.
        The y factor also multiplies the covariance for cal1xdepth1 
        wherever it appears in the covariance to account for how the
        changing calibration factor affects the covariance.
    """
    numerator = (cal1xcal2 - y*cal1xdepth1)**2
    covariance = ((1/nu_cal1_cal2)*(cal1xcal1*cal2xcal2+cal1xcal2**2)
                 +y**2*(1/nu_cal1_depth1)*(depth1xdepth1*cal1xcal1+cal1xdepth1**2)
               -2*y*(1/nu_all_three)*(cal1xdepth1*cal1xcal2+cal1xcal1*cal2xdepth1))
    denominator = 2*covariance
    likelihood = np.exp(-np.sum(numerator/denominator))
    return likelihood

def fwhm_fit(angles, likelihood, fwhm_point=0.5):
    """
       Finds the peak of the normalized likelihood, then finds the angle (in units of input angles)
       of the FWHM on either side of the maximum (default is to use the range 0-1 instead
       of the range of the function such that the FWHM always occurs at 0.5)

       Returns the peak and total FWHM in units of input angles.
    """
    # Find peak - should only be one due to normalization
    idx_peaks = np.where(likelihood==np.max(likelihood))[0]
    if idx_peaks.size > 1: 
        print(f"Multiple peaks found in fwhm_fit! idx_peak={idx_peak}. Using first peak found.")
    idx_peak = idx_peaks[0]
    peak_angle = angles[idx_peak]
    
    # Find FWHM on each side of peak
    if idx_peak != len(angles)-1 and idx_peak != 0:
        idx_above = np.argmin(np.abs(likelihood[idx_peak:-1] - fwhm_point))
        idx_below = np.argmin(np.abs(likelihood[0:idx_peak] - fwhm_point))
        if idx_above==len(likelihood[idx_peak:-1])-1:
            # If either of the half-width half-maxes are at the boundary
            # return a bad fit code for that HWHM
            hwhm_above = -9999
        else:
            hwhm_above = angles[idx_peak:-1][idx_above]
        if idx_below==0:
            hwhm_below = -9999
        else:
            hwhm_below = angles[0:idx_peak][idx_below]
        if hwhm_above == -9999 or hwhm_below == -9999:
            fwhm = -9999
        else:
            fwhm = hwhm_above - hwhm_below
    else:
        # If the peak is at the boundary of the fit range, 
        # return a bad fit code for everything
        fwhm = -9999
        hwhm_above = -9999
        hwhm_below = -9999  
    
    return peak_angle, fwhm, hwhm_above, hwhm_below

def skew_normal_fit_curvefit(angles,data,guess=None):
    """
        Uses scipy.optimize.curve_fit() to fit a skew normal distribution
        to the likelihood to get its mu, sigma, and alpha parameters. The first two
        parameters are NOT necessarily the mean and standard deviation because there
        are multiple sets of these three parameters that can give similar looking curves.
        Only when alpha=0 are they the conventional Gaussian mean and standard deviation.

        Assumes everything is in radians when fitting angles.

        Default starting values for the three parameters for curve_fit may be passed in 
        as a list with guess, but the default is to calculate an optimal guess from the
        data. That generally works well to avoid issues with failed fits, which can 
        happen when the initial guess if far from the final values.
    """    
    if guess is None:
        # default guess is the peak and std of likelihood plus mild asymmetry
        # The real default guess for the width (possible future upgrade) should be
        # the FHWM or similar, but not all our likelihoods have good FWHMs.
        # This is a simpler thing that gives good results with fewer checks.
        guess = [angles[np.argmax(data)], np.std(data), 1.0]
    popt, _ = op.curve_fit(utils.norm_skewnorm_pdf,angles,data,guess,maxfev=50000)
    mu = popt[0]
    sigma = np.abs(popt[1])
    alpha = popt[2]
    return mu, sigma, alpha

def gaussian_fit_curvefit(angles,data,guess=None):
    """
        Uses scipy.optimize.curve_fit() to fit a Gaussian to the likelihood to
        get the mean and standard deviation.

        Assumes everything is in radians when fitting angles.

        Default starting values for curve_fit may be passed in as a list with guess,
        but the default is to calculate an optimal guess from the data. That generally
        works well to avoid issues with failed fits, which can happen when the initial
        guess if far from the final values.
    """
    if guess is None:
        # default guess is the peak and std of likelihood
        # The real default guess for the width (possible future upgrade) should be
        # the FHWM or similar, but not all our likelihoods have good FWHMs.
        # This is a simpler thing that gives good results with fewer checks.
        guess = [angles[np.argmax(data)], np.std(data)]
    popt, _ = op.curve_fit(utils.gaussian,angles,data,guess,maxfev=50000)
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

def calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood, mean, stddev):
    """Calculating mean and sum of abs of residual between likelihood and best fit approximate Gaussian
    within 1 sigma of mean value - two possible ways of identifying and cutting poor S/N or bad fits
    """
    residual = norm_sampled_likelihood - utils.gaussian(angles_rad,mean,stddev)
    minus_sigma_idx = np.searchsorted(angles_rad, mean-stddev)
    plus_sigma_idx = np.searchsorted(angles_rad, mean+stddev)
    residual_mean = np.mean(np.abs(residual[minus_sigma_idx:plus_sigma_idx]))
    residual_sum = np.sum(np.abs(residual[minus_sigma_idx:plus_sigma_idx]))
    return residual_mean, residual_sum, residual

def sample_likelihood_and_fit(estimator,covariance,theory_ClEE,angle_min_deg=-45.0,angle_max_deg=45.0,
                              num_pts=50000,fit_method='fwhm',plot_like=False,output_dir=None,map_fname=None):
    """
       Samples likelihood for a range of angles and returns the best fit parameters for the
       resulting distribution in degrees.

       Has the option to do the fitting with scipy.optimize.curve_fit() (set fit_method='curvefit'),
       a method estimating the FWHM directly to account for non-Gaussianity (set fit_method='fwhm'),
       a method using a skew normal distribution (set fit_method='skewnorm'),
       or a method using moments of the Gaussian. The moments method is faster but less
       accurate when the likelihood deviates from Gaussianity in any way. The FWHM method is the most
       general and most accurately gets the peak value for likelihoods that start to deviate from
       an ideal Gaussian likelihood, but the skew normal method tends to account for the full shape
       better than the FWHM method while picking out the peak better than the Gaussian fit.

       Can also set fit_method='all' to use all methods on each likelihood.
    """
    likelihood_output_dict = {}
    
    if(angle_min_deg >= angle_max_deg): 
        raise ValueError("The min angle must be smaller than the max!")
    angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
    angles_rad = np.deg2rad(angles_deg)
    
    bin_sampled_likelihood = [estimator_likelihood(angle,estimator,covariance,theory_ClEE) for angle in angles_rad]
    norm_sampled_likelihood = bin_sampled_likelihood/np.max(bin_sampled_likelihood)
    
    if fit_method=='all':
        # Does all the methods
        # Start with Gaussian curve_fit
        fit_values = gaussian_fit_curvefit(angles_rad,norm_sampled_likelihood)
        likelihood_output_dict['meas_angle_gauss-curvefit'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['meas_errbar_gauss-curvefit'] = np.rad2deg(fit_values[1])
        # Does the residuals and plotting params for the Gaussian curve_fit part only
        residual_mean, residual_sum, residual = calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood,
                                                                 fit_values[0],fit_values[1])
        likelihood_output_dict['residual_mean'] = residual_mean
        likelihood_output_dict['residual_sum'] = residual_sum
        gauss_params_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]

        # FHWM fit
        fit_values = fwhm_fit(angles_rad,norm_sampled_likelihood)
        likelihood_output_dict['meas_angle_fwhm-method'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['fwhm_fwhm-method'] = np.rad2deg(fit_values[1])
        likelihood_output_dict['meas_errbar_fwhm-method'] = np.rad2deg(fit_values[1] / np.sqrt(8*np.log(2)))
        likelihood_output_dict['hwhm_above_fwhm-method'] = np.rad2deg(fit_values[2])
        likelihood_output_dict['hwhm_below_fwhm-method'] = np.rad2deg(fit_values[3])

        # Skew normal fit
        fit_values = skew_normal_fit_curvefit(angles_rad,norm_sampled_likelihood)
        angles_deg_skewnorm =  np.linspace(2*angle_min_deg,2*angle_max_deg,num=num_pts)
        skewnorm_fitted = utils.norm_skewnorm_pdf(angles_deg_skewnorm, np.rad2deg(fit_values[0]), 
                                                  np.rad2deg(fit_values[1]), fit_values[2])
        skewnorm_fwhm_params = fwhm_fit(angles_deg_skewnorm,skewnorm_fitted) # Results already in deg
        likelihood_output_dict['mu_skewnorm-method'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['sigma_skewnorm-method'] = np.rad2deg(fit_values[1])
        likelihood_output_dict['alpha_skewnorm-method'] = fit_values[2] # alpha is unitless
        likelihood_output_dict['meas_angle_skewnorm-method'] = skewnorm_fwhm_params[0]
        likelihood_output_dict['fwhm_skewnorm-method'] = skewnorm_fwhm_params[1]
        likelihood_output_dict['meas_errbar_skewnorm-method'] = skewnorm_fwhm_params[1] / np.sqrt(8*np.log(2))
        likelihood_output_dict['hwhm_above_skewnorm-method'] = skewnorm_fwhm_params[2]
        likelihood_output_dict['hwhm_below_skewnorm-method'] = skewnorm_fwhm_params[3]

        # Moment method
        fit_values = gaussian_fit_moment(angles_rad,norm_sampled_likelihood)
        likelihood_output_dict['meas_angle_gauss-moment'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['meas_errbar_gauss-moment'] = np.rad2deg(fit_values[1])
    elif fit_method=='curvefit':
        # Using default guess starting values set in function
        fit_values = gaussian_fit_curvefit(angles_rad,norm_sampled_likelihood)
        # Could add some flag or option to redo the fit for a given map if curve_fit() returns
        # a bad stddev value from failing to fit - for now I will leave it so I can see easily by the stddev that it fails
        likelihood_output_dict['meas_angle_gauss-curvefit'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['meas_errbar_gauss-curvefit'] = np.rad2deg(fit_values[1])

        residual_mean, residual_sum, residual = calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood,
                                                                 fit_values[0],fit_values[1])
        likelihood_output_dict['residual_mean'] = residual_mean
        likelihood_output_dict['residual_sum'] = residual_sum
        gauss_params_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]
    elif fit_method=='fwhm':
        fit_values = fwhm_fit(angles_rad,norm_sampled_likelihood)

        likelihood_output_dict['meas_angle_fwhm-method'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['fwhm_fwhm-method'] = np.rad2deg(fit_values[1])
        # Convert FWHM to the standard deviation of an equivalent width Gaussian
        # Good S/N likelihoods are already Gaussian. Lower S/N ones that can still
        # be fit have a central peak with width close to the width of a Gaussian.
        likelihood_output_dict['meas_errbar_fwhm-method'] = np.rad2deg(fit_values[1] / np.sqrt(8*np.log(2)))
        likelihood_output_dict['hwhm_above_fwhm-method'] = np.rad2deg(fit_values[2])
        likelihood_output_dict['hwhm_below_fwhm-method'] = np.rad2deg(fit_values[3])

        residual_mean, residual_sum, residual = calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood,
                                                                 fit_values[0],fit_values[1]/np.sqrt(8*np.log(2)))
        likelihood_output_dict['residual_mean'] = residual_mean
        likelihood_output_dict['residual_sum'] = residual_sum
        gauss_params_deg = [likelihood_output_dict['meas_angle_fwhm-method'], likelihood_output_dict['meas_errbar_fwhm-method']]
    elif fit_method=='skewnorm':
        fit_values = skew_normal_fit_curvefit(angles_rad,norm_sampled_likelihood)
        # Then fitting the FWHM of the best fit skew normal distribution
        # over a range 2x the width of the fitting range
        angles_deg_skewnorm =  np.linspace(2*angle_min_deg,2*angle_max_deg,num=num_pts)
        skewnorm_fitted = utils.norm_skewnorm_pdf(angles_deg_skewnorm, np.rad2deg(fit_values[0]), 
                                                  np.rad2deg(fit_values[1]), fit_values[2])
        skewnorm_fwhm_params = fwhm_fit(angles_deg_skewnorm,skewnorm_fitted) # Results already in deg

        likelihood_output_dict['mu_skewnorm-method'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['sigma_skewnorm-method'] = np.rad2deg(fit_values[1])
        likelihood_output_dict['alpha_skewnorm-method'] = fit_values[2] # alpha is unitless
        likelihood_output_dict['meas_angle_skewnorm-method'] = skewnorm_fwhm_params[0]
        likelihood_output_dict['fwhm_skewnorm-method'] = skewnorm_fwhm_params[1]
        likelihood_output_dict['meas_errbar_skewnorm-method'] = skewnorm_fwhm_params[1] / np.sqrt(8*np.log(2))
        likelihood_output_dict['hwhm_above_skewnorm-method'] = skewnorm_fwhm_params[2]
        likelihood_output_dict['hwhm_below_skewnorm-method'] = skewnorm_fwhm_params[3]

        residual_mean, residual_sum, residual = calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood,
                                                            np.deg2rad(skewnorm_fwhm_params[0]),
                                                            np.deg2rad(skewnorm_fwhm_params[1] / np.sqrt(8*np.log(2))))
        likelihood_output_dict['residual_mean'] = residual_mean
        likelihood_output_dict['residual_sum'] = residual_sum
        gauss_params_deg = [likelihood_output_dict['meas_angle_skewnorm-method'], likelihood_output_dict['meas_errbar_skewnorm-method']]
    else:
        fit_values = gaussian_fit_moment(angles_rad,norm_sampled_likelihood)
        likelihood_output_dict['meas_angle_gauss-moment'] = np.rad2deg(fit_values[0])
        likelihood_output_dict['meas_errbar_gauss-moment'] = np.rad2deg(fit_values[1])
        residual_mean, residual_sum, residual = calc_residual_mean_and_sum(angles_rad, norm_sampled_likelihood,
                                                                 fit_values[0],fit_values[1])
        likelihood_output_dict['residual_mean'] = residual_mean
        likelihood_output_dict['residual_sum'] = residual_sum
        gauss_params_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]

    if plot_like:
        map_name = os.path.split(map_fname)[1][:-9] # removing "_map.fits"
        aop.plot_likelihood(output_dir, map_name, angles_deg, norm_sampled_likelihood,gauss_params_deg,residual)

    return likelihood_output_dict

def estimate_pol_angle(map_path, line, logger, ref_maps, ref_ivar, galaxy_mask,
                       kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                       plot_maps, plot_like, output_dir_path, 
                       bins, centers, CAMB_ClEE_binned, CAMB_ClBB_binned, 
                       depth1_beam, ref_beam, tfunc, 
                       num_pts, angle_min_deg, angle_max_deg, fit_method):
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
                       'name_timestamp': -9999, 'median_timestamp': -9999, 
                       'initial_timestamp': -9999, 'final_timestamp': -9999,
                       'ivar_sum': -9999, 'residual_mean': -9999, 
                       'residual_sum': -9999, 'map_cut': 1}

        if fit_method == 'all':
            output_dict.update({'meas_angle_gauss-curvefit': -9999, 'meas_errbar_gauss-curvefit': -9999,
                                'meas_angle_fwhm-method': -9999, 'fwhm_fwhm-method': -9999,
                                'meas_errbar_fwhm-method': -9999, 'hwhm_above_fwhm-method': -9999,
                                'hwhm_below_fwhm-method': -9999, 'mu_skewnorm-method': -9999,
                                'sigma_skewnorm-method': -9999, 'alpha_skewnorm-method': -9999,
                                'meas_angle_skewnorm-method': -9999, 'fwhm_skewnorm-method': -9999,
                                'meas_errbar_skewnorm-method': -9999, 'hwhm_above_skewnorm-method': -9999,
                                'hwhm_below_skewnorm-method': -9999, 'meas_angle_gauss-moment': -9999,
                                'meas_errbar_gauss-moment': -9999})
        elif fit_method == 'curvefit':
            output_dict.update({'meas_angle_gauss-curvefit': -9999, 'meas_errbar_gauss-curvefit': -9999})
        elif fit_method == 'fwhm':
            output_dict.update({'meas_angle_fwhm-method': -9999, 'fwhm_fwhm-method': -9999,
                                'meas_errbar_fwhm-method': -9999, 'hwhm_above_fwhm-method': -9999,
                                'hwhm_below_fwhm-method': -9999})
        elif fit_method == 'skewnorm':
            output_dict.update({'mu_skewnorm-method': -9999,
                                'sigma_skewnorm-method': -9999, 'alpha_skewnorm-method': -9999,
                                'meas_angle_skewnorm-method': -9999, 'fwhm_skewnorm-method': -9999,
                                'meas_errbar_skewnorm-method': -9999, 'hwhm_above_skewnorm-method': -9999,
                                'hwhm_below_skewnorm-method': -9999})
        else:
            output_dict.update({'meas_angle_gauss-moment': -9999, 'meas_errbar_gauss-moment': -9999})

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
        w2w4_depth1 = w2_depth1**2 / np.mean(w_depth1**4)
        w2w4_cross = w2_cross**2 / np.mean(w_depth1**2 * w_ref**2)
        w2w4_ref = w2_ref**2 / np.mean(w_ref**4)

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

        likelihood_output_dict = sample_likelihood_and_fit(estimator,covariance,CAMB_ClEE_binned,num_pts=num_pts,
                                                            angle_min_deg=angle_min_deg, angle_max_deg=angle_max_deg,
                                                            fit_method=fit_method,plot_like=plot_like,
                                                            output_dir=output_dir_path,map_fname=line)

        output_dict = {'ell': centers, 'E1xB2': binned_E1xB2, 'E2xB1': binned_E2xB1, 
                        'E1xE1': binned_E1xE1, 'B2xB2': binned_B2xB2, 'E2xE2': binned_E2xE2,
                        'B1xB1': binned_B1xB1, 'E1xE2': binned_E1xE2, 'B1xB2': binned_B1xB2,
                        'E1xB1': binned_E1xB1, 'E2xB2': binned_E2xB2, 'bincount': bincount,
                        'estimator': estimator, 'covariance': covariance,
                        'CAMB_EE': CAMB_ClEE_binned, 'CAMB_BB': CAMB_ClBB_binned,
                        'w2_depth1': w2_depth1, 'w2_cross': w2_cross, 'w2_ref': w2_ref, 'fsky': fsky,
                        'w2w4_depth1': w2w4_depth1, 'w2w4_cross': w2w4_cross, 'w2w4_ref': w2w4_ref,
                        'ivar_sum': ivar_sum, 'map_cut': 0}
        output_dict.update(likelihood_output_dict)
        return output_dict, (depth1_mask, depth1_mask_indices, depth1_ivar, depth1_TEB[0], w_depth1)


def cal_sample_likelihood_and_fit(cal1xcal2, cal1xdepth1, cal1xcal1, cal2xcal2, cal2xdepth1, 
                                  depth1xdepth1, nu_cal1_cal2, nu_cal1_depth1, nu_all_three, 
                                  y_min=0.7, y_max=1.3, num_pts=50000, cal_fit_method='fwhm'):
    """
        Samples the likelihood for the TT calibration at num_pts values of the calibration
        factor, y, between the values y_min and y_max.

        The cal_fit_method variable selects which fitter to use for extracting the peak
        and width of likelihood. Options are 'fwhm' (uses fwhm_fit() to find peak and FWHM,
        which is converted to the std dev for an equivalent Gaussian), 'curvefit' (uses scipy's
        curve_fit() to fit a Gaussian to the likelihood), 'skewnorm' (uses scipy's curve_fit() to 
        fit a skew normal distribution to the likelihood), or 'moment' (calculates the first and
        second moments of a Gaussian). The FWHM or skewnorm option is the most general and robust to non-
        Gaussianities in the lower S/N likelihoods.

        Cal1 and Cal2 here may refer to either the pa5 coadd or the pa6 coadd, depending on
        which array the depth-1 map is.
    """
    if(y_min >= y_max): 
        raise ValueError("The min y value must be smaller than the max!")
    y_values = np.linspace(y_min,y_max,num=num_pts)

    cal_sampled_likelihood = [cal_likelihood(y, cal1xcal2, cal1xdepth1, cal1xcal1, 
                                             cal2xcal2, cal2xdepth1, depth1xdepth1,
                                             nu_cal1_cal2, nu_cal1_depth1, nu_all_three) for y in y_values]
    norm_sampled_likelihood = cal_sampled_likelihood/np.max(cal_sampled_likelihood)

    if cal_fit_method=='curvefit':
        # The default guess starting values are for the angle likelihood, so pass new ones here
        guess = [y_values[np.argmax(norm_sampled_likelihood)], 0.2] # Guessing a center value of the data peak with std dev 0.2
        fit_values = gaussian_fit_curvefit(y_values,norm_sampled_likelihood, guess=guess)
        # Testing revealed this sometimes fails to fit good ones that are far from the guess
        # Easiest thing to do is just refit them with moment method - usually still easy to separate out bad ones
        if fit_values[1] < 0.005:
            fit_values = gaussian_fit_moment(y_values,norm_sampled_likelihood)
    elif cal_fit_method=='fwhm':
        fit_values = fwhm_fit(y_values,norm_sampled_likelihood)
        # Convert FWHM to the standard deviation of an equivalent width Gaussian
        # Good S/N likelihoods are already Gaussian. Lower S/N ones that can still
        # be fit have a central peak with width close to the width of a Gaussian.
        sigma = fit_values[1] / np.sqrt(8*np.log(2))
        fit_values = [fit_values[0], sigma]
    elif cal_fit_method=='skewnorm':
        skew_fit_values = skew_normal_fit_curvefit(y_values,norm_sampled_likelihood)
        # Then fitting the FWHM of the best fit skew normal distribution
        # over a range 2x the width of the fitting range
        y_skewnorm =  np.linspace(0.5*y_min,2*y_max,num=num_pts)
        skewnorm_fitted = utils.norm_skewnorm_pdf(y_skewnorm, skew_fit_values[0], 
                                                  skew_fit_values[1], skew_fit_values[2])
        skewnorm_fwhm_params = fwhm_fit(y_skewnorm,skewnorm_fitted)
        fit_values = [skewnorm_fwhm_params[0], skewnorm_fwhm_params[1] / np.sqrt(8*np.log(2))]
    else:
        fit_values = gaussian_fit_moment(y_values,norm_sampled_likelihood)
    
    return fit_values

def cross_calibrate(map_array, cal_T_map1_act_footprint, cal_T_map2_act_footprint,
                    cal_T_ivar1_act_footprint, cal_T_ivar2_act_footprint,
                    depth1_ivar, depth1_mask, depth1_mask_indices,
                    galaxy_mask, depth1_T, w_depth1, w2_depth1, cal_centers, cal_bins, tfunc,
                    kx_cut, ky_cut, unpixwin, filter_radius, use_ivar_weight,
                    depth1_beam, cal_T_beam1, cal_T_beam2, y_min, y_max, cal_num_pts, cal_fit_method):
    """
        High-level function to do the TT cross-calibration. 
        Allows intermediate maps to be cleaned up when function closes for better RAM usage.

        cal_T_map1_act_footprint must be a pa5 coadd and cal_T_map2_act_footprint must be a pa6 coadd
        to avoid biasing the results from elevated noise in the depth1xcoadd when they belong to the
        same array.

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
    w2_depth1xpa5 = np.mean(w_depth1*w_cal1)
    w2_depth1xpa6 = np.mean(w_depth1*w_cal2)
    w2_pa5xpa6 = np.mean(w_cal1*w_cal2)
    w2_pa5xpa5 = np.mean(w_cal1*w_cal1)
    w2_pa6xpa6 = np.mean(w_cal2*w_cal2)
    w2w4_depth1xpa5 = w2_depth1xpa5**2 / np.mean(w_depth1**2 * w_cal1**2)
    w2w4_depth1xpa6 = w2_depth1xpa6**2 / np.mean(w_depth1**2 * w_cal2**2)
    w2w4_pa5xpa6 = w2_pa5xpa6**2 / np.mean(w_cal1**2 * w_cal2**2)
    w2w4_556d = w2_pa5xpa6*w2_depth1xpa5 / np.mean(w_cal1*w_cal1*w_cal2*w_depth1)
    w2w4_665d = w2_pa5xpa6*w2_depth1xpa6 / np.mean(w_cal1*w_cal2*w_cal2*w_depth1)
    # Depth-1 T cross cal map 1 T (pa5 coadd)
    binned_T1xpa5T, cal_bincount = spectrum_from_maps(depth1_T,cal_map1_fourier,b_ell_bin_1=depth1_beam,b_ell_bin_2=cal_T_beam1,w2=w2_depth1xpa5,bins=cal_bins)
    binned_T1xpa5T /= tfunc
    # cal map 1 T (pa5 coadd) cross cal map 2 T (pa6 coadd)
    binned_pa5Txpa6T, _ = spectrum_from_maps(cal_map1_fourier,cal_map2_fourier,b_ell_bin_1=cal_T_beam1,b_ell_bin_2=cal_T_beam2,w2=w2_pa5xpa6,bins=cal_bins)
    binned_pa5Txpa6T /= tfunc
    # Getting spectra for the covariance
    # cal map 1 T (pa5 coadd) cross cal map 1 T (pa5 coadd)
    binned_pa5Txpa5T, _ = spectrum_from_maps(cal_map1_fourier,cal_map1_fourier,b_ell_bin_1=cal_T_beam1,b_ell_bin_2=cal_T_beam1,w2=w2_pa5xpa5,bins=cal_bins)
    binned_pa5Txpa5T /= tfunc
    # cal map 2 T (pa6 coadd) cross cal map 2 T (pa6 coadd)
    binned_pa6Txpa6T, _ = spectrum_from_maps(cal_map2_fourier,cal_map2_fourier,b_ell_bin_1=cal_T_beam2,b_ell_bin_2=cal_T_beam2,w2=w2_pa6xpa6,bins=cal_bins)
    binned_pa6Txpa6T /= tfunc
    # Depth-1 T cross depth-1 T
    binned_T1xT1, _ = spectrum_from_maps(depth1_T,depth1_T,b_ell_bin_1=depth1_beam,b_ell_bin_2=depth1_beam,w2=w2_depth1,bins=cal_bins)
    binned_T1xT1 /= tfunc
    # Depth-1 T cross cal map 2 T (pa6 coadd)
    binned_T1xpa6T, _ = spectrum_from_maps(depth1_T,cal_map2_fourier,b_ell_bin_1=depth1_beam,b_ell_bin_2=cal_T_beam2,w2=w2_depth1xpa6,bins=cal_bins)
    binned_T1xpa6T /= tfunc

    if map_array=='pa4' or map_array=='pa6':
        # Calibrating the pa4 and pa6 maps to the pa6 ref maps to avoid depth1xpa6 bias
        # The estimator is then pa5xpa6-y*depth1xpa5 for the pa6 (and pa4) depth-1 maps
        # Constructing covariance mode count factors
        cal_binned_nu_pa5_pa6 = cal_bincount*w2w4_pa5xpa6*np.sqrt(tfunc)
        cal_binned_nu_pa5_depth1 = cal_bincount*w2w4_depth1xpa5*np.sqrt(tfunc)
        cal_binned_nu_556d = cal_bincount*w2w4_556d*np.sqrt(tfunc)

        # Evaluating likelihood and fitting Gaussian for best fit calibration factor
        cal_fit_values = cal_sample_likelihood_and_fit(binned_pa5Txpa6T, binned_T1xpa5T, binned_pa5Txpa5T,
                                                    binned_pa6Txpa6T, binned_T1xpa6T, binned_T1xT1,
                                                    cal_binned_nu_pa5_pa6, cal_binned_nu_pa5_depth1, cal_binned_nu_556d,
                                                    y_min=y_min,y_max=y_max,num_pts=cal_num_pts,cal_fit_method=cal_fit_method)
    else:
        # Calibrating the pa5 maps to the pa5 ref maps to avoid depth1xpa5 bias
        # The estimator is then pa5xpa6-y*depth1xpa6 for the pa5 depth-1 maps
        # Constructing covariance mode count factors
        cal_binned_nu_pa5_pa6 = cal_bincount*w2w4_pa5xpa6*np.sqrt(tfunc)
        cal_binned_nu_pa6_depth1 = cal_bincount*w2w4_depth1xpa6*np.sqrt(tfunc)
        cal_binned_nu_665d = cal_bincount*w2w4_665d*np.sqrt(tfunc)

        # Evaluating likelihood and fitting Gaussian for best fit calibration factor
        cal_fit_values = cal_sample_likelihood_and_fit(binned_pa5Txpa6T, binned_T1xpa6T, binned_pa6Txpa6T,
                                                    binned_pa5Txpa5T, binned_T1xpa5T, binned_T1xT1,
                                                    cal_binned_nu_pa5_pa6, cal_binned_nu_pa6_depth1, cal_binned_nu_665d,
                                                    y_min=y_min,y_max=y_max,num_pts=cal_num_pts,cal_fit_method=cal_fit_method)        
    # Returning all the variables that I want to store in the output
    cal_output_dict = {'cal_ell': cal_centers, 'T1xpa5T': binned_T1xpa5T, 'pa5Txpa6T': binned_pa5Txpa6T,
                       'pa5Txpa5T': binned_pa5Txpa5T, 'pa6Txpa6T': binned_pa6Txpa6T, 
                       'T1xT1': binned_T1xT1, 'T1xpa6T': binned_T1xpa6T, 'cal_bincount': cal_bincount,
                       'cal_factor': cal_fit_values[0], 'cal_factor_errbar': cal_fit_values[1],
                       'w2_depth1xpa5': w2_depth1xpa5, 'w2_depth1xpa6': w2_depth1xpa6,
                       'w2_pa5xpa6': w2_pa5xpa6, 'w2_pa5xpa5': w2_pa5xpa5, 'w2_pa6xpa6': w2_pa6xpa6,
                       'w2w4_556d': w2w4_556d, 'w2w4_665d': w2w4_665d, 'w2w4_depth1xpa5': w2w4_depth1xpa5, 
                       'w2w4_pa5xpa6': w2w4_pa5xpa6, 'w2w4_depth1xpa6': w2w4_depth1xpa6}

    return cal_output_dict
