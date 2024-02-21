import numpy as np
from pixell import enmap, enplot
import nawrapper as nw

#########################################################
# Functions for viewing maps if needed
keys_eshow = {"downgrade": 2, "ticks": 5, "colorbar": True, "font_size": 40}

def eshow(x,**kwargs): 
    ''' Function to plot the maps for debugging '''
    plots = enplot.get_plots(x, **kwargs)
    enplot.show(plots, method = "auto")

##########################################################
# Functions for loading config file and maps

"""def load_camb_theory_spectrum(camb_file,lmax,bin_size,start_index):
    
        Loads the DlEE spectrum from a CAMB output file, converts to ClEE, and
        bins it appropriately for use in estimator.
        
        Ignores bins up to start_index to get rid of bins that are filtered out
        in the power spectrum estimation in the maps.
    

    ell,DlEE = np.loadtxt(camb_file, usecols=(0,2), unpack=True)
    
    # Note that ell runs from 2 to 5400
    ClEE = DlEE * 2 * np.pi / (ell*(ell+1.))
    
    bins = np.arange(0,lmax,bin_size)
    digitized = np.digitize(ell, bins, right=True)
    ClEE_binned = np.bincount(digitized, ClEE.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]

    return ClEE_binned[start_index:]"""

def load_and_bin_beam(fname,lmax,bins):
    """
       	For a given fname containing an ACT beam, loads in the ell and b_ell
        of the beam. Normalizes the beam and bins it into the same binning as
        the power spectra.
    """
    beam_ell, beam_tform = np.loadtxt(fname,usecols=(0,1),unpack=True)
    lmax = int(bins[-1])+2
    beam_ell_norm = beam_ell[1:lmax]
    beam_tform_norm = beam_tform[1:lmax]/np.max(beam_tform[1:lmax])

    digitized = np.digitize(beam_ell_norm, bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    beam_tform_norm_binned = np.bincount(digitized, beam_tform_norm.reshape(-1))[1:-1]/bincount
    return beam_tform_norm_binned
    
def load_ref_map_and_beam(fname_ref,fname_ref_beam,lmax,bins):
    """
        Loads in the full reference map (T,Q,U) and the beam for this map. 
        Needs to be done once.
        
        The beam is also binned to match the binning of all spectra.
    """
    maps = enmap.read_map(fname_ref)
    ivar_name = fname_ref[:-8] + "ivar.fits"
    ivar = 0.5*enmap.read_map(ivar_name) # 0.5 for polarization noise
    
    b_ell = load_and_bin_beam(fname_ref_beam,lmax,bins)

    return maps, ivar, b_ell

def apply_kspace_filter(maps, kx_cut, ky_cut, unpixwin):
    """
       	Takes in a set of T/Q/U maps that already have a taper applied, apply
        a k-space filter to remove ground pickup, and returns the E and B maps
    """
    singleobs_TEB = enmap.map2harm(maps, normalize = "phys")

    if unpixwin:  # remove pixel window in Fourier space
        for i in range(len(singleobs_TEB)):
            wy, wx = enmap.calc_window(singleobs_TEB[i].shape)
            singleobs_TEB[i] /= wy[:, np.newaxis]
            singleobs_TEB[i] /= wx[np.newaxis, :]

    ly, lx = singleobs_TEB.lmap()
    kfilter_x = np.abs(lx) >= kx_cut
    kfilter_y = np.abs(ly) >= ky_cut
    filtered_TEB = singleobs_TEB * kfilter_x * kfilter_y

    return filtered_TEB
    
def load_test_map_and_filter(fname, beam_path, footprint, kx_cut, ky_cut, unpixwin, lmax, bins):
    """
       	Loads in the maps located at fname to the shape and wcs specified by the
        footprint. Also filters the maps after applying the taper.

        Also loads in appropriate beam and bins it correctly.
    """
    maps = enmap.read_map(fname)
    maps = enmap.extract(maps,footprint.shape,footprint.wcs)

    b_ell = load_and_bin_beam(beam_path,lmax,bins)

    filtered_TEB = apply_kspace_filter(maps*footprint, kx_cut, ky_cut, unpixwin)

    return filtered_TEB, b_ell

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

def make_tapered_mask(map_to_mask,filter_radius=1.0,plot=False):
    """
        Makes a mask for a given map based on where the ivar map is nonzero.
        Also apodizes the mask and gets the indices of where the apodized
        mask is not equal to one (everything tapered or outside the mask)
        in order to set all points but those to zero after filtering.
    """
    footprint = 1*map_to_mask.astype(bool)
    mask = nw.apod_C2(footprint,filter_radius)
    
    # Getting points to set to zero after filtering
    indices = np.nonzero(mask != 1)
    
    if plot:
        eshow(mask, **keys_eshow)
    
    return mask, indices

def load_and_filter_depth1(fname, ref_maps, kx_cut, ky_cut, unpixwin):
    depth1_maps, depth1_ivar, shape, wcs = load_depth1_with_T(fname)
    ref_cut = trim_ref_with_T(ref_maps,shape,wcs)
        
    # Apodize and filter depth-1
    depth1_mask, depth1_indices = make_tapered_mask(depth1_ivar,filter_radius=1.0)
    filtered_depth1_TEB = apply_kspace_filter(depth1_maps*depth1_mask, kx_cut, ky_cut, unpixwin=True)
        
    # Apodize and filter coadd
    ref_cut_TEB = apply_kspace_filter(ref_cut*depth1_mask, kx_cut, ky_cut, unpixwin=True)
    
    return filtered_depth1_TEB, depth1_ivar, depth1_mask, ref_cut_TEB
##########################################################


##########################################################
# Functions for doing PS calculations and estimator

def spectrum_from_maps(map1, map2, b_ell_bin_1, b_ell_bin_2, w2, bins):
    """Function modified from the one in ACT DR4/5 NB7 for binning a power spectrum for two maps.
       This function does account for a window correction for the apodizing at this point.
       Also accounts for a beam correction using a beam defined by b_ell.
    """
    spectrum = np.real(map1*np.conj(map2))

    # Dividing by an approx. correction for the loss of power from tapering
    spectrum /= w2

    modlmap = map1.modlmap()

    # Bin the power spectrum
    digitized = np.digitize(np.ndarray.flatten(modlmap), bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    binned = np.bincount(digitized, spectrum.reshape(-1))[1:-1]/bincount

    binned /= (b_ell_bin_1*b_ell_bin_2)

    return binned, bincount

def get_tfunc(kx,ky,lmax, bins):
    cut = (ky+kx)*4
    ell = np.arange(lmax)
    tfunc = np.zeros(lmax)
    tfunc[1:] = 1 - cut / (2*np.pi*ell[1:])

    digitized = np.digitize(ell, bins, right=True)
    bincount = np.bincount(digitized)[1:-1]
    tfunc_binned = np.bincount(digitized, tfunc.reshape(-1))[1:-1]/bincount
    return tfunc_binned

"""def calc_all_spectra_and_estimator(test_map_E,test_map_B,test_map_beam,ref_map_E,ref_map_B,ref_map_beam,w2,lmax,bin_size,start_index):
    
        Calculates all of the relevant auto and cross-spectra between two maps
        for the estimator and covariance in our PS method.
        
        All spectra are binned. Bins prior to start_index are cut in order
        to get rid of the parts of the spectra affected by filtering out ground
        pickup.
        
        **To-do: save the calculated spectra for calibration purposes
    
    map1_E = test_map_E
    map1_B = test_map_B
    map1_beam = test_map_beam
    map2_E = ref_map_E
    map2_B = ref_map_B
    map2_beam = ref_map_beam    
    
    # Spectra for estimator - bincount and ell_b should be the same for all
    ell_b, binned_E1xB2, bincount = spectrum_from_maps(map1_E, map2_B, b_ell_bin_1=map1_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_E2xB1, _ = spectrum_from_maps(map1_B, map2_E, b_ell_bin_1=map1_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)

    # Spectra for covariance
    _, binned_E1xE1, _ = spectrum_from_maps(map1_E, map1_E, b_ell_bin_1=map1_beam, b_ell_bin_2=map1_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_B2xB2, _ = spectrum_from_maps(map2_B, map2_B, b_ell_bin_1=map2_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_E2xE2, _ = spectrum_from_maps(map2_E, map2_E, b_ell_bin_1=map2_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_B1xB1, _ = spectrum_from_maps(map1_B, map1_B, b_ell_bin_1=map1_beam, b_ell_bin_2=map1_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_E1xE2, _ = spectrum_from_maps(map1_E, map2_E, b_ell_bin_1=map1_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_B1xB2, _ = spectrum_from_maps(map1_B, map2_B, b_ell_bin_1=map1_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_E1xB1, _ = spectrum_from_maps(map1_E, map1_B, b_ell_bin_1=map1_beam, b_ell_bin_2=map1_beam, w2=w2, lmax=lmax, bin_size=bin_size)
    _, binned_E2xB2, _ = spectrum_from_maps(map2_E, map2_B, b_ell_bin_1=map2_beam, b_ell_bin_2=map2_beam, w2=w2, lmax=lmax, bin_size=bin_size)

    binned_nu = bincount*w2

    # Start index is supposed to ignore that parts of ell space that
    # are filtered out
    bin_estimator = binned_E1xB2[start_index:]-binned_E2xB1[start_index:]
    bin_covariance = ((1/binned_nu[start_index:])*((binned_E1xE1[start_index:]*binned_B2xB2[start_index:]+binned_E1xB2[start_index:]**2)
                                    +(binned_E2xE2[start_index:]*binned_B1xB1[start_index:]+binned_E2xB1[start_index:]**2)
                                    -2*(binned_E1xE2[start_index:]*binned_B1xB2[start_index:]+binned_E1xB1[start_index:]*binned_E2xB2[start_index:])))

    return bin_estimator, bin_covariance"""

    
def estimator_likelihood(angle, estimator, covariance, ClEE):
    """For a given difference in angle between the depth-1 map (map 1) and the coadd (map 2),
       returns the value of the normalized likelihood for our estimator.
       
       ClEE is the theory EE spectrum from CAMB"""

    numerator = (estimator - ClEE*np.sin(angle))**2
    denominator = 2*covariance
    likelihood = np.exp(-np.sum(numerator/denominator))
    return likelihood
    
def gaussian_fit_moment(angles,data):
    """
       Uses moments to quickly find mean and standard deviation of a Gaussian
       for the likelihood.
    """
    mean = np.sum(angles*data)/np.sum(data)
    std_dev = np.sqrt(abs(np.sum((angles-mean)**2*data)/np.sum(data)))
    return mean, std_dev
    
def sample_likelihood_and_fit(estimator,covariance,theory_ClEE,angle_min_deg=-20.0,angle_max_deg=20.0,num_pts=10000):
    """
       Samples likelihood for a range of angles and returns the best fit for the
       mean and median of the resulting Gaussian.  
    """
    if(angle_min_deg >= angle_max_deg): 
        raise ValueError("The min angle must be smaller than the max!")
    angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
    angles_rad = np.deg2rad(angles_deg)
    
    bin_sampled_likelihood = [estimator_likelihood(angle,estimator,covariance,theory_ClEE) for angle in angles_rad]
    norm_sampled_likelihood = bin_sampled_likelihood/np.max(bin_sampled_likelihood)
    
    fit_values = gaussian_fit_moment(angles_deg,norm_sampled_likelihood)
    
    return fit_values

