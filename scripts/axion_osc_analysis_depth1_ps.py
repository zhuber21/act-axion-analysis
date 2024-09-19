import numpy as np
from pixell import enmap, enplot
import nawrapper as nw
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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
    
def load_ref_map_and_beam(fname_ref,fname_ref_beam,bins):
    """
        Loads in the full reference map (T,Q,U) and the beam for this map. 
        Needs to be done once.
        
        The beam is also binned to match the binning of all spectra.
    """
    maps = enmap.read_map(fname_ref)
    ivar_name = fname_ref[:-8] + "ivar.fits"
    ivar = 0.5*enmap.read_map(ivar_name) # 0.5 for polarization noise
    
    b_ell = load_and_bin_beam(fname_ref_beam,bins)

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

def load_and_filter_depth1(fname, ref_maps, kx_cut, ky_cut, unpixwin,filter_radius=1.0,plot_maps=False,output_dir=None):
    depth1_maps, depth1_ivar, shape, wcs = load_depth1_with_T(fname)
    ref_cut = trim_ref_with_T(ref_maps,shape,wcs)
       
    # Apodize and filter depth-1
    depth1_mask, depth1_indices = make_tapered_mask(depth1_ivar,filter_radius=filter_radius)
    filtered_depth1_TEB = apply_kspace_filter(depth1_maps*depth1_mask, kx_cut, ky_cut, unpixwin=True)
        
    # Apodize and filter coadd
    ref_cut_TEB = apply_kspace_filter(ref_cut*depth1_mask, kx_cut, ky_cut, unpixwin=True)

    if plot_maps:
        map_fname = os.path.split(fname)[1][:-9] # removing "_map.fits"
        plot_T_maps(output_dir, map_fname, depth1_maps[0], **keys_ewrite_T)
        plot_QU_maps(output_dir, map_fname, [depth1_maps[1], depth1_maps[2]], **keys_ewrite_QU)
        plot_T_ref_maps(output_dir, map_fname, depth1_mask*ref_cut[0], **keys_ewrite_ref_T)
        plot_QU_ref_maps(output_dir, map_fname, [depth1_mask*ref_cut[1], depth1_mask*ref_cut[2]],**keys_ewrite_ref_QU)
        plot_mask(output_dir, map_fname, depth1_mask, **keys_ewrite_mask)
        plot_EB_filtered_maps(output_dir, map_fname, filtered_depth1_TEB, depth1_mask, **keys_ewrite_EB)
    
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
    
def gaussian_fit_moment(angles,data):
    """
       Uses moments to quickly find mean and standard deviation of a Gaussian
       for the likelihood.
    """
    mean = np.sum(angles*data)/np.sum(data)
    std_dev = np.sqrt(abs(np.sum((angles-mean)**2*data)/np.sum(data)))
    return mean, std_dev
    
def sample_likelihood_and_fit(estimator,covariance,theory_ClEE,angle_min_deg=-20.0,angle_max_deg=20.0,
                              num_pts=10000,plot_like=False,output_dir=None,map_fname=None):
    """
       Samples likelihood for a range of angles and returns the best fit for the
       mean and std dev of the resulting Gaussian in degrees.  
    """
    if(angle_min_deg >= angle_max_deg): 
        raise ValueError("The min angle must be smaller than the max!")
    angles_deg = np.linspace(angle_min_deg,angle_max_deg,num=num_pts)
    angles_rad = np.deg2rad(angles_deg)
    
    bin_sampled_likelihood = [estimator_likelihood(angle,estimator,covariance,theory_ClEE) for angle in angles_rad]
    norm_sampled_likelihood = bin_sampled_likelihood/np.max(bin_sampled_likelihood)
    
    fit_values = gaussian_fit_moment(angles_rad,norm_sampled_likelihood)
    fit_values_deg = [np.rad2deg(fit_values[0]), np.rad2deg(fit_values[1])]

    if plot_like:
        map_name = os.path.split(map_fname)[1][:-9] # removing "_map.fits"
        plot_likelihood(output_dir, map_name, angles_deg, norm_sampled_likelihood,fit_values_deg)

    return fit_values_deg

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
keys_ewrite_ref_QU = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 1000, "mask": 0}
keys_ewrite_T = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 2500, "mask": 0}
keys_ewrite_ref_T = {"downgrade": 4, "ticks": 5, "colorbar": True, "font_size": 40, "range": 2000, "mask": 0}
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

def plot_EB_filtered_maps(output_dir, map_name, depth1_TEB, map_mask, **kwargs):
    """Converts the filtered Fourier space E and B maps back to real space and plots them.
       Assumes you pass in TEB, but only saves EB. Also multiplies by mask again to ignore
       any leakage from Fourier transforming."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_E = map_name + "_filtered_E"
    save_fname_B = map_name + "_filtered_B"
    maps_realspace = enmap.harm2map(depth1_TEB, normalize = "phys")
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
        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E1xE1'],ell_b),alpha=1.0)
        plt.plot(ell_b, cl_to_dl(CAMB_ClEE_binned,ell_b), 'r--', label="CAMB EE")
        plt.ylabel("$D_{\ell}^{E1xE1}$")
        plt.xlabel("$\ell$")
        plt.title("E1xE1 " + maps[i][:-9])
        plt.grid()
        plt.legend()
        output_fname = save_dir + maps[i][:-9] + "_e1xe1_spectrum_withCAMBee.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()
        
        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E2xE2'],ell_b),alpha=1.0)
        plt.plot(ell_b, cl_to_dl(CAMB_ClEE_binned,ell_b), 'r--', label="CAMB EE")
        plt.ylabel("$D_{\ell}^{E2xE2}$")
        plt.xlabel("$\ell$") 
        plt.title("E2xE2 " + maps[i][:-9])
        plt.grid()
        plt.legend()
        output_fname = save_dir + maps[i][:-9] + "_e2xe2_spectrum_withCAMBee.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()

        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B1xB1'],ell_b),alpha=1.0)
        plt.plot(ell_b, cl_to_dl(CAMB_ClBB_binned,ell_b), 'r--', label="CAMB BB")
        plt.ylabel("$D_{\ell}^{B1xB1}$")
        plt.xlabel("$\ell$")
        plt.title("B1xB1 " + maps[i][:-9])
        plt.grid()
        plt.legend()
        output_fname = save_dir + maps[i][:-9] + "_b1xb1_spectrum_withCAMBbb.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()
        
        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B2xB2'],ell_b),alpha=1.0)
        plt.plot(ell_b, cl_to_dl(CAMB_ClBB_binned,ell_b), 'r--', label="CAMB BB")
        plt.ylabel("$D_{\ell}^{B2xB2}$")
        plt.xlabel("$\ell$") 
        plt.title("B2xB2 " + maps[i][:-9])
        plt.grid()
        plt.legend()
        output_fname = save_dir + maps[i][:-9] + "_b2xb2_spectrum_withCAMBbb.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()

        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E1xB2'],ell_b),alpha=1.0)
        plt.ylabel("$D_{\ell}^{E1xB2}$")
        plt.xlabel("$\ell$")
        plt.title("E1xB2 " + maps[i][:-9])
        plt.grid()
        output_fname = save_dir + maps[i][:-9] + "_e1xb2_spectrum.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()
        
        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['E2xB1'],ell_b),alpha=1.0)
        plt.ylabel("$D_{\ell}^{E2xB1}$")
        plt.xlabel("$\ell$") 
        plt.title("E2xB1 " + maps[i][:-9])
        plt.grid()
        output_fname = save_dir + maps[i][:-9] + "_e2xb1_spectrum.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()

        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['estimator'],ell_b),alpha=1.0)
        # Could also add plotting the expected signal given the fit for this map
        plt.ylabel("$D_{\ell}^{E1xB2} - D_{\ell}^{E2xB1}$")
        plt.xlabel("$\ell$")
        plt.title("Estimator " + maps[i][:-9])
        plt.grid()
        output_fname = save_dir + maps[i][:-9] + "_estimator.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()

        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['covariance'],ell_b),alpha=1.0)
        plt.ylabel("Covariance")
        plt.xlabel("$\ell$")
        plt.title("Covariance " + maps[i][:-9])
        plt.grid()
        output_fname = save_dir + maps[i][:-9] + "_covariance.png"
        plt.savefig(output_fname, dpi=300)
        plt.close()

        fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
        plt.plot(ell_b, cl_to_dl(spectra[maps[i]]['binned_nu'],ell_b),alpha=1.0)
        plt.ylabel("Effective modes per bin")
        plt.xlabel("$\ell$")
        plt.title("$\\nu_b$ " + maps[i][:-9])
        plt.grid()
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
    plt.plot(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_EE'],ell_b), 'r--', label="CAMB EE")
    for i in range(len(maps)):
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E1xE1'],ell_b),alpha=0.3)
    plt.ylabel("$D_{\ell}^{EE}$")
    plt.xlabel("$\ell$")
    plt.grid()
    plt.legend()
    plt.title("EE Autospectra for All Depth-1 Maps")
    plt.savefig(output_path_ee, dpi=300)
    plt.close()

    # Plotting all the reference map EE autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    output_path_ee_ref = save_dir + "ee_autospectra_all_ref_maps.png"
    plt.plot(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_EE'],ell_b), 'r--', label="CAMB EE")
    for i in range(len(maps)):
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['E2xE2'],ell_b),alpha=0.3)
    plt.ylabel("$D_{\ell}^{EE}$")
    plt.xlabel("$\ell$")
    plt.grid()
    plt.legend()
    plt.title("EE Ref Map Autospectra in All Depth-1 Footprints")
    plt.savefig(output_path_ee_ref, dpi=300)
    plt.close()

    # Plotting all the BB autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained') 
    output_path_bb = save_dir + "bb_autospectra_all_test_maps.png"
    plt.plot(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_BB'],ell_b), 'r--', label="CAMB BB")
    for i in range(len(maps)):
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B1xB1'],ell_b),alpha=0.3)
    plt.ylabel("$D_{\ell}^{BB}$")
    plt.xlabel("$\ell$")
    plt.grid()
    plt.legend()
    plt.title("BB Autospectra for All Depth-1 Maps")
    plt.savefig(output_path_bb, dpi=300)
    plt.close()

    # Plotting all the reference map BB autospectra together
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    output_path_bb_ref = save_dir + "bb_autospectra_all_ref_maps.png"
    plt.plot(ell_b, cl_to_dl(spectra[maps[0]]['CAMB_BB'],ell_b), 'r--', label="CAMB BB")
    for i in range(len(maps)):
        plt.semilogy(ell_b, cl_to_dl(spectra[maps[i]]['B2xB2'],ell_b),alpha=0.3)
    plt.ylabel("$D_{\ell}^{BB}$")
    plt.xlabel("$\ell$")
    plt.grid()
    plt.legend()
    plt.title("BB Ref Map Autospectra in All Depth-1 Footprints")
    plt.savefig(output_path_bb_ref, dpi=300)
    plt.close()

def plot_likelihood(output_dir, map_name, angles, likelihood, gauss_fits):
    """Plotting likelihood for angles in degrees."""
    save_dir = output_dir + "/plots/"
    if not os.path.exists(save_dir): # Make new folder for this run - should be unique
        os.makedirs(save_dir)
    save_fname_likelihood = map_name + "_likelihood.png"
    mean = gauss_fits[0]
    stddev = gauss_fits[1]
    fig = plt.figure(figsize=(6.4,4.8), layout='constrained')
    plt.plot(angles, likelihood, 'b.', label='Mean={:1.3f}\n$\sigma$={:1.3f}'.format(mean,stddev))
    plt.axvline(mean,alpha=0.3,color='black')
    # Could also add 1 sigma shading for stddev at some point
    # Could also move label to plt.figtext, but legend will auto adjust for me
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
    plt.xlabel("$\ell$")
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
    plt.xlabel("$\ell$")
    plt.grid()
    plt.title("Filtering tfunc for kx,ky: " + str(kx) + ',' + str(ky))
    plt.savefig(save_dir+save_fname_tfunc, dpi=300)
    plt.close()