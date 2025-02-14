#########################################################
# Module containing all plotting functions for code
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot
from tqdm import tqdm
import os
from act_axion_analysis.utils import gaussian

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
    plt.title(f"Beam Profile: {beam_name}")
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
    plt.title(f"Filtering tfunc for kx,ky: {kx},{ky}")
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
             label=f"{num_maps} total maps \n {num_cut_maps} cut maps")
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
                 label=f"{num_maps} total maps, array {array} \n {num_cut_maps} cut maps")
        plt.title(f"Histogram of measured angles, array {array}")
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
