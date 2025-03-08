"""
    get_camb_spectra_from_ini.py

    Script showing exactly how I generated the CAMB
    power spectra results from CAMB ini files.

    camb_planck2018acc_lensedtotcls.dat was generated
    using CAMB v1.5.9 in a fresh conda-forge installation
    with Python 3.12.

    The ini files were copied from the CAMB GitHub in Feb 2025.
    They are preserved in act_axion_analysis to ensure reproducibility.
"""
import camb

planck_params = camb.read_ini('planck_2018_acc.ini')
planck_results = camb.get_results(planck_params)
planck_spectra = planck_results.get_cmb_power_spectra(planck_params,
                                                      CMB_unit='muK')
planck_results.save_cmb_power_spectra('camb_planck2018acc_lensedtotcls.dat',
                                      lmax=5400, CMB_unit='muK')
