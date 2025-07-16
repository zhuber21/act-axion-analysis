import numpy as np

def calc_marg_amp(input_fs, As, phases, angles, errorbars, durations, mean_times):
    """This function calculates the best fit amplitude after marginalizing over phase
       for a given timestream. The timestream could be a split for a null test or the
       full timestream for calculating upper limits."""
    
    # Precalculating factors involving amplitude and phase
    cos_phases = np.cos(phases)
    sin_phases = np.sin(phases)
    cos2_phases = cos_phases**2
    sin2_phases = sin_phases**2
    A2s = As**2
    # Also precalculating common factors in trig and sinc 
    # functions to try to speed up loop even more
    sinc_factor_f = np.pi*input_fs 
    trig_factor_f = 2*np.pi*input_fs

    phase_step = np.diff(phases)[0]

    # Output array
    best_fit_amps = np.empty(input_fs.size)

    # Defining factors for chi2 calculation
    factor2 = np.empty(input_fs.size)
    factor3 = np.empty(input_fs.size)
    factor4 = np.empty(input_fs.size)
    factor5 = np.empty(input_fs.size)
    factor6 = np.empty(input_fs.size)
    # This one is frequency independent
    factor1 = np.sum(angles**2/errorbars**2)
    # For the rest, there is a whole different number for each frequency
    for j in range(input_fs.size):
        # The factors of np.pi in np.sinc() account for the numpy normalization
        # in order to make np.sinc(x) = sin(x)/x, as expected. 
        factor2[j] = np.sum(-2.0*angles*np.sinc(sinc_factor_f[j]*durations/np.pi)
                                       *np.sin(trig_factor_f[j]*mean_times)/errorbars**2)
        factor3[j] = np.sum(-2.0*angles*np.sinc(sinc_factor_f[j]*durations/np.pi)
                                       *np.cos(trig_factor_f[j]*mean_times)/errorbars**2)
        factor4[j] = np.sum(np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.sin(trig_factor_f[j]*mean_times)**2/errorbars**2)
        factor5[j] = np.sum(2.0*np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.sin(trig_factor_f[j]*mean_times)
                                       *np.cos(trig_factor_f[j]*mean_times)/errorbars**2)
        factor6[j] = np.sum(np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.cos(trig_factor_f[j]*mean_times)**2/errorbars**2)

    for i in range(input_fs.size):
        sampled_chi2s = np.empty((phases.size,As.size)) # Allowing dtype to be set automatically for now - may need to shrink to float32 
        # Switching to iterating over phase since I want more amp resolution than phase
        for k in range(phases.size):
            sampled_chi2s[k] = (factor1 + As*cos_phases[k]*factor2[i]
                                        + As*sin_phases[k]*factor3[i] 
                                        + A2s*cos2_phases[k]*factor4[i]
                                        + A2s*sin_phases[k]*cos_phases[k]*factor5[i] 
                                        + A2s*sin2_phases[k]*factor6[i])
        sampled_chi2s -= np.min(sampled_chi2s) # Normalization to make numerically tractable when exponentiating
        sampled_chi2s *= -0.5                  # Likelihood is e^{-0.5*chi2}
        likelihood = np.exp(sampled_chi2s)     # At least on Alpher, this is the slow step b/c of the underlying C library
        # Marginalizing over full range of phases (one deg less than 2*pi because we don't include 2*pi in the range)
        trap_marg_like = np.trapz(likelihood,dx=phase_step,axis=0)/(phases[-1]-phases[0])
        norm_trap_marg_like = trap_marg_like / np.trapz(trap_marg_like,x=As)

        best_fit_amps[i] = As[np.argmax(norm_trap_marg_like)]

    return best_fit_amps

def calc_upper_limits(input_fs, As, phases, angles, errorbars, durations, mean_times):
    """This function calculates the 95% upper limit after marginalizing over phase
       for a given timestream. The timestream could be a split for a null test or the
       full timestream for calculating upper limits."""
    
    # Precalculating factors involving amplitude and phase
    cos_phases = np.cos(phases)
    sin_phases = np.sin(phases)
    cos2_phases = cos_phases**2
    sin2_phases = sin_phases**2
    A2s = As**2
    # Also precalculating common factors in trig and sinc 
    # functions to try to speed up loop even more
    sinc_factor_f = np.pi*input_fs 
    trig_factor_f = 2*np.pi*input_fs

    phase_step = np.diff(phases)[0]
    A_step = np.diff(As)[0]

    # Output array
    amp_upper_limits = np.empty(input_fs.size)

    # Defining factors for chi2 calculation
    factor2 = np.empty(input_fs.size)
    factor3 = np.empty(input_fs.size)
    factor4 = np.empty(input_fs.size)
    factor5 = np.empty(input_fs.size)
    factor6 = np.empty(input_fs.size)
    # This one is frequency independent
    factor1 = np.sum(angles**2/errorbars**2)
    # For the rest, there is a whole different number for each frequency
    for j in range(input_fs.size):
        # The factors of np.pi in np.sinc() account for the numpy normalization
        # in order to make np.sinc(x) = sin(x)/x, as expected. 
        factor2[j] = np.sum(-2.0*angles*np.sinc(sinc_factor_f[j]*durations/np.pi)
                                       *np.sin(trig_factor_f[j]*mean_times)/errorbars**2)
        factor3[j] = np.sum(-2.0*angles*np.sinc(sinc_factor_f[j]*durations/np.pi)
                                       *np.cos(trig_factor_f[j]*mean_times)/errorbars**2)
        factor4[j] = np.sum(np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.sin(trig_factor_f[j]*mean_times)**2/errorbars**2)
        factor5[j] = np.sum(2.0*np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.sin(trig_factor_f[j]*mean_times)
                                       *np.cos(trig_factor_f[j]*mean_times)/errorbars**2)
        factor6[j] = np.sum(np.sinc(sinc_factor_f[j]*durations/np.pi)**2
                                       *np.cos(trig_factor_f[j]*mean_times)**2/errorbars**2)

    for i in range(input_fs.size):
        sampled_chi2s = np.empty((phases.size,As.size)) # Allowing dtype to be set automatically for now - may need to shrink to float32 
        # Switching to iterating over phase since I want more amp resolution than phase
        for k in range(phases.size):
            sampled_chi2s[k] = (factor1 + As*cos_phases[k]*factor2[i]
                                        + As*sin_phases[k]*factor3[i] 
                                        + A2s*cos2_phases[k]*factor4[i]
                                        + A2s*sin_phases[k]*cos_phases[k]*factor5[i] 
                                        + A2s*sin2_phases[k]*factor6[i])
        sampled_chi2s -= np.min(sampled_chi2s)
        sampled_chi2s *= -0.5
        likelihood = np.exp(sampled_chi2s)
        # Marginalizing over full range of phases (one deg less than 2*pi because we don't include 2*pi in the range)
        trap_marg_like = np.trapz(likelihood,dx=phase_step,axis=0)/(phases[-1]-phases[0])
        norm_trap_marg_like = trap_marg_like / np.trapz(trap_marg_like,x=As)

        # Calculating 95% upper limit on amplitude 
        partial_sums = np.cumsum(norm_trap_marg_like*A_step)
        try:
            limit_idx = next(a for a, x in enumerate(partial_sums) if x > 0.95)
        except StopIteration:
            limit_idx = None
        amp_upper_limits[i] = As[limit_idx]

    return amp_upper_limits
