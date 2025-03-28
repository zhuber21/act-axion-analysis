# Small functions needed in common by multiple packages

import numpy as np
import scipy

def gaussian(x,mean,sigma):
    """Normalized Gaussian for curve_fit"""
    amp = 1.0
    return amp*np.exp(-(x-mean)**2/(2*sigma**2))

def norm_skewnorm_pdf(x, mu, sigma, alpha):
    """Normalized skew normal distribution for curve_fit
       alpha is the skew parameter. 
       mu and sigma are no longer exactly their Gaussian counterparts.
       Most notably, the peak is not at mu unless alpha=0.
    """
    skewnorm = np.exp(-(x-mu)**2/(2*sigma**2))*(1+scipy.special.erf(alpha*((x-mu)/(sigma*np.sqrt(2)))))
    return skewnorm/np.max(skewnorm)
