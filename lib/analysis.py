#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: 
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from lib.util import functimer

#===============================================================================
# CONSTANTS
#===============================================================================


#===============================================================================
# STATISTICAL METHODS
#===============================================================================
# @functimer
def get_transient(data:np.ndarray, ddof:int=1, min_samples:int=10) -> int:
    """
    Discards the initial {d} samples and calculates the SEM for the {data}.

    Parameters
    ----------
    data : np.ndarray
        1d-array.
    ddof : int, optional
        Degree of freedom for calculation of std. The default is 1.
    min_samples : int, optional
        Minimum number of samples to calculate the SEM from. Default is 10.

    Returns
    -------
    SEM: np.ndarray: SEM for the data with {d} discarded samples
    d: int: the index of the lowest SEM

    """
    """
    # SEM = np.zeros(L - min_samples)
    # for d in np.arange(L - min_samples):
    #     SEM[d] = np.sqrt(1 / (L - d)) * data[d:].std(ddof=ddof)
    # return SEM, np.argmin(SEM)
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data must be a 1D array")
    
    L = data.size
    
    # suffix sizes for d = 0..L-1: n[d] = L - d
    n_all = np.arange(L, 0, -1, dtype=np.int64)
    
    suf_sum = np.cumsum(data[::-1])[::-1]
    suf_sumsq = np.cumsum((data**2)[::-1])[::-1]
    
    # keep only d where n >= min_samples
    max_d = L - min_samples
    n = n_all[: max_d + 1].astype(np.float64)
    s = suf_sum[: max_d + 1]
    q = suf_sumsq[: max_d + 1]

    # variance with requested ddof
    if ddof == 0:
        var = q / n - (s / n) ** 2
    else:
        denom = n - ddof
        if np.any(denom <= 0):
            raise ValueError("Not enough samples for the chosen ddof.")
        var = (q - (s * s) / n) / denom
        
    # numeric guard: tiny negative due to roundoff
    var = np.maximum(var, 0.0)

    # SEM(d) = std(d) / sqrt(n) = sqrt(var / n)
    SEM = np.sqrt(var / n)

    d_star = int(np.argmin(SEM))
    return SEM, d_star


#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
