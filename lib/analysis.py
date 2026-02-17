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


#===============================================================================
# CONSTANTS
#===============================================================================


#===============================================================================
# STATISTICAL METHODS
#===============================================================================

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
    l = data.size
    SEM = np.zeros(l - min_samples)
    for d in np.arange(l - min_samples):
        SEM[d] = np.sqrt(1 / (l - d)) * data[d:].std(ddof=ddof)
    return SEM, np.argmin(SEM)

#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
