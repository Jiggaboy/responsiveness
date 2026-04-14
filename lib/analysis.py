#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Contains statistical methods and other methods to organize analysis.

History:
    - v0.1: get_transient added.
    - v0.1a: bootstrap added.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from lib.util import functimer

from lib.responsehdf5 import load_and_merge_spikes
from lib.conversion import spikecount_to_FR

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
def bootstrap(hfile:object, run_ids:np.ndarray, params:object, rep:int, samples_per_strap:int, hist_binwidth:float):
    """
    Bootstraps the run_ids and obtains the firing rate and transient delay estimates for each bootstrap.
    
    :param hfile: hdf5-file.
    :type hfile: object
    :param run_ids: Array of ids obtained by the hdf5 file.
    :type run_ids: np.ndarray
    :param params: Configuration object.
    :type params: object
    :param t_bins: Time bins of the simulations data - used for binning.
    :type t_bins: np.ndarray
    :param rep: Repetition of bootstrapping.
    :type rep: int
    :param samples_per_strap: Number of samples per bootstrap.
    :type samples_per_strap: int
    :param hist_binwidth: Binwidth for binning the FR.
    :type hist_binwidth: float
    """
    
    # Time management    
    t_pre  = np.arange(0., -params.duration_pre, -hist_binwidth, dtype=float)[::-1][:-1] + params.warmup + params.duration_pre - params.dt / 2
    t_post = np.arange(0.,  params.duration_post, hist_binwidth, dtype=float) + params.warmup + params.duration_pre - params.dt / 2
    t_bins = np.concat((t_pre, t_post))
    
    assert not np.any(t_bins >= params.warmup + params.duration_pre + params.duration_post)
    assert np.count_nonzero(t_bins == params.warmup + params.duration_pre - params.dt / 2) == 1
    
    t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup
    
    delay_estimates = np.zeros(rep)
    population_FR   = np.zeros((rep, t_bins.size-1))
           
    for b in range(rep):
        np.random.seed(b) # The index is shuffled internally, so independent of the values/run_ids, the order remains.   
        samples = np.random.choice(run_ids, samples_per_strap, replace=True)
        
        # SPIKECOUNTS 
        spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)[:-1] # offsetting the last bin to avoid boundary effects.
        
        # DELAY ESTIMATION
        SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
        delay_estimates[b] = delay * hist_binwidth
                
        # FIRING RATE
        FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
        population_FR[b] = FRs
    return t_bins, delay_estimates, population_FR


#===============================================================================
if __name__ == '__main__':
    main()
