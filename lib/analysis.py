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

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.util import functimer

from lib.responsehdf5 import load_and_merge_spikes
from lib.conversion import spikecount_to_FR
from lib.responsehdf5 import inh_tag

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
def bootstrap(hfile:object, run_ids:np.ndarray, params:object, rep:int, samples_per_strap:int, subgroup:str=None):
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
    t_bins = get_tbins(params)
    t_start = get_tstart(params)
    index = (t_bins >= t_start).argmax() - 1 # Gets first value that is larger than t_start
    
    delay_estimates = np.zeros(rep)
    population_FR   = np.zeros((rep, t_bins.size-1))
           
    for b in range(rep):
        np.random.seed(b) # The index is shuffled internally, so independent of the values/run_ids, the order remains.   
        samples = np.random.choice(run_ids, samples_per_strap, replace=True)
        
        # SPIKECOUNTS 
        spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins, subgroup=subgroup)[:-1] # offsetting the last bin to avoid boundary effects.
        
        # DELAY ESTIMATION
        SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
        delay_estimates[b] = delay * params.hist_binwidth
                
        # FIRING RATE
        N = params.N // 4 if subgroup == inh_tag else params.N
        FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), N, params.hist_binwidth)
        population_FR[b] = FRs
    return t_bins, delay_estimates, population_FR


def get_response_kernel(params:object, df:pd.DataFrame, delay:pd.Series, threshold:int=8, std_margin:float=0.1):
    # Identifies how many True values are consecutive.
    consecutive_True = lambda condition: [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
    
    t_bins = get_tbins(params)
    t_start = params.warmup + params.duration_pre
    index_start = (t_bins >= t_start).argmax() - 1
    
    halved = df[index_start:].size // 2
    std_latter = df[-halved:].std(ddof=1)
    mu_latter = df[-halved:].mean()
    
    index = (t_bins >= t_start + delay).argmax() - 1# Gets the last relevant index as the transient is over after the delay
    
    # Gets all those FRs that are above 1.5 the std of the target post_FR/mu_latter
    osc_over    = df[index_start:index] > (mu_latter + std_margin*std_latter)
    # Gets the first True value
    osc_first_over = osc_over.idxmax() if any(osc_over) else 0
    # Gets all those FRs that are below 1.5 the std of the target post_FR/mu_latter after the overshoot!
    osc_under   = df[osc_first_over:index] < (mu_latter - std_margin*std_latter)
    # Counts as oscillatory if there are {threshold} values above AND {threshold} below the post_FR/mu_latter
    osc = True if np.count_nonzero(osc_over) > threshold and np.count_nonzero(osc_under) > threshold else False

    # Strong overshoot as more than 3 times the std
    above = df[:index] > (std_margin*std_latter + mu_latter)
    # And also more than {threshold} values being above.
    overshoot = True if np.any(above) and np.max(consecutive_True(above)) > threshold else False
    overshoot = False if osc else overshoot

    undershoot = False if osc or overshoot else True

    return np.asarray([overshoot, osc, undershoot])

def get_response_kernels(params:object, df:pd.DataFrame, delays:pd.Series, threshold:int=8, std_margin:float=0.1):
    # Identifies how many True values are consecutive.
    consecutive_True = lambda condition: [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
    
    t_bins = get_tbins(params)
    t_start = params.warmup + params.duration_pre
    index_start = (t_bins >= t_start).argmax() - 1
    
    mu = df.mean(axis=0)
    std = df.std(axis=0)
    
    halved = mu[index_start:].size // 2
    std_latter = std[-halved:].mean()
    mu_latter = mu[-halved:].mean()
    
    # Response kernel
    kernel_type = np.asarray([0, 0, 0], dtype=float)
    
    # Iter over FR & delay combinations
    for (_, fr), delay in zip(df.iterrows(), delays):
        index = (t_bins >= t_start + delay).argmax() - 1# Gets the last relevant index as the transient is over after the delay
    
        # Gets all those FRs that are above 1.5 the std of the target post_FR/mu_latter
        osc_over    = fr[index_start:index] > (mu_latter + std_margin*std_latter)
        # Gets the first True value
        osc_first_over = osc_over.idxmax() if any(osc_over) else 0
        # Gets all those FRs that are below 1.5 the std of the target post_FR/mu_latter after the overshoot!
        osc_under   = fr[osc_first_over:index] < (mu_latter - std_margin*std_latter)
        # Counts as oscillatory if there are {threshold} values above AND {threshold} below the post_FR/mu_latter
        osc = True if np.count_nonzero(osc_over) > threshold and np.count_nonzero(osc_under) > threshold else False
    
        # Strong overshoot as more than 3 times the std
        above = fr[:index] > (std_margin*std_latter + mu_latter)
        # And also more than {threshold} values being above.
        overshoot = True if np.any(above) and np.max(consecutive_True(above)) > threshold else False
        overshoot = False if osc else overshoot
    
        undershoot = False if osc or overshoot else True

        kernel_type += [overshoot, osc, undershoot]
    return kernel_type


def get_tbins(params:object) -> np.ndarray:
    """
    Generates the time-line that starts at the time of change. Goes backwards from there to cover the time prior to the change.
    
    :param params: Config-object that contains durations, and hist_binwidth as attributes.
    :type params: object
    
    Test:
    
    plt.figure()
    for stim_rep in stim_reps:
        for stim_duration in stim_durations:
            params.stim_duration = stim_duration
            params.stim_reps = stim_rep
            tbins = get_tbins(params)
            tstart = get_tstart(params)
            plt.step(np.arange(len(tbins)), tbins)
            plt.axhline(tstart)
    plt.show()
    quit()
    """
    t_reference = get_tstart(params)
    t_remaining = params.duration_post + params.duration_pre + params.warmup - t_reference
    t_pre  = np.arange(0., -t_reference, -params.hist_binwidth, dtype=float)[::-1][:-1]
    t_post = np.arange(0.,  t_remaining - params.hist_binwidth / 2, params.hist_binwidth, dtype=float)
    # print(t_pre.shape)
    # print(t_post.shape)
    # t_pre  = np.arange(0., -params.duration_pre, -params.hist_binwidth, dtype=float)[::-1][:-1] + params.warmup + params.duration_pre - params.dt / 2
    # t_post = np.arange(0.,  params.duration_post, params.hist_binwidth, dtype=float) + params.warmup + params.duration_pre - params.dt / 2
    t_bins = np.concat((t_pre, t_post)) + t_reference - params.dt / 2
    
    assert not np.any(t_bins >= params.warmup + params.duration_pre + params.duration_post)
    assert np.count_nonzero(t_bins == t_reference - params.dt / 2) == 1
    
    return t_bins


def get_tstart(params:object) -> float:
    tstart = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
    return tstart
#===============================================================================
if __name__ == '__main__':
    main()
