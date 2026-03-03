#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


from constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color
from config import load_config
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient

from lib.conversion import spikecount_to_FR


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

plot_rate_and_delays = True
plot_rate_and_delays = False

hue_order = [mean_tag, std_tag, mean_std_tag]

ylim_delay = (0, 125)

#===============================================================================
# CONSTANTS
#===============================================================================
hist_binwidth = 5. #ms

bootstraps = 50     #50
samples_per_strap = 25 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    main2()
    return
    theta = 25
    tau = 12e-3
    mu = 30
    sigma = 1
    tau_ref = 2e-3
    V_min = -25
    # lif_stationary_density(theta, tau, mu, sigma, V_reset, E_L, V_min, N)
    V, p = lif_stationary_density(theta, tau, mu, sigma, 0, 0, V_min=V_min)
    plt.plot(V, p)
    

@functimer  # .6s per seed (25 straps x 10 samples)
def main2():
    control, params = load_config()
    t_bins = np.arange(0., params.duration_pre + params.duration_post + hist_binwidth, float(hist_binwidth)) + params.warmup
    
    pre_FR = 2.
    post_FR = 4.
    
    pre_FR = 5.
    post_FR = 10.
    
    # pre_FR = 4.
    # # # post_FR = 6.
    # post_FR = 12.
    #
    # pre_FR = 10.
    # post_FR = 5.
    means = np.arange(240, 320+1, 40.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)
    
    
    
    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
    
        ##### ALL ANALYSES ######################################
        for tag in (mean_tag, std_tag, mean_std_tag):
        # for tag in (mean_tag,):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                if tag in (mean_tag, std_tag):
                    rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]] 
                elif tag == mean_std_tag:
                    mask = np.logical_and(rows[f"pre_{mean_tag}"] != rows[f"post_{mean_tag}"], rows[f"pre_{std_tag}"] != rows[f"post_{std_tag}"])
                    rows_filtered = rows[mask]
                else:
                    raise ValueError("No valid tag given...")

                stim_mask = np.logical_and(rows_filtered["stim_duration"] == params.stim_duration, 
                                           rows_filtered["break_duration"] == params.break_duration, 
                                           rows_filtered["stim_reps"] == params.stim_reps)
                rows_filtered = rows_filtered[stim_mask]
                run_ids = rows_filtered[id_tag]


                spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
                t_start = params.warmup + params.duration_pre + hist_binwidth
                index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                # FIRING RATE
                FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0)[index:], params.N, hist_binwidth)
                

                new_rows = pd.DataFrame(FRs)
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(FRs.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_metrics.append(new_rows)
    df_metrics = pd.concat(all_metrics)
    
    bin_center = (t_bins[:-1] + t_bins[1:]) / 2
    bin_center = bin_center[index:] - bin_center[index:].min()
    dt = bin_center[1]-bin_center[0]
    for m, mean in enumerate(means):
        plt.figure(f"{mean}")
        df = df_metrics.xs(mean, level=("mean"))
        for tag, g in df.groupby(level="tag"):
            g = g.squeeze()
            label = Label[tag]
            color = Color[tag]
            
            print(g.shape)
            plt.plot(bin_center, g-g.mean(), color=color)
            lags, acf = autocorrelation(g)
            tau = fit_time_constant(lags, acf, dt=dt)
            # plt.figure()
            plt.plot(lags*dt, acf, ls="dotted", color=color)
            plt.plot(bin_center, (g.max()-g.mean())*exp_decay(bin_center, tau), ls="--", color=color)
            
            
            # alpha, A = fit_power_law(lags*dt, acf)
            # plt.plot(lags*dt, power_law(lags*dt, A, alpha))
            #
            # A, alpha = fit_power_law_nl(lags*dt, acf)
            # plt.plot(lags*dt, power_law(lags*dt, A, alpha))

    
def autocorrelation_control():
    exp = lambda x, tau=2: np.exp(-x/tau)
    
    x = np.linspace(0, 2, 100)
    
    f = exp(x, tau=.1)
    
    plt.figure()
    plt.plot(x, f)

    lags, corr = autocorrelation(f)
    tau = fit_time_constant(lags, corr, dt=x[1])
    # plt.figure()
    plt.plot(lags*x[1], corr)
    plt.plot(x, exp_decay(x, tau), ls="--")
    print(tau)

@functimer  # .6s per seed (25 straps x 10 samples)
def variance():
    control, params = load_config()
    t_bins = np.arange(0., params.duration_pre + params.duration_post + hist_binwidth, float(hist_binwidth)) + params.warmup
    
    pre_FR = 2.
    post_FR = 4.
    
    pre_FR = 5.
    post_FR = 10.
    
    # pre_FR = 4.
    # # # post_FR = 6.
    # post_FR = 12.
    #
    # pre_FR = 10.
    # post_FR = 5.
    means = np.arange(240, 320+1, 40.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)

    
    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
    
        plt.figure()
        ##### ALL ANALYSES ######################################
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                if tag in (mean_tag, std_tag):
                    rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]] 
                elif tag == mean_std_tag:
                    mask = np.logical_and(rows[f"pre_{mean_tag}"] != rows[f"post_{mean_tag}"], rows[f"pre_{std_tag}"] != rows[f"post_{std_tag}"])
                    rows_filtered = rows[mask]
                else:
                    raise ValueError("No valid tag given...")

                stim_mask = np.logical_and(rows_filtered["stim_duration"] == params.stim_duration, 
                                           rows_filtered["break_duration"] == params.break_duration, 
                                           rows_filtered["stim_reps"] == params.stim_reps)
                rows_filtered = rows_filtered[stim_mask]
                run_ids = rows_filtered[id_tag]

    
                # Detailed feature analysis
                stds   = np.zeros(bootstraps)
                stds2   = np.zeros(bootstraps)
                for b in range(bootstraps):
                    np.random.shuffle(run_ids)
                    samples = run_ids[:samples_per_strap] # Bootstrapping
    
                    # DELAY 
                    spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)
                    t_start = params.warmup + params.duration_pre + 0.2 * params.duration_post
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                    # FIRING RATE
                    FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                    stds[b] = FRs[index:].std(ddof=1)
                    plt.plot(t_bins[index:-1], FRs[index:])
                    
                    t_start = params.warmup + params.duration_pre + 0.5 * params.duration_post
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                    # FIRING RATE
                    FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                    stds2[b] = FRs[index:].std(ddof=1)
                    
                    
                    t_start = params.warmup + params.duration_pre
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                new_rows = pd.DataFrame({
                    "std": stds, "std2": stds2, #"fr": FRs[index:].mean(axis=0)
                })
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(stds))],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_metrics.append(new_rows)
    df_metrics = pd.concat(all_metrics)
    
    plt.figure()
    sns.violinplot(df_metrics, x="mean", y="std", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order)
    
    plt.figure()
    sns.violinplot(df_metrics, x="mean", y="std2", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order)
    
    
    #===============================================================================
    # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
    #===============================================================================
    plt.figure()
    sns.lineplot(df_metrics, y="fr", hue="tag", hue_order=hue_order) 
    # lags, corr = autocorrelation(x)
    


#===============================================================================
# METHODS
#===============================================================================

def autocorrelation(x, max_lag=None):
    """
    Compute normalized autocorrelation function.

    Returns
    -------
    lags : array of lag indices
    acf  : normalized autocorrelation
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    corr = np.correlate(x, x, mode="full")
    corr = corr[n-1:n+max_lag]

    # unbiased normalization
    norm = np.arange(n, n-max_lag-1, -1)
    corr /= norm

    corr /= corr[0]  # normalize to 1 at lag 0

    lags = np.arange(0, max_lag+1)
    return lags, corr

from scipy.optimize import curve_fit

def exp_decay(t, tau):
    return np.exp(-t / tau)

def fit_time_constant(lags, acf, dt, fit_until=None):
    """
    Fit exponential to ACF.

    Parameters
    ----------
    lags : lag indices
    acf  : autocorrelation
    dt   : time step
    fit_until : maximum lag (in time units) to fit

    Returns
    -------
    tau_est
    """
    t = lags * dt

    # exclude lag 0
    mask = t > 0

    if fit_until is not None:
        mask &= (t <= fit_until)

    t_fit = t[mask]
    acf_fit = acf[mask]

    # only fit positive region
    pos = acf_fit > 0
    t_fit = t_fit[pos]
    acf_fit = acf_fit[pos]

    popt, _ = curve_fit(exp_decay, t_fit, acf_fit)
    return popt[0]


def fit_power_law(t, acf, t_min=None, t_max=None):
    """
    Fit C(t) = A * t^{-alpha}
    using log-log linear regression.
    """

    t = np.asarray(t)
    acf = np.asarray(acf)

    # Only positive values
    mask = (acf > 0) & (t > 0)

    if t_min is not None:
        mask &= (t >= t_min)
    if t_max is not None:
        mask &= (t <= t_max)

    t_fit = t[mask]
    acf_fit = acf[mask]

    log_t = np.log(t_fit)
    log_c = np.log(acf_fit)

    slope, intercept = np.polyfit(log_t, log_c, 1)

    alpha = -slope
    A = np.exp(intercept)

    return alpha, A


def power_law(t, A, alpha):
    return A * t**(-alpha)

def fit_power_law_nl(t, acf, t_min=None, t_max=None):

    mask = (acf > 0) & (t > 0)

    if t_min is not None:
        mask &= (t >= t_min)
    if t_max is not None:
        mask &= (t <= t_max)

    t_fit = t[mask]
    acf_fit = acf[mask]

    popt, _ = curve_fit(power_law, t_fit, acf_fit)
    return popt  # A, alpha


from scipy.integrate import cumulative_trapezoid


def lif_stationary_density(theta, tau, mu, sigma,
                           V_reset=0.0,
                           E_L=0.0,
                           V_min=-1.0,
                           N=2000):

    V = np.linspace(V_min, theta, N)
    dV = V[1] - V[0]

    # Drift and diffusion
    a = (-(V - E_L) + mu) / tau
    D = sigma**2 / (2 * tau**2)

    # Compute Phi(V)
    integrand = a / D
    Phi = cumulative_trapezoid(integrand, V, initial=0.0)

    # Compute integral term for each V
    exp_minus_Phi = np.exp(-Phi)
    integral = np.zeros_like(V)

    # backward cumulative integral
    integral = cumulative_trapezoid(exp_minus_Phi[::-1], V[::-1], initial=0.0)[::-1]

    # density up to normalization constant r
    p_unnorm = np.exp(Phi) * integral / D

    # normalize to 1
    Z = np.trapezoid(p_unnorm, V)
    p = p_unnorm / Z

    return V, p




def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
