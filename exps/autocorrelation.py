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
hist_binwidth = 2. #ms

bootstraps = 50     #50
samples_per_strap = 25 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  # .6s per seed (25 straps x 10 samples)
def main():
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
    means = np.arange(220, 320+1, 140.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)

    
    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_rates = []
    
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
                t_start = params.warmup + params.duration_pre
                index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    
                population_FR   = np.zeros((bootstraps, t_bins.size-index-1))
                for b in range(bootstraps):
                    np.random.shuffle(run_ids)
                    samples = run_ids[:samples_per_strap] # Bootstrapping
    
                    # DELAY 
                    spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)
                    
                    # FIRING RATE
                    FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0)[index:], params.N, hist_binwidth)
                    population_FR[b] = FRs              

                # Extend the array of firing rates
                new_rows = pd.DataFrame(population_FR)
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(population_FR.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_rates.append(new_rows)
    df_rates = pd.concat(all_rates)
    
    plt.figure()
    plt.xlabel("time")
    plt.ylabel("FR")
    
    for m, mean in enumerate(means): 
        # for tag in (mean_tag, std_tag, mean_std_tag):
        #     label = Label[tag]
        #     color = Color[tag]
            
            # df = df_rates.xs((mean, tag), level=("mean", "tag"))
            # plt.plot(t_bins[index+1:]-t_start, df.T)
            
        df = df_rates.xs(mean, level=("mean"))
        df.reset_index()

             
        acfs = df.apply(lambda row: autocorrelation(row), axis=1)
        acfs = acfs.apply(pd.Series) 
        acfs.rename(columns={0: "index", 1: "ac"}, inplace=True)
        
        # idx = acfs["index"]
        # values = acfs["ac"]
        # plt.plot()
        
        acfs_2d = pd.DataFrame(
            np.vstack(acfs['ac'].to_numpy()),
            index=acfs.index,
            columns=acfs.iloc[0]['index']   # use the first row's index list as column names
        )
        
        long_acfs = acfs.reset_index().explode(['index', 'ac'])

        sns.lineplot(
            data=long_acfs,
            x="index",
            y="ac",
            hue="tag",
            marker="o",
            # errorbar="sd",
            # estimator="mean",
            hue_order=hue_order,
            # ax=ax_meandelay,
        )
        
        
        params_df = acfs.apply(fit_row_ordered, axis=1)
        t = np.asarray(acfs.iloc[0]['index'], dtype=float)
        
        fit_df = pd.DataFrame(
            {
                idx: double_exp_ordered(
                    t,
                    row['w'],
                    row['tau_fast'],
                    row['tau_slow']-row['tau_fast'],
                )
                for idx, row in params_df.iterrows()
            },
        ).T
        fit_df.index.names = ("tag", "bootstrap_id")
        
        # fit_df = params_df.apply(lambda x: double_exp_ordered, axis=1)
        # fit_df = pd.DataFrame({"fit": }, index=params_df.index)
        long_fit = (
            fit_df
            .stack()
            .rename("fit")
            .reset_index()
            .rename(columns={"level_2": "lag"})
        )
        
        sns.lineplot(
            data=long_fit,
            x="lag",
            y="fit",
            hue="tag",
            marker="^",
            # units="bootstrap_id",
            # errorbar="sd",
            # estimator="mean",
            hue_order=hue_order,
            # ax=ax_meandelay,
            linestyle="--",
        )
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


def double_exp_ordered(t, w, tau_fast, delta_tau, **kwargs):
    tau_slow = tau_fast + delta_tau
    return (
        w * np.exp(-t / tau_fast) +
        (1 - w) * np.exp(-t / tau_slow)
    )

def fit_row_ordered(row):
    t = np.asarray(row["index"], dtype=float)
    y = np.asarray(row["ac"], dtype=float)

    try:
        popt, pcov = curve_fit(
            double_exp_ordered,
            t,
            y,
            p0=[0.5, 1.0, 10.0],                 # w, tau_fast, delta_tau
            bounds=([0.0, 1e-8, 1e-8], [1.0, np.inf, np.inf]),
            maxfev=20000
        )

        w, tau_fast, delta_tau = popt
        tau_slow = tau_fast + delta_tau

        return pd.Series({
            "w": w,
            "tau_fast": tau_fast,
            "tau_slow": tau_slow,
            "success": True,
            "fail_reason": ""
        })

    except Exception as e:
        return pd.Series({
            "w": np.nan,
            "tau_fast": np.nan,
            "tau_slow": np.nan,
            "success": False,
            "fail_reason": str(e)
        })
if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
