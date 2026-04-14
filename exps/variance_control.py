#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Aim is to plot the link between delay estimation and variance.
    This puts the test onto the artifact that larger variance may lead to shorter estimates.

Description:
    Derived from variance.py

History:
    v0.2a: Adjustment to allow the analysis for the network setup.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from constants import mean_tag, std_tag, mean_std_tag, Label, Color
from config import load_config
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, exc_tag

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient

from lib.conversion import spikecount_to_FR


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

is_network = True
is_network = False

plot_rate_and_delays = True
plot_rate_and_delays = False

hue_order = [mean_tag, std_tag, mean_std_tag]

ylim_delay = (0, 125)

#===============================================================================
# CONSTANTS
#===============================================================================
# hist_binwidth = 2.5 #ms

bootstraps = 200     #50
samples_per_strap = 10 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    control, params = load_config(is_network=is_network)

    
    pre_FR = 2.
    post_FR = 4.
    
    # pre_FR = 5.
    # post_FR = 10.
    
    # pre_FR = 4.
    # # # post_FR = 6.
    # post_FR = 12.
    #
    # pre_FR = 10.
    # post_FR = 5.
    means = np.arange(220, 320+1, 40.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)

    
    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
    
        for hist_binwidth in (1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6):
            print(f"Binwidth: {hist_binwidth}")
    
            t_pre  = np.arange(0., -params.duration_pre, -hist_binwidth, dtype=float)[::-1][:-1] + params.warmup + params.duration_pre - params.dt / 2
            t_post = np.arange(0.,  params.duration_post, hist_binwidth, dtype=float) + params.warmup + params.duration_pre - params.dt / 2
            t_bins = np.concat((t_pre, t_post))
            
            assert not np.any(t_bins >= params.warmup + params.duration_pre + params.duration_post)
            assert np.count_nonzero(t_bins == params.warmup + params.duration_pre - params.dt / 2) == 1
            
            t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
            index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup

            #  Get spikes with buffer
            t_start_buffer = params.warmup + params.duration_pre + 0.25 * params.duration_post
            index_buffered = (t_bins >= t_start_buffer).argmax() # Gets first value that is larger than duration_pre + warmup  
            
            
            # plt.figure(f"binwidht: {hist_binwidth}")
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
                    stds            = np.zeros(bootstraps)
                    delay_estimates = np.zeros(bootstraps)
                    for b in range(bootstraps):
                        np.random.seed(b) # The index is shuffled internally, so independent of the values/run_ids, the order remains.
                        # np.random.shuffle(run_ids)
                        # samples = run_ids[:samples_per_strap] # Bootstrapping
                        samples = np.random.choice(run_ids, samples_per_strap, replace=True)

                        #  Get spikes
                        subgroup = exc_tag if is_network else None
                        spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins, subgroup=subgroup)
                                          
                        # FIRING RATE after Buffer
                        FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                        stds[b] = FRs[index_buffered:].std(ddof=1)
                        # plt.plot(t_bins[index:-1-1], FRs[index:])
                        
                        
                        # # DELAY(No Buffer)
                        # t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
                        # index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup
        
                        SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                        delay_estimates[b] = delay * hist_binwidth
                        
    
                    new_rows = pd.DataFrame({
                        "std": stds, "delay": delay_estimates,
                    })
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], [hist_binwidth], range(len(stds))],
                        names=["tag", "mean", "binwidth", "bootstrap_id"]
                    )
                    all_metrics.append(new_rows)
                # break
    df = pd.concat(all_metrics)
    # return
    
    for binwidth, g in df.groupby(level="binwidth"):
        plt.figure(f"Variance (network={is_network}) and binwidth: {binwidth}")
        sns.violinplot(g, x="mean", y="std", hue="tag", 
                       cut=0, density_norm="width", common_norm=True, 
                       hue_order=hue_order)
        
    h = (
        df.reset_index()
         .groupby(['tag', 'mean', 'binwidth'], as_index=False)[['std', 'delay']]
         .mean()
    )
    
    m = (
        df.reset_index()
         .groupby(['tag', 'mean', 'binwidth'], as_index=False)[['std', 'delay']]
         .median()
    )


    plt.figure(f"Mean: Delay over Variance (network={is_network})")
    sns.lineplot(h, x="std", y="delay", hue="tag", style="mean",
                    hue_order=hue_order)
    
    plt.figure(f"Median: Delay over Variance (network={is_network})")
    sns.lineplot(m, x="std", y="delay", hue="tag", style="mean",
                    hue_order=hue_order, markers=True)
    
    
    
    return
    # Statistical tests
    import scipy.stats as st
    for tag in (mean_tag, std_tag, mean_std_tag):
        for m in means:
            samples = df_metrics.xs((tag, m), level=("tag", "mean"))["std"]
            res = st.shapiro(samples)
            print(f"std: {tag} {m}", res.statistic, res.pvalue)
            
    print("Kruskal-Wallis")
    for m in means:
        df = df_metrics.xs(m, level=("mean"))
        arr2d = df["std"].unstack(level="tag")
        res = st.kruskal(arr2d.to_numpy().T)
        print(f"std: {tag} {m}", res.statistic, res.pvalue)
        
        


if __name__ == '__main__':
    main()
    plt.show()
