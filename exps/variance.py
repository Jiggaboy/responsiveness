#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:

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
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


from plot_constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color
from config import load_config
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, exc_tag, inh_tag

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
hist_binwidth = 2.5 #ms

bootstraps = 50     #50
samples_per_strap = 25 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    control, params = load_config(is_network=is_network)
    t_bins = np.arange(0., params.duration_pre + params.duration_post + hist_binwidth, float(hist_binwidth)) + params.warmup
    
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
    means = np.arange(220, 320+1, 10.)
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
                for b in range(bootstraps):
                    np.random.shuffle(run_ids)
                    samples = run_ids[:samples_per_strap] # Bootstrapping
    
                    # DELAY 
                    subgroup = exc_tag if is_network else None
                    spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins, subgroup=subgroup)
                    t_start = params.warmup + params.duration_pre + 0.25 * params.duration_post
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                    # FIRING RATE
                    FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                    stds[b] = FRs[index:].std(ddof=1)
                    plt.plot(t_bins[index:-1], FRs[index:])
        
                    
                    
                    t_start = params.warmup + params.duration_pre
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup                    

                new_rows = pd.DataFrame({
                    "std": stds,
                })
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(stds))],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_metrics.append(new_rows)
    df_metrics = pd.concat(all_metrics)
    
    plt.figure(f"Variance (network={is_network}) and binwidth: {hist_binwidth}")
    sns.violinplot(df_metrics, x="mean", y="std", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order)
    
    
    
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
