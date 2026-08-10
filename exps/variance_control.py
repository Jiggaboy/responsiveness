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
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids, exc_tag

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient, get_tbins, get_tstart, bootstrap

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
bootstraps        = 100
samples_per_strap =  15 if is_network else 5

    
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
# means = np.arange(240, 290+1, 110.)
# means = np.append(means, 320.)


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
# def main():
#     control, params = load_config(is_network=is_network)
#     plt.figure()
#     t_start = get_tstart(params)
#     plt.axhline(t_start)
#     plt.axhline(499.95, c="yellow")
#     print(t_start)
#     for h, hist_binwidth in enumerate((1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6)):
#         params.hist_binwidth = hist_binwidth
#         t_bins = get_tbins(params)
#         index = (t_bins >= t_start).argmax() - 1 # Gets first value that is larger than t_start
#
#         plt.plot(t_bins, marker=".")
#         plt.scatter(index, t_bins[index], marker=h)
#         plt.scatter(index, t_bins[index-1], marker=h)
#         assert t_bins[index-1] == 499.95

def main():
    control, params = load_config(is_network=is_network)
    
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
    
        # for hist_binwidth in (1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6):
        for hist_binwidth in (1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6)[::3]:
            params.hist_binwidth = hist_binwidth
            print(f"Binwidth: {params.hist_binwidth}")
            t_bins = get_tbins(params)
            t_start = get_tstart(params)

            #  Get spikes with buffer
            t_start_buffer = t_start + 0.25 * params.duration_post
            index_buffered = (t_bins >= t_start_buffer).argmax() # Index of the first value being larger than the buffered time.
            ##### ALL ANALYSES ######################################
            for tag in (mean_tag, std_tag, mean_std_tag):
                for m, mean in enumerate(means):
                    logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                    rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                    run_ids = get_run_ids(rows, params, tag)
                    
                    
                    subgroup = exc_tag if is_network else None
                    t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=subgroup)
                    
                    stds = population_FR[:, index_buffered:].std(axis=1, ddof=1)
                        
                    # Plots the sample standard deviation across bootstraps.
                    # plt.plot(population_FR.std(axis=0, ddof=1), c=Color[tag])
    
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
    plt.figure(f"Mean: Delay over Variance (network={is_network})")
    sns.lineplot(h, x="std", y="delay", hue="tag", style="mean",
                    hue_order=hue_order)
    
    
    m = (
        df.reset_index()
         .groupby(['tag', 'mean', 'binwidth'], as_index=False)[['std', 'delay']]
         .median()
    )
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
