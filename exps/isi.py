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
from cflogger import logger

import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from config import load_config
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order, KTH_sky, KTH_blue, KTH_navy, KTH_grey

from lib.analysis import bootstrap, get_tbins
from lib.conversion import spikecount_to_FR
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids
from lib.util import save_figure

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
pre_FR = 5
post_FR = 10

# bootstraps = 20
# samples_per_strap = 50


means = np.arange(220, 320+1, 40)

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config(no_stim=True)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    
    all_spikes = []
    df = pd.DataFrame(columns=["spikes", "tag", "mean"])
    
    t_start = params.warmup + params.duration_pre
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)
                
                allspikes_by_sender = [] # All spikes
                for run_id in run_ids:
                    print(run_id)
                    spikes_by_sender = hfile.get_node(hfile.data, f"run{run_id}").spikes_by_sender.read()
                    for spikes in spikes_by_sender:
                        spikes_tmp = spikes[spikes >= t_start]
                        allspikes_by_sender.append(spikes_tmp - t_start)
                        
                new_rows = pd.DataFrame({"spikes": allspikes_by_sender})
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(allspikes_by_sender))],
                    names=["tag", "mean", "id"]
                )
                all_spikes.append(new_rows)
                
                            
                            
    #             new_rows = pd.DataFrame({"firstspike": all_first_spike})
    #             new_rows.index = pd.MultiIndex.from_product(
    #                 [[tag], [mean], range(len(all_first_spike))],
    #                 names=["tag", "mean", "neuron_id"]
    #             )
    #             time_to_first_spike.append(new_rows)
    df = pd.concat(all_spikes)
    
    for (mean, tag), gb in df.groupby(level=["mean", "tag"]):
        plt.figure(f"{mean}-{tag}")
        for n in [1, 2, 3]:
            nthspike = map(lambda x: get_isi(x, nth=n), gb["spikes"])
            H, edges = np.histogram(np.fromiter(nthspike, dtype=float), bins=np.arange(0, 200, 20))
            edge_center = (edges[:-1] + edges[1:]) / 2
            plt.plot(edge_center, H)
    return
    
    #===============================================================================
    # PLOT -  TIME TO FIRST SPIKE
    #===============================================================================
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]", title=f"Time to First Spike\nDistribution")
    ax0.set_ylim(-5, 150)

    plot_means = [240, 280, 320]
    df_tmp = df[df.index.isin(plot_means, level="mean")] - (params.warmup + params.duration_pre)
    sns.violinplot(df_tmp, x="mean", y="firstspike", hue="tag", 
                       cut=0, density_norm="width", common_norm=True, 
                       inner=None,
                       hue_order=hue_order, ax=ax0,
                       native_scale=True,
            palette=Color,
            legend=False,
            )
    
    
    
    ax1.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]", ylim=(25, 87), xticks=(means[::4]), title="Time to First Spike\n")
        
    sns.lineplot(
        data=df - (params.warmup + params.duration_pre),
        x="mean",
        y="firstspike",
        hue="tag",
        marker="o",
        errorbar=("se", 1.96), # 95% interval
        estimator="mean",
        hue_order=hue_order,
        ax=ax1,
        palette=Color,
    )
    sns.lineplot(
        data=df - (params.warmup + params.duration_pre),
        x="mean",
        y="firstspike",
        hue="tag",
        marker="^",
        linestyle="--",
        # errorbar=("pi", 50),
        errorbar=None,
        estimator="median",
        hue_order=hue_order,
        ax=ax1,
        palette=Color,
    )
    labels = []
    for stat in ("mean", "median"):
        for tag in hue_order:
            labels.append(rf"{stat.capitalize()}")
            # labels.append(rf"{stat.capitalize()} ({Label[tag]})")
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, ncols=2, handlelength=3)



#===============================================================================
# METHODS
#===============================================================================
def get_isi(arr, nth:int = 1):
    if len(arr) < nth:
        return np.nan
    if nth == 1:
        return arr[0]
    return arr[nth-1] - arr[nth-2]


#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()