#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Auxiliary functions for plotting.
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
import pandas as pd

from constants import Label, Color


#===============================================================================
# METHODS
#===============================================================================


def panel_FR_with_delay(mean:float, df_rates:pd.DataFrame, t_bins:np.ndarray, ax:object, **plot_kwargs):
    bin_center = (t_bins[:-1] + t_bins[1:]) / 2
    
    
    df = df_rates.xs(mean, level=("mean"))
    for tag, g in df.groupby(level="tag"):
        # Set Colors & Labels
        label = Label[tag]
        color = Color[tag]
        
        # Filter a potential "all" id:
        gb = g[g.index.get_level_values("bootstrap_id") != "all"]

        # Delay Estimation across all runs
        # d = df_all_runs_delays.xs((tag, mean), level=("tag", "mean"))[delay_tag].squeeze()
        # ax1.axvline(d + t_start, c=color, lw=2, ls="--", zorder=15)

        # g: rows = sims, cols = points
        mu = gb.mean(axis=0)
        std = gb.std(axis=0)


        ax.plot(bin_center, mu, label=label, color=color, **plot_kwargs)
        ax.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=-5)
        
        halved = mu.size // 2
        ax.axhline(mu[halved:].mean(), color=color, **plot_kwargs)
    
def hist_delays(mean:float, df_delays:pd.DataFrame, t_bins:np.ndarray, ax:object, t_start:float=0.):
    
    df = df_delays.xs(mean, level=("mean"))
    for tag, g in df.groupby(level="tag"):
        # Set Colors & Labels
        label = Label[tag]
        color = Color[tag]
        
        # Hist delays
        delays = g["delay"]
        ax.hist(delays + t_start, bins=t_bins, color=color, density=True, zorder=-4, rwidth=0.9, alpha=0.5)


#===============================================================================
# AUX - METHODS
#===============================================================================

def plot_axvline_at_change(params:object, control:object, ax:object, **plot_kwargs):
    """
    Plots vertical lines when the input changes. 
    Also adjusts the labels on the xaxis accordingly.
    
    History: 
        - v0.1: Initial implementation.
    
    :param params: Param-object containing the simulation parameter.
    :type params: object
    :param control: Control-object containing control variables.
    :type control: object
    :param ax: The axis which to plot it on.
    :type ax: Axis-object of matplotlib.
    """
    plot_kwargs["color"] = plot_kwargs.get("color", "red")
    plot_kwargs["ls"] = plot_kwargs.get("ls", "--")
    plot_kwargs["zorder"] = plot_kwargs.get("zorder", 10)
    
    t_onset = params.warmup + params.duration_pre
    
    ax.axvline(t_onset, **plot_kwargs)
    if control.brief_stimulus:
        for i in range(params.stim_reps):
            d = params.stim_duration + params.break_duration
            ax.axvline(t_onset + d*i + params.stim_duration, **plot_kwargs)
            ax.axvline(t_onset + d*i, **plot_kwargs)

    if control.brief_stimulus:
        ax.set_xticks(list(ax.get_xticks()) + [t_onset, t_onset+params.stim_duration], 
                      list(ax.get_xticklabels(minor=False)) + [r"$t_\Delta$", r"$t_\Delta'$"])  
    else:
        ax.set_xticks(list(ax.get_xticks()) + [t_onset, ], 
                      list(ax.get_xticklabels(minor=False)) + [r"$t_\Delta$", ]) 


#===============================================================================
if __name__ == '__main__':
    main()
