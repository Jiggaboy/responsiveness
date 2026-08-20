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
from cplot.constants import KTH_grey
    
#===============================================================================
# METHODS
#===============================================================================


def plot_FRs(mean:float, df_rates:pd.DataFrame, t_bins:np.ndarray, ax:object, add_traces:bool=False, hue_order:tuple=None, **plot_kwargs):
    bin_center = (t_bins[:-1] + t_bins[1:]) / 2
    
    
    df = df_rates.xs(mean, level=("mean"))
    
    if hue_order is not None:
        groups = ((tag, df.xs(tag, level="tag")) for tag in hue_order)
    else:
        groups = df.groupby(level="tag")

    for tag, g in groups:
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

        if add_traces:
            ax.plot(bin_center, gb.T, alpha=0.05, color="grey", zorder=-10)

        ax.plot(bin_center, mu, label=label, color=color, **plot_kwargs)
        ax.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=-5)
        
        halved = mu.size // 2
        # ax.axhline(mu[halved:].mean(), color=color, **plot_kwargs)
        
    
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
    plot_kwargs["color"] = plot_kwargs.get("color", KTH_grey)
    plot_kwargs["ls"] = plot_kwargs.get("ls", "--")
    plot_kwargs["zorder"] = plot_kwargs.get("zorder", 10)
    plot_kwargs["ymax"] = plot_kwargs.get("ymax", 0.89)
    
    ytext = plot_kwargs["ymax"] + 0.03
    
    t_onset = params.warmup + params.duration_pre
    
    label_start = r"$t_\Delta$"
    label_end   = r"$t_\Delta'$"
    
    if control.brief_stimulus:
        for i in range(params.stim_reps):
            d = params.stim_duration + params.break_duration
            
            xstart  = t_onset + d*i
            ax.axvline(xstart, **plot_kwargs)
            
            xend    = t_onset + d*i + params.stim_duration
            plot_kwargs_tmp = dict(plot_kwargs)
            plot_kwargs_tmp["color"] = "salmon"
            ax.axvline(xend, **plot_kwargs_tmp)
            if i == 0:
                add_toplabel(ax, xstart, label_start, y=ytext)
                add_toplabel(ax, xend, label_end, y=ytext)
    else:
        ax.axvline(t_onset, **plot_kwargs)
        add_toplabel(ax, t_onset, label_start, y=ytext)


    # if control.brief_stimulus:
    #     ax.set_xticks(list(ax.get_xticks()) + [t_onset, t_onset+params.stim_duration], 
    #                   list(ax.get_xticklabels(minor=False)) + [r"$t_\Delta$", r"$t_\Delta'$"])  
    # else:
    #     ax.set_xticks(list(ax.get_xticks()) + [t_onset, ], 
    #                   list(ax.get_xticklabels(minor=False)) + [r"$t_\Delta$", ]) 
        
def add_toplabel(ax:object, x:float, label:str, y=0.92):
    ax.text(
        x,
        y,                     # slightly above axes
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="baseline"
    )
    

# From ChatGPT
def align_zero(ax_ref, ax_other):
    """Align y=0 of ax_other with y=0 of ax_ref."""
    ref_min, ref_max = ax_ref.get_ylim()

    if not ref_min < 0 < ref_max:
        raise ValueError("The reference axis must contain zero.")

    # Relative position of zero, measured from the bottom
    zero_position = -ref_min / (ref_max - ref_min)

    other_min, other_max = ax_other.get_ylim()

    # Find a span that retains all current values
    required_span = max(
        -other_min / zero_position if other_min < 0 else 0,
        other_max / (1 - zero_position) if other_max > 0 else 0,
    )

    ax_other.set_ylim(
        -zero_position * required_span,
        (1 - zero_position) * required_span,
    )
