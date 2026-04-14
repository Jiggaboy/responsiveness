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


#===============================================================================
# CONSTANTS
#===============================================================================



#===============================================================================
# METHODS
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
