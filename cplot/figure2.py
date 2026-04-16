# !/usr/bin/env python3
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
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order

from lib.analysis import bootstrap
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, panel_FR_with_delay, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)


force = False
# force = True

fn_rate = "dfrate"
fn_delay = "dfdelay"
    
pre_FR = 2
post_FRs = np.arange(4, 8, 2, dtype=float)
    
bootstraps = 20
samples_per_strap = 50

hist_binwidth = 2.

means = np.asarray([220., 260., 300.])


metadata = {
    "post_FRs": post_FRs,
    "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": hist_binwidth,
    "means": means,
}
    
    
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config()
    
    # Data collection
    overwrite = False
    if not force:
        df_rates = pd.read_pickle(fn_rate)
        df_delays = pd.read_pickle(fn_delay)
        for key, value in metadata.items():
            if isinstance(value, np.ndarray) and len(value) > 1:
                equal_r = all(df_rates.attrs[key] == value)
                equal_d = all(df_delays.attrs[key] == value)
            else:
                equal_r = df_rates.attrs[key] == value
                equal_d = df_delays.attrs[key] == value
            if not equal_r or not equal_d:
                overwrite = True
                print("Old metadata, rewrite data...")
    
    if force or overwrite:
        all_metrics = []
        all_rates = []
        with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
            for post_FR in post_FRs:
                for tag in (mean_tag, std_tag, mean_std_tag):
                # for tag in (std_tag, ):
                    for m, mean in enumerate(means):
                        logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                        run_ids = get_run_ids(rows, params, tag)
                        t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, hist_binwidth=hist_binwidth)
                
                        new_rows = pd.DataFrame({delay_tag: delay_estimates,})
                        new_rows.index = pd.MultiIndex.from_product(
                            [[tag], [mean], range(len(delay_estimates))],
                            names=["tag", "mean", "bootstrap_id"]
                        )
                        all_metrics.append(new_rows)
            
                        # Extend the array of firing rates
                        new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
                        new_rows.index = pd.MultiIndex.from_product(
                            [[tag], [mean], range(population_FR.shape[0])],
                            names=["tag", "mean", "bootstrap_id"]
                        )
                        all_rates.append(new_rows)
    
    
        # Conversion and save
        df_rates = pd.concat(all_rates)
        df_delays = pd.concat(all_metrics)
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            df_rates.attrs[key] = value
            df_delays.attrs[key] = value
        df_rates.to_pickle(fn_rate)
        df_delays.to_pickle(fn_delay)
            
    
    
    #===============================================================================
    # PLOTS
    #===============================================================================
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=3) #width_ratios=()
    fig.subplots_adjust(
        # left=0.06,
        # right=0.95,
        # bottom=0.1,
        # top=0.93,
        # wspace=0.2,
        # hspace=0.3
    )
    
    ax_kwargs = {
        "xlabel": "Time [ms]", "ylabel": "FR [Hz]", "ylim": (0, 21),
        "xlim": (params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125),
    }    
    axt_kwargs = {
        "ylabel": "Density of delays", "yticks": np.linspace(0, 0.5, 3), "ylim": (0, 0.5),
    }
    
    plot_kwargs = {"markersize": 2}
    ax = fig.add_subplot(gs[0, 0])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    panel_FR_with_delay(means[0], df_rates, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    panel_FR_with_delay(means[1], df_rates, t_bins, ax, marker="*", **plot_kwargs)
    panel_FR_with_delay(means[2], df_rates, t_bins, ax, marker="^", **plot_kwargs)

    

#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
