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

from lib.analysis import bootstrap, get_tbins, get_response_kernels
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, panel_FR_with_delay, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
ylim_delay = (0, 80)

force = False
# force = True

fn_rate = "dfrate"
fn_delay = "dfdelay"
    
pre_FR = 2.
post_FRs = np.arange(3, 8+1, 1, dtype=float)
# pre_FR = 4.
# post_FRs = np.arange(10, 10+1, 1, dtype=float)
# post_FRs = [12.]
    
bootstraps = 100
samples_per_strap = 50

means = np.asarray([220., 260., 300.])
means = np.arange(220, 320+1, 20.)

plot_means = np.asarray([320., 260., 300.])
plot_FRs = np.asarray([3., 4., 5.])



    
    
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config()
    
    metadata = {
        "post_FRs": post_FRs,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": params.hist_binwidth,
        "means": means,
    }
    
    # Data collection
    overwrite = False
    if not force:
        df_rates = pd.read_pickle(fn_rate)
        df_delays = pd.read_pickle(fn_delay)
        for key, value in metadata.items():
            if isinstance(value, np.ndarray) and len(value) > 1 and \
                isinstance(df_rates.attrs[key], np.ndarray) and len(df_rates.attrs[key]) > 1:
                try:
                    equal_r = all(df_rates.attrs[key] == value)
                    equal_d = all(df_delays.attrs[key] == value)
                except ValueError:
                    logger.info("Error while comparing metadata. Reset...")
                    equal_r, equal_d = False, False
            else:
                equal_r = df_rates.attrs[key] == value
                equal_d = df_delays.attrs[key] == value
                if not isinstance(equal_r, bool) and len(equal_r) > 1:
                    equal_r, equal_d = False, False
            if not equal_r or not equal_d:
                overwrite = True
                print("Old metadata, rewrite data...")
                break
    
    t_bins = get_tbins(params)
    if force or overwrite:
        all_metrics = []
        all_rates = []
        for post_FR in post_FRs:
            base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
            tmp_filename = base_filename + f"_{pre_FR}_{post_FR}_" + f".{suffix}"
            
            with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
                for tag in (mean_tag, std_tag, mean_std_tag):
                # for tag in (std_tag, ):
                    for m, mean in enumerate(means):
                        logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                        run_ids = get_run_ids(rows, params, tag)
                        t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
                
                        new_rows = pd.DataFrame({delay_tag: delay_estimates,})
                        new_rows.index = pd.MultiIndex.from_product(
                            [[post_FR], [tag], [mean], range(len(delay_estimates))],
                            names=["post_FR", "tag", "mean", "bootstrap_id"]
                        )
                        all_metrics.append(new_rows)
            
                        # Extend the array of firing rates
                        new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
                        new_rows.index = pd.MultiIndex.from_product(
                            [[post_FR], [tag], [mean], range(population_FR.shape[0])],
                            names=["post_FR", "tag", "mean", "bootstrap_id"]
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
    gs = fig.add_gridspec(nrows=3, ncols=3) #width_ratios=()
    fig.subplots_adjust(
        # left=0.06,
        # right=0.95,
        # bottom=0.1,
        # top=0.93,
        # wspace=0.2,
        # hspace=0.3
    )
    
    #===============================================================================
    # PLOTS - Row 1: Firing rates across means and post FRs
    #===============================================================================
    ax_kwargs = {
        "xlabel": "Time [ms]", "ylabel": "FR [Hz]", "ylim": (0, 16),
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
    df_tmp = df_rates.xs(plot_FRs[0], level="post_FR")
    panel_FR_with_delay(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # panel_FR_with_delay(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # panel_FR_with_delay(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)
    
    ax = fig.add_subplot(gs[0, 1])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    df_tmp = df_rates.xs(plot_FRs[1], level="post_FR")
    panel_FR_with_delay(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # panel_FR_with_delay(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # panel_FR_with_delay(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)
    
    ax = fig.add_subplot(gs[0, 2])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    df_tmp = df_rates.xs(plot_FRs[2], level="post_FR")
    panel_FR_with_delay(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # panel_FR_with_delay(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # panel_FR_with_delay(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)

    #===============================================================================
    # PLOTS - Row 2: Estimations of the transients
    #===============================================================================
    
    ax_transient = fig.add_subplot(gs[1, 0])
    post_FR = plot_FRs[0]
    title = f"Delay estimates\n(FR: {pre_FR} to {post_FR})"

    ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay, title=title)
    ax_transient.set_xticks(ticks=np.arange(len(means)), labels=means)
    df_tmp = df_delays.xs(post_FR, level="post_FR")
    sns.violinplot(df_tmp, x="mean", y="delay", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order, ax=ax_transient, native_scale=True,)
    
    
    # stats = df.groupby("category")["value"].agg(["mean", "median"]).reset_index()


    handles = []
    for tag in hue_order:
        handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
    plt.legend(handles=handles)
    

    ax_mean_delay = fig.add_subplot(gs[2, 0])
    ax_mean_delay.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay)

    sns.lineplot(
        data=df_tmp,
        x="mean",
        y="delay",
        hue="tag",
        marker="o",
        errorbar="sd",
        estimator="mean",
        hue_order=hue_order,
        ax=ax_mean_delay,
    )
    sns.lineplot(
        data=df_tmp,
        x="mean",
        y="delay",
        hue="tag",
        marker="^",
        linestyle="--",
        # errorbar=("pi", 50),
        estimator="median",
        hue_order=hue_order,
        ax=ax_mean_delay,
    )
    
    labels = []
    for stat in ("mean", "median"):
        for tag in hue_order:
            labels.append(rf"{stat.capitalize()} delay ({Label[tag]})")
    handles, _ = ax_mean_delay.get_legend_handles_labels()
    ax_mean_delay.legend(handles, labels, fontsize="x-small")
    
    
    #===============================================================================
    # PLOT: Delay over response kernel
    #===============================================================================
    
    ax_transient = fig.add_subplot(gs[1:, 1:])
                    
    # Get minimum delay
    # for post_FR in post_FRs:
    for (tag, post_FR), gb in df_delays.groupby(level=("tag", "post_FR")):
        delay_by_mean = gb.groupby(level="mean").mean()
        min_delay = delay_by_mean["delay"].min()
        min_mean  = delay_by_mean[delay_by_mean["delay"] == min_delay]
        min_mean  = min_mean.index.to_numpy()
    
        # # Get the corresponding response kernel
        # df_tmp = df_rates
        # # [overshoot, osc, undershoot]
        # response_kernel = get_response_kernels(params, gb, delays)
    
        for mean in min_mean:
            df_tmp = df_rates.xs((tag, mean, post_FR), level=("tag", "mean", "post_FR"))
            delays = df_delays.xs((tag, mean, post_FR), level=("tag", "mean", "post_FR"))["delay"]
            # [overshoot, osc, undershoot]
            response_kernel = get_response_kernels(params, df_tmp, delays, threshold=3)
            
            fraction_overshoot = response_kernel[0] / response_kernel.sum()
            
            plt.plot(mean + np.random.normal(scale=1, size=mean.size), fraction_overshoot + np.random.normal(scale=.05, size=mean.size),
                      marker=f"${int(post_FR)}$", color=Color[tag])
            
                
#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
