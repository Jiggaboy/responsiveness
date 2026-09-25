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
import sklearn.neighbors as skn

from config import load_config
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order

from lib.analysis import bootstrap, get_tbins, get_response_kernels, get_response_kernel
from lib.responsehdf5 import ResponseHdf5, get_run_ids
from lib.util import save_figure

from cplot.plot_constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
fname = "figure2_recovery"

    
ylim_delay = (0, 95)
ylim_delay_example = (0, 58)
FR_lim = (0, 12)

force = False
# force = True

    
pre_FR = 2.
post_FRs = np.arange(3, 8+1, 1, dtype=float)

# pre_FR = 4.
# post_FRs = np.arange(6, 12+1, 2, dtype=float)
# post_FRs = [12.]
    
bootstraps = 100
samples_per_strap = 50

means = np.asarray([220., 260., 300.])
means = np.arange(220, 320+1, 20.)

plot_means = np.asarray([280., 260., 300.])
FRs_to_plot = np.asarray([4., 6., 8.])
# FRs_to_plot = np.asarray([6., 10., 12.])



fn_id    = f"_{pre_FR}"
fn_rate  = "dfrate" + fn_id
fn_delay = "dfdelay" + fn_id
    
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config()
    xlim_time = (params.warmup + params.duration_pre - 15, params.warmup + params.duration_pre + 85)
    
    metadata = {
        "post_FRs": post_FRs,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": params.hist_binwidth,
        "means": means,
    }
    
    # Data collection
    overwrite = False
    if not force:
        try:
            df_rates = pd.read_pickle(fn_rate)
            df_delays = pd.read_pickle(fn_delay)
        except FileNotFoundError:
            logger.info("File not found. Start analysis...")
            overwrite = True
        else:    
            for key, value in metadata.items():
                if isinstance(value, np.ndarray) and len(value) > 1 and \
                    isinstance(df_rates.attrs[key], np.ndarray) and len(df_rates.attrs[key]) > 1:
                    try:
                        # Value needs only to be in the set
                        equal_r = all(np.isin(value, df_rates.attrs[key]))
                        equal_d = all(np.isin(value, df_delays.attrs[key]))
                        # equal_r = all(df_rates.attrs[key] == value)
                        # equal_d = all(df_delays.attrs[key] == value)
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
            tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
            
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
        left=0.06,
        right=0.96,
        bottom=0.08,
        top=0.93,
        wspace=0.25,
        hspace=0.6
    )
    
    #===============================================================================
    # PLOTS - Row 1: Firing rates across means and post FRs
    #===============================================================================
    ax_kwargs = {
        "xlabel": xlabel_time, "ylabel": ylabel_fr, "ylim": FR_lim,
        "xlim": xlim_time,
        "yticks": np.arange(*FR_lim, 2),
    }    
    axt_kwargs = {
        "ylabel": "Density of delays", "yticks": np.linspace(0, 0.5, 3), "ylim": (0, 0.5),
    }
    
    plot_kwargs = {"markersize": 2}
    for i, fr in enumerate(FRs_to_plot):
        ax = fig.add_subplot(gs[0, i])
        title = f"Activity over Time\n(FR: {pre_FR:.0f}Hz" + r"$\rightarrow$" + f"{fr:.0f}Hz)"
        if i == 0:
            ax.set(title=title, **ax_kwargs)
        else:
            ax_kwargs_tmp = dict(ax_kwargs)
            ax_kwargs_tmp.pop("ylabel", None)
            ax.set(title=title, **ax_kwargs_tmp)
            ax.tick_params(labelleft=False)
            
        plot_axvline_at_change(params, control, ax)
        df_tmp = df_rates.xs(FRs_to_plot[i], level="post_FR")
        plot_FRs(plot_means[0], df_tmp, t_bins, ax, **plot_kwargs)
        
        ax.set_xlim(ax_kwargs["xlim"])
    
        if i == 1:
            ax.legend(reverse=True)
    
    # axt = ax.twinx()   
    # axt.set(**axt_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    # plot_FRs(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # plot_FRs(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)
    

    #===============================================================================
    # PLOTS - Row 2: Estimations of the transients
    #===============================================================================
    
    gs_delays = gs[1:, :2].subgridspec(nrows=2, ncols=1,
        wspace=0.3,
        hspace=0.4,
    )
    
    ax_transient = fig.add_subplot(gs_delays[0, 0])
    post_FR = FRs_to_plot[0]
    title = f"Recovery Time\n(FR: {pre_FR:.0f}Hz" + r"$\rightarrow$" + f"{post_FR:.0f}Hz)"
    

    ax_transient.set(ylabel=ylabel_recovery, ylim=ylim_delay_example, title=title)
    ax_transient.set_xticks(ticks=np.arange(len(means)), labels=means)
    df_tmp = df_delays.xs(post_FR, level="post_FR")
    sns.violinplot(df_tmp, x="mean", y="delay", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order, ax=ax_transient, native_scale=True,
                   inner=None,
                   palette=Color,
    )
    ax_transient.set(xlabel=None, xticks=means)
    
    handles = []
    for tag in hue_order:
        handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
    plt.legend(handles=handles, ncols=3)
    

    # ax_mean_delay = fig.add_subplot(gs[1, 0])
    ax_mean_delay = fig.add_subplot(gs_delays[1, 0])
    title = "Mean " + title
    ax_mean_delay.set(xlabel=xlabel_drive, ylabel=ylabel_recovery, ylim=ylim_delay_example, title=title)

    sns.lineplot(
        data=df_tmp,
        x="mean",
        y="delay",
        hue="tag",
        marker=marker_mean_recovery,
        errorbar=None,
        estimator="mean",
        hue_order=hue_order,
        ax=ax_mean_delay,
        palette=Color,
    )
    sns.lineplot(
        data=df_tmp,
        x="mean",
        y="delay",
        hue="tag",
        marker=marker_median_recovery,
        linestyle="--",
        errorbar=None,
        estimator="median",
        hue_order=hue_order,
        ax=ax_mean_delay,
        palette=Color,
    )
    
    labels = []
    for stat in ("mean", "median"):
        for tag in hue_order:
            # labels.append(rf"{stat.capitalize()} delay ({Label[tag]})")
            labels.append(rf"{stat.capitalize()}")
    handles, _ = ax_mean_delay.get_legend_handles_labels()
    ax_mean_delay.legend(handles, labels, ncols=2, loc="upper left")
    

    #===============================================================================
    # PLOT: Delay over response kernel
    #===============================================================================
    
    if True:
        # ax_transient = fig.add_subplot(gs[1:, 1])
        fig_trans, ax_transient = plt.subplots()
                        
        avg_delay = (
            df_delays["delay"]
            .groupby(level=["tag", "mean", "post_FR"])
            .mean()
        )
        idx = avg_delay.groupby(level=("tag", "post_FR")).idxmin()
        out = avg_delay.loc[idx].reset_index(name="avg_delay")
            
            
        for tag, rows in out.groupby("tag"):
            for i in range(len(rows)-1):
                ax_transient.plot(rows["mean"].iloc[i:i+1+1],
                                  rows["avg_delay"].iloc[i:i+1+1],
                                  alpha=rows["post_FR"].iloc[i:i+1+1].mean() / rows["post_FR"].max(),
                                  color=Color[tag], marker="*")
                
            
    
    def compute(row):
        post_FR, tag, mean, bootstrap_id = row.name
        return get_response_kernel(
            params, 
            row, 
            df_delays.loc[row.name, "delay"],
            threshold = 3,
        )
        
    df_delays[["over", "osc", "under"]] = df_rates.apply(
        compute,
        axis=1,
        result_type="expand"
    )
    
    # ax_response = fig.add_subplot(gs[1:, 2])
    gs_kernel = gs[1:, 2].subgridspec(nrows=3, ncols=1, hspace=0.2)
    ax_response_mean = fig.add_subplot(gs_kernel[0])
    ax_response_std  = fig.add_subplot(gs_kernel[1])
    ax_response_both = fig.add_subplot(gs_kernel[2])
        
    xbuffer = 5
    ax_kwargs = {
        "ylim": ylim_delay, 
        "yticks": np.arange(ylim_delay[0], ylim_delay[1]+15, 20), 
        "xlim": (means[0] - xbuffer, means[-1] + xbuffer),
        "xticks": means[::2],
    }
    title = "Mean Recovery Time"
    for tag, gb in df_delays.groupby(level="tag"):
        if tag == std_tag:
            ax = ax_response_mean
            ax.tick_params(labelbottom=False)
            ax.set(title=title)
        elif tag == mean_tag:
            ax = ax_response_std
            ax.tick_params(labelbottom=False)
            ax.set_ylabel(ylabel_recovery)
        elif tag == mean_std_tag:
            ax = ax_response_both
            ax.set_xlabel(xlabel_drive)
        else:
            raise ValueError
        ax.set(**ax_kwargs)
        
        under_tag = "under"
        over_tag  = "over"
        osc_tag   = "osc"
        markersize = 10
        kcolor = {under_tag: CUNDERSHOOT, over_tag: COVERSHOOT, osc_tag: COSCILLATORY}
        # kcolor = {under_tag: KTH_sky, over_tag: KTH_blue, osc_tag: KTH_navy}
        kmarker = {under_tag: "v", over_tag: "^", osc_tag: r"$\sim$"}
        klabel = {under_tag: "undershoot", over_tag: "overshoot", osc_tag: "damped osc."}
        for post_FR, gb_fr in gb.groupby(level=("post_FR")):
            gb_delay = gb_fr.groupby(level=("mean")).mean()
            values = gb_delay.reset_index()
            ax.plot(values["mean"], values["delay"],
                    color=Color[tag], alpha=post_FR / post_FRs.max())
            # ax.plot(gb_delay.index.get_level_values(level=0), gb_delay["delay"],
            #                 color=Color[tag], alpha=post_FR / post_FRs.max())
            
            kernel_max = values[["over", "under", "osc"]].max(axis=1)
            kernel_max_idx = values[["over", "under", "osc"]].idxmax(axis=1)
            for idx, row in values.iterrows():
                ax.scatter(row["mean"], row["delay"],
                            marker=kmarker[kernel_max_idx[idx]],
                            alpha=kernel_max[idx], c=kcolor[kernel_max_idx[idx]], s=markersize,
                            zorder=10)
        # Add legend
        if tag == mean_tag:
            for key in kcolor.keys():
                ax.scatter(-100, -100, marker=kmarker[key], c=kcolor[key], s=markersize, label=klabel[key].capitalize())
            ax.legend()

            
            
    save_figure(fname, fig, is_latex=True)
        
#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
