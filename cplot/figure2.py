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
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
ylim_delay = (0, 80)

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
FRs_to_plot = np.asarray([3., 4., 5.])
# FRs_to_plot = np.asarray([6., 10., 12.])



fn_id    = f"_{pre_FR}"
fn_rate  = "dfrate" + fn_id
fn_delay = "dfdelay" + fn_id
    
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
    df_tmp = df_rates.xs(FRs_to_plot[0], level="post_FR")
    plot_FRs(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # plot_FRs(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # plot_FRs(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)
    
    ax = fig.add_subplot(gs[0, 1])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    df_tmp = df_rates.xs(FRs_to_plot[1], level="post_FR")
    plot_FRs(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # plot_FRs(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # plot_FRs(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)
    
    ax = fig.add_subplot(gs[0, 2])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    df_tmp = df_rates.xs(FRs_to_plot[2], level="post_FR")
    plot_FRs(plot_means[0], df_tmp, t_bins, ax, marker="o", **plot_kwargs)
    # hist_delays(plot_means[0], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()
    # plot_FRs(plot_means[1], df_tmp, t_bins, ax, marker="*", **plot_kwargs)
    # plot_FRs(plot_means[2], df_tmp, t_bins, ax, marker="^", **plot_kwargs)

    #===============================================================================
    # PLOTS - Row 2: Estimations of the transients
    #===============================================================================
    
    ax_transient = fig.add_subplot(gs[1, 0])
    post_FR = FRs_to_plot[0]
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
    
    ax_transient = fig.add_subplot(gs[1:, 1])
                    
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
            
        
    # # Get minimum delay
    # # for post_FR in post_FRs:
    # for (tag, post_FR), gb in df_delays.groupby(level=("tag", "post_FR")):
    #     delay_by_mean = gb.groupby(level="mean").mean()
    #     min_delay = delay_by_mean["delay"].min()
    #     min_mean  = delay_by_mean[delay_by_mean["delay"] == min_delay]
    #     min_mean  = min_mean.index.to_numpy()
    #
    #
    #     # # Get the corresponding response kernel
    #     # df_tmp = df_rates
    #     # # [overshoot, osc, undershoot]
    #     # response_kernel = get_response_kernels(params, gb, delays)
    #
    #     for mean in min_mean:
    #         df_tmp = df_rates.xs((tag, mean, post_FR), level=("tag", "mean", "post_FR"))
    #         delays = df_delays.xs((tag, mean, post_FR), level=("tag", "mean", "post_FR"))["delay"]
    #         # [overshoot, osc, undershoot]
    #         response_kernel = get_response_kernels(params, df_tmp, delays, threshold=3)
    #
    #         fraction_overshoot = response_kernel[0] / response_kernel.sum()
    #
    #         plt.plot(mean + np.random.normal(scale=1, size=mean.size), fraction_overshoot + np.random.normal(scale=.05, size=mean.size),
    #                   marker=f"${int(post_FR)}$", color=Color[tag])
    #

            


    # if tag == mean_tag:
    #     ax = ax_response_mean
    # elif tag == std_tag:
    #     ax = ax_response_std
    # elif tag == mean_std_tag:
    #     ax = ax_response_both
    # else:
    #     raise ValueError

    def compute(row):
        post_FR, tag, mean, bootstrap_id = row.name
        return get_response_kernel(
            params, 
            row, 
            df_delays.loc[row.name, "delay"],
            threshold = 3,
        )
        
    from time import perf_counter
    before = perf_counter()
    df_delays[["over", "osc", "under"]] = df_rates.apply(
        compute,
        axis=1,
        result_type="expand"
    )
    
    # ax_response = fig.add_subplot(gs[1:, 2])
    gs_kernel = gs[1:, 2].subgridspec(nrows=3, ncols=1)
    ax_response_mean = fig.add_subplot(gs_kernel[0])
    ax_response_std  = fig.add_subplot(gs_kernel[1])
    ax_response_both = fig.add_subplot(gs_kernel[2])
        
    for tag, gb in df_delays.groupby(level="tag"):
        if tag == mean_tag:
            ax = ax_response_mean
        elif tag == std_tag:
            ax = ax_response_std
        elif tag == mean_std_tag:
            ax = ax_response_both
        else:
            raise ValueError
        
        under_tag = "under"
        over_tag  = "over"
        osc_tag   = "osc"
        kcolor = {under_tag: "grey", over_tag: "purple", osc_tag: "yellow"}
        for post_FR, gb_fr in gb.groupby(level=("post_FR")):
            gb_delay = gb_fr.groupby(level=("mean")).mean()
            values = gb_delay.reset_index()
            ax.plot(values["mean"], values["delay"],
                    color=Color[tag], alpha=post_FR / post_FRs.max())
            # ax.plot(gb_delay.index.get_level_values(level=0), gb_delay["delay"],
            #                 color=Color[tag], alpha=post_FR / post_FRs.max())
            
            kernel_max = values[["over", "under", "osc"]].max(axis=1)
            kernel_max_idx = values[["over", "under", "osc"]].idxmax(axis=1)
            ax.scatter(values["mean"], values["delay"],
                        marker="o", alpha=kernel_max, c=kernel_max_idx.map(kcolor))

        ##### DENSITY PLOTS
        # under_tag = "under"
        # over_tag  = "over"
        # osc_tag   = "osc"
        # kmaps = {under_tag: "Greys", over_tag: "Purples", osc_tag: "YlOrBr"}
        # kcolor = {under_tag: "grey", over_tag: "purple", osc_tag: "yellow"}
        # for ktyp in [under_tag, over_tag, osc_tag]:
        #     values = gb_fr.reset_index()[["mean", "delay", ktyp]]
        #     if not values[[ktyp]].any().item():
        #         continue
        #     sns.kdeplot(values, x="mean", y="delay", weights=values[[ktyp]].squeeze(), 
        #                 bw_method=1,
        #                 ax=ax, color=kcolor[ktyp], zorder=-8)
    



    
    after = perf_counter()
    print(f"Time elapsed: {after - before}")  
    return
    for (tag, post_FR, mean), gb in df_rates.groupby(level=("tag", "post_FR", "mean")):
        gb_delay = df_delays.xs((tag, post_FR, mean), level=("tag", "post_FR", "mean"))
        # for post_FR, gb in df_tmp.groupby(level=("post_FR")):
        # for (tag, post_FR), gb in df.groupby(level=("tag", "post_FR")):
        response_kernel = get_response_kernel(params, gb.iloc[0], gb_delay.iloc[0]["delay"], threshold=3)
        df_rates
        
        
        
        response_kernel = get_response_kernels(params, gb, gb_delay["delay"], threshold=3)
            
        kernels.append(response_kernel / response_kernel.sum())
        
            
        gb_delay = df_delays.xs((tag, post_FR), level=("tag", "post_FR"))
        delay_by_mean = gb_delay.groupby(level="mean").mean()
        
        kernels = []
        for mean, gbmean in gb.groupby(level="mean"):
            delays = df_delays.xs((tag, mean, post_FR), level=("tag", "mean", "post_FR"))["delay"]

            response_kernel = get_response_kernels(params, gbmean, delays, threshold=3)
            
            kernels.append(response_kernel / response_kernel.sum())
        kernels = np.asarray(kernels)
        
        # Overshoot
        kernel_density = skn.KernelDensity()
        kernel_density.fit()
        
        ax.plot(delay_by_mean.index.get_level_values(level=0), delay_by_mean["delay"],
                        color=Color[tag], alpha=post_FR / post_FRs.max())
            
            
            
            # ax_response.scatter(delay_by_mean.index.get_level_values(level=0), delay_by_mean["delay"], 
            #                     marker="o", alpha=post_FR / post_FRs.max(), c=kernels)
                
            
        
#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
