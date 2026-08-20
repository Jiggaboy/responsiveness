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

from lib.analysis import bootstrap, get_tbins
from lib.conversion import spikecount_to_FR
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids
from lib.util import save_figure

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays


#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
fname = "figure1_response_kernels"

ylim_timetospike = (-2, 125)
ylim_fr = (0, 21)

    
pre_FR = 5
post_FR = 10

bootstraps = 20
samples_per_strap = 50
    

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config(is_network=False, no_stim=True)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=3, ncols=3) #width_ratios=()
    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.06,
        top=0.94,
        wspace=0.3,
        hspace=0.65
    )
    
    ### Schematic
    ax = fig.add_subplot(gs[0, 2])
    ax.set(xlabel=xlabel_time, ylabel=ylabel_fr, ylim=(0.4, 3), xlim=(0, 65), title="Response Kernels")
    ax.set_yticks([1, 2], [r"$FR_{pre}$", r"$FR_{post}$"])
    xticks = np.arange(0, 60+1, 20)
    xlabels = list(xticks)
    xlabels[1] = r"$t_\Delta$"
    ax.set_xticks(xticks, xlabels)
    panel_schematic(ax)
    ax.legend()

    ### Example trace 1
    # How to get all the data we need?
    all_metrics = []
    all_rates = []
    
    means = [240., 280., 320.]
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)
                t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
    
    
    
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

    df_rates = pd.concat(all_rates)
    df_delays = pd.concat(all_metrics)
        
    ax_kwargs = {
        "xlabel": xlabel_time, "ylabel": ylabel_fr, "ylim": ylim_fr,
        "xlim": (params.warmup + params.duration_pre - 15, params.warmup + params.duration_pre + 85),
    }    
    axt_kwargs = {
        "ylabel": "Density of delays", "yticks": np.linspace(0, 0.5, 3), "ylim": (0, 0.5),
    }
    
    
    for m, mean in enumerate(means):
        ax = fig.add_subplot(gs[1, m])
        title = "Activity over Time\n" + r"$\mu_{pre}$" + f"={int(mean)}mA"
        if m == 0:
            ax.set(title=title, **ax_kwargs)
        else:
            ax_kwargs_tmp = dict(ax_kwargs)
            ax_kwargs_tmp.pop("ylabel", None)
            ax.set(title=title, **ax_kwargs_tmp)
            ax.tick_params(labelleft=False)

        
        plot_axvline_at_change(params, control, ax)
        plot_FRs(mean, df_rates, t_bins, ax)
        
        # axt = ax.twinx()   
        # axt.set(**axt_kwargs)
        # hist_delays(mean, df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
        if m == 1:
            ax.legend(reverse=True)
    #
    # ax = fig.add_subplot(gs[0, 2])
    # ax.set(**ax_kwargs)
    # ax.tick_params(labelleft=False)
    # plot_axvline_at_change(params, control, ax)
    # plot_FRs(means[1], df_rates, t_bins, ax)
    #
    # # axt = ax.twinx()   
    # # axt.set(**axt_kwargs)
    # # hist_delays(means[1], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    #
    # ax.legend()
    #
    # ax = fig.add_subplot(gs[0, 2])
    # ax.set(**ax_kwargs)
    # ax.tick_params(labelleft=False)
    # plot_axvline_at_change(params, control, ax)
    # plot_FRs(means[1], df_rates, t_bins, ax)
    #
    # # axt = ax.twinx()   
    # # axt.set(**axt_kwargs)
    # # hist_delays(means[1], df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    #
    # ax.legend()
    #===============================================================================
    # PLOT -  INDIVIDUAL FR
    #===============================================================================
    mean = means[0]
    all_rates = []
    t_bins = get_tbins(params)
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        # for tag in (mean_tag, std_tag, mean_std_tag):
        for tag in (std_tag, ):
            rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
            run_ids = get_run_ids(rows, params, tag)
            
            
            for run_id in run_ids:
                spikecount = load_and_merge_spikes(hfile, [run_id], t_bins)
                FRs = spikecount_to_FR(spikecount, params.N, params.hist_binwidth)
                
                # Extend the array of firing rates
                new_rows = pd.DataFrame(FRs) # Shape 1 x time
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], [run_id]],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_rates.append(new_rows)
    df_rates = pd.concat(all_rates)
    
    ax = fig.add_subplot(gs[2, 0])
    ax.set(title="Ind. Traces", **ax_kwargs)
    plot_FRs(mean, df_rates, t_bins, ax, add_traces=True)
    
    # bin_center = (t_bins[:-1] + t_bins[1:]) / 2
    #
    # df = df_rates.xs(mean, level=("mean"))
    # for tag, g in df.groupby(level="tag"):
    #     # Set Colors & Labels
    #     label = Label[tag]
    #     color = Color[tag]
    #
    #     # Filter a potential "all" id:
    #     gb = g[g.index.get_level_values("run_id") != "all"]
    #
    #     # g: rows = sims, cols = points
    #     mu = gb.mean(axis=0)
    #     std = gb.std(axis=0)
    #
    #     ax.plot(bin_center, df.T, alpha=0.05, color="grey", zorder=-10)
    #
    #     ax.plot(bin_center, mu, label=label, color=color)
    #     ax.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=-5)
        
    #===============================================================================
    # DATA -  TIME TO FIRST SPIKE
    #===============================================================================
    means = np.arange(220, 320+1, 10)
    
    time_to_first_spike = []
    df = pd.DataFrame(columns=["firstspike", "tag", "mean"])
    
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)
                
                all_first_spike = [] # All spikes 
                for run_id in run_ids:
                    if hfile.has_firstspike(run_id):
                        firstspike_tmp = hfile.get_node(hfile.data, f"run{run_id}").firstspike.read()
                        all_first_spike.extend(firstspike_tmp)
                        continue
                        
                    run_first_spike = []
                    spikes_by_sender = hfile.get_node(hfile.data, f"run{run_id}").spikes_by_sender.read()
                    for spikes in spikes_by_sender:
                        spikes_tmp = spikes[spikes >= params.warmup+params.duration_pre]
                        if len(spikes_tmp) > 0:
                            run_first_spike.append(spikes_tmp[0])
                    all_first_spike.extend(run_first_spike)
                    
                    hfile.add_firstspike(run_id, np.asarray(run_first_spike))
                hfile.flush()
                            
                            
                new_rows = pd.DataFrame({"firstspike": all_first_spike})
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(all_first_spike))],
                    names=["tag", "mean", "neuron_id"]
                )
                time_to_first_spike.append(new_rows)
    df = pd.concat(time_to_first_spike)
    
    
    
    #===============================================================================
    # PLOT -  TIME TO FIRST SPIKE
    #===============================================================================
    plot_means = [240, 280, 320]
    
    ax = fig.add_subplot(gs[2, 1])
    
    ax.set(
        xlabel=xlabel_drive, 
        ylabel=ylabel_reaction,
        title=f"Reaction Time",
        ylim=ylim_timetospike
    )

    df_tmp = df[df.index.isin(plot_means, level="mean")] - (params.warmup + params.duration_pre)
    sns.violinplot(
        df_tmp, 
        x="mean", 
        y="firstspike", 
        hue="tag", 
        cut=0, density_norm="width", common_norm=True, 
        inner=None,
        hue_order=hue_order, 
        ax=ax,
        native_scale=True,
        palette=Color,
        legend=False,
    )
    
    
    # handles = []
    # for tag in hue_order:
    #     handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
    # ax.legend(handles=handles)
        
    
    ax = fig.add_subplot(gs[2, 2])
    ax.set(
        xlabel=xlabel_drive, 
        ylabel=ylabel_reaction,
        ylim=(25, 87), 
        xticks=(means[::4]), 
        title="Reaction Time"
    )
        
    sns.lineplot(
        data=df - (params.warmup + params.duration_pre),
        x="mean",
        y="firstspike",
        hue="tag",
        marker="o",
        errorbar=("se", 1.96), # 95% interval
        estimator="mean",
        hue_order=hue_order,
        ax=ax,
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
        ax=ax,
        palette=Color,
    )
    labels = []
    for stat in ("mean", "median"):
        for tag in hue_order:
            labels.append(rf"{stat.capitalize()}")
            # labels.append(rf"{stat.capitalize()} ({Label[tag]})")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncols=2)


    save_figure(fname, fig, is_latex=True)

#===============================================================================
# METHODS
#===============================================================================
def panel_schematic(ax:object):
    dt = 0.1
    t_split = 20
    tau = 10
    f = 50e-3
    offset = 1
    
    t_pre  = np.arange(0, t_split, dt)
    t_post = np.arange(t_split, 80, dt)
    t = np.concatenate([t_pre, t_post])

    t_decay = t_post - t_split
    
    undershoot = np.zeros(t_pre.size + t_post.size, dtype=float)
    undershoot[-t_post.size:] = 1 - exp_decay(t_decay, tau)
    
    overshoot = np.zeros(t_pre.size + t_post.size, dtype=float)
    overshoot[-t_post.size:] = 0.25 * (t_post - t_split) * exp_decay(t_decay, 5) + 1 - exp_decay(t_decay)
    
    damped_osc = np.zeros(t_pre.size + t_post.size, dtype=float)
    damped_osc[-t_post.size:] = np.sin(2*np.pi * f * t_decay) * 2 * exp_decay(t_decay, 6) + 1 - exp_decay(t_decay, 2)
    # damped_osc[-t_post.size:] = np.sin(2*np.pi * f * t_decay) * exp_decay(t_decay, 6) + 1 - exp_decay(t_decay)
    
    ax.plot(t, undershoot + offset, color=CUNDERSHOOT, label="Undershoot", zorder=10) 
    ax.plot(t, overshoot + offset, color=COVERSHOOT, label="Overshoot", zorder=20) 
    ax.plot(t, damped_osc + offset, color=COSCILLATORY, label="Damped osc.", zorder=15)
    ax.axvline(t_split, c=KTH_grey, ls="--")
    
def exp_decay(t:np.ndarray, tau:float=1):
    return np.exp(-t / tau)


#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
