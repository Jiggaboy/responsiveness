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

from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from config import load_config, Control, Params
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order

from lib.analysis import bootstrap, get_tbins, get_tstart
from lib.conversion import spikecount_to_FR
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids
import lib.nest_interface as nif
from lib.util import save_figure

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
fname = "figure3_stimulus"

force = False
force = True

control, params = load_config(no_stim=True)
xlim_time = (params.warmup + params.duration_pre - 25, params.warmup + params.duration_pre + 65)
xlim_extendedtime = (params.warmup + params.duration_pre - 25, params.warmup + params.duration_pre + 125)
ylim_recovery = (0, 90)
# ylim_recovery = (15, 40)
yticks_recovery = np.arange(0, 100, 25)
ylim_FR = (0, 15)
ylabel_recovery = "Recovery Time [ms]"
ylabel_fr = "FR [Hz]"
xlabel_time = r"Time [ms]"
xlabel_mean = r"Mean drive $\mu_{pre}$"

marker_up = "^"
marker_down = "v"
marker = (marker_up, marker_down)
ls_up = "solid"
ls_down = "dashed"
lss = (ls_up, ls_down)

duration_marker = ["1", "x", "+"]
    
pre_FR_up = 5
post_FR_up = 10
pre_FR_down = 10
post_FR_down = 5

stim_reps = np.asarray([1, 2, 5])

stim_durations = np.round(np.asarray([nif.tau / 3, nif.tau / 2, nif.tau, nif.tau * 2]))
# stim_durations = np.round(np.asarray([nif.tau]))

bootstraps = 100
samples_per_strap = 50
    

means = np.asarray([220., 260., 300.])
means = np.arange(220, 320+1, 20.)


plot_mean = 240.
xticks_mean = means[::2]

fn_id    = f"_{pre_FR_up}"
fn_rate  = "stim_dfrate" + fn_id
fn_delay = "stim_dfdelay" + fn_id

fn_id    = f"_{pre_FR_up}_{post_FR_up}"
fn_bwrate  = "bandwidth_dfrate" + fn_id
fn_bwdelay = "bandwidth_dfdelay" + fn_id
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    
    metadata = {
        "pre_FR_up": pre_FR_up, "post_FR_up": post_FR_up, "pre_FR_down": pre_FR_down, "post_FR_down": post_FR_down,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": params.hist_binwidth,
        "means": means,
    }
    
    #===============================================================================
    # DATA: UP AND DOWN
    #===============================================================================
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
        for pre_FR, post_FR in zip((pre_FR_up, pre_FR_down), (post_FR_up, post_FR_down)):
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
                            [[pre_FR], [post_FR], [tag], [mean], range(len(delay_estimates))],
                            names=["pre_FR", "post_FR", "tag", "mean", "bootstrap_id"]
                        )
                        all_metrics.append(new_rows)
            
                        # Extend the array of firing rates
                        new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
                        new_rows.index = pd.MultiIndex.from_product(
                            [[pre_FR], [post_FR], [tag], [mean], range(population_FR.shape[0])],
                            names=["pre_FR", "post_FR", "tag", "mean", "bootstrap_id"]
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
    # DATA: STIM/BANDWIDTH
    #===============================================================================
    pre_FR = pre_FR_up
    post_FR = post_FR_up

    bwcontrol = Control()
    bwcontrol.brief_stimulus = True
    bwparams = Params(bwcontrol)
    bwparams.stim_duration = stim_durations[0]
    bwparams.stim_reps = stim_reps[0]
        
    overwrite = False
    metadata_tmp = dict(metadata)
    for key in ("pre_FR_down", "post_FR_down"):
        metadata_tmp.pop(key, None)
    metadata_tmp["stim_durations"] = stim_durations
    metadata_tmp["stim_reps"] = stim_reps
    if not force:
        try:
            df_bwrates = pd.read_pickle(fn_bwrate)
            df_bwdelays = pd.read_pickle(fn_bwdelay)
        except FileNotFoundError:
            logger.info("File not found. Start analysis...")
            overwrite = True
        else:    
            for key, value in metadata_tmp.items():
                if isinstance(value, np.ndarray) and len(value) > 1 and \
                    isinstance(df_bwrates.attrs[key], np.ndarray) and len(df_bwrates.attrs[key]) > 1:
                    try:
                        # Value needs only to be in the set
                        equal_r = all(np.isin(value, df_bwrates.attrs[key]))
                        equal_d = all(np.isin(value, df_bwdelays.attrs[key]))
                    except ValueError:
                        logger.info("Error while comparing metadata. Reset...")
                        equal_r, equal_d = False, False
                else:
                    try:
                        equal_r = df_bwrates.attrs[key] == value
                        equal_d = df_bwdelays.attrs[key] == value
                        if not isinstance(equal_r, bool) and len(equal_r) > 1:
                            equal_r, equal_d = False, False
                    except KeyError:
                        equal_r = equal_d = False
                if not equal_r or not equal_d:
                    overwrite = True
                    print("Old metadata, rewrite data...")
                    break
    
    t_bins = get_tbins(params)
    if force or overwrite:
        
        all_metrics = []
        all_rates = []
        for stim_rep, stim_duration in product(stim_reps, stim_durations):
            bwparams.stim_duration = stim_duration
            bwparams.stim_reps = stim_rep
            base_filename, suffix = bwparams.filename.rsplit(".", maxsplit=1)
            tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
            
            with ResponseHdf5(tmp_filename, "a", metadata=bwparams.metadata) as hfile:
                for tag in (mean_tag, std_tag, mean_std_tag):
                # for tag in (std_tag, ):
                    for m, mean in enumerate(means):
                        logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                        run_ids = get_run_ids(rows, bwparams, tag)
                        t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, bwparams, rep=bootstraps, samples_per_strap=samples_per_strap)
                
                        new_rows = pd.DataFrame({delay_tag: delay_estimates,})
                        new_rows.index = pd.MultiIndex.from_product(
                            [[pre_FR], [post_FR], [stim_duration], [stim_rep], [tag], [mean], range(len(delay_estimates))],
                            names=["pre_FR", "post_FR", "stim_duration", "stim_reps", "tag", "mean", "bootstrap_id"]
                        )
                        all_metrics.append(new_rows)
            
                        # Extend the array of firing rates
                        new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
                        new_rows.index = pd.MultiIndex.from_product(
                            [[pre_FR], [post_FR], [stim_duration], [stim_rep], [tag], [mean], range(population_FR.shape[0])],
                            names=["pre_FR", "post_FR", "stim_duration", "stim_reps", "tag", "mean", "bootstrap_id"]
                        )
                        all_rates.append(new_rows)
    
    
        # Conversion and save
        df_bwrates = pd.concat(all_rates)
        df_bwdelays = pd.concat(all_metrics)
        for key, value in metadata_tmp.items():
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            df_bwrates.attrs[key] = value
            df_bwdelays.attrs[key] = value
        df_bwrates.to_pickle(fn_bwrate)
        df_bwdelays.to_pickle(fn_bwdelay)
            
            
    #===============================================================================
    # PLOTS - Up and down step
    #===============================================================================
    fig = plt.figure(figsize=figsize)
    # gs = fig.add_gridspec(nrows=4, ncols=3)
    gs = fig.add_gridspec(nrows=4, ncols=3)
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.25,
        hspace=1,
    )
    
    ax_kwargs = {
        "xlabel": xlabel_time, "ylabel": ylabel_fr, "ylim": ylim_FR, "yticks": [0, int(pre_FR), int(post_FR)],
        "xlim": xlim_time,
    }    
    axt_kwargs = {
        "ylabel": "Density of delays", "yticks": np.linspace(0, 0.5, 3), "ylim": (0, 0.5),
    }
    
    # for idx, (pre_FR, post_FR) in enumerate(zip((pre_FR_down, pre_FR_up, ), (post_FR_down, post_FR_up, ))):
    for idx, (pre_FR, post_FR) in enumerate(zip((pre_FR_up, pre_FR_down), (post_FR_up, post_FR_down))):
        df_rates_tmp  = df_rates.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
        df_delays_tmp = df_delays.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
        
        ax = fig.add_subplot(gs[:1, idx])
        if idx == 0:
            ax.set(**ax_kwargs)
        elif idx == 1:
            ax_kwargs_tmp = dict(ax_kwargs)
            ax_kwargs_tmp.pop("ylabel", None)
            ax.set(**ax_kwargs_tmp)
            ax.tick_params(labelleft=False)
            
        title = f"Activity over Time\n(FR: {pre_FR}Hz" + r"$\rightarrow$" + f"{post_FR}Hz)"
        ax.set_title(title)
        
        plot_axvline_at_change(params, control, ax)
        plot_FRs(plot_mean, df_rates_tmp, t_bins, ax)
        
        # axt = ax.twinx()   
        # axt.set(**axt_kwargs)
        # hist_delays(plot_mean, df_delays_tmp, t_bins, axt, t_start=params.warmup+params.duration_pre)
        
        ax.set_xlim(ax_kwargs["xlim"])
        ax.legend()
   

    ax = fig.add_subplot(gs[:1, 2])
    ax.set(xlabel=xlabel_mean, ylabel=ylabel_recovery, ylim=ylim_recovery, xticks=xticks_mean)
    ax.set_title("Recovery Time\n")
    
    for idx, (pre_FR, post_FR, mark, ls) in enumerate(zip((pre_FR_up, pre_FR_down), (post_FR_up, post_FR_down), marker, lss)):
    # for idx, (pre_FR, post_FR, mark, ls) in enumerate(zip((pre_FR_up, ), (post_FR_up, ), marker[::1], lss[::1])):
    # for idx, (pre_FR, post_FR, mark, ls) in enumerate(zip((pre_FR_down, pre_FR_down), (post_FR_down, ), marker[::-1], lss[::-1])):
        df_delays_tmp = df_delays.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
            
        sns.lineplot(
            data=df_delays_tmp,
            x="mean",
            y="delay",
            hue="tag",
            marker=mark,
            linestyle=ls,
            errorbar=None,
            estimator="mean",
            hue_order=hue_order,
            palette=Color,
            ax=ax,
            alpha=0.75,
        )
        # sns.lineplot(
        #     data=df_delays_tmp,
        #     x="mean",
        #     y="delay",
        #     hue="tag",
        #     marker="^",
        #     linestyle="--",
        #     # errorbar=("pi", 50),
        #     estimator="median",
        #     hue_order=hue_order,
        #     ax=ax_mean_delay,
        # )
    
    labels = []
    # for stat in ("\N{DOWNWARDS ARROW}"):
    for stat in ("\N{UPWARDS ARROW}", "\N{DOWNWARDS ARROW}"):
        for tag in hue_order:
            labels.append(rf"Mean ({stat})")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncols=2)
    
    #===============================================================================
    # PLOTS - TRANSIENTS
    #===============================================================================
    gs_activity = gs[1:, 0].subgridspec(nrows=3, ncols=1,
       wspace=0.2,
       hspace=0.5,
    )
    gs_recovery = gs[1:, 1:].subgridspec(nrows=3, ncols=3,
        wspace=0.2,
        hspace=0.5,
    )
    pre_FR = pre_FR_up
    post_FR = post_FR_up
    stim_duration = np.round(nif.tau)
    
    bwparams.stim_duration = stim_duration
    for s, stim_rep in enumerate(stim_reps):
        bwparams.stim_reps = stim_rep
        t_start = get_tstart(bwparams)
        
        df_rates_tmp  = df_bwrates.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
        df_delays_tmp = df_bwdelays.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
        df_rates_tmp  = df_rates_tmp.xs((stim_rep, stim_duration), level=("stim_reps", "stim_duration"))
        df_delays_tmp = df_delays_tmp.xs((stim_rep, stim_duration), level=("stim_reps", "stim_duration"))

    
        # ax = fig.add_subplot(gs[s+1, 0])
        ax = fig.add_subplot(gs_activity[s])
        title = "Activity over Time\n" + f"({stim_rep} stimulation" + "s" * (stim_rep.item() > 1) + ")"
        ax.set(title=title, **ax_kwargs)
        
        if s == 2:
            ax.set(xlabel=xlabel_time)
        else:
            ax.set(xlabel=None)
            ax.tick_params(labelbottom=False)
                
        plot_axvline_at_change(bwparams, bwcontrol, ax)
        plot_FRs(plot_mean, df_rates_tmp, t_bins, ax)
        
        ax.set_xlim(xlim_extendedtime)
        # ax.legend()
        
        # axt = ax.twinx()   
        # axt.set(**axt_kwargs)
        # hist_delays(plot_mean, df_delays_tmp, t_bins, axt, t_start=t_start)
   
   
        #===============================================================================
        # PLOTS - Single Up - VIOLIN
        #===============================================================================
        # pre_FR = pre_FR_up
        # post_FR = post_FR_up
        # ax_violin = fig.add_subplot(gs[s+1, 1])
        # title = f"Delay estimates\n(FR: {pre_FR} to {post_FR})"
        #
        # ax_violin.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_recovery, title=title)
        # ax_violin.set_xticks(ticks=np.arange(len(means)), labels=means)
        #
        # df_tmp = df_bwdelays.xs((pre_FR, post_FR, stim_rep, stim_duration), level=("pre_FR", "post_FR", "stim_reps", "stim_duration"))
        # sns.violinplot(df_tmp, x="mean", y="delay", hue="tag", 
        #                cut=0, density_norm="width", common_norm=True, 
        #                hue_order=hue_order, ax=ax_violin, native_scale=True,)


        #===============================================================================
        # PLOTS - Single Up - MEANS across 
        #===============================================================================
        # ax = fig.add_subplot(gs[s+1, 1:])
        
        # ax = fig.add_subplot(gs_recovery[0, 0])
        
        
        
        # for idx, (stim_duration, mark) in enumerate(zip(stim_durations, duration_marker)):
        df_delays_tmp = df_bwdelays.xs((pre_FR, post_FR, stim_rep), level=("pre_FR", "post_FR", "stim_reps"))
        
        ax_response_mean = fig.add_subplot(gs_recovery[s, 0])
        ax_response_std  = fig.add_subplot(gs_recovery[s, 1])
        ax_response_both = fig.add_subplot(gs_recovery[s, 2])
        # ax_response_mean = fig.add_subplot(gs_recovery[0, s])
        # ax_response_std  = fig.add_subplot(gs_recovery[1, s])
        # ax_response_both = fig.add_subplot(gs_recovery[2, s])
            
        # ax_kwargs = {"ylim": ylim_recovery, "ylabel": ylabel, 
        #              "yticks": np.arange(ylim_recovery[0], ylim_recovery[1]+15, 20), 
        #              "xticks": means[::2]}
        title = f"Recovery Time ({stim_rep} stimulation" + "s" * (stim_rep.item() > 1) + ")" + "\n"
        for tag, gb in df_delays_tmp.groupby(level="tag"):
            if tag == mean_tag:
                ax = ax_response_mean
                ax.set_ylabel(ylabel_recovery)
            elif tag == std_tag:
                ax = ax_response_std
                ax.set(title=title)
                ax.tick_params(labelleft=False)
            elif tag == mean_std_tag:
                ax = ax_response_both
                ax.tick_params(labelleft=False)
            else:
                raise ValueError
            
            if s == 2:
                ax.set(xlabel=xlabel_mean)
            else:
                ax.tick_params(labelbottom=False)
                
            ax.set(ylim=ylim_recovery, xticks=xticks_mean, yticks=yticks_recovery)
        
        
            sns.lineplot(
                # data=df_delays_tmp,
                data=gb,
                x="mean",
                y="delay",
                style="stim_duration",
                # hue="tag",
                markers=True,
                # markers=duration_marker,
                linestyle=ls,
                errorbar=None,
                estimator="mean",
                # hue_order=hue_order,
                ax=ax,
                # palette=Color,
                color = Color[tag],
            )
            
        
            labels = []
            handles, legend_labels = ax.get_legend_handles_labels()
            for label in legend_labels:
                labels.append(rf"{int(float(label))}ms")
            ax.legend(handles, labels, ncols=2, alignment="right")
    
    save_figure(fname, fig, is_latex=True)
    
    
    # Supplementary Figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=3, ncols=3)
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.25,
        hspace=.3,
    )
    df_delays_tmp = df_bwdelays.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
    df_delays_tmp = df_delays_tmp[df_delays_tmp.index.isin(means[::2], level="mean")]
    
    
    for i, ((tag, mean), gb) in enumerate(df_delays_tmp.groupby(level=("tag", "mean"))):
        
        title = f"{tag} is constant"
        
        if tag == mean_tag:
            ax = fig.add_subplot(gs[0, i%3])
            ax.set_ylabel(ylabel_recovery)
            ax.tick_params(labelbottom=False)
        elif tag == std_tag:
            ax = fig.add_subplot(gs[1, i%3])
            ax.tick_params(labelbottom=False)
        elif tag == mean_std_tag:
            ax = fig.add_subplot(gs[2, i%3])
        else:
            raise ValueError
        
        if (i%3) == 1:
            ax.set(title=title)

        sns.lineplot(
            data = gb,
            x = "stim_reps",
            y = "delay",
            style = "stim_duration",
            markers=True,
            errorbar=None,
            estimator="mean",
            ax = ax,
            color = Color[tag],
        )
        
#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
