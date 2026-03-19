#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


from constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color
from config import load_config
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient

from lib.conversion import spikecount_to_FR


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

plot_rate_and_delays = True
# plot_rate_and_delays = False

plot_transient_estimates = True
plot_transient_estimates = False


hue_order = [mean_tag, std_tag, mean_std_tag]

ylim_delay = (0, 125)

#===============================================================================
# CONSTANTS
#===============================================================================
hist_binwidth = 5. #ms

bootstraps = 50     #50
samples_per_strap = 25 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  # .6s per seed (25 straps x 10 samples)
def main():
    control, params = load_config()
    
    pre_FR = 2.
    post_FR = 4.
    
    pre_FR = 5.
    post_FR = 10.
    
    # pre_FR = 4.
    # # # post_FR = 6.
    # post_FR = 12.
    #
    # pre_FR = 10.
    # post_FR = 5.
    means = np.arange(240, 320+1, 20.)
    means = np.arange(220, 320+1, 120.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)

    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        for hist_binwidth in (1., 2., 2.5, 5.):
            t_bins = np.arange(0., params.duration_pre+params.duration_post+hist_binwidth, float(hist_binwidth)) + params.warmup
            #===============================================================================
            # MORE METHODS
            #===============================================================================    
            all_metrics = []
            all_rates = []
            all_runs_delays = []
        
            ##### ALL ANALYSES ######################################
            for tag in (mean_tag, std_tag, mean_std_tag):
                for m, mean in enumerate(means):
                    ### Filter the rows
                    logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")

                    rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                    
                    if tag in (mean_tag, std_tag):
                        rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]] 
                    elif tag == mean_std_tag:
                        mask = np.logical_and(rows[f"pre_{mean_tag}"] != rows[f"post_{mean_tag}"], rows[f"pre_{std_tag}"] != rows[f"post_{std_tag}"])
                        rows_filtered = rows[mask]
                    else:
                        raise ValueError("No valid tag given...")
                    
                    stim_mask = np.logical_and(rows_filtered["stim_duration"] == params.stim_duration, 
                                               rows_filtered["break_duration"] == params.break_duration, 
                                               rows_filtered["stim_reps"] == params.stim_reps)
                    rows_filtered = rows_filtered[stim_mask]
                    run_ids = rows_filtered[id_tag]
        
                    # DELAY ACROSS ALL RUNS
                    spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
                    # t_start = params.warmup + params.duration_pre + (control.brief_stimulus * params.stim_duration)
                    t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
                    index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup
        
                    SEM, delay_all_runs = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
        
                    new_rows = pd.DataFrame({delay_tag: [delay_all_runs * hist_binwidth]})
                    new_rows.index = pd.MultiIndex.from_tuples([(tag, mean)], names=["tag", "mean"])
                    all_runs_delays.append(new_rows)
        
                    FR_all_runs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                    # Extend the array of firing rates
                    new_rows = pd.DataFrame([FR_all_runs])
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], ["all"]],
                        names=["tag", "mean", "bootstrap_id"]
                    )
                    all_rates.append(new_rows)
        
                    # Detailed feature analysis
                    entropies_pre   = np.zeros(bootstraps)
                    delay_estimates = np.zeros(bootstraps)
                    population_FR   = np.zeros((bootstraps, t_bins.size-1))
                    for b in range(bootstraps):
                        np.random.shuffle(run_ids)
                        samples = run_ids[:samples_per_strap] # Bootstrapping
        
                        # DELAY 
                        spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)
                        t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
                        index = (t_bins >= t_start).argmax() # Gets first value that is larger than duration_pre + warmup
        
                        SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                        delay_estimates[b] = delay * hist_binwidth
                        
    
                        # FIRING RATE
                        FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
                        population_FR[b] = FRs
        
                        # ENTROPY
                        pre_entropies  = hfile.read_rows(samples)["pre_entropy"]
                        entropies_pre[b] = pre_entropies.mean()
        
        
                    new_rows = pd.DataFrame({
                        entropy_tag: entropies_pre,
                        delay_tag: delay_estimates,
                    })
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], range(len(delay_estimates))],
                        names=["tag", "mean", "bootstrap_id"]
                    )
                    all_metrics.append(new_rows)
        
                    # Extend the array of firing rates
                    new_rows = pd.DataFrame(population_FR)
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], range(population_FR.shape[0])],
                        names=["tag", "mean", "bootstrap_id"]
                    )
                    all_rates.append(new_rows)
        
        
            df_rates = pd.concat(all_rates)
            df_metrics = pd.concat(all_metrics)
            df_all_runs_delays = pd.concat(all_runs_delays)
        
        
        
        #===============================================================================
        # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
        #=============================================================================== 
            if plot_rate_and_delays:
                t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
                for m, mean in enumerate(means):
                    figname = f"{hist_binwidth} Firing rates (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
                    fig, ax1 = plt.subplots(num=figname)
                    ax1.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
                            xlim=(params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125))
            
                    # Indicate the time point of change
                    ax1.axvline(params.warmup + params.duration_pre, color="red", zorder=10, ls="--")
                    if control.brief_stimulus:
                        offset = params.warmup + params.duration_pre
                        for i in range(params.stim_reps):
                            d = params.stim_duration + params.break_duration
                            ax1.axvline(offset + d*i + params.stim_duration, color="red", zorder=10, ls="--")
                            ax1.axvline(offset + d*i, color="red", zorder=10, ls="--")
                        # TODO: Set ticks properly for multiple reps
                        ax1.set_xticks(list(plt.xticks()[0]) + [params.warmup+params.duration_pre, params.warmup+params.duration_pre+params.stim_duration], list(plt.xticks()[0]) + [r"$t_\Delta$", r"$t_\Delta'$"])  
                    else:
                        ax1.set_xticks(list(plt.xticks()[0]) + [params.warmup+params.duration_pre, ], list(plt.xticks()[0]) + [r"$t_\Delta$", ])                
            
                    ax2 = ax1.twinx()   
                    ax2.set(ylabel="Density of delays", yticks=np.linspace(0, 0.5, 3), ylim=(0, 0.5))
            
                    bin_center = (t_bins[:-1] + t_bins[1:]) / 2
            
            
                    df = df_rates.xs(mean, level=("mean"))
                    for tag, g in df.groupby(level="tag"):
                        gb = g[g.index.get_level_values("bootstrap_id") != "all"]
                        label = Label[tag]
                        color = Color[tag]
            
                        # Delay Estimation across all runs
                        d = df_all_runs_delays.xs((tag, mean), level=("tag", "mean"))[delay_tag].squeeze()
                        ax1.axvline(d + t_start, c=color, lw=2, ls="--", zorder=15)
            
                        # g: rows = sims, cols = points
                        mu = gb.mean(axis=0)
                        std = gb.std(axis=0)
            
            
                        ax1.plot(bin_center, mu, label=label, color=color)
                        ax1.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=-5)
                        
                        halved = mu.size // 2
                        ax1.axhline(mu[halved:].mean(), color=color)
            
                        ax1.plot(bin_center, g.xs("all", level="bootstrap_id").squeeze(), ls="dotted", color=color)
            
                        # Hist delays
                        delays = df_metrics.xs((tag, mean), level=("tag", "mean"))["delay"]
                        ax2.hist(delays + t_start, bins=t_bins, color=color, density=True, zorder=-4, rwidth=0.9, alpha=0.5)
            
                    ax1.set_ylim(bottom=0) 
                    ax1.legend()
            
                    # save_figure(figname, fig)
    
        
            #===============================================================================
            # PLOT - TRANSIENT ESTIMATES
            #=============================================================================== 
            if plot_transient_estimates:
                figname_transient = f"{hist_binwidth} Delay estimates (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
                fig, ax_transient = plt.subplots(num=figname_transient)
                ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay,)
                ax_transient.set_xticks(ticks=np.arange(len(means)), labels=means)
                sns.violinplot(df_metrics, x="mean", y="delay", hue="tag", 
                               cut=0, density_norm="width", common_norm=True, 
                               hue_order=hue_order, ax=ax_transient)
                handles = []
                for tag in hue_order:
                    handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
                plt.legend(handles=handles)
                # save_figure(figname_transient, fig)
        
        
            
                figname_transient_means = f"Mean delay estimates (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
                fig, ax_meandelay = plt.subplots(num=figname_transient_means)
                ax_meandelay.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay,)
                ax_meandelay.set_xticks(ticks=means, labels=means)
            
                sns.lineplot(
                    data=df_metrics,
                    x="mean",
                    y="delay",
                    hue="tag",
                    marker="o",
                    errorbar="sd",
                    estimator="mean",
                    hue_order=hue_order,
                    ax=ax_meandelay,
                )
                sns.lineplot(
                    data=df_metrics,
                    x="mean",
                    y="delay",
                    hue="tag",
                    marker="^",
                    linestyle="--",
                    # errorbar=("pi", 50),
                    estimator="median",
                    hue_order=hue_order,
                    ax=ax_meandelay,
                )
            
                labels = []
                for stat in ("mean", "median"):
                    for tag in hue_order:
                        labels.append(rf"{stat.capitalize()} delay ({Label[tag]})")
                handles, _ = ax_meandelay.get_legend_handles_labels()
                ax_meandelay.legend(handles, labels)
                # save_figure(figname_transient_means, fig)
                     
        







#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
