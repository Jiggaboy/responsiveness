#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:

History:
    - 0.2: Adjustment of t_bins with dt/2
    - 0.2b: Added get_run_ids.
    - 0.2c: Moved get_run_ids to responsehdf5.

"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2c'

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


from plot_constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color, hue_order
from config import load_config
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient, bootstrap

from lib.conversion import spikecount_to_FR

from cplot.aux import plot_axvline_at_change


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

plot_rate_and_delays = True
# plot_rate_and_delays = False

plot_entropy_over_delay = True
plot_entropy_over_delay = False

plot_transient_estimates = True
plot_transient_estimates = False

plot_time_to_first_spike = True
plot_time_to_first_spike = False

plot_dist_first_and_second_spike = True
plot_dist_first_and_second_spike = False


ylim_delay = (0, 75)

#===============================================================================
# CONSTANTS
#===============================================================================
bootstraps        = 100    
samples_per_strap =  50



pre_FR  = 2.
post_FR = 4.
post_FR = 6.

pre_FR = 5.
post_FR = 10.
#
# pre_FR = 4.
# # # # post_FR = 6.
# post_FR = 8.
# post_FR = 10.
# post_FR = 12.

# pre_FR = 10.
# post_FR = 5.

means = np.arange(320, 320+1, 40.)
# means = np.arange(220, 320+1, 10.)

# Equivalent for network with J=0.075
# pre_FR = 5.
# post_FR = 10.
# means = np.asarray([-2.5, 35])
# # Equivalent for network with J=0.075
# pre_FR = 5.
# post_FR = 10.
# means = np.asarray([-2.5, 35])

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  # .6s per seed (25 straps x 10 samples)
def main():
    control, params = load_config()

    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{pre_FR}_{post_FR}" + f".{suffix}"
    
        
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
        all_rates = []
        all_runs_delays = []
    
        ##### ALL ANALYSES ######################################
        for tag in (mean_tag, std_tag, mean_std_tag):
        # for tag in (std_tag, ):
            for m, mean in enumerate(means):
                logger.info(f"{tag}: Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)

    
                # # DELAY ACROSS ALL RUNS
                # spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
                #
                # SEM, delay_all_runs = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                #
                # new_rows = pd.DataFrame({delay_tag: [delay_all_runs * params.hist_binwidth]})
                # new_rows.index = pd.MultiIndex.from_tuples([(tag, mean)], names=["tag", "mean"])
                # all_runs_delays.append(new_rows)
                #
                # FR_all_runs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, params.hist_binwidth)
                # plt.plot(t_bins[:-1], FR_all_runs)
                # # Extend the array of firing rates
                # new_rows = pd.DataFrame([FR_all_runs])
                # new_rows.index = pd.MultiIndex.from_product(
                #     [[tag], [mean], ["all"]],
                #     names=["tag", "mean", "bootstrap_id"]
                # )
                # all_rates.append(new_rows)
                #

                
                t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
                
                
                new_rows = pd.DataFrame({
                    # entropy_tag: entropies_pre,
                    delay_tag: delay_estimates,
                })
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
        df_metrics = pd.concat(all_metrics)
        # df_all_runs_delays = pd.concat(all_runs_delays)
        # plt.xlim(params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125)
    
    
    
    #===============================================================================
    # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
    #=============================================================================== 
    if plot_rate_and_delays:
        t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
        for m, mean in enumerate(means):
            figname = f"Firing rates (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
            fig, ax1 = plt.subplots(num=figname)
            ax1.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
                    xlim=(params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125))
    
            plot_axvline_at_change(params, control, ax1)              
    
            ax2 = ax1.twinx()   
            ax2.set(ylabel="Density of delays", yticks=np.linspace(0, 0.5, 3), ylim=(0, 0.5))
    
            bin_center = (t_bins[:-1] + t_bins[1:]) / 2
            # bin_center = bin_center[:-1]
    
    
            df = df_rates.xs(mean, level=("mean"))
            for tag, g in df.groupby(level="tag"):
                gb = g[g.index.get_level_values("bootstrap_id") != "all"]
                label = Label[tag]
                color = Color[tag]
    
                # Delay Estimation across all runs
                # d = df_all_runs_delays.xs((tag, mean), level=("tag", "mean"))[delay_tag].squeeze()
                # ax1.axvline(d + t_start, c=color, lw=2, ls="--", zorder=15)
    
                # g: rows = sims, cols = points
                mu = gb.mean(axis=0)
                std = gb.std(axis=0)
    
    
                ax1.plot(bin_center, gb.T, color=color)
                ax1.plot(bin_center, mu, label=label, color="k")
                ax1.fill_between(bin_center, mu+std, mu-std, color="k", alpha=0.25, zorder=5)
                # break
                halved = mu.size // 2
                ax1.axhline(mu[halved:].mean(), color=color)
    
                # ax1.plot(bin_center, g.xs("all", level="bootstrap_id").squeeze(), ls="dotted", color=color)
    
                # Hist delays
                delays = df_metrics.xs((tag, mean), level=("tag", "mean"))["delay"]
                ax2.hist(delays + t_start, bins=t_bins, color=color, density=True, zorder=-4, rwidth=0.9, alpha=0.5)
    
    
    
                # rgb = np.asarray([0, 0, 0], dtype=float)
                # threshold = 8
                #
                # import itertools
                # consecutive_True = lambda condition: [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
                #
                # std_latter = std[-std.size:].mean()
                # mu_latter = mu[-mu.size:].mean()
                # for (_, fr), delay in zip(gb.iterrows(), delays):
                #     index = (t_bins >= t_start + delay).argmax() # Gets first value that is larger than duration_pre
                #
                #     osc_over    = fr[:index] > (1.5*std_latter + mu_latter)
                #     osc_first_over = osc_over.argmax() # Gets the first True value
                #     osc_under   = fr[osc_first_over:index] < (1.5*std_latter + mu_latter)
                #     osc = True if np.count_nonzero(osc_first_over) > threshold and np.count_nonzero(osc_under) > threshold else False
                #
                #     above = fr[:index] > (3*std_latter + mu_latter)
                #     overshoot = True if np.any(above) and np.max(consecutive_True(above)) > threshold else False
                #
                #     undershoot = False if osc or overshoot else True
                #
                #     rgb += [overshoot, osc, undershoot]
                #
                # rgb /= len(gb)
                # # print(rgb)
                # ax1.scatter(t_start + delays.mean(), 10, color=rgb.reshape([1, 3]), marker="o", zorder=20)
                # ax1.text(t_start + delays.mean(), 10, Label[tag], va="bottom", ha="center")
                #
    
    
    
            ax1.set_ylim(bottom=0) 
            ax1.legend()
    
            save_figure(figname, fig)
    
    
    #===============================================================================
    # PLOT - ENTROPY OVER DELAY
    #=============================================================================== 
    if plot_entropy_over_delay:
        figname_preentropy = f"preentropy (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
        fig, ax_entropy = plt.subplots(num=figname_preentropy)
        ax_entropy.set(xlabel="Entropy [nats]", ylabel="Delay [ms]", ylim=ylim_delay)
        handles = []
        for tag in (mean_tag, std_tag, mean_std_tag):
    
            label = Label[tag]
            color = Color[tag]
            handles.append((mpatches.Patch(facecolor=color, label=label)))
            kwargs = {
                "levels": 4, 
                "fill": False, 
                "color": color,
            }
    
            for m, mean in enumerate(means):
                metrics_tmp = df_metrics.xs((tag, mean), level=("tag", "mean"))
                sns.kdeplot(x=metrics_tmp[entropy_tag], y=metrics_tmp[delay_tag], **kwargs, ax=ax_entropy)
        ax_entropy.legend(handles=handles)
    
    
    #===============================================================================
    # PLOT - TRANSIENT ESTIMATES
    #=============================================================================== 
    if plot_transient_estimates:
        figname_transient = f"Delay estimates (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
        fig, ax_transient = plt.subplots(num=figname_transient)
        ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay,)
        sns.violinplot(df_metrics, x="mean", y="delay", hue="tag", 
                       cut=0, density_norm="width", common_norm=True, 
                       hue_order=hue_order, ax=ax_transient, native_scale=True,)
        handles = []
        for tag in hue_order:
            handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
        plt.legend(handles=handles)
        save_figure(figname_transient, fig)
    
    
    
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
        save_figure(figname_transient_means, fig)
    
    

    #===============================================================================
    # PLOT -  TIME TO FIRST SPIKE
    #===============================================================================
    if plot_time_to_first_spike:
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
        figname_spike = f"Spike to first spike (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms)"
        fig, ax_firstspike = plt.subplots(num=figname_spike)
        ax_firstspike.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]")
                
        bins = np.arange(0, 500, 1, dtype=float) + params.warmup+params.duration_pre
        bins = np.asarray(bins)
        print(bins)
        bins = range(int(params.warmup+params.duration_pre), int(params.warmup+params.duration_pre+params.duration_post), 1)
        

        sns.violinplot(df, x="mean", y="firstspike", hue="tag", 
                           cut=0, density_norm="width", common_norm=True, 
                           hue_order=hue_order, ax=ax_firstspike)
        
        handles = []
        for tag in hue_order:
            handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
        plt.legend(handles=handles)
        
    if plot_time_to_first_spike:    
        figname_spike = f"Mean first spike timing (FR: {pre_FR} to {post_FR})"
        fig, ax_firstspike = plt.subplots(num=figname_spike)
        ax_firstspike.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]")
        
        sns.lineplot(
            data=df,
            x="mean",
            y="firstspike",
            hue="tag",
            marker="o",
            errorbar=("se", 1.96), # 95% interval
            estimator="mean",
            hue_order=hue_order,
            ax=ax_firstspike,
        )
        sns.lineplot(
            data=df,
            x="mean",
            y="firstspike",
            hue="tag",
            marker="^",
            linestyle="--",
            # errorbar=("pi", 50),
            errorbar=None,
            estimator="median",
            hue_order=hue_order,
            ax=ax_firstspike,
        )
        labels = []
        for stat in ("mean", "median"):
            for tag in hue_order:
                labels.append(rf"{stat.capitalize()} delay ({Label[tag]})")
        handles, _ = ax_firstspike.get_legend_handles_labels()
        ax_firstspike.legend(handles, labels)
    
    
        # save_figure(figname_spike, fig)

    #===============================================================================
    # PLOT - DISTRIBUTION OF FIRST AND SECOND SPIKE
    #===============================================================================
    if plot_dist_first_and_second_spike:
        time_to_first_spike = []
        time_to_second_spike = []
        df = pd.DataFrame(columns=["firstspike", "tag", "mean"])

        
        with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
            for tag in (mean_tag, std_tag, mean_std_tag):
                for m, mean in enumerate(means):
                    rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                    run_ids = get_run_ids(rows, params, tag)
                    
                    
                    all_first_spike = [] # All spikes 
                    all_second_spike = [] # All spikes 
                    all_third_spike = [] # All spikes 
                    for run_id in run_ids:    
                        run_first_spike = []
                        run_second_spike = []
                        run_third_spike = []
                        spikes_by_sender = hfile.get_node(hfile.data, f"run{run_id}").spikes_by_sender.read()
                        for spikes in spikes_by_sender:
                            spikes_tmp = spikes[spikes >= params.warmup+params.duration_pre]
                            if len(spikes_tmp) > 0:
                                run_first_spike.append(spikes_tmp[0])
                            if len(spikes_tmp) > 1:
                                run_second_spike.append(spikes_tmp[1])
                            else:
                                run_second_spike.append(np.nan)
                            if len(spikes_tmp) > 2:
                                run_third_spike.append(spikes_tmp[2])
                            else:
                                run_third_spike.append(np.nan)
                        all_first_spike.extend(run_first_spike)
                        all_second_spike.extend(run_second_spike)
                        all_third_spike.extend(run_third_spike)
                                
                                
                    new_rows = pd.DataFrame({"firstspike": all_first_spike, "secondspike": all_second_spike, "thirdspike": all_third_spike})
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], range(len(all_first_spike))],
                        names=["tag", "mean", "neuron_id"]
                    )
                    time_to_first_spike.append(new_rows)


        df = pd.concat(time_to_first_spike)
        figname_spike = f"Spike to first spike (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms)"
        fig, ax_firstspike = plt.subplots(num=figname_spike)
        ax_firstspike.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]")
                
        bins = np.arange(0, 500, 1, dtype=float) + params.warmup + params.duration_pre
        bins = np.asarray(bins)
        bins = range(int(params.warmup+params.duration_pre), int(params.warmup+params.duration_pre+params.duration_post), 1)
        
        for (tag, mean), gb in df.groupby(level=("tag", "mean")):
            plt.figure(f"Test - {tag} {mean}")
            plt.hist(gb["firstspike"], bins=bins, density=True)
            plt.errorbar(gb["firstspike"].mean(), 0.011, xerr=gb["firstspike"].std(ddof=1), marker="o")
            plt.hist(gb["secondspike"], bins=bins, density=True, rwidth=0.9, alpha=0.9)
            plt.errorbar(gb["secondspike"].mean(), 0.010, xerr=gb["secondspike"].std(ddof=1), marker="o")
            plt.hist(gb["thirdspike"], bins=bins, density=True, rwidth=0.75, alpha=0.75)
            plt.errorbar(gb["thirdspike"].mean(), 0.009, xerr=gb["thirdspike"].std(ddof=1), marker="o")

            plt.ylim(0, 0.0125)
            plt.xlim(495, 1200)
            plt.legend()
            
        sns.violinplot(df, x="mean", y="firstspike", hue="tag", 
                           cut=0, density_norm="width", common_norm=True, 
                           hue_order=hue_order, ax=ax_firstspike)
        
        handles = []
        for tag in hue_order:
            handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
        plt.legend(handles=handles)
        
        # save_figure(figname_spike, fig)
            
    # pre_means = np.zeros(len(means))
    # post_means = np.zeros(len(means))
    # pre_stds = np.zeros(len(means))
    # post_stds = np.zeros(len(means))
    # # for delta in (mean_tag, ):
    # fig, axes = plt.subplots(nrows=2)
    # for delta in (mean_tag, std_tag, mean_std_tag):
    #     for m, mean in enumerate(means):
    #         pre_means[m] = mean
    #         pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=params.dt).root, 2)
    #         pre_stds[m] = pre_std
    #
    #         if delta == mean_tag:
    #             post_means[m] = round(siegert.find_parameter(pre_std, target_FR=post_FR, given_parameter=std_tag, dt=params.dt).root, 2) # ie delta mean
    #             post_stds[m] = pre_std
    #         elif delta == std_tag:
    #             post_means[m] = mean
    #             post_stds[m] = round(siegert.find_parameter(mean, target_FR=post_FR, dt=params.dt).root, 2) # ie delta std
    #         elif delta == mean_std_tag:
    #             post_std = round(siegert.find_parameter(mean, target_FR=post_FR, dt=params.dt).root, 2)
    #             post_stds[m] = pre_std + (post_std - pre_std) / 2
    #             post_mean  = round(siegert.find_parameter(pre_std, target_FR=post_FR, dt=params.dt, given_parameter=std_tag).root, 2)
    #             post_means[m] = mean + (post_mean - mean) / 2
    #         else:
    #             raise ValueError("No valid delta chosen")
    #
    #     if delta == mean_tag:
    #         ls = "solid"
    #         marker = "o"
    #     elif delta == std_tag:
    #         ls = "dashed"
    #         marker = ">"
    #     else:
    #         ls = "dotted"
    #         marker = "x"
    #
    #     axes[0].plot(pre_means, pre_stds, marker=marker, ls=ls)
    #     axes[0].plot(post_means, post_stds, marker=marker, ls=ls)
        
        axes[1].plot(pre_means, post_means/pre_means, marker="o", ls=ls)
        axes[1].plot(pre_means, post_stds /pre_stds, marker="x", ls=ls)
        

    #===============================================================================
    # PLOT - EMD
    #===============================================================================
    # TODO: MEANING?
    # from scipy.stats import wasserstein_distance
    # df = pd.DataFrame(columns=["emd", "tag", "mean"])
    #
    # with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
    #     for tag in (mean_tag, std_tag):
    #         for m, mean in enumerate(means):
    #             rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
    #             rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
    #             run_ids = rows_filtered[id_tag]
    #
    #             # INSERT HERE
    #             emds = np.zeros(len(run_ids))
    #             for r, run_id in enumerate(run_ids):
    #                 Vdist_pre  = hfile.get_node(hfile.data, f"run{run_id}").dist_pre.read()
    #                 Vdist_post = hfile.get_node(hfile.data, f"run{run_id}").dist_post.read()
    #
    #                 emd = wasserstein_distance(Vdist_pre, Vdist_post)
    #                 emds[r] = emd
    #
    #             new_rows = pd.DataFrame({
    #                 "emd": emds,
    #                 "tag": [tag] * len(emds),
    #                 "mean": [mean] * len(emds),
    #             })
    #             df = pd.concat([df, new_rows], ignore_index=True)
    #
    #     figname_spike = f"Earth-Mover-Distance between p(V) pre and post (FR: {pre_FR} to {post_FR})"
    #     fig = plt.figure(figname_spike)
    #     sns.violinplot(df, x="mean", y="emd", hue="tag")
    #     plt.xlabel(r"mean drive $\mu_{pre}$")
    #     plt.ylabel("emd [au?]")
    #     # plt.xticks(plt.xticks()[0], plt.xticks()[1])
    #     handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
    #                mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
    #     plt.legend(handles=handles)
    #     # save_figure(figname_spike, fig)



               
        







#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()


