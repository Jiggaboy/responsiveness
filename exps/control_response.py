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


from constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color, hue_order
from config import load_config, NetworkParams
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from lib import siegert
from lib.rnn import RNN
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient, bootstrap

from lib.conversion import spikecount_to_FR

from cplot.aux import plot_axvline_at_change, plot_FRs


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



pre_FR = 5.
post_FR = 10.


# Set to the same values as in network_simulate.py
Imean_ext   = 260.
FR_I = pre_FR

means = [260., 300.,]
means = np.arange(220, 320+1, 40.)

tags = (mean_tag, std_tag, mean_std_tag)
tags = (mean_tag, std_tag, mean_std_tag)

# # Equivalent for network with J = 0.075
# pre_FR = 5.
# post_FR = 10.
# means = np.asarray([-2.5, 35])
#
# pre_mean  = -2.5
# post_mean =  35
#
# # Equivalent for network with J = 0.25
# # mean = 220
# pre_mean  = 132.5
# post_mean = 181
# # mean = 260
# pre_mean  = 132.5
# post_mean = 214
# # mean = 300
# # pre_mean  = 132.5
# # post_mean = 246
#
# # Equivalent for network with J = 0.25
# # mean = 220
# exp220 = {
#     mean_tag: (132.5, 145),
#     std_tag: (132.5, 181),
#     mean_std_tag: (132.5, 164),
# }
# # mean = 260
# exp260 = {
#     mean_tag: (172.5, 185),
#     std_tag: (172.5, 214),
#     mean_std_tag: (172.5, 199),
# }
# # mean = 300
# exp300 = {
#     mean_tag: (212.5, 225),
#     std_tag: (212.5, 246),
#     mean_std_tag: (212.5, 236),
# }
#
#
# exps = [exp220, exp260, exp300]
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer 
def main():
    control, params = load_config(no_stim=True)
    networkparams = NetworkParams(control)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    # tmp_filename = base_filename + f"_{pre_mean}_{post_mean}" + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    tmp_filename = base_filename + f"_control_{networkparams.J}" + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"

    rnn = RNN(networkparams, pre_FR, post_FR, FR_I, drive_Imean=Imean_ext)
    
    recoveries = []
    rates = []
    

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        for delta in tags:
            for mean in means:
                rnn.set_up_network(mean, delta)
                pre_mean, pre_std = rnn.pre_Esetpoint
                post_mean, post_std = rnn.post_Esetpoint
            
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, 
                                         pre_mean=pre_mean, post_mean=post_mean)
                run_ids = get_run_ids(rows, params, tag=None)
                t_bins, recovery_samples, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
                
                new_rows = pd.DataFrame({
                    delay_tag: recovery_samples,
                })
                new_rows.index = pd.MultiIndex.from_product(
                    [[delta], [pre_mean], range(len(recovery_samples))],
                    names=["tag", "mean", "bootstrap_id"]
                )
                recoveries.append(new_rows)
                
                # Extend the array of firing rates
                new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
                new_rows.index = pd.MultiIndex.from_product(
                    [[delta], [pre_mean], range(population_FR.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                rates.append(new_rows)
                
    df_recovery = pd.concat(recoveries)
    df_rates = pd.concat(rates)
    
    
    # all_metrics = []
    # all_rates = []
    # all_runs_delays = []   
    # for exp in exps:
    #     for tag, (pre_mean, post_mean) in exp.items():
    #         tmp_filename = base_filename + f"_{pre_mean}_{post_mean}" + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    #
    #         with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
    #
    #             ##### ALL ANALYSES ######################################
    #             rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=pre_mean, post_mean=post_mean)
    #             run_ids = get_run_ids(rows, params, tag=None)
    #             t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
    #
    #
    #             new_rows = pd.DataFrame({
    #                 delay_tag: delay_estimates,
    #             })
    #             new_rows.index = pd.MultiIndex.from_product(
    #                 [[tag], [pre_mean], range(len(delay_estimates))],
    #                 names=["tag", "mean", "bootstrap_id"]
    #             )
    #             all_metrics.append(new_rows)
    #
    #             # Extend the array of firing rates
    #             new_rows = pd.DataFrame(population_FR) # Shape Bootstraps x time
    #             new_rows.index = pd.MultiIndex.from_product(
    #                 [[tag], [pre_mean], range(population_FR.shape[0])],
    #                 names=["tag", "mean", "bootstrap_id"]
    #             )
    #             all_rates.append(new_rows)
    #
    #
    # df_rates = pd.concat(all_rates)
    # df_metrics = pd.concat(all_metrics)
    #

    
    
    #===============================================================================
    # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
    #=============================================================================== 
    if plot_rate_and_delays:
        for mean, gb in df_rates.groupby(level=("mean")):
            figname = f"Activity (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR})"
            fig, ax = plt.subplots(num=figname)
            ax.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
                    xlim=(params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125))
    
            plot_axvline_at_change(params, control, ax)
            
            plot_FRs(mean, df_rates, t_bins, ax)
        
            # bin_center = (t_bins[:-1] + t_bins[1:]) / 2
            #
            # for tag, gb_tmp in gb.groupby(level="tag"):
            #     label = Label[tag]
            #     color = Color[tag]
                
            
        
        # for exp in exps:
        #     pre_mean = exp[mean_tag][0] # Could be any tag
        #     t_start = params.warmup + params.duration_pre + (params.stim_reps * params.stim_duration + (params.stim_reps-1) * params.break_duration)
        #     figname = f"Firing rates (mean: {pre_mean}; pre_FR: {pre_FR}; post_FR: {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
        #     fig, ax1 = plt.subplots(num=figname)
        #     ax1.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
        #             xlim=(params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 125))
        #
        #     plot_axvline_at_change(params, control, ax1)              
        #
        #     ax2 = ax1.twinx()   
        #     ax2.set(ylabel="Density of delays", yticks=np.linspace(0, 0.5, 3), ylim=(0, 0.5))
        #
        #     bin_center = (t_bins[:-1] + t_bins[1:]) / 2
        #
        #     df = df_rates.xs(pre_mean, level=("mean"))
        #     for tag, gb in df.groupby(level="tag"):
        #         label = Label[tag]
        #         color = Color[tag]
        #
        #
        #         # g: rows = sims, cols = points
        #         mu = gb.mean(axis=0)
        #         std = gb.std(axis=0)
        #
        #
        #         ax1.plot(bin_center, mu, label=label, color=color)
        #         ax1.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=5)
        #         # break
        #         halved = mu.size // 2
        #         ax1.axhline(mu[halved:].mean())
        #
        #         # Hist delays
        #         delays = df_metrics.xs((tag, pre_mean), level=("tag", "mean"))["delay"]
        #         ax2.hist(delays + t_start, bins=t_bins, density=True, zorder=-4, rwidth=0.9, alpha=0.5, color=color)
        #
        #
        #
        #     ax1.set_ylim(bottom=0) 
        #     ax1.legend()
        #
        # # save_figure(figname, fig)

    
    #===============================================================================
    # PLOT - TRANSIENT ESTIMATES
    #=============================================================================== 
    if plot_transient_estimates:
        figname_transient = f"Network Control Delay estimates (FR: {pre_FR} to {post_FR})"
        fig, ax_transient = plt.subplots(num=figname_transient)
        ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay,)

        sns.violinplot(df_metrics, x="mean", y="delay", hue="tag", 
                       cut=0, density_norm="width", common_norm=True, 
                       hue_order=hue_order, ax=ax_transient, native_scale=True,)
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


