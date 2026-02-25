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


import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


from constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color
from config import load_config
import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient

from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm, spikecount_to_FR


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

plot_rate_and_delays = True
# plot_rate_and_delays = False

plot_entropy_over_delay = True
plot_entropy_over_delay = False

plot_transient_estimates = True
plot_transient_estimates = False



#===============================================================================
# CONSTANTS
#===============================================================================
hist_binwidth = 2. #ms

bootstraps = 50     #50
samples_per_strap = 20 #25
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  # .6s per seed (25 straps x 10 samples)
def main():
    control, params = load_config()
    t_bins = np.arange(0., params.duration_pre+params.duration_post+hist_binwidth, float(hist_binwidth)) + params.warmup
    
    pre_FR = 2.
    post_FR = 4.
    pre_FR = 5.
    post_FR = 10.
    # pre_FR = 4.
    # post_FR = 6.
    # post_FR = 12.
    # pre_FR = 4.
    # post_FR = 2.
    means = np.arange(240, 320+1, 130.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)

    
    with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
        all_rates = []
        all_runs_delays = []
        
        
        ##### ALL ANALYSES ######################################
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                if tag in (mean_tag, std_tag):
                    rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]] 
                elif tag == mean_std_tag:
                    mask = np.logical_and(rows[f"pre_{mean_tag}"] != rows[f"post_{mean_tag}"], rows[f"pre_{std_tag}"] != rows[f"post_{std_tag}"])
                    rows_filtered = rows[mask]
                else:
                    raise ValueError("No valid tag given...")
                run_ids = rows_filtered[id_tag]
                # print(len(run_ids), np.sum(np.asarray(run_ids)))
                # continue
                
                # DELAY ACROSS ALL RUNS
                spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
                index = (t_bins >= params.warmup+params.duration_pre).argmax() # Gets first value that is larger than duration_pre
        
                _, delay_all_runs = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                
                
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
                    t_start = params.warmup + params.duration_pre + (control.double_step * params.delta_step)
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
        t_start = params.warmup + params.duration_pre + (control.double_step * params.delta_step)
        for m, mean in enumerate(means):
            figname = f"Firing rates (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR})"
            fig, ax1 = plt.subplots(num=figname)
            ax1.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
                    xlim=(params.warmup+params.duration_pre - 10, params.warmup+params.duration_pre + 125))
        
            # Indicate the time point of change
            ax1.axvline(params.warmup+params.duration_pre, color="red", zorder=10, ls="--")
            if control.double_step:
                ax1.axvline(params.warmup+params.duration_pre+params.delta_step, color="red", zorder=10, ls="--")
                ax1.set_xticks(list(plt.xticks()[0]) + [params.warmup+params.duration_pre, params.warmup+params.duration_pre+params.delta_step], list(plt.xticks()[0]) + [r"$t_\Delta$", r"$t_\Delta'$"])  
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
 
                ax1.plot(bin_center, g.xs("all", level="bootstrap_id").squeeze(), ls="dotted", color=color)
        
                delays = df_metrics.xs((tag, mean), level=("tag", "mean"))["delay"]
                ax2.hist(delays + t_start, bins=t_bins, color=color, density=True, zorder=-4, rwidth=0.9, alpha=0.5)
        
        
                
                rgb = np.asarray([0, 0, 0], dtype=float)
                threshold = 8
                
                import itertools
                consecutive_True = lambda condition: [ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
                
                std_latter = std[-std.size:].mean()
                mu_latter = mu[-mu.size:].mean()
                for (_, fr), delay in zip(gb.iterrows(), delays):
                    index = (t_bins >= t_start + delay).argmax() # Gets first value that is larger than duration_pre

                    osc_over    = fr[:index] > (1.5*std_latter + mu_latter)
                    osc_first_over = osc_over.argmax() # Gets the first True value
                    osc_under   = fr[osc_first_over:index] < (1.5*std_latter + mu_latter)
                    osc = True if np.count_nonzero(osc_first_over) > threshold and np.count_nonzero(osc_under) > threshold else False
                    
                    above = fr[:index] > (3*std_latter + mu_latter)
                    overshoot = True if np.any(above) and np.max(consecutive_True(above)) > threshold else False
                    
                    undershoot = False if osc or overshoot else True
                    
                    rgb += [overshoot, osc, undershoot]
                    
                    # print(fr)
                    # print(delay)
                    # print(std_latter)
                    # print(above)
                    
                rgb /= len(gb)
                print(rgb)
                ax1.scatter(t_start + delay, 10, color=rgb.reshape([1, 3]), marker="o", zorder=20)
                ax1.text(t_start + delay, 10, Label[tag], va="bottom", ha="center")
                    
                    
        
            ax1.set_ylim(bottom=0) 
            ax1.legend()
        
            save_figure(figname, fig)
    
    
    #===============================================================================
    # PLOT - ENTROPY OVER DELAY
    #=============================================================================== 
    if plot_entropy_over_delay:
        figname_preentropy = f"preentropy (FR: {pre_FR} to {post_FR})"
        fig, ax_entropy = plt.subplots(num=figname_preentropy)
        ax_entropy.set(xlabel="Entropy [nats]", ylabel="Delay [ms]", ylim=(0, 80))
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
        figname_transient = f"Delay estimates (FR: {pre_FR} to {post_FR})"
        # fig = plt.figure(figname)
        fig, ax_transient = plt.subplots(num=figname_transient)
        ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=(0, 80),)
        ax_transient.set_xticks(ticks=np.arange(len(means)), labels=means)
        sns.violinplot(df_metrics, x="mean", y="delay", hue="tag", cut=0, density_norm="width", common_norm=True)
        handles = []
        for tag in (mean_tag, std_tag, mean_std_tag):
            handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
        plt.legend(handles=handles)
        # save_figure(figname, fig)
        



                
    #===============================================================================
    # PLOT -  TIME TO FIRST SPIKE
    #===============================================================================
    # df = pd.DataFrame(columns=["firstspike", "tag", "mean"])
    #
    # with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
    #     for tag in (mean_tag, std_tag):
    #         for m, mean in enumerate(means):
    #             rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
    #             rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
    #             run_ids = rows_filtered[id_tag]
    #
    #             first_spike = []
    #             for run_id in run_ids:
    #                 spikes_by_sender = hfile.get_node(hfile.data, f"run{run_id}").spikes_by_sender.read()
    #                 for spikes in spikes_by_sender:
    #                     spikes_tmp = spikes[spikes >= params.warmup+params.duration_pre]
    #                     if len(spikes_tmp) > 0:
    #                         first_spike.append(spikes_tmp[0])
    #             new_rows = pd.DataFrame({
    #                 "firstspike": np.asarray(first_spike)-params.warmup-params.duration_pre,
    #                 "tag": [tag] * len(first_spike),
    #                 "mean": [mean] * len(first_spike)
    #             })
    #             df = pd.concat([df, new_rows], ignore_index=True)
    #
    # figname_spike = f"Spike to first spike (FR: {pre_FR} to {post_FR})"
    #
    # fig, ax_firstspike = plt.subplots(num=figname_spike)
    # ax_firstspike.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Time to first spike [ms]")
    #
    # sns.violinplot(df, x="mean", y="firstspike", hue="tag")
    # handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
    #            mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
    # plt.legend(handles=handles)
    # save_figure(figname_spike, fig)
        
        

    #===============================================================================
    # PLOT - EMD
    #===============================================================================
    # TODO: MEANING?
    # from scipy.stats import wasserstein_distance
    # df = pd.DataFrame(columns=["emd", "tag", "mean"])
    #
    # with ResponseHdf5(params.filename, "a", metadata=params.metadata) as hfile:
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
