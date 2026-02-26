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



#===============================================================================
# CONSTANTS
#===============================================================================
hist_binwidth = 2.5 #ms

bootstraps = 25     #50
samples_per_strap = 10 #25


pre_FR = 2.
post_FR = 4.
pre_FR = 5.
post_FR = 10.
# pre_FR = 4.
# post_FR = 6.
# post_FR = 12.
# pre_FR = 4.
# post_FR = 2.
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  # .6s per seed (25 straps x 10 samples)
def main():
    control, params = load_config()
    t_bins = np.arange(0., params.duration_pre+params.duration_post+hist_binwidth, float(hist_binwidth)) + params.warmup
    

    
    with ResponseHdf5(params.poisson_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
        all_rates = []
        all_runs_delays = []
        
        
        ##### ALL ANALYSES ######################################
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        run_ids = rows[id_tag]
        
        # DELAY ACROSS ALL RUNS
        spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
        index = (t_bins >= params.warmup+params.duration_pre).argmax() # Gets first value that is larger than duration_pre

        _, delay_all_runs = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
        
                
        new_rows = pd.DataFrame({delay_tag: [delay_all_runs * hist_binwidth]})
        all_runs_delays.append(new_rows)
                
        FR_all_runs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
        # Extend the array of firing rates
        new_rows = pd.DataFrame([FR_all_runs],
            index=pd.RangeIndex(1, name="bootstrap_id"),            
        )
        all_rates.append(new_rows)

        # Detailed feature analysis
        delay_estimates = np.zeros(bootstraps)
        population_FR   = np.zeros((bootstraps, t_bins.size-1))
        for b in range(bootstraps):
            np.random.shuffle(run_ids)
            samples = run_ids[:samples_per_strap] # Bootstrapping

            # DELAY 
            spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)
            index = (t_bins >= params.warmup + params.duration_pre).argmax() # Gets first value that is larger than duration_pre

            SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
            delay_estimates[b] = delay * hist_binwidth

            # FIRING RATE
            FRs = spikecount_to_FR(spikecounts_all_runs.mean(axis=0), params.N, hist_binwidth)
            population_FR[b] = FRs
            
            
        new_rows = pd.DataFrame(
            {delay_tag: delay_estimates}, 
            index=pd.RangeIndex(len(delay_estimates), name="bootstrap_id")
        )
        all_metrics.append(new_rows)
        
        # Extend the array of firing rates
        new_rows = pd.DataFrame(population_FR, 
            index=pd.RangeIndex(population_FR.shape[0], name="bootstrap_id")
        )
        all_rates.append(new_rows)
        
            
    df_rates = pd.concat(all_rates)
    df_metrics = pd.concat(all_metrics)
    df_all_runs_delays = pd.concat(all_runs_delays)
    
        
    
    #===============================================================================
    # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
    #=============================================================================== 
    if plot_rate_and_delays:
        figname = f"Firing rates (pre_FR: {pre_FR}; post_FR: {post_FR})"
        fig, ax1 = plt.subplots(num=figname)
        ax1.set(xlabel="Time [ms]", ylabel="Firing rate [Hz]",
                xlim=(params.warmup+params.duration_pre - 10, params.warmup+params.duration_pre + 125))
    
        # Indicate the time point of change
        ax1.axvline(params.warmup+params.duration_pre, color="red", zorder=10, ls="--")
        if control.brief_stimulus:
            ax1.axvline(params.warmup+params.duration_pre+params.stim_duration, color="red", zorder=10, ls="--")
            ax1.set_xticks(list(plt.xticks()[0]) + [params.warmup+params.duration_pre, params.warmup+params.duration_pre+params.stim_duration], list(plt.xticks()[0]) + [r"$t_\Delta$", r"$t_\Delta'$"])  
        else:
            ax1.set_xticks(list(plt.xticks()[0]) + [params.warmup+params.duration_pre, ], list(plt.xticks()[0]) + [r"$t_\Delta$", ])                
    
        ax2 = ax1.twinx()   
        ax2.set(ylabel="Density of delays", yticks=np.linspace(0, 0.5, 3), ylim=(0, 0.5))
    
        bin_center = (t_bins[:-1] + t_bins[1:]) / 2
        
        
        color = "tab:olive"
        label = "poisson"
        
        df = df_rates
        
        
        
        # Delay Estimation across all runs
        d = df_all_runs_delays[delay_tag].squeeze()
        ax1.axvline(d + params.warmup + params.duration_pre, c=color, lw=2, ls="--", zorder=15)
        
        # g: rows = sims, cols = points
        mu = df.mean(axis=0)
        std = df.std(axis=0)
        
        
        ax1.plot(bin_center, mu, label=label, color=color)
        ax1.fill_between(bin_center, mu+std, mu-std, color=color, alpha=0.25, zorder=-5)
        
        
        delays = df_metrics["delay"]
        ax2.hist(delays + params.warmup + params.duration_pre, bins=t_bins, color=color, density=True, zorder=-4, rwidth=0.9, alpha=0.5)

        
        ax1.set_ylim(bottom=0) 
        ax1.legend()
        
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
