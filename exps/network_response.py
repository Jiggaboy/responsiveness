#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History:
    - 0.2a: Added to plot the transients.
    - 0.2b: Refactor to use general function for plotting.

"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2b'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
import seaborn as sns


from constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color, hue_order
from config import load_config
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, get_run_ids, get_spikes_by_sender, exc_tag, inh_tag

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import bootstrap, get_tbins, get_tstart

from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm, spikecount_to_FR

from cplot.aux import plot_axvline_at_change, plot_FRs
from cplot.plot_constants import xlabel_time, ylabel_fr

#===============================================================================
# CONTROL VARIABLES
#===============================================================================

plot_rate_and_delays = True
# plot_rate_and_delays = False

plot_transient_estimates = True
plot_transient_estimates = False



#===============================================================================
# CONSTANTS
#===============================================================================
tags = (mean_tag, std_tag, mean_std_tag)
# tags = (mean_tag, )

pre_FR = 5.
post_FR = 10.

means = [220, ]
means = [260, ]
means = np.arange(220, 320+1, 40.)
    

bootstraps = 100     #50
samples_per_strap = 25


ylim_delay = (0, 100)
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer  
def main():
    control, params = load_config(is_network=True)  
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_metrics = []
        Erates = []
        Irates = []
        
        
        ##### ALL ANALYSES ######################################
        for tag in tags:
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)
                
                
                t_bins, Edelay_estimates, Epopulation_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=exc_tag)
                t_bins, Idelay_estimates, Ipopulation_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=inh_tag)
                
                
                    
                new_rows = pd.DataFrame({
                    delay_tag: Edelay_estimates,
                })
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(Edelay_estimates))],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_metrics.append(new_rows)
                
                # Extend the array of firing rates
                new_rows = pd.DataFrame(Epopulation_FR)
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(Epopulation_FR.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                Erates.append(new_rows)
                
                # Extend the array of firing rates
                new_rows = pd.DataFrame(Ipopulation_FR)
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(Ipopulation_FR.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                Irates.append(new_rows)
                
                
        df_Erates = pd.concat(Erates)
        df_Irates = pd.concat(Irates)
        df_metrics = pd.concat(all_metrics)        
        
    
    #===============================================================================
    # PLOT - FIRING RATE AND INDIVIDUAL DELAY ESTIMATES
    #===============================================================================
    if plot_rate_and_delays:
        t_start = get_tstart(params)
        t_bins  = get_tbins(params)
        for m, mean in enumerate(means):
            figname = f"Network: Firing rates (mean: {mean}; J: {params.J})"
            fig, ax = plt.subplots(num=figname)
            ax.set(
                xlabel=xlabel_time, ylabel=ylabel_fr,
                xlim=(params.warmup+params.duration_pre - 10, params.warmup+params.duration_pre + 125)
            )
            
        
            # Indicate the time point of change
            plot_axvline_at_change(params, control, ax)      
            plot_FRs(mean, df_Erates, t_bins, ax)
            plot_FRs(mean, df_Irates, t_bins, ax, ls="dashed")

            ax.set_ylim(bottom=0) 
            ax.legend()
        
            # save_figure(figname, fig)


    

    #===============================================================================
    # PLOT - TRANSIENT ESTIMATES
    #=============================================================================== 
    if plot_transient_estimates:
        figname_transient = f"Delay estimates (FR: {pre_FR} to {post_FR}; stim: {params.stim_reps} with {params.stim_duration}ms and break {params.break_duration}ms)"
        fig, ax_transient = plt.subplots(num=figname_transient)
        ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay,)

        sns.violinplot(df_metrics, x="mean", y="delay", hue="tag", 
                       cut=0, density_norm="width", common_norm=True, 
                       hue_order=hue_order, ax=ax_transient, native_scale=True, )
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
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
