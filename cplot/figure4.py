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

from lib.analysis import bootstrap, get_tbins, get_response_kernels, get_response_kernel, get_tstart
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids, exc_tag, inh_tag

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)
ylim_delay = (0, 80)
    
force = False
force = True

pre_FR  =  5.
post_FR = 10.
# post_FR = 12.

J = 0.075

bootstraps        = 100
samples_per_strap =  10#50

means = np.asarray([220., 260., 300.])
means = np.arange(220, 320+1, 40.)
means = np.arange(220, 320+1, 140.)


plot_mean = 260.
plot_mean = 220.



fn_id    = f"_{pre_FR}"
fn_Erate  = "dfnetwork_Erate" + fn_id
fn_Irate  = "dfnetwork_Irate" + fn_id
fn_delay = "dfnetwork_delay" + fn_id
    
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config(is_network=True, no_stim=True)
    params.J = J
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
        
    metadata = {
        "pre_FR": pre_FR, "post_FR": post_FR,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": params.hist_binwidth,
        "means": means,
    }
    
        
    #===============================================================================
    # DATA COLLECTION
    overwrite = False
    if not force:
        try:        
            df_Erates = pd.read_pickle(fn_Erate)
            df_Irates = pd.read_pickle(fn_Irate)
            df_delays = pd.read_pickle(fn_delay)
        except FileNotFoundError:
            logger.info("File not found. Start analysis...")
            overwrite = True
        else:
            for df in [df_Erates, df_Irates, df_delays]:
                if overwrite:
                    break
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray) and len(value) > 1 and \
                        isinstance(df.attrs[key], np.ndarray):
                        try:
                            # Value needs only to be in the set
                            is_equal = all(np.isin(value, df.attrs[key]))
                        except ValueError:
                            logger.info("Error while comparing metadata. Reset...")
                            is_equal = False
                    else:
                        is_equal = df.attrs[key] == value
                        if not isinstance(is_equal, bool) and len(is_equal) > 1:
                            is_equal = False
                    if not is_equal:
                        overwrite = True
                        print("Old metadata, rewrite data...")
                        break
    
    t_bins = get_tbins(params)
    if force or overwrite:
        delays = []
        Erates = []
        Irates = []
            
        with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
            for tag in (mean_tag, std_tag, mean_std_tag):
            # for tag in (std_tag, ):
                for m, mean in enumerate(means):
                    logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                    rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                    run_ids = get_run_ids(rows, params, tag)
                    
                    t_bins, Edelay_estimates, Epopulation_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=exc_tag)
                    t_bins, Idelay_estimates, Ipopulation_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=inh_tag)
            
                    new_rows = pd.DataFrame({delay_tag: Edelay_estimates,})
                    new_rows.index = pd.MultiIndex.from_product(
                        [[tag], [mean], range(len(Edelay_estimates))],
                        names=["tag", "mean", "bootstrap_id"]
                    )
                    delays.append(new_rows)
        
                    
                
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
    
        # Conversion and save
        df_Erates = pd.concat(Erates)
        df_Irates = pd.concat(Irates)
        df_delays = pd.concat(delays)
        
        dfs = [df_Erates, df_Irates, df_delays]
        fns = [fn_Erate, fn_Irate, fn_delay]
        for df, fn in zip(dfs, fns):
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    value = np.asarray(value)
                df.attrs[key] = value
                df.to_pickle(fn)

    #===============================================================================
    # PLOTS
    #===============================================================================
    
    fig = plt.figure(figsize=figsize, num=f"{J}")

    gs = fig.add_gridspec(nrows=3, ncols=3) 
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        # bottom=0.1,
        # top=0.93,
        wspace=0.45,
        # hspace=0.3
    )      
    
    
    #===============================================================================
    ## FIRING RATE
    ax_kwargs = {
        "xlabel": "Time [ms]", "ylabel": "FR [Hz]", "ylim": (0, 12),
        "xlim": (params.warmup + params.duration_pre - 10, params.warmup + params.duration_pre + 100),
    }    
    axt_kwargs = {
        "ylabel": "Density of delays", "yticks": np.linspace(0, 0.5, 3), "ylim": (0, 0.5),
    }
    plot_kwargs = {"markersize": 2}
    
    ax = fig.add_subplot(gs[0, 1])
    ax.set(**ax_kwargs)
    axt = ax.twinx()   
    axt.set(**axt_kwargs)
    plot_axvline_at_change(params, control, ax)
    
    plot_FRs(plot_mean, df_Erates, t_bins, ax, **plot_kwargs)
    plot_FRs(plot_mean, df_Irates, t_bins, ax, ls="--", **plot_kwargs)
    hist_delays(plot_mean, df_delays, t_bins, axt, t_start=params.warmup+params.duration_pre)
    ax.legend()


    #===============================================================================
    ## DELAY OVER MEAN -- VIOLIN
    ax_transient = fig.add_subplot(gs[0, 2])
    title = f"Delay estimates\n(FR: {pre_FR} to {post_FR})"

    ax_transient.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay, title=title)
    ax_transient.set_xticks(means)
    sns.violinplot(df_delays, x="mean", y="delay", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order, ax=ax_transient, native_scale=True,)

    handles = []
    for tag in hue_order:
        handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
    plt.legend(handles=handles)
    

    #===============================================================================
    ## DELAY OVER MEAN -- MEAN AND MEDIAN
    ax_mean_delay = fig.add_subplot(gs[1, 0])
    ax_mean_delay.set(xlabel=r"Mean drive $\mu_{pre}$", ylabel="Delay [ms]", ylim=ylim_delay)

    sns.lineplot(
        data=df_delays,
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
        data=df_delays,
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
    ## COEFFICIENT OF VARIATION
    t_start = get_tstart(params)
    buffer = 0.2
    index = (t_bins >= t_start + buffer * params.duration_post).argmax() - 1 # Gets first value that is larger than t_start
    
    
    
    ax_CV = fig.add_subplot(gs[1, 1])
    ax_CV.set(ylabel=r"Mean drive $\mu_{pre}$", xlabel="CV")
    ax_CV.set_yticks(means)
    
    # df_Erates, t_bins
    Erates_mean = df_Erates.loc[:, index:].mean(axis=1)
    Erates_std  = df_Erates.loc[:, index:].std(axis=1)
    
    
    Erates_CV = Erates_std / Erates_mean
    Erates_CV = Erates_CV.reset_index(name="CV")
    # ax_CV.set(xlabel=r"Mean drive $\mu_{pre}$")
    # ax_CV.set_xticks(ticks=np.arange(len(means)), labels=means)
    sns.violinplot(Erates_CV, x="CV", y="mean", hue="tag", 
                   cut=0, density_norm="width", common_norm=True, 
                   hue_order=hue_order, ax=ax_CV, native_scale=True, orient="h")

    handles = []
    for tag in hue_order:
        handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
    plt.legend(handles=handles)
    

    #===============================================================================
    ## RASTER PLOT
    ax_raster = fig.add_subplot(gs[1, 2])
    ax_raster.set(xlabel="Time [ms]", ylabel="Neuron ID", xlim=(1000, 1050))
    
    tmp_tag = "std"
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=plot_mean)
        run_ids = get_run_ids(rows, params, tmp_tag)
        target = hfile.get_node(hfile.data, f"run{run_ids[0]}")
        
        for subgroup in (exc_tag, inh_tag):
            color = Color[tmp_tag] if subgroup == exc_tag else "k"
            
            subtarget = target[subgroup]
            spike_times = subtarget.spikes.read()
            senders     = subtarget.senders.read()
            
            ax_raster.scatter(spike_times, senders, color=color, marker=".", s=1)
    
        
        
        
    
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
