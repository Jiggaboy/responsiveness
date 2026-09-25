#!/usr/bin/env python3
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
import pathlib
import shutil
import seaborn as sns

from config import load_config, Control, Params
from constants import DATA_DIR, mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order

from lib.analysis import bootstrap, get_tbins, get_tstart, get_transient
from lib.conversion import spikecount_to_FR
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids
import lib.nest_interface as nif
from lib.util import save_figure

from cplot.plot_constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays

#===============================================================================
# CONSTANTS
#===============================================================================

figsize = (17.6*cm, 18*cm)
fname = "suppfigure5_stats"

tags = (mean_tag, std_tag, mean_std_tag)
# tags = (mean_tag, )
force = False
# force = True

control = Control()
control.brief_stimulus = True
params = Params(control)


pre_FR  = 5
post_FR = 10

stim_reps = np.asarray([1, 2, 5])
# stim_reps = np.asarray([5, ])
stim_durations = np.round(np.asarray([nif.tau / 3, nif.tau / 2, nif.tau, nif.tau * 2]))
# stim_durations = np.round(np.asarray([nif.tau / 2, nif.tau]))
# stim_durations = np.round(np.asarray([nif.tau, ]))


bootstraps = 100
samples_per_strap = 50
    

means = np.asarray([220., 260., 300.])
# means = np.arange(220, 320+1, 40.)
means = np.arange(220, 320+1, 20.)


fn_id    = f"_{pre_FR}_{post_FR}"
fn_rate  = "recovery_dist_dfrate" + fn_id
fn_delay = "recovery_dist_dfrec"  + fn_id
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    
    #===============================================================================
    # DATA
    #===============================================================================
    control = Control()
    control.brief_stimulus = True
    params = Params(control)
    params.stim_duration = stim_durations[0]
    params.stim_reps = stim_reps[0]
        
    
    metadata = {
        "pre_FR": pre_FR, "post_FR": post_FR,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap, "hist_binwidth": params.hist_binwidth,
        "means": means,
        "stim_durations": stim_durations,
        "stim_reps": stim_reps,
    }
    
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
                    except ValueError:
                        logger.info("Error while comparing metadata. Reset...")
                        equal_r, equal_d = False, False
                else:
                    try:
                        equal_r = df_rates.attrs[key] == value
                        equal_d = df_rates.attrs[key] == value
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
            params.stim_duration = stim_duration
            params.stim_reps = stim_rep
            base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
            tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
            
            with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
                for tag in (mean_tag, std_tag, mean_std_tag):
                # for tag in (std_tag, ):
                    for m, mean in enumerate(means):
                        logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                        run_ids = get_run_ids(rows, params, tag)
                        bwt_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
                
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
        df_rates = pd.concat(all_rates)
        df_delays = pd.concat(all_metrics)
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            df_rates.attrs[key] = value
            df_delays.attrs[key] = value
        df_rates.to_pickle(fn_rate)
        df_delays.to_pickle(fn_delay)
            
            

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=3, ncols=3)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.94,
        wspace=0.1,
        hspace=0.3,
    )
    axes = gs.subplots(sharex=True, sharey=True)

        
    # Filter for pre_FR and post_FR
    df_delays_tmp = df_delays.xs((pre_FR, post_FR), level=("pre_FR", "post_FR"))
        
    for (tag, stim_rep), gb in df_delays_tmp.groupby(level=("tag", "stim_reps")):
        row = np.flatnonzero(stim_reps == stim_rep).item()
        col = np.flatnonzero(np.asarray(hue_order) == tag).item()

        ax = axes[row, col]
        ax.set_xticks(means[::2])
        
        ax_kwargs = {}
        title = ""
        if row == 0:
            title += f"Modality {Label[tag]}" + "\n"
        ax_kwargs["title"] = title + f"{stim_rep} Stimulation Pulse" + "s" * (stim_rep > 1)
        if col == 0:
            ax_kwargs["ylabel"] = ylabel_recovery
        if row == axes[:, 0].size - 1:
            ax_kwargs["xlabel"] = xlabel_drive
            
        ax.set(**ax_kwargs)
        
        palette = sns.light_palette(Color[tag], n_colors=4)
        vp = sns.violinplot(
            data = gb,
            x = "mean",
            y = "delay",
            hue = "stim_duration",
            inner = None,
            palette = palette,
            cut = 0, 
            density_norm = "width", 
            common_norm = True,
            native_scale = True,
            ax = ax,
        )
        vp.legend_.remove()
        
        if row == 0:
            labels = []
            handles, legend_labels = ax.get_legend_handles_labels()
            for label in legend_labels:
                labels.append(rf"{int(float(label))}ms")
            ax.legend(handles, labels, ncols=2, loc="upper center")
        
        #===============================================================================
        # STATISTICAL TESTS 
        #===============================================================================        
        ## Nonparametric
        ## Holm Bonferroni correction
        
        fontdict = {
            "y": 90, # height
            "horizontalalignment": "center",
            "fontweight": "bold",
            "fontfamily": "monospace",
            "fontsize": "xx-large",
        }
        
        from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon
        
        df_statistic = df_delays_tmp.xs((tag, stim_rep), level=("tag", "stim_reps"))
        
        print()
        print(f"{tag} with {stim_rep}")
        for mean, gb in df_statistic.groupby(level="mean"):
            # print(mean, tag, stim_rep)
            # if tag == std_tag:
            #     ax = ax_response_mean
            # elif tag == mean_tag:
            #     ax = ax_response_std
            # elif tag == mean_std_tag:
            #     ax = ax_response_both
            # else:
            #     raise ValueError
            # Kruskal-Wallis H-test for independent samples.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html?utm_source=chatgpt.com
            # The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. 
            # It is a non-parametric version of ANOVA.
            H, p = kruskal(
                *(x["delay"] for _, x in gb.groupby("stim_duration", sort=True))
            )
            print(f"Mean: {mean}")
            print("Kruskal-Wallis:", p)
            # print(p / 54)
        
        
            # Perform the Mann-Whitney U rank test on two independent samples.
            # Wilcoxon: Paired
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
            ps = []
            wis = []
            for stim_duration in stim_durations:
                if stim_duration == nif.tau:
                    continue
        
                st = mannwhitneyu(
                    gb.xs(nif.tau, level="stim_duration", drop_level=False),
                    gb.xs(stim_duration, level="stim_duration", drop_level=False),
                    alternative="greater", 
                )
                print(f"Significance ({nif.tau} vs {stim_duration}):", st.pvalue)
                ps.append(st.pvalue)
                
        
                # wil = wilcoxon(
                #     gb.xs(nif.tau, level="stim_duration", drop_level=False),
                #     gb.xs(stim_duration, level="stim_duration", drop_level=False),
                #     alternative="greater", 
                # )
                # print(f"wilcoxon ({nif.tau} vs {stim_duration}):", wil.pvalue)
                # wis.append(wil.pvalue)
                
                # st = mannwhitneyu(
                #     gb.xs(nif.tau, level="stim_duration", drop_level=False),
                #     gb.xs(stim_duration, level="stim_duration", drop_level=False),
                #     alternative="less", 
                # )
                # print("Significance: (lower)", st.pvalue)
            ps = np.asarray(ps)
            wis = np.asarray(wis)
            # 6 means, 3 tags, 3 stim_pulses, 4 durations (test vs 3)
            factor = 6 * 3 * 3 * 3
            # plt.figure(f"{tag} - {stim_rep}")
            ax.set_ylim(0, 105)
            fontdict["y"] = 80
            if np.all(ps < (0.001 / factor)):
                print(10*"^")
                print(3*"*")
                ax.text(mean, s=3*r"$\star\!\!$", **fontdict)
            elif np.all(ps < (0.01 / factor)):
                print(15*"v")
                print(2*"*")
                ax.text(mean,s=2*r"$\star\!\!$", **fontdict)
            elif np.all(ps < (0.05 / factor)):
                print(15*"x")
                print(1*"*")
                ax.text(mean, s=1*r"$\star$", **fontdict)
            else:
                pass
            
            # fontdict["y"] = 100
            # if np.all(wis < (0.0001 / factor)):
            #     print(15*"-")
            #     ax.text(mean, s=3*r"$\star\!\!$", **fontdict)
            # elif np.all(wis < (0.001 / factor)):
            #     print(10*"*")
            #     ax.text(mean,s=2*r"$\star\!\!$", **fontdict)
            # elif np.all(wis < (0.05 / factor)):
            #     print(5*"*")
            #     ax.text(mean, s=1*r"$\star$", **fontdict)
            # else:
            #     pass
    save_figure(fname, fig, is_latex=True)
    
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()