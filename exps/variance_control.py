#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Aim is to plot the link between delay estimation and variance.
    This puts the test onto the artifact that larger variance may lead to shorter estimates.

Description:
    Derived from variance.py

History:
    v0.2a: Adjustment to allow the analysis for the network setup.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
import seaborn as sns


from config import load_config
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids, exc_tag

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient, get_tbins, get_tstart, bootstrap

from lib.conversion import spikecount_to_FR

from cplot.plot_constants import *

#===============================================================================
# CONTROL VARIABLES
#===============================================================================
figsize = (17.6*cm, 15*cm)
fname = "suppfigure1_variance"


is_network = True
is_network = False

plot_rate_and_delays = True
plot_rate_and_delays = False

force = False
force = True
#===============================================================================
# CONSTANTS
#===============================================================================
bootstraps        = 100
samples_per_strap =  10 if is_network else 50

    
pre_FR = 2.
post_FR = 4.

# pre_FR = 5.
# post_FR = 10.

# pre_FR = 4.
# # # post_FR = 6.
# post_FR = 12.
#
# pre_FR = 10.
# post_FR = 5.
means = np.arange(220, 320+1, 40.)
# means = np.arange(240, 290+1, 110.)
# means = np.append(means, 320.)

binwidths = np.asarray([1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])

control, params = load_config(is_network=is_network)
    
fn_id    = f"_{pre_FR}"
fn_var   = "dfvar" + fn_id



ylim_var = (0.22, 0.38)
ylabel = "Std of FR [Hz]"
    
ax_kwargs = {
    "xlabel": ylabel,
    "ylabel": ylabel_recovery,
    "xlim": (0.259, 0.32),
    "ylim": (0, 36),
}
line_kwargs = {
    "hue": "tag",
    "hue_order": hue_order,
    "palette": Color,
    "markers": True,
}
violin_kwargs = {
    "hue": "tag",
    "cut": 0,
    "common_norm": True,
    "density_norm": "width",
    "hue_order": hue_order,
    "palette": Color,
    "legend": None,
    "inner": None,
    "native_scale": True,
}
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
# def main():
#     control, params = load_config(is_network=is_network)
#     plt.figure()
#     t_start = get_tstart(params)
#     plt.axhline(t_start)
#     plt.axhline(499.95, c="yellow")
#     print(t_start)
#     for h, hist_binwidth in enumerate((1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6)):
#         params.hist_binwidth = hist_binwidth
#         t_bins = get_tbins(params)
#         index = (t_bins >= t_start).argmax() - 1 # Gets first value that is larger than t_start
#
#         plt.plot(t_bins, marker=".")
#         plt.scatter(index, t_bins[index], marker=h)
#         plt.scatter(index, t_bins[index-1], marker=h)
#         assert t_bins[index-1] == 499.95

def main():
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
        
    metadata = {
        "pre_FR": pre_FR, "post_FR": post_FR,
        "bootstraps": bootstraps, "samples_per_strap": samples_per_strap,
        "binwidths": binwidths,
        "means": means,
    }
    
        # Data collection
    overwrite = False
    if not force:
        try:
            df = pd.read_pickle(fn_var)
        except FileNotFoundError:
            logger.info("File not found. Start analysis...")
            overwrite = True
        else:    
            for key, value in metadata.items():
                if isinstance(value, np.ndarray) and len(value) > 1 and \
                    isinstance(df.attrs[key], np.ndarray) and len(df.attrs[key]) > 1:
                    try:
                        # Value needs only to be in the set
                        isequal = all(np.isin(value, df.attrs[key]))
                    except ValueError:
                        logger.info("Error while comparing metadata. Reset...")
                        isequal = False
                else:
                    # if len(df.attrs[key]) < value:
                    #     isequal = False
                    #     break
                    isequal = df.attrs[key] == value
                    if not isinstance(isequal, bool) and len(isequal) > 1:
                        isequal = False
                if not isequal:
                    overwrite = True
                    print("Old metadata, rewrite data...")
                    break
                
    
    if force or overwrite:
        with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
            #===============================================================================
            # MORE METHODS
            #===============================================================================    
            all_metrics = []
        
            for hist_binwidth in binwidths:
                params.hist_binwidth = hist_binwidth
                print(f"Binwidth: {params.hist_binwidth}")
                t_bins = get_tbins(params)
                t_start = get_tstart(params)
    
                #  Get spikes with buffer
                t_start_buffer = t_start + 0.25 * params.duration_post
                index_buffered = (t_bins >= t_start_buffer).argmax() # Index of the first value being larger than the buffered time.
                
                ##### ALL ANALYSES ######################################
                for tag in (mean_tag, std_tag, mean_std_tag):
                    for m, mean in enumerate(means):
                        logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                        run_ids = get_run_ids(rows, params, tag)
                        
                        subgroup = exc_tag if is_network else None
                        t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap, subgroup=subgroup)
                        
                        stds = population_FR[:, index_buffered:].std(axis=1, ddof=1)
                            
                        # Plots the sample standard deviation across bootstraps.
                        # plt.plot(population_FR.std(axis=0, ddof=1), c=Color[tag])
        
                        new_rows = pd.DataFrame({
                            "std": stds, "delay": delay_estimates,
                        })
                        new_rows.index = pd.MultiIndex.from_product(
                            [[tag], [mean], [hist_binwidth], range(len(stds))],
                            names=["tag", "mean", "binwidth", "bootstrap_id"]
                        )
                        all_metrics.append(new_rows)
                    # break
        # return
        
        # Conversion and save
        df = pd.concat(all_metrics)
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            df.attrs[key] = value
        df.to_pickle(fn_var)
    
    
    
        
    #===============================================================================
    # PLOTS - Up and down step
    #===============================================================================
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=(2, 1, 1, 1))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.93,
        wspace=0.2,
        hspace=.8,
    )
    
    axmean   = fig.add_subplot(gs[0, 0])
    axmean.set(title="Mean Recovery Time" + "\n", **ax_kwargs)
    axmean.set_yticks([0, 15, 30])
    axmean.set_xticks(np.linspace(0.26, 0.32, 4))
    
    axmedian = fig.add_subplot(gs[0, 1], sharex=axmean, sharey=axmean)
    axmedian.set(title="Median Recovery Time" + "\n", **ax_kwargs)
    
    nrows = 3
    ncols = 3
    gs_var = gs[1:, :].subgridspec(nrows=nrows, ncols=ncols,
       wspace=0.1,
       hspace=0.35,
    )
    indexoffset = len(fig.axes)
    
    
    
    
    
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
    
            if i == 0 and j == 0:
                ax = fig.add_subplot(gs_var[i, j])
            else:
                ax = fig.add_subplot(
                    gs_var[i, j],
                    sharex=axes[0, 0],
                    sharey=axes[0, 0],
                )
            axes[i, j] = ax
    
    
    
    for binwidth, g in df.groupby(level="binwidth"):
        # plt.figure(f"Variance (network={is_network}) and binwidth: {binwidth}")
        index = np.where(binwidths == binwidth)[0].item()
        ax = fig.axes[index + indexoffset]
        ax.set(title=f"Binwidth: {binwidth}ms")
        sns.violinplot(
            g, x="mean", y="std",
            ax = ax,
            **violin_kwargs,
        )
        # sns.pointplot(
        #     data=g,
        #     x="mean",
        #     y="std",
        #     hue="tag",
        #     estimator="mean",
        #     errorbar=None,
        #     dodge=True,
        #     linestyle="none",
        #     markers="_",
        #     ax=ax,
        #     legend=False,
        #     color="k",
        #     native_scale=True,
        # )
        if index == 4:
            handles = []
            for tag in hue_order:
                handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
            ax.legend(handles=handles, ncols=3, loc="upper center")
    
    
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.set(ylim = ylim_var)
            ax.set_xticks(means)
            if j == 0:
                ax.set(ylabel=ylabel)
            else:
                ax.tick_params(labelleft=False)
                ax.set_ylabel(None)
                
            if i == nrows - 1:
                ax.set(xlabel=xlabel_drive)
            else:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel(None)
                
                
    # Mean
    h = (
        df.reset_index()
            .groupby(['tag', 'mean', 'binwidth'], as_index=False)
            .agg(std=('std', 'mean'), delay=('delay', 'mean'))
    )
    # plt.figure(f"Mean: Delay over Variance (network={is_network})")
    sns.lineplot(h, x="std", y="delay", style="mean",
        ax = axmean,
        **line_kwargs,
    )
    update_legend(axmean)
    
    

    
    
    # Median
    m = (
        df.reset_index()
          .groupby(['tag', 'mean', 'binwidth'], as_index=False)
          .agg(std=('std', 'mean'), delay=('delay', 'median'))
    )
    # plt.figure(f"Median: Delay over Variance (network={is_network})")
    sns.lineplot(m, x="std", y="delay", style="mean",
        ax = axmedian,
        **line_kwargs,
    )
    update_legend(axmedian)
    
    
    save_figure(fname, fig, is_latex=True)

    # # Statistical tests
    # import scipy.stats as st
    # for tag in (mean_tag, std_tag, mean_std_tag):
    #     for m in means:
    #         samples = df_metrics.xs((tag, m), level=("tag", "mean"))["std"]
    #         res = st.shapiro(samples)
    #         print(f"std: {tag} {m}", res.statistic, res.pvalue)
    #
    # print("Kruskal-Wallis")
    # for m in means:
    #     df = df_metrics.xs(m, level=("mean"))
    #     arr2d = df["std"].unstack(level="tag")
    #     res = st.kruskal(arr2d.to_numpy().T)
    #     print(f"std: {tag} {m}", res.statistic, res.pvalue)
        
        
def update_legend(ax:object):
    handles, labels = ax.get_legend_handles_labels()

    # Replace section headers
    labels[0] = "Mod."
    for l, label in enumerate(labels[1:4]):
        labels[l+1] = Label[label]
    labels[4] = r"$\mu_{pre}$"
    
    # Format numerical labels
    labels[5:] = [f"{float(x):.0f}" for x in labels[5:]]
    
    ax.legend(
        handles, labels,
        ncols=2,
        loc="upper right",
        bbox_to_anchor=(0, 1, 1., 0.15),
    )

if __name__ == '__main__':
    main()
    plt.show()
