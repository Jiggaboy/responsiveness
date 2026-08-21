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


from plot_constants import mean_tag, std_tag, delay_tag, entropy_tag, mean_std_tag, Label, Color, hue_order
from config import load_config
import lib.nest_interface as nif
from lib.analysis import bootstrap, get_tbins, get_tstart
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from lib import siegert
from lib.util import pairwise, save_figure, functimer

from cplot.constants import *
from cplot.aux import plot_axvline_at_change, plot_FRs, hist_delays


#===============================================================================
# CONTROL VARIABLES
#===============================================================================

figsize = (17.6*cm, 10*cm)
fname = "suppfigure3_autocorrelation"

xlabel_acf = "Time lag [ms]"
ylabel_acf = "Correlation coefficient"
ax_kwargs = {
    "xlim": (0, 50),
    "ylim": (-0.25, 1.),
}
plot_rate_and_delays = True
plot_rate_and_delays = False

ylim_delay = (0, 125)

#===============================================================================
# CONSTANTS
#===============================================================================
hist_binwidth = 2. #ms

bootstraps          = 10
samples_per_strap   = 50


pre_FR = 2.
post_FR = 4.

pre_FR = 5.
post_FR = 10.

means = np.arange(220, 320+1, 20.)
# means = np.arange(240, 290+1, 10.)
# means = np.append(means, 320.)

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

@functimer
def main():
    control, params = load_config(is_network=False, no_stim=True)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    

    all_metrics = []
    all_rates = []
    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        for tag in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                run_ids = get_run_ids(rows, params, tag)
                t_bins, delay_estimates, population_FR = bootstrap(hfile, run_ids, params, rep=bootstraps, samples_per_strap=samples_per_strap)
    
                start_index = (t_bins > get_tstart(params)).argmax()
    
                new_rows = pd.DataFrame({delay_tag: delay_estimates,})
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(len(delay_estimates))],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_metrics.append(new_rows)
    
                # Extend the array of firing rates
                new_rows = pd.DataFrame(population_FR[:, start_index-1:]) # Shape Bootstraps x time
                new_rows.index = pd.MultiIndex.from_product(
                    [[tag], [mean], range(population_FR.shape[0])],
                    names=["tag", "mean", "bootstrap_id"]
                )
                all_rates.append(new_rows)
    
    df_rates = pd.concat(all_rates)
    df_delays = pd.concat(all_metrics)
    

    
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Autocorrelation Functions" + "\n"+ r"FR: $5 \rightarrow 10$Hz")
    gs = fig.add_gridspec(nrows=2, ncols=3)
    fig.subplots_adjust(
        left=0.1,
        right=0.96,
        bottom=0.1,
        top=0.82,
        wspace=0.2,
        hspace=0.5
    )
    
    for m, mean in enumerate(means):
        row = m // 3
        col = m % 3
        ax = fig.add_subplot(gs[row, col])
        
        title = r"$\mu_{pre}$" + f"={int(mean)}pA"
        
        # TODO: Set the Figure title
        # plt.figure(f"Autocorrelation {mean}")
        ax.set(title=title, **ax_kwargs, xlabel=xlabel_acf)
        # if row == 1:
        #     ax.set(xlabel=xlabel_acf)
        # elif col == 2:
        #     ax.set(xlabel=xlabel_acf)
        # else:
        #     ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set(ylabel=ylabel_acf)
        else:
            ax.tick_params(labelleft=False)
        ax.axhline(0, ls="--", c="k")
        
        if m == 5:
            ax.set(xlim=(0, 130))
        
        # Selects only the rows of the currently iterated mean
        df = df_rates.xs(mean, level=("mean"))
        
        # Row: The FR per time points
        acfs = df.apply(lambda row: autocorrelation(row), axis=1)
        # Expands the tuples into columns, and renames them
        acfs = acfs.apply(pd.Series) 
        acfs.rename(columns={0: "index", 1: "ac"}, inplace=True)
        acfs["index"] *= hist_binwidth
        
        # Converts from having a long list in a single cell to having them in the full column.
        long_acfs = acfs.reset_index().explode(['index', 'ac'])

        sns.lineplot(
            data=long_acfs,
            x="index",
            y="ac",
            hue="tag",
            errorbar="sd",
            estimator="mean",
            hue_order=hue_order,
            palette=Color,
            legend=None,
            ax=ax,
        )
        
        
        if row == 0 and col == 1:
            handles = []
            for tag in hue_order:
                handles.extend([mpatches.Patch(facecolor=Color[tag], label=Label[tag])])
            plt.legend(handles=handles, loc="upper center") # ncols=3?
            
            
    save_figure(fname, fig, is_latex=True)
#===============================================================================
# METHODS
#===============================================================================
# TODO: Normalize correctly!
def autocorrelation(x, max_lag=None):
    """
    Compute normalized autocorrelation function.

    Returns
    -------
    lags : array of lag indices
    acf  : normalized autocorrelation
    """
    x = np.asarray(x, dtype=np.float64)
    x = (x - np.mean(x)) / x.std()

    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    corr = np.correlate(x, x, mode="full")
    corr = corr[n-1:n+max_lag]

    # unbiased normalization
    norm = np.arange(n, n-max_lag-1, -1)
    corr /= norm

    lags = np.arange(0, max_lag+1)
    return lags, corr


from scipy.optimize import curve_fit

def exp_decay(t, tau):
    return np.exp(-t / tau)


def double_exp_ordered(t, w, tau_fast, delta_tau, **kwargs):
    tau_slow = tau_fast + delta_tau
    return (
        w * np.exp(-t / tau_fast) +
        (1 - w) * np.exp(-t / tau_slow)
    )

def fit_row_ordered(row):
    t = np.asarray(row["index"], dtype=float)
    y = np.asarray(row["ac"], dtype=float)

    try:
        popt, pcov = curve_fit(
            double_exp_ordered,
            t,
            y,
            p0=[0.5, 1.0, 10.0],                 # w, tau_fast, delta_tau
            bounds=([0.0, 1e-8, 1e-8], [1.0, np.inf, np.inf]),
            maxfev=20000
        )

        w, tau_fast, delta_tau = popt
        tau_slow = tau_fast + delta_tau

        return pd.Series({
            "w": w,
            "tau_fast": tau_fast,
            "tau_slow": tau_slow,
            "success": True,
            "fail_reason": ""
        })

    except Exception as e:
        return pd.Series({
            "w": np.nan,
            "tau_fast": np.nan,
            "tau_slow": np.nan,
            "success": False,
            "fail_reason": str(e)
        })
if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
