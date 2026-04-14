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

from config import load_config
from constants import mean_tag, std_tag, delay_tag, mean_std_tag, Label, Color, hue_order

from lib.analysis import bootstrap
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, get_run_ids

from cplot.constants import *
from cplot.aux import plot_axvline_at_change

#===============================================================================
# CONSTANTS
#===============================================================================
figsize = (17.6*cm, 15*cm)


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config()
    
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=3) #width_ratios=()
    fig.subplots_adjust(
        # left=0.06,
        # right=0.95,
        # bottom=0.1,
        # top=0.93,
        # wspace=0.2,
        # hspace=0.3
    )
    
    ### Schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set(xlabel="Time [ms]", ylabel="FR [Hz]", ylim=(0, 3))
    ax.set_yticks([0, 1, 2], [0, r"$FR_{pre}$", r"$FR_{post}$"])

#===============================================================================
# METHODS
#===============================================================================

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
