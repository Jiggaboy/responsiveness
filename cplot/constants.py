#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

############ COLORS #########################

## Updated on Jan 24. 2025
KTH_GREEN = np.asarray((77, 160, 97)) / 255
KTH_PINK = np.asarray((232, 106, 88)) / 255
KTH_GREY = np.asarray((50, 50, 50)) / 255
KTH_BLUE = np.asarray((0, 71, 145)) / 255
KTH_LIGHT_BLUE = np.asarray((98, 152, 210)) / 255
KTH_YELLOW = np.asarray((255, 190, 0)) / 255

COLORS = (KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE, KTH_LIGHT_BLUE, KTH_YELLOW)
BS_COLOR = "magenta"

cm = 1 / 2.54

############ LIMITS #########################
max_delay = 75


rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.labelpad"] = 2
rcParams["errorbar.capsize"] = 2
rcParams["font.size"] = 8
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 0.6
rcParams["legend.handlelength"] = 1.25
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = 1
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["legend.framealpha"] = 1
rcParams["xtick.major.pad"] = 2
rcParams["ytick.major.pad"] = 2


title_style = {
    "fontsize": rcParams["axes.titlesize"],
    "fontweight": rcParams["axes.titleweight"],
    "fontfamily": rcParams["font.family"],
    "ha": "center",
    "va": "center"
}