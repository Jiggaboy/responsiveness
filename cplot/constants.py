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
from matplotlib import rcParams

cm = 1 / 2.54
#===============================================================================
# KTH COLORS
#===============================================================================
## Updated on Aug 06. 2026
KTH_green       = "#4DA060"
KTH_turquoise   = "#339C9C"
KTH_brick       = "#E86A58"
KTH_yellow      = "#FFBE00"

KTH_blue        = "#004791"
KTH_navy        = "#000061"
KTH_sky         = "#6298D2"

KTH_grey        = "#A5A5A5"

############ COLORS #########################
EXC_NEURON = "#78001aff"
INH_NEURON = "#004791ff"

COVERSHOOT      = "mediumpurple"
CUNDERSHOOT     = "yellowgreen"
COSCILLATORY    = "cornflowerblue"




############ LABELS #########################
label_drive_std    = r"Fluctuation level $\sigma$ [pA]"
label_drive_mean   = r"Mean drive $\mu$ [pA]"

xlabel_time     = "Time [ms]"
xlabel_drive    = r"Mean drive $\mu_{pre}$ [pA]"
xlabel_stimulus = "# of stimulus pulses"

ylabel_fr       = "FR [Hz]"
ylabel_reaction = "Reaction time [ms]"
ylabel_recovery = "Recovery time [ms]"

marker_mean_recovery = "p"
marker_median_recovery = "*"

#===============================================================================
# rcParams and styles
#===============================================================================
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.labelpad"] = 2
rcParams["errorbar.capsize"] = 2
rcParams["font.size"] = 8
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 1
rcParams["legend.handlelength"] = 1.8
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = .6
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["legend.framealpha"] = 1
rcParams["xtick.major.pad"] = 2
rcParams["ytick.major.pad"] = 2
rcParams["lines.linewidth"] = 1
rcParams["lines.markersize"] = 5

rcParams["figure.titlesize"] = "x-large"


title_style = {
    "fontsize": rcParams["axes.titlesize"],
    "fontweight": rcParams["axes.titleweight"],
    "fontfamily": rcParams["font.family"],
    "ha": "center",
    "va": "center"
}

quiver_style = {
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1,
    "units": "xy",
    "headaxislength": 4,
    "headlength": 4,
}
