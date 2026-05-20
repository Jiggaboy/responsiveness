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
__version__ = '0.1a'

#===============================================================================
# rcParams
#===============================================================================
from matplotlib.pyplot import rcParams

rcParams["legend.fontsize"] = "small"
rcParams["legend.handlelength"] = 4

#===============================================================================
# KTH COLORS
#===============================================================================
KTH_green       = "#4DA060"
KTH_turquoise   = "#339C9C"
KTH_brick       = "#E86A58"
KTH_yellow      = "#FFBE00"

KTH_blue        = "#004791"
KTH_navy        = "#000061"
KTH_sky         = "#6298D2"

KTH_grey        = "#A5A5A5"


#===============================================================================
# TAGS
#===============================================================================
mean_tag = "mean"
std_tag = "std"
mean_std_tag = "both"

delay_tag = "delay"
entropy_tag = "entropy"

hue_order = [mean_tag, std_tag, mean_std_tag]

Color = {mean_tag: "tab:blue", std_tag: "tab:orange", mean_std_tag: "tab:green"}
Color = {mean_tag: KTH_turquoise, std_tag: KTH_brick, mean_std_tag: KTH_yellow}
Label = {mean_tag: r"$\Delta \, \sigma$", std_tag: r"$\Delta \, \mu$", mean_std_tag: r"$\Delta \, \mu$&$\Delta \, \sigma$"}
#===============================================================================
# DIRECTORIES
#===============================================================================

DATA_DIR        = "data"
FIGURE_DIR      = "figures"
LATEXFIGURE_DIR = "latex_figures"

#===============================================================================
# FIGURE SUFFICES
#===============================================================================

ANIMATION_SUFFIX = ".gif" # ".mp4"
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"