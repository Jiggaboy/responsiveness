#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History: 
    - v0.1b: Moving KTH colors to cplot/constants.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1b'

from cplot.plot_constants import KTH_brick, KTH_turquoise, KTH_yellow

#===============================================================================
# TAGS - Indicates the value that is kept constant
#===============================================================================
mean_tag = "mean"
std_tag = "std"
mean_std_tag = "both"

delay_tag = "delay"
entropy_tag = "entropy"

hue_order = [std_tag, mean_tag, mean_std_tag]

Color = {
    mean_tag: KTH_turquoise, 
    std_tag: KTH_brick, 
    mean_std_tag: KTH_yellow
}
Label = {
    mean_tag: r"$\Delta \, \sigma$", 
    std_tag: r"$\Delta \, \mu$", 
    mean_std_tag: r"$\Delta \, \mu$&$\Delta \, \sigma$"
}

#===============================================================================
# DIRECTORIES
#===============================================================================
DATA_DIR        = "data"
FIGURE_DIR      = "figures"
LATEXFIGURE_DIR = "latex_figures"

#===============================================================================
# FIGURE SUFFICES
#===============================================================================
# https://matplotlib.org/stable/users/explain/animations/animations.html
ANIMATION_SUFFIX = ".gif"
# https://matplotlib.org/stable/users/explain/configuration.html#rcparam-savefig-format
# {png, ps, pdf, svg}
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"