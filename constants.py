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
# TAGS
#===============================================================================
mean_tag = "mean"
std_tag = "std"
mean_std_tag = "both"

delay_tag = "delay"
entropy_tag = "entropy"

Color = {mean_tag: "tab:blue", std_tag: "tab:orange", mean_std_tag: "tab:green"}
Label = {mean_tag: r"$\Delta \, \sigma$", std_tag: r"$\Delta \, \mu$", mean_std_tag: r"$\Delta \, \mu$&$\Delta \, \sigma$"}
#===============================================================================
# DIRECTORIES
#===============================================================================

DATA_DIR = "data"
FIGURE_DIR = "figures"

#===============================================================================
# FIGURE SUFFICES
#===============================================================================

ANIMATION_SUFFIX = ".gif" # ".mp4"
FIGURE_SUFFIX = ".svg"
FIGURE_ALTERNATIVE_SUFFIX = ".png"