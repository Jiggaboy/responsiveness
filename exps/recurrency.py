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

import matplotlib.pyplot as plt
import numpy as np


import lib.nest_interface as nif

from constants import mean_tag, std_tag, mean_std_tag, Color
from cplot.constants import EXC_NEURON, INH_NEURON, quiver_style
from cplot.aux import align_zero
from lib import siegert
from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm

from lib.rnn import RNN

#===============================================================================
# CONSTANTS
#===============================================================================

from config import load_config
control, params = load_config(is_network=True)

Imean_ext   = 260.
means       = [260., ]
# means       = [220., 300.]


pre_FR = 5 # for E and I
post_FR = 10
FR_I = pre_FR


x0 = 0
x1 = 1
xshift = 3
offset = 7
          
ext_color = "darkgrey"
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig, ax0 = plt.subplots(num=f"test")
    ax0.set(xlabel="mean drive", ylabel="fluctuations")
    
    rnn = RNN(params, pre_FR, post_FR, FR_I, drive_Imean=Imean_ext)
    for delta in (mean_tag, std_tag, mean_std_tag):
        for m, mean in enumerate(means):
            rnn.set_up_network(mean, delta=delta)
            
            # plt.scatter(x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths)
            # Start: FF
            plt.scatter(rnn.pre_drive_Emean, rnn.pre_drive_Estd, c=ext_color, zorder = 10,)
            # EE contribution
            plt.quiver(
                rnn.pre_drive_Emean, rnn.pre_drive_Estd, 
                rnn.pre_EE_mean, rnn.pre_EE_std,
                color=EXC_NEURON,
                zorder = 5,
                **quiver_style,
            )
            # EI contribution
            plt.quiver(
                rnn.pre_drive_Emean + rnn.pre_EE_mean, rnn.pre_drive_Estd + rnn.pre_EE_std,
                rnn.pre_EI_mean, rnn.pre_EI_std,
                color=INH_NEURON,
                zorder = 4,
                **quiver_style,
            )
            
            # Start: Pre
            plt.scatter(
                *rnn.pre_Esetpoint,
                c = ext_color,
                zorder = 10,
            )
            #####################################################################################
            if delta == mean_tag:
                color = Color[std_tag]
            elif delta == std_tag:
                color = Color[mean_tag]
            else:
                color = Color[mean_std_tag]
            
            deltaEE_mean = rnn.post_EE_mean - rnn.pre_EE_mean
            deltaEE_std  = rnn.post_EE_std - rnn.pre_EE_std
            
            # Delta Generator
            plt.quiver(
                *rnn.pre_Esetpoint,
                rnn.post_Esetpoint[0] - rnn.pre_Esetpoint[0] - deltaEE_mean, rnn.post_Esetpoint[1] - rnn.pre_Esetpoint[1] - deltaEE_std,
                color=ext_color,
                zorder = 2,
                **quiver_style,
            )
            
        
            plt.quiver(
                rnn.post_Esetpoint[0] - deltaEE_mean, rnn.post_Esetpoint[1] - deltaEE_std,
                deltaEE_mean, deltaEE_std,
                color=EXC_NEURON,
                zorder = 5,
                **quiver_style,
            )
            
            plt.quiver(
                *rnn.pre_Esetpoint,
                rnn.post_Esetpoint[0] - rnn.pre_Esetpoint[0], rnn.post_Esetpoint[1] - rnn.pre_Esetpoint[1],
                color=color,
                zorder = 4,
                **quiver_style,
            )
            continue
            #
            # fig, ax0 = plt.subplots(num=f"{delta}")
            # ax0.set(ylabel="mean drive")
            # ax0.spines.right.set_visible(True)
            #
            # ax1 = ax0.twinx()
            # ax1.set(ylabel="fluctuations")
            #
            #
            # ax0.bar(
            #     np.asarray([x0, x0, x0]),
            #     [mean, from_free_Vm_to_generator(EEmean_pre), from_free_Vm_to_generator(EImean_pre)],
            #     # bottom = [0, mean, mean + from_free_Vm_to_generator(EEmean_pre)],
            #     bottom = [0, pre_Emeans[m], 0],
            #     color=[ext_color, EXC_NEURON, INH_NEURON],
            #     **mean_bar_kwargs,
            # )
            # total_mean_pre = pre_Emeans[m] + from_free_Vm_to_generator(EEmean_pre) + from_free_Vm_to_generator(EImean_pre)
            # ax0.plot([x0-1, x0+1], [total_mean_pre, total_mean_pre], c="k")
            #
            # ax0.bar(
            #     np.asarray([x0, x0, x0]) + offset,
            #     [post_Emeans[m], from_free_Vm_to_generator(EEmean_post), from_free_Vm_to_generator(EImean_post)],
            #     # bottom = [0, post_Emeans[m], post_Emeans[m] + from_free_Vm_to_generator(EEmean_post)],
            #     bottom = [0, post_Emeans[m], 0],
            #     color=[ext_color, EXC_NEURON, INH_NEURON],
            #     **mean_bar_kwargs,
            # )
            # total_mean_post = post_Emeans[m] + from_free_Vm_to_generator(EEmean_post) + from_free_Vm_to_generator(EImean_post)
            # ax0.plot([x0-1  + offset, x0+1 + offset], [total_mean_post, total_mean_post], c="k")
            #
            # ax0.axhline(0, c="k")
            #
            #
            #
            # # FF
            # std = siegert.find_parameter(mean, target_FR=pre_FR, dt=params.dt).root            
            # ax0.bar(x0 - offset, mean, color=ext_color, **mean_bar_kwargs)
            # ax0.plot([x0 - offset - 1, x0 - offset + 1], [mean, mean], c="k")
            # ax1.bar(x0 - offset + xshift, std, color=ext_color, **std_bar_kwargs)
            #
            #
            #
            # EEstd_drive_pre = np.sqrt( from_free_Vm_to_generator(var_V=(EEvar_pre), dt=params.dt) )
            # EIstd_drive_pre = np.sqrt( from_free_Vm_to_generator(var_V=(EIvar_pre), dt=params.dt) )
            # # ax1.bar(
            # #     np.asarray([x0, x0, x0]) + xshift,
            # #     [Estd_pre, EEstd_drive_pre, EIstd_drive_pre],
            # #     bottom = [0, Estd_pre, Estd_pre + EEstd_drive_pre],
            # #     color=[ext_color, EXC_NEURON, INH_NEURON],
            # #     **std_bar_kwargs,
            # # )
            # #
            # # EEstd_drive_post = np.sqrt( from_free_Vm_to_generator(var_V=(EEvar_post), dt=params.dt) )
            # # EIstd_drive_post = np.sqrt( from_free_Vm_to_generator(var_V=(EIvar_post), dt=params.dt) )
            # # ax1.bar(
            # #     np.asarray([x0, x0, x0]) + xshift + offset,
            # #     [Estd_pre, EEstd_drive_post, EIstd_drive_post],
            # #     bottom = [0, Estd_pre, Estd_pre + EEstd_drive_post],
            # #     color=[ext_color, EXC_NEURON, INH_NEURON],
            # #     **std_bar_kwargs,
            # # )
            # ax1.bar(
            #     np.asarray([x0, x0, x0]) + xshift,
            #     [Estd_pre, EEstd_drive_pre, EIstd_drive_pre],
            #     bottom = [0, Estd_pre, Estd_pre + EEstd_drive_pre],
            #     color=[ext_color, EXC_NEURON, INH_NEURON],
            #     **std_bar_kwargs,
            # )
            #
            # EEstd_drive_post = np.sqrt( from_free_Vm_to_generator(var_V=(EEvar_post), dt=params.dt) )
            # EIstd_drive_post = np.sqrt( from_free_Vm_to_generator(var_V=(EIvar_post), dt=params.dt) )
            # ax1.bar(
            #     np.asarray([x0, x0, x0]) + xshift + offset,
            #     [Estd_pre, EEstd_drive_post, EIstd_drive_post],
            #     bottom = [0, Estd_pre, Estd_pre + EEstd_drive_post],
            #     color=[ext_color, EXC_NEURON, INH_NEURON],
            #     **std_bar_kwargs,
            # )
            # align_zero(ax0, ax1)
            # #
            # # yticks = (
            # #
            # # )
            # #
            # # ax0.
            # return
        
#===============================================================================
# CLASS
#===============================================================================
#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
