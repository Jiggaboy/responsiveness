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

#===============================================================================
# CONSTANTS
#===============================================================================

#===============================================================================
# METHODS
#===============================================================================

def spikecount_to_FR(spikecount:np.ndarray, N:int, binwidth:float):
    """binwidth in [ms]"""
    return spikecount / N / (binwidth*1e-3)


def from_generator_to_free_Vm(mean_pA:float=None, var_pA:float=None, dt:float=None):
    """
    var_factor was taken from https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html
    Requires nest_interface as nif.
    """
    mean_factor = (nif.tau*1e-3) / (nif.capacitance*1e-12)    
    if var_pA:
        if dt is None:
            raise ValueError("If var_pA is given, dt is required.")
        else:
            var_factor  = (dt*1e-3) * (nif.tau*1e-3) / (2*(nif.capacitance*1e-12)**2)
            
    if var_pA is None:
        return mean_factor * (mean_pA*1e-12)
    if mean_pA is None:
        return var_factor * (var_pA*1e-12**2)
    return mean_factor * (mean_pA*1e-12), var_factor * (var_pA*1e-12**2)


def from_free_Vm_to_generator(mean_V:float=None, var_V:float=None, dt:float=None):
    """
    returns currents in pA.
    var_factor was taken from https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html
    Requires nest_interface as nif.
    """
    mean_factor = (nif.tau*1e-3) / (nif.capacitance*1e-12)    
    if var_V:
        if dt is None:
            raise ValueError("If var_pA is given, dt is required.")
        else:
            var_factor  = (dt*1e-3) * (nif.tau*1e-3) / (2*(nif.capacitance*1e-12)**2)
            
    if var_V is None:
        return (1/mean_factor) * (mean_V*1e12)
    if mean_V is None:
        return (1/var_factor) * (var_V*1e12**2)
    return (1/mean_factor) * (mean_V*1e12), (1/var_factor) * (var_V*1e12**2)
