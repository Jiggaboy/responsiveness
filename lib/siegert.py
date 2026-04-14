#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from collections.abc import Iterable
from functools import partial
import numpy as np
from scipy import integrate, special
from scipy.optimize import root_scalar

import lib.nest_interface as nif

#===============================================================================
# METHODS - HIGH LEVEL
#===============================================================================

def find_parameter(value:float, target_FR:float, dt:float, given_parameter:str="mean", low:float=1., high:float=10_000.):
    "Value is in the generator space"
    if given_parameter not in ("mean", "std"):
        raise ValueError
    if given_parameter == "mean":
        search = "std"
    else:
        search = "mean"
    kwargs = {given_parameter: value, "dt": dt}
    
    loaded = partial(FR_from_siegert, **kwargs)
    def objective(param):
        return loaded(**{search: param}) - target_FR
    return root_scalar(objective, bracket=[low, high])


def FR_from_siegert(mean:float, std:float, dt:float)->float:
    """
    Requires nest_interface for default values of the neurons.

    Parameters
    ----------
    mean : float
        Value of the Noise Generator (nest).
    std : float
        Value of the Noise Generator (nest).
    dt : float
        Time step of the Noise Generator, default is 10*dt of the simulation (nest: https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html).

    Returns
    -------
    float
        Predicted FR.

    """
    mean_tmp, var_tmp = potential_from_moments(mean, std**2, tau_ms=nif.tau, dt=dt, membrane_capacitance=nif.capacitance)
    return siegert(mean_tmp, np.sqrt(var_tmp), tau_ref=nif.t_ref*1e-3, tau_m=nif.tau*1e-3, threshold=nif.V_th*1e-3)

#===============================================================================
# METHODS - LOW LEVEL
#===============================================================================
def error_integral(mu:float, sigma:float, threshold:float, reset_potential:float=0., **kwargs):
    # Instead of the normal error function, we use the scaled complementary error function to avoid arithmetic underflow.
    lower_bound = (reset_potential - mu) / sigma
    upper_bound = (threshold - mu) / sigma
    return integrate.quad(lambda u:special.erfcx(-u), lower_bound, upper_bound)


def siegert_plain(mu:float, sigma:float, tau_ref:float, tau_m:float, **kwargs):
    """{tau_m} and {tau_ref} are in seconds."""
    integral, max_error = error_integral(mu, sigma, **kwargs)
    ### error routine
    # print(f"Error: {max_error}; Relative error: {max_error/integral}.")
    # isi_min = tau_ref + tau_m * np.sqrt(np.pi) * (integral + max_error)
    # isi_max = tau_ref + tau_m * np.sqrt(np.pi) * (integral - max_error)
    # print(f"Estimated ISI boundaries: {1 / isi_min} to {1 / isi_max}.")
    # print(f"tau_ref: {tau_ref} relative to second term {tau_m * np.sqrt(np.pi) * integral}")
    
    isi = tau_ref + tau_m * np.sqrt(np.pi) * integral
    return (1 / isi if not np.isnan(isi) else 0)


def siegert(mu:(float, Iterable), sigma:(float, Iterable), **kwargs):
    mu_is_iterable = isinstance(mu, Iterable)
    sigma_is_iterable = isinstance(sigma, Iterable)
    if mu_is_iterable and sigma_is_iterable:
        return np.fromiter(map(partial(siegert_plain, **kwargs), mu, sigma), dtype=float)
    elif mu_is_iterable:
        return np.fromiter(map(partial(siegert_plain, sigma=sigma, **kwargs), mu), dtype=float)
    elif sigma_is_iterable:
        return np.fromiter(map(partial(siegert_plain, mu, **kwargs), sigma), dtype=float)
    else:
        return siegert_plain(mu, sigma, **kwargs)


def brunel_delta_synapse(rate:float, synaptic_weight_mV:float, tau_membrane_ms:float):
    """
    Calculates the mean and variance of the membrane potential of a neuron that receives spiking
    input and incorporates dirac-delta synapses.
    NOTE: The variances has to be divided by 2 to get the actual variance of the membrane potential. Here the factor
    is omitted that the result can be plugged in the siegert formula immediately.

    Input parameter tau_membrane is given in ms.
    Input parameter synaptic_weight is given in mV.
    """
    synaptic_weight_V = synaptic_weight_mV * 1e-3
    tau_membrane_s = tau_membrane_ms * 1e-3
    mu = rate * synaptic_weight_V * tau_membrane_s
    var = rate * synaptic_weight_V**2 * tau_membrane_s
    return mu, var


# Misleading name of the function.
def potential_from_moments(GWN_mean_pA:np.ndarray, GWN_variance_pA:np.ndarray, tau_ms:float, dt:float, syn_weight:float=1., membrane_capacitance=250.):
    """
    Takes the moments of a current source and
    calculates the mean and variance of the free membrane potential.
    Immediate transformation to the values one can plug-in into the Siegert formula.
    See: https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html
    Note: The factor 2 is missing as the siegert equation takes 2*var (cf. Tsodys 1991)
    """
    mean_potential = GWN_mean_pA * tau_ms / membrane_capacitance * syn_weight
    # Units: 1e-12 * 1e-3 / 1e-12 = 1e-3
    mean_potential *= 1e-3

    # Free membrane potential
    var_potential = dt * tau_ms * GWN_variance_pA / membrane_capacitance**2 * syn_weight**2 / 2
    # Units: 1e-3 * 1e-3 1e-24 / 1e-24= 1e-6
    var_potential *= 1e-6
    return mean_potential, 2*var_potential # as it is used for the siegert formula



def get_drive_moments(tau_ms:float, syn_weight:float, C:int, FR:float):
    """See Brunel 2000: Eq. 20(?)"""
    mean = tau_ms*1e-3 * C * (syn_weight*1e-3)    * FR
    var  = tau_ms*1e-3 * C * (syn_weight*1e-3)**2 * FR
    return mean, var

