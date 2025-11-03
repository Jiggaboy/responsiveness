#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-12-10

@author: Hauke Wernecke
"""

from collections.abc import Iterable
from functools import partial
import numpy as np
from scipy import integrate, special


def error_integral(mu:float, sigma:float, threshold:float, reset_potential:float=0., **kwargs):
    # Instead of the normal error function, we use the scaled complementary error function to avoid arithmetic underflow.
    lower_bound = (reset_potential - mu) / sigma
    upper_bound = (threshold - mu) / sigma
    return integrate.quad(lambda u:special.erfcx(-u), lower_bound, upper_bound)


def siegert_plain(mu:float, sigma:float, tau_ref:float, tau_m:float, **kwargs):
    """{tau_m} and {tau_ref} are in seconds."""
    integral, max_error = error_integral(mu, sigma, **kwargs)
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
    """
    mean_potential = GWN_mean_pA * tau_ms / membrane_capacitance * syn_weight
    # Units: 1e-12 * 1e-3 / 1e-12 = 1e-3
    mean_potential *= 1e-3

    var_potential = dt * tau_ms * GWN_variance_pA / membrane_capacitance**2 * syn_weight**2
    # Units: 1e-3 * 1e-3 1e-24 / 1e-24= 1e-6
    var_potential *= 1e-6
    return mean_potential, var_potential
