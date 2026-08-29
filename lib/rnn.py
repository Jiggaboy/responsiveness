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
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import lib.nest_interface as nif

from constants import mean_tag, std_tag, mean_std_tag, Color
from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm
from lib import siegert

#===============================================================================
# CONSTANTS
#===============================================================================


#===============================================================================
# CLASS
#===============================================================================

@dataclass
class RNN:
    # Requires siegert and nif.
    # Prefix E is for the excitatory population, prefix I for the inhibitory one.
    # Prefix AB (A, B elem of {E, I}) indicate the connections (target-source notation).
    params: object
    pre_FR: float
    post_FR: float
    FR_I: float
    drive_Imean: float
    
    @property
    def pre_Esetpoint(self):
        return (
            self.pre_drive_Emean + self.pre_EE_mean + self.pre_EI_mean,
            np.sqrt(self.pre_drive_Estd**2 + self.pre_EE_std**2 + self.pre_EI_std**2)
        )
        
    
    @property
    def post_Esetpoint(self):
        return (
            self.post_drive_Emean + self.post_EE_mean + self.post_EI_mean,
            np.sqrt(self.post_drive_Estd**2 + self.post_EE_std**2 + self.post_EI_std**2)
        )
        
     
    def set_up_network(self, pre_mean:float, delta:str):
        ### PRE
        params = self.params
        # pre_EE_var has no prefactor like mili, and is the variance of the free membrane potential Vm induced by the external drive;
        # Evar_int_pre is the variance of the external drive
        pre_EE_mean, pre_EE_var = siegert.get_drive_moments(nif.tau,           params.J, params.C_EE, self.pre_FR)  # these are the moments of the free Vm
        pre_EI_mean, pre_EI_var = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_EI, self.FR_I)    # free Vm
        pre_int_Emean = from_free_Vm_to_generator(pre_EE_mean + pre_EI_mean)              # given in pA
        pre_int_Evar  = from_free_Vm_to_generator(var_V=(pre_EE_var + pre_EI_var), dt=params.dt) # given in pA
        
        pre_drive_Emean = pre_mean
        # Total fluctuation level with network recurrency
        pre_total_Estd  = siegert.find_parameter(pre_drive_Emean + pre_int_Emean, target_FR=self.pre_FR, dt=params.dt).root # Estd_tmp in Generator space/pA
        pre_drive_Estd  = np.sqrt(pre_total_Estd**2 - pre_int_Evar) # recurrent network compensated
        
        ### POST
        # Recurrent network effects
        post_EE_mean, post_EE_var = siegert.get_drive_moments(nif.tau,           params.J, params.C_EE, self.post_FR) # these are the moments of the free Vm
        post_EI_mean, post_EI_var = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_EI, self.FR_I)
        post_int_Emean = from_free_Vm_to_generator(post_EE_mean + post_EI_mean)              # given in pA
        post_int_Evar  = from_free_Vm_to_generator(var_V=(post_EE_var + post_EI_var), dt=params.dt) # given in pA

        if delta == mean_tag:
            # Delta mean (Factor sqrt(2) required, cf Tsodyks 1991)
            # Add the external and internal fluctuation level (variances),
            # and find the total mean drive to the E population.
            post_total_Emean = siegert.find_parameter(np.sqrt(pre_drive_Estd**2 + post_int_Evar), target_FR=self.post_FR, given_parameter="std", dt=params.dt).root # Estd_tmp in Generator space

            # Compensate the total drive by the internal drive
            post_drive_Emean = post_total_Emean - post_int_Emean
            post_drive_Estd  = pre_drive_Estd
            assert np.isclose(siegert.FR_from_siegert(post_total_Emean, np.sqrt(pre_drive_Estd**2 + post_int_Evar), dt=params.dt), self.post_FR)
        elif delta == std_tag:
            # Delta std - keeping the same ext. drive; 
            # update with new internal network effects.
            post_total_Estd = siegert.find_parameter(pre_drive_Emean+post_int_Emean, target_FR=self.post_FR, dt=params.dt).root # Estd_tmp in Generator space
            
            post_drive_Emean = pre_drive_Emean
            post_drive_Estd  = np.sqrt(post_total_Estd**2 - post_int_Evar) # Remove updated internal network effects
        elif delta == mean_std_tag:                    
            # Delta both - keeping the same ext. drive; update with new internal network effects
            post_total_Estd = siegert.find_parameter(pre_drive_Emean+post_int_Emean, target_FR=self.post_FR, dt=params.dt).root # Estd_tmp in Generator space
            post_drive_Estd_tmp = np.sqrt(post_total_Estd**2 - post_int_Evar) # Remove updated internal network effects
            
            # Set the std post change to the pre + delta/2
            post_drive_Estd = pre_drive_Estd + (post_drive_Estd_tmp - pre_drive_Estd) / 2
            
            # Update the post mean accordingly
            post_total_Emean = siegert.find_parameter(np.sqrt(post_drive_Estd**2 + post_int_Evar), target_FR=self.post_FR, given_parameter="std", dt=params.dt).root # Estd_tmp in Generator space
            post_drive_Emean = post_total_Emean - post_int_Emean
        else:
            raise ValueError("No valid delta chosen")

        ### INHIBITION
        ## PRE
        pre_IE_mean, pre_IE_var = siegert.get_drive_moments(nif.tau,           params.J, params.C_IE, self.pre_FR)
        pre_II_mean, pre_II_var = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_II, self.FR_I)
        pre_int_Imean = from_free_Vm_to_generator(pre_IE_mean + pre_II_mean)              # given in pA
        pre_int_Ivar  = from_free_Vm_to_generator(var_V=(pre_IE_var + pre_II_var), dt=params.dt) # given in pA
        
        pre_total_Istd = siegert.find_parameter(self.drive_Imean + pre_int_Imean, target_FR=self.FR_I, dt=params.dt).root
        pre_drive_Istd = np.sqrt(pre_total_Istd**2 - pre_int_Ivar) # recurrent network compensated

        ## POST
        # Recurrent networks effects
        post_IE_mean, post_IE_var = siegert.get_drive_moments(nif.tau,           params.J, params.C_IE, self.post_FR)   # these are the moments of the free Vm
        post_II_mean, post_II_var = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_II, self.FR_I)      # free Vm
        post_int_Imean = from_free_Vm_to_generator(post_IE_mean + post_II_mean)               # given in pA
        post_int_Ivar  = from_free_Vm_to_generator(var_V=(post_IE_var + post_II_var), dt=params.dt)  # given in pA
        
        # Delta mean - Compensation for the increased exc. FR
        post_total_Imean = siegert.find_parameter(np.sqrt(pre_drive_Istd**2 + post_int_Ivar), target_FR=self.FR_I, given_parameter="std", dt=params.dt).root
        post_drive_Imean = post_total_Imean - post_int_Imean
        
        #### Assignment of variables
        ### PRE
        # Drive variables
        self.pre_drive_Emean = pre_drive_Emean
        self.pre_drive_Estd  = pre_drive_Estd
        self.pre_drive_Imean = self.drive_Imean
        self.pre_drive_Istd  = pre_drive_Istd
        
        # Internal excitatory variables
        # Mean
        self.pre_EE_mean = from_free_Vm_to_generator( pre_EE_mean )
        self.pre_EI_mean = from_free_Vm_to_generator( pre_EI_mean )
        self.pre_int_Emean = pre_int_Emean
        
        # Std
        self.pre_EE_std = np.sqrt( from_free_Vm_to_generator( var_V=pre_EE_var, dt=params.dt ))
        self.pre_EI_std = np.sqrt( from_free_Vm_to_generator( var_V=pre_EI_var, dt=params.dt ))
        
        self.pre_int_Estd  = np.sqrt( from_free_Vm_to_generator(var_V=(pre_EE_var + pre_EI_var), dt=params.dt)) # given in pA
        
        # Internal inhibitory variables
        # Mean
        self.pre_IE_mean = from_free_Vm_to_generator( pre_IE_mean )
        self.pre_II_mean = from_free_Vm_to_generator( pre_II_mean )
        
        # Std
        self.pre_IE_std = np.sqrt( from_free_Vm_to_generator( var_V=pre_IE_var, dt=params.dt ))
        self.pre_II_std = np.sqrt( from_free_Vm_to_generator( var_V=pre_II_var, dt=params.dt ))
        
        self.pre_int_Istd  = np.sqrt(pre_int_Evar) # given in pA
        
        
        # Mean
        ### POST
        # Drive variables
        self.post_drive_Emean = post_drive_Emean
        self.post_drive_Estd  = post_drive_Estd
        
        self.post_drive_Imean = post_drive_Imean
        self.post_drive_Istd  = pre_drive_Istd
        
        # Internal excitatory variables
        # Mean
        self.post_EE_mean = from_free_Vm_to_generator( post_EE_mean )
        self.post_EI_mean = from_free_Vm_to_generator( post_EI_mean )
        self.post_int_Emean = post_int_Emean

        # Std
        self.post_EE_std = np.sqrt( from_free_Vm_to_generator( var_V=post_EE_var, dt=params.dt ))
        self.post_EI_std = np.sqrt( from_free_Vm_to_generator( var_V=post_EI_var, dt=params.dt ))
        
        self.post_int_Estd  = np.sqrt( from_free_Vm_to_generator(var_V=(post_EE_var + post_EI_var), dt=params.dt)) # given in pA
        
        # Internal inhibitory variables
        # Mean
        self.post_IE_mean = from_free_Vm_to_generator( post_IE_mean )
        self.post_II_mean = from_free_Vm_to_generator( post_II_mean )

        # Std
        self.post_IE_std = np.sqrt( from_free_Vm_to_generator( var_V=post_IE_var, dt=params.dt ))
        self.post_II_std = np.sqrt( from_free_Vm_to_generator( var_V=post_II_var, dt=params.dt ))

