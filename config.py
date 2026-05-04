#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Config-file

Usage:
    from config import force, brief_stimulus, stim_duration, double
    
History:
    - v0.2: NetworkParams added. load_config adjusted accordingly.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'
#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

import lib.nest_interface as nif

#===============================================================================
# CONTROL
#===============================================================================
class Control:
    test = True
    test = False
    force= True
    force= False
    
    brief_stimulus = True
    brief_stimulus = False
    


#===============================================================================
# PARAMS
#===============================================================================
@dataclass
class Params:
    control: object
    N: int                = 500
    dt: float             = 0.1
    warmup: float         = 100.
    duration_pre: float   = 400.
    duration_post: float  = 1000.
    # stim_duration: float  = nif.tau #* 2
    stim_duration: float  = nif.tau / 2
    # break_duration: float = nif.tau * 0.5
    break_duration: float = 5.
    stim_reps: int        = 5
    
    hist_binwidth = 2.
    
    poisson_filename: str = "poisson.hdf5"
    
    def __post_init__(self):
        if not self.control.brief_stimulus:
            self.stim_duration = 0.
            self.break_duration = 0.
            self.stim_reps = 0
        filename = self.filename


    @property
    def metadata(self):
        keys = ["N", "dt", "warmup", "duration_pre", "duration_post"]
        return {k: getattr(self, k) for k in keys}


    @property
    def filename(self):
        c = self.control
        if not c.brief_stimulus:
            filename        = "sim_data.hdf5"
        elif c.brief_stimulus:
            self.stim_duration = np.round(self.stim_duration)
            filename        = f"stimulus_{self.stim_duration}.hdf5"
        else:
            raise ValueError("Invalid arguments")
            
        if c.test:
            self.N             = 500  
            self.duration_pre  = 200.
            self.duration_post = 500.
            filename = "test_" + filename
            
        logger.info(f"Filename: {filename}") 
        return filename

@dataclass
class NetworkParams(Params):    
    # Target-Source notation
    # Indegree definition
    C_EE: int               = 400
    C_EI: int               = 400
    C_IE: int               = 100
    C_II: int               = 100
    
    J = 0.075
    g = 8
    
    def __post_init__(self):
        super().__post_init__()
        self.N = 5000
        
        
    @property
    def filename(self):
        filename        = "network"
        filename = filename + f"_J_{self.J}"
        if self.control.brief_stimulus:
            self.stim_duration = np.round(self.stim_duration)
            filename        = filename + f"_stim_{self.stim_duration}"
    
            
        if self.control.test:
            self.N             = 500  
            self.duration_pre  = 200.
            self.duration_post = 500.
            filename = "test_" + filename
            
        logger.info(f"Filename: {filename}") 
        return filename + ".hdf5"
        
    @property
    def metadata(self):
        base = super().metadata
        conn_keys = ["C_EE", "C_EI", "C_IE", "C_II", "J", "g"]
        conn = {k: getattr(self, k) for k in conn_keys}
        base.update(conn)
        return base
#===============================================================================
# METHODS
#===============================================================================
def load_config(is_network:bool = False, no_stim:bool = False):
    control = Control()
    if no_stim:
        control.brief_stimulus = False
    if is_network:
        return control, NetworkParams(control)
    return control, Params(control)
