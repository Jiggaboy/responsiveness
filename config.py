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
    
    long_sim = True
    long_sim = False


#===============================================================================
# PARAMS
#===============================================================================
@dataclass
class Params:
    N: int                = 2500
    dt: float             = 0.1
    warmup: float         = 100.
    duration_pre: float   = 400.
    duration_post: float  = 1000.
    stim_duration: float  = nif.tau
    # break_duration: float = nif.tau * 0.5
    break_duration: float = 5.
    stim_reps: int        = 1
    
    poisson_filename: str = "poisson.hdf5"
    
    def __post_init__(self):
        c = Control()
        if not c.brief_stimulus:
            self.stim_duration = 0.
            self.break_duration = 0.
            self.stim_reps = 0
            self.filename        = "sim_data.hdf5"
            # self.filename        = "randomseeds_data.hdf5"
        elif c.brief_stimulus:
            self.filename        = "brief_stimulus.hdf5"
        else:
            raise ValueError("Invalid arguments")
            
        if c.long_sim:
            self.filename = "long_" + self.filename
            self.duration_post *= 2
            
        if c.test:
            self.N             = 500  
            self.duration_pre  = 200.
            self.duration_post = 500.
            self.filename = "test_" + self.filename
            
        logger.info(f"Filename: {self.filename}") 

    @property
    def metadata(self):
        keys = ["N", "dt", "warmup", "duration_pre", "duration_post"]
        return {k: getattr(self, k) for k in keys}
        # return {"N": self.N, "dt": self.dt, "warmup": self.warmup, "duration_pre": self.duration_pre, "duration_post": self.duration_post, }


@dataclass
class NetworkParams(Params):    
    # Target-Source notation
    # Indegree definition
    C_EE: int               = 100
    C_EI: int               = 200
    C_IE: int               = 200
    C_II: int               = 100
    
    J = 0.1
    g = 8
    
    def __post_init__(self):
        super().__post_init__()
        c = Control()
        if not c.brief_stimulus:
            self.network_filename        = "network.hdf5"
        elif c.brief_stimulus:
            self.network_filename        = "network_stim.hdf5"
        else:
            raise ValueError("Invalid arguments")
    
            
        if c.test:
            self.N             = 500  
            self.duration_pre  = 200.
            self.duration_post = 500.
            self.network_filename = "test_" + self.network_filename
            
        logger.info(f"Filename: {self.network_filename}") 
        
        
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
def load_config(is_network:bool = False):
    if is_network:
        return Control(), NetworkParams()
    return Control(), Params()
