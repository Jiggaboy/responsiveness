#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Config-file

Usage:
    from config import force, double_step, delta_step, double
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
    
    double_step = True
    double_step = False


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
    delta_step: float     = nif.tau
    
    poisson_filename: str = "poisson.hdf5"
    
    def __post_init__(self):
        c = Control()
        if not c.double_step:
            self.filename        = "sim_data.hdf5"
            # self.filename        = "randomseeds_data.hdf5"
        elif c.double_step:
            self.filename        = "double_step.hdf5"
        else:
            raise ValueError("Invalid arguments")
            
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
    network_filename: str = "network.hdf5"
    
    # Target-Source notation
    # Indegree definition
    C_EE: int               = 100
    C_EI: int               = 200
    C_IE: int               = 200
    C_II: int               = 100
    
    J = 0.1
    g = 8
    
    
    @property
    def metadata(self):
        base = super().metadata
        conn_keys = ["C_EE", "C_EI", "C_IE", "C_II"]
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
