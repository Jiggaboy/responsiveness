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

from config import load_config
from lib.util import replace_table_with_new_description
from lib.responsehdf5 import StimRun, ResponseHdf5

#===============================================================================
# CONSTANTS
#===============================================================================


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    control, params = load_config()
    with ResponseHdf5("_poisson.hdf5", "a", metadata=params.metadata) as hfile:
        replace_table_with_new_description(hfile, hfile.run, StimRun)
    print("Finished")

#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
