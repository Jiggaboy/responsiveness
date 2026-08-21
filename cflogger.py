#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Custom Logger - Parametrized

Usage:
    from cflogger import logger
    # logger is an object and ready to use
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

__all__ = [
    "logger"
]

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

# standard libs
import logging
import sys
import numpy as np

# Printing to a file is corrupted if linewidth is an integer.
np.set_printoptions(linewidth=np.nan)

# plot_constants
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(funcName)s: %(message)s"
DEF_LOG_FILE = "./debug.log"


def getLogger()->object:
    """
    Returns the logger of the logging - lib. 
    Also ensures, that the initialization is done only once.
    """
    try:
        if not getLogger.initialized:
            raise AttributeError
    except AttributeError:
        set_up()
        getLogger.initialized = True
    return logging.getLogger()


def set_up()->None:
    """Set up the logger including the format, the log level and the handlers."""
    filename = DEF_LOG_FILE
    logging.basicConfig(level=LOG_LEVEL,
                        format=LOG_FORMAT,
                        handlers=[
                            logging.FileHandler(filename, mode="w"),
                            logging.StreamHandler(sys.stdout),
                        ],
    )

logger = getLogger()
