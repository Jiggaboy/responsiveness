#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Utility methods

Methods can be used across projects.
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

from collections.abc import Iterable
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from functools import wraps, partial
from time import perf_counter

from pathlib import Path

from constants import DATA_DIR, FIGURE_DIR, FIGURE_SUFFIX, FIGURE_ALTERNATIVE_SUFFIX


#===============================================================================
# CONSTANTS
#===============================================================================

#===============================================================================
# CLASSES
#===============================================================================

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#===============================================================================
# METHODS
#===============================================================================
def pairwise(iterable):
    """Template from itertools."""
    # pairwise('ABCDEFG') → AB BC CD DE EF FG

    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b
        
def functimer(func=None, *, logger=None):
    ''' Measures the elapsed time and prints it (optionally the same as a logger output.'''
    if func is None:
        return partial(functimer, logger=logger)

    printer = print if logger is None else logger.error

    @wraps(func)
    def wrapper(*args, **kwargs):
        printer(f"Start method: {func.__name__}")
        pre = perf_counter()
        result = func(*args, **kwargs)
        post = perf_counter()
        printer(f"Time elapsed ({func.__name__}): {post - pre}")
        return result
    return wrapper


def mkdir(filename:str) -> None:
    """Creates directories such that the filename is valid."""
    path = Path(filename)
    path.parent.absolute().mkdir(parents=True, exist_ok=True)


#===============================================================================
# FIGURES
#===============================================================================

def save_figure(filename:str, figure:object, sub_directory:str=None, **kwargs):
    """
    Saves the figure-directory in the subdirectory.
    """
    if sub_directory:
        filename = prepend_dir(filename, sub_directory)
    filename = prepend_dir(filename, FIGURE_DIR)
    mkdir(filename)

    figure.savefig(str(filename) + FIGURE_SUFFIX, **kwargs)
    figure.savefig(str(filename) + FIGURE_ALTERNATIVE_SUFFIX, **kwargs)
    

def prepend_dir(filename: str | Path, directory: str | Path = DATA_DIR) -> Path:
    filename = Path(filename)
    directory = Path(directory)
    return directory / filename

#===============================================================================
# UTIL
#===============================================================================

def make_iterable(element:Any) -> Iterable:
    """
    Returns the {element} as an Iterable-object.
    Returns {element} if {element} is already Iterable (except str).

    :param element: Single element or collection of elements.
    :type element: Any
    """
    if isinstance(element, str):
        return (element, )
    if not isinstance(element, Iterable):
        return (element, )
    return element


def yes_no(question:str, answer:bool=None) -> bool:
    if answer is not None:
        return answer
    answer = input(question + " (y/n)")
    return answer.lower().strip() == "y"


def play_beep(repeat:int=3, pause:float=0.2):
    import os
    beep = lambda x: os.system(f"echo -n '\a'; sleep {pause};" * x)
    beep(repeat)

