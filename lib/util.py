#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Utility methods

Methods can be used across projects.

History:
    v0.1a: h5path added.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from collections.abc import Iterable
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from pathlib import PurePosixPath

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


def h5path(*parts):
    return str(PurePosixPath(*parts))


def replace_table_with_new_description(h5file:tb.file, table_path:str, new_description:object):
    """
    Recreate a PyTables table with a new description.

    Parameters
    ----------
    h5file : tables.File
        Open PyTables file handle (mode="a").
    table_path : str
        Full path of the existing table.
    new_description : dict or IsDescription subclass
        New table description.
    """

    old_table = h5file.get_node(table_path)
    parent = old_table._v_parent
    name = old_table._v_name

    tmp_name = name + "_tmp"

    # Create new table
    new_table = h5file.create_table(
        parent,
        tmp_name,
        description=new_description,
        filters=old_table.filters
    )

    # Read old data
    old_data = old_table.read()
    new_data = np.zeros(old_data.shape, dtype=new_table.dtype)

    # Copy overlapping columns only
    for col in old_table.colnames:
        if col in new_table.colnames:
            new_data[col] = old_data[col]

    new_table.append(new_data)
    new_table.flush()

    # Replace old table
    old_table.remove()
    new_table.move(parent, name)

    return new_table

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

