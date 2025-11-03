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
from cflogger import logger


import datetime
import tables as tb
import numpy as np

from pathlib import PosixPath, Path


#===============================================================================
# CONSTANTS
#===============================================================================
DATA_DIR = "data"

suffix = ".hdf5"

metadata_tag = "metadata"
data_tag = "data"
run_tag = "run"
id_tag = "run_id"
senders_tag = "senders"
spikes_tag = "spikes"
time_tag = "time"
Vm_tag = "Vm"
seed_tag = "seed"
spikes_by_sender_tag = "spikes_by_sender"
binwidth_tag = "binwidth"
entropy_tag = "entropy"
delay_tag = "delay_steps"
SEM_tag = "SEM"

dateformat = "%y%m%d_%H%M%S"

#===============================================================================
# CLASS
#===============================================================================
class ResponseHdf5(tb.File):
    def __init__(self, *args, metadata, **kwargs):
        if kwargs.get("file", None):
            file = kwargs.pop("file")
        else:
            args = list(args)
            file = args.pop(0)
        logger.info(f"Update filename with dir: {file}")
        path = prepend_dir(file, DATA_DIR)
        super().__init__(path, *args, **kwargs)
        
        if not metadata_tag in self.root:
            logger.info("No Metadata found. Set Metadata...")
            self.set_metadata(metadata)
            
        if not self.is_current_metadata(metadata):
            if yes_no("Different metadata: Rename (y) or abort (n)?"):
                prefix = datetime.datetime.now().strftime(dateformat)
                path = prepend_dir(prefix + "_" + file, DATA_DIR)
                self.copy_file(path)
                for child in self.root._v_children.values():
                    self.remove_node(child, recursive=True)
                self.set_metadata(metadata)
            else:
                raise FileExistsError
        
        self.data = self.require_group(self.root, data_tag)
        self.run = self.require_table(self.data, run_tag, Run)


    def is_current_metadata(self, metadata:dict) -> bool:
        metadata_grp = self.require_group(self.root, metadata_tag)
        for key, value in metadata.items():
            if not metadata_grp._v_attrs[key] == value:
                logger.warning(f"Unequal attribute ({key}): {value}")
                return False
        return True
    
    
    def set_metadata(self, metadata:dict) -> None:
        metadata_grp = self.require_group(self.root, metadata_tag)
        for key, value in metadata.items():
            metadata_grp._v_attrs[key] = value
        
        
    def require_group(self, where:str, name:str, *args, **kwargs)->tb.Group:
        """Extension of the method {create_group} with same signature."""
        try:
            return self.get_node(where, name)
        except tb.NoSuchNodeError:
            logger.info(f"Create new Group: {name}")
            return self.create_group(where, name, *args, **kwargs)
        
        
    def require_table(self, where:str, name:str, *args, **kwargs)->tb.Table:
        """Extension of the method {create_table} with same signature."""
        try:
            return self.get_node(where, name)
        except tb.NoSuchNodeError:
            logger.info(f"Create new Table: {name}")
            return self.create_table(where, name, *args, **kwargs)
        
        
    @property
    def next_run_id(self):
        node = self.get_node(self.data, run_tag)
        return len(node)
        
        
    def filter_rows(self, node:tb.Node, **kwargs) -> (None, tb.Group):
        """Usage:        
            t = self.filter_rows(run, pre_FR=pre_FR, post_FR=post_FR,
                                  pre_mean=pre_mean, post_mean=post_mean,
                                  pre_std=pre_std, post_std=post_std)
        """
        condition = ""
        c = "({} == {})"
        for key, value in kwargs.items():
            if condition:
                condition += " & "
            condition += c.format(key, value)
        return node.read_where(condition)
            
            
    def read_rows(self, run_ids:np.ndarray) -> tb.Node:
        condition = " | ".join([f"({id_tag} == {run_id})" for run_id in run_ids])
        return self.run.read_where(condition)
    
    
    def add_run(self, 
                pre_FR:int, post_FR:int, 
                pre_mean:float, post_mean:float,
                pre_std:float, post_std:float, seed:int):
        
        row = self.run.row
        run_id = self.next_run_id
        logger.info(f"New run ID: {run_id}")
        row[id_tag] = run_id
        row["pre_FR"] = pre_FR
        row["post_FR"] = post_FR
        row["pre_mean"] = pre_mean
        row["post_mean"] = post_mean
        row["pre_std"] = pre_std
        row["post_std"] = post_std
        row[seed_tag] = seed
        row.append()
        self.flush()
        return run_id
        
        
    def add_data_to_run(self, run_id:int, senders:np.ndarray, spikes:np.ndarray, time:np.ndarray, Vm:np.ndarray):
        run_data = self.create_group(self.data, run_tag+str(run_id))
        
        self.create_array(run_data, senders_tag, senders.astype(np.int16))
        self.create_array(run_data, spikes_tag, spikes.astype(np.float32))
        
        self.create_array(run_data, time_tag, time)
        self.create_array(run_data, Vm_tag, Vm.astype(np.float32))
        self.flush()
        
    
    def has_entropy(self, run_id:int) -> bool:
        rows = self.filter_rows(self.run, **{id_tag: run_id})
        if len(rows) != 1:
            raise ValueError
        return rows[0]["pre_entropy"] > 0. # 0. is the default value.
    
    
    def add_entropy(self, run_id:int, pre_entropy:np.ndarray) -> None:
        """
        Adds the entropy of the membrane potentials to the file.
        
        :param run_id: Run ID.
        :type run_id:int 
        :param pre_entropy: Entropy of the membrane potentials before the change.
        :type pre_entropy:np.ndarray
        """

        for row in self.run.where(f"{id_tag} == {run_id}"):
            row["pre_entropy"] = pre_entropy
            row.update()


    def has_spikes_by_sender(self, run_id:int) -> bool:
        run = self.get_node(self.data, f"run{run_id}")
        if spikes_by_sender_tag in run:
            return True
        return False
        
    def add_spikes_by_sender(self, run_id, spikes_by_sender) -> None:
        run = self.get_node(self.data, f"run{run_id}")
        vlarray = self.create_vlarray(run, spikes_by_sender_tag, tb.Float32Atom())
        for spikes in spikes_by_sender:
            vlarray.append(spikes)
        self.flush()
        
        
#===============================================================================
# DESCRIPTOR CLASSES
#===============================================================================
class Run(tb.IsDescription):
    run_id      = tb.UInt32Col()   
    pre_FR      = tb.Float64Col()   
    post_FR     = tb.Float64Col()   
    pre_mean    = tb.Float64Col()      
    post_mean   = tb.Float64Col()
    pre_std     = tb.Float64Col()      
    post_std    = tb.Float64Col()
    seed        = tb.UInt8Col()  
    pre_entropy = tb.Float32Col()
    

#===============================================================================
# METHODS
#===============================================================================

def prepend_dir(filename: str, directory: str = DATA_DIR) -> PosixPath:
    # Added in v0.1
    return Path(directory).joinpath(filename)


def yes_no(question:str, answer:bool=None) -> bool:
    if answer is not None:
        return answer
    answer = input(question + " (y/n)")
    return answer.lower().strip() == "y"
#===============================================================================
if __name__ == '__main__':
    main()
