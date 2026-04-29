#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History:
    - v0.1b: Add firstspike methods.
    - v0.2: Add StimRun as extension of Run.
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


import datetime
import tables as tb
import numpy as np

from pathlib import PosixPath, Path
from lib.util import yes_no

from constants import mean_tag, std_tag, mean_std_tag

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
exc_tag = "excitatory"
inh_tag = "inhibitory"
time_tag = "time"
Vm_tag = "Vm"
seed_tag = "seed"
spikes_by_sender_tag = "spikes_by_sender"
binwidth_tag = "binwidth"
firstspike_tag = "firstspike" # First spike after change per neuron.
entropy_tag = "entropy"
dist_pre_tag = "dist_pre"
dist_post_tag = "dist_post"
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
        self.run = self.require_table(self.data, run_tag, StimRun)


    def is_current_metadata(self, metadata:dict) -> bool:
        metadata_grp = self.require_group(self.root, metadata_tag)
        for key, value in metadata.items():
            try:
                if not metadata_grp._v_attrs[key] == value:
                    logger.warning(f"Unequal attribute ({key}): {value}")
                    return False
            except KeyError:
                if yes_no(f"{key} not found: Extend current metadata?"):
                    logger.warning(f"Add attribute to metadata ({key} = {value})...")
                    metadata_grp._v_attrs[key] = value
                else:
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
        
        
    def filter_rows(self, node:tb.Node, f={}, **kwargs) -> (None, tb.Group):
        """Usage:        
            t = self.filter_rows(run, pre_FR=pre_FR, post_FR=post_FR,
                                  pre_mean=pre_mean, post_mean=post_mean,
                                  pre_std=pre_std, post_std=post_std)
        """
        # TODO: str.join?
        condition = ""
        c = "({} == {})"
        for key, value in kwargs.items():
            if condition:
                condition += " & "
            condition += c.format(key, value)
        for key, value in f.items():
            if condition:
                condition += " & "
            condition += c.format(key, value)
        return node.read_where(condition)
            
            
    def read_rows(self, run_ids:np.ndarray) -> tb.Node:
        condition = " | ".join([f"({id_tag} == {run_id})" for run_id in run_ids])
        return self.run.read_where(condition)
    
    
    def add_run(self, 
                pre_FR: int, post_FR: int, 
                pre_mean: float, post_mean: float,
                pre_std: float, post_std: float, seed:int,
                stim_duration: float=0., break_duration:float=0., stim_reps:int=0):
        """
        History:
            - v0.1a: Remove self.flush() -> Requires to flush in main script now.
            - v0.2 : Added the stimulation parameters (stim_duration, break_duration, stim_reps).
        """
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
        # StimRun addition
        row["stim_duration"] = stim_duration
        row["break_duration"] = break_duration
        row["stim_reps"] = stim_reps
        row.append()
        return run_id
        
        
    def add_data_to_run(self, run_id:int, senders:np.ndarray, spikes:np.ndarray, time:np.ndarray=None, Vm:np.ndarray=None, subgroup:str=None) -> None:
        """
        History:
            - v0.1a: Remove self.flush() -> Requires to flush in main script now.
        """
        run_data = self.require_group(self.data, run_tag+str(run_id))
        if subgroup is not None:
            target = self.require_group(run_data, subgroup)
        else:
            target = run_data
        
        # Refactor to chunked array create_carray with filters
        filters = tb.Filters(complevel=2, complib="zlib", shuffle=True)  # good default
        
        self.create_carray(target, senders_tag, obj=senders.astype(np.int16), filters=filters)
        self.create_carray(target, spikes_tag, obj=spikes.astype(np.float32), filters=filters)
        # self.create_array(target, senders_tag, senders.astype(np.int16))
        # self.create_array(target, spikes_tag, spikes.astype(np.float32))
        
        if time is not None and Vm is not None:
            self.create_array(target, time_tag, time)
            self.create_array(target, Vm_tag, Vm.astype(np.float32))
              
        
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
            
            
    def has_Vdistribution(self, run_id:int) -> None:
        run = self.get_node(self.data, f"run{run_id}")
        if dist_pre_tag in run and dist_post_tag in run:
            return True
        return False
    
    
    def add_Vdistribution(self, run_id:int, dist_pre:np.ndarray, dist_post:np.ndarray) -> None:
        """
        Adds the entropy of the membrane potentials to the file.
        
        :param run_id: Run ID.
        :type run_id:int 
        :param dist_pre: Distribution of membrane potentials before the change
        :type dist_pre:np.ndarray
        :param dist_post: Distribution of membrane potentials after the change
        :type dist_post:np.ndarray
        
        History:
            - v0.1a: Remove self.flush() -> Requires to flush in main script now.
        """
        run = self.get_node(self.data, f"run{run_id}")
        if dist_pre_tag in run:
            self.remove_node(run, dist_pre_tag, recursive=True)
            self.create_array(run, dist_pre_tag, dist_pre.astype(np.float32))
        else:
            self.create_array(run, dist_pre_tag, dist_pre.astype(np.float32))
        
        if dist_post_tag in run:
            self.remove_node(run, dist_post_tag, recursive=True)
            self.create_array(run, dist_post_tag, dist_post.astype(np.float32))
        else:
            self.create_array(run, dist_post_tag, dist_post.astype(np.float32))


    def has_spikes_by_sender(self, run_id:int, subgroup:str=None) -> bool:
        run_data = self.require_group(self.data, run_tag+str(run_id))
        if subgroup is not None:
            target = self.require_group(run_data, subgroup)
        else:
            target = run_data
        if spikes_by_sender_tag in target:
            return True
        return False
        
        
    def add_spikes_by_sender(self, run_id:int, spikes_by_sender:np.ndarray, subgroup:str=None) -> None:
        """History:
            - v0.1a: Remove self.flush() -> Requires to flush in main script now.
        """
        run_data = self.require_group(self.data, run_tag+str(run_id))
        if subgroup is not None:
            target = self.require_group(run_data, subgroup)
        else:
            target = run_data
        vlarray = self.create_vlarray(target, spikes_by_sender_tag, tb.Float32Atom())
        for spikes in spikes_by_sender:
            vlarray.append(spikes)

        
    def has_firstspike(self, run_id:int, subgroup:str=None) -> bool:
        """
        Assumption: True if tag is there, hence the assumption that the array is also filled with values.
        
        History:
            - v0.1b: Initial addition.
        """
        run_data = self.require_group(self.data, run_tag+str(run_id))
        if subgroup is not None:
            target = self.require_group(run_data, subgroup)
        else:
            target = run_data
        if firstspike_tag in target:
            return True
        return False
        
        
    def add_firstspike(self, run_id:int, firstspikes:np.ndarray, subgroup:str=None) -> None:
        """History:
            - v0.1b: Initial addition.
        """
        run_data = self.require_group(self.data, run_tag+str(run_id))
        if subgroup is not None:
            target = self.require_group(run_data, subgroup)
        else:
            target = run_data
        self.create_array(target, firstspike_tag, firstspikes)
        
#===============================================================================
# DESCRIPTOR CLASSES
#===============================================================================
class Run(tb.IsDescription):
    run_id      = tb.UInt64Col()   
    pre_FR      = tb.Float64Col()   
    post_FR     = tb.Float64Col()   
    pre_mean    = tb.Float64Col()      
    post_mean   = tb.Float64Col()
    pre_std     = tb.Float64Col()      
    post_std    = tb.Float64Col()
    seed        = tb.UInt32Col()  
    pre_entropy = tb.Float32Col()
    
    
class StimRun(Run):
    stim_duration  = tb.Float32Col()
    break_duration = tb.Float32Col()
    stim_reps      = tb.Int8Col()

#===============================================================================
# METHODS
#===============================================================================
 
def load_and_merge_spikes(hfile:object, run_ids:np.ndarray, t_bins:np.ndarray, subgroup:str=None) -> np.ndarray:  
    """
    Pools and histograms the spikes of the selected run ids.
    Discards the last histogram interval as it has different behavior than remaining intervals (cf. numpy docs).
    
    :param hfile: hdf5file-object.
    :type hfile: object
    :param run_ids: The ids for which the spikes are merged.
    :type run_ids: np.ndarray
    :param t_bins: time bins passed on to np.histogram
    :type t_bins: np.ndarray
    """
    spikecounts_all_runs = []
    for run_id in run_ids:
        target = hfile.get_node(hfile.data, f"run{run_id}")
        if subgroup is not None:
            target = target[subgroup]
        spike_times = target.spikes.read()
        # Mask all spikes that are on the edge of the last interval (cf. https://numpy.org/doc/stable/reference/generated/numpy.histogram.html)
        mask = spike_times >= t_bins[-1]
        spikecounts, _ = np.histogram(spike_times[~mask], bins=t_bins)
        spikecounts_all_runs.append(spikecounts)
    return np.asarray(spikecounts_all_runs)


def get_spikes_by_sender(spikes:np.ndarray, senders:np.ndarray, N:int) -> dict:
    spikes_per_sender = np.empty(N, dtype=object)
    # Initialize all sender with zeros (in case not all neurons fire)
    for i in range(N):
        spikes_per_sender[i] = np.zeros(0)
    for i, s in enumerate(set(senders)):
        spikes_per_sender[i] = spikes[senders == s]
    return spikes_per_sender


def get_run_ids(rows:np.ndarray, params:object, tag:str):
    """
    Filters the rows to match the conditions imposed by {params} and {tag}.
    
    :param rows:
    :type rows:
    :param params: Configuration.
    :type params: object
    :param tag: The parameter that is kept constant across the change. Options are "mean", "std", "both".
    :type tag: str
    
    History:
        - added in 0.2b.
    """
    
    if tag in (mean_tag, std_tag):
        rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]
    elif tag == mean_std_tag:
        mask = np.logical_and(rows[f"pre_{mean_tag}"] != rows[f"post_{mean_tag}"], rows[f"pre_{std_tag}"] != rows[f"post_{std_tag}"])
        rows_filtered = rows[mask]
    else:
        raise ValueError("No valid tag given...")
    stim_mask = np.logical_and.reduce((rows_filtered["stim_duration"] == params.stim_duration,
        rows_filtered["break_duration"] == params.break_duration,
        rows_filtered["stim_reps"] == params.stim_reps))
    rows_filtered = rows_filtered[stim_mask]
    run_ids = rows_filtered[id_tag]
    return run_ids


def prepend_dir(filename: str, directory: str = DATA_DIR) -> PosixPath:
    # Added in v0.1
    return Path(directory).joinpath(filename)


#===============================================================================
if __name__ == '__main__':
    main()
