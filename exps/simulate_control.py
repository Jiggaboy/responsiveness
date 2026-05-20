#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: A single population can be stimulated in various ways and the data is processed immediately after.

History:
    - v0.2: Add more flexibilty with respect to the stimulus, now more stimuli with breaks in between are allowed.
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
logger.setLevel("WARNING")

import nest
nest.set_verbosity("M_WARNING")
nest.print_time = True

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


from constants import mean_tag, std_tag, mean_std_tag
from config import load_config
import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

from lib import siegert
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient

from exps.simulate import simulate


#===============================================================================
# CONSTANTS
#===============================================================================
pre_FR = 5.
post_FR = 10.

# J = 0.75
pre_mean  = -2.5
post_mean =  35


### J = 0.25
## delta mean
# mean = 220
pre_mean  = 132.5
post_mean = 181
# mean = 260
pre_mean  = 172.5
post_mean = 214
# mean = 300
pre_mean  = 212.5
post_mean = 246

## delta std
# mean = 220
pre_mean  = 132.5
post_mean = 145
# mean = 260
pre_mean  = 172.5
post_mean = 185
# # mean = 300
pre_mean  = 212.5
post_mean = 225

## delta both
# mean = 220
pre_mean  = 132.5
post_mean = 164
# mean = 260
pre_mean  = 172.5
post_mean = 199
# # mean = 300
pre_mean  = 212.5
post_mean = 236


seeds = np.arange(200, dtype=int) #40
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
@functimer
def main():
    control, params = load_config(no_stim=True)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{pre_mean}_{post_mean}" + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        pre_std  = round(siegert.find_parameter(pre_mean, target_FR=pre_FR, dt=params.dt).root, 2)
        post_std = round(siegert.find_parameter(post_mean, target_FR=post_FR, dt=params.dt).root, 2) # ie delta std
                
        # Run simulations
        for seed in seeds:
            # TODO: Add condition here for stimulus, break, and duration.
            if not control.force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                   pre_mean=pre_mean, post_mean=post_mean,
                                                   pre_std=pre_std, post_std=post_std, seed=seed,
                                                   stim_duration=params.stim_duration, break_duration=params.break_duration, stim_reps=params.stim_reps)):
                logger.info("Skip simulation...")
                continue
            logger.info("Run simulation...")
            senders, spike_times, time, Vm = simulate(params, control, pre_mean, pre_std, post_mean, post_std, seed=seed)
            
            logger.info("Save simulation...")
            run_id = hfile.add_run(pre_FR, post_FR, pre_mean, post_mean, pre_std, post_std, seed=seed, stim_duration=params.stim_duration, break_duration=params.break_duration, stim_reps=params.stim_reps)
            hfile.add_data_to_run(run_id, senders, spike_times, time, Vm)
            hfile.flush()

        #===============================================================================
        # POST-PROCESSING
        #===============================================================================
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        run_ids = rows[id_tag] # id_tag is the tag for all runs
        for run_id in run_ids:
            logger.info(f"Analyze run {run_id}...")           
                
            if not hfile.has_spikes_by_sender(run_id):
                spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
                senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
                spikes_by_sender = get_spikes_by_sender(spike_times, senders, params.N)
                hfile.add_spikes_by_sender(run_id, spikes_by_sender)
        hfile.flush()
    
    logger.info("Finished...")           



#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    from lib.util import play_beep
    play_beep(repeat=1)
    main()
    play_beep(pause = 0.5)
    import time
    time.sleep(1)
    plt.show()

