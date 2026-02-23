#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: A single population can be stimulated in various ways and the data is processed immediately after.
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

import nest
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
from lib.util import pairwise, save_figure
from lib.analysis import get_transient

from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm

#===============================================================================
# CONSTANTS
#===============================================================================
# hist_binwidth = 2.5 #ms
Vm_entropy_bins = np.arange(nif.V_reset-5, nif.V_th+1, .1)
    
    
pre_FR = 2.
post_FR = 4.
pre_FR = 5.
post_FR = 10.
# pre_FR = 4.
# # post_FR = 6.
# post_FR = 12.
# pre_FR = 4.
# post_FR = 2.
seeds = np.arange(20, dtype=int)


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    control, params = load_config()
    
    with ResponseHdf5(params.poisson_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # SIMULATION
        #===============================================================================
        for seed in seeds:
            if not control.force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, seed=seed)):
                logger.info("Skip simulation...")
                continue
            logger.info("Run simulation...")
            senders, spike_times = simulate(params, control, params.dt, seed=seed)
                
            logger.info("Save simulation...")
            run_id = hfile.add_run(pre_FR, post_FR, None, None, None, None, seed=seed)
            hfile.add_data_to_run(run_id, senders, spike_times)

        #===============================================================================
        # POST-PROCESSING
        #===============================================================================
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        run_ids = rows[id_tag] # id_tag is the tag for all runs
        for run_id in run_ids:                
            if not hfile.has_spikes_by_sender(run_id):
                spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
                senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
                spikes_by_sender = get_spikes_by_sender(spike_times, senders, params.N)
                hfile.add_spikes_by_sender(run_id, spikes_by_sender)
    
    logger.info("Finished...")           


def simulate(params:object, control:object, dt:float, seed:None) -> tuple:
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    logger.info(f"Seed: {seed}")
    nest.SetKernelStatus({
        "resolution": dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 2,
    })

    logger.info("Create Network...")
    generator = nest.Create(Generator.poisson_generator, params.N, params={"rate": pre_FR})
    spike_detector = nif.create_spike_detector()
    nif.measure_neuron(generator, spike_detector=spike_detector)

    logger.info("Run simulation...")
    nest.Simulate(params.warmup)
    nest.Simulate(params.duration_pre)

    logger.info("Change generator settings...")
    generator.rate = post_FR
    
    logger.info("Stimulate after changing the input...")    
    if control.double_step:
        logger.info("Stimulate after changing the input...")
        nest.Simulate(delta_step)
        logger.info("Change generator settings...")
        generator.rate = pre_FR
        nest.Simulate(params.duration_post-delta_step)
    else:
        nest.Simulate(params.duration_post)
        

    logger.info("Collect Spikes...")
    senders, spike_times = nif.collect_spikes(spike_detector).values()
    mask = np.logical_and(spike_times >= params.warmup, spike_times < params.duration_pre+params.warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_pre)}")
    mask = np.logical_and(spike_times >= params.duration_pre+params.warmup, spike_times < params.duration_pre+params.duration_post+params.warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_post)}")

    return senders, spike_times





#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
