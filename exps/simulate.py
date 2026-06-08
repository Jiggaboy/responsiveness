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


#===============================================================================
# CONSTANTS
#===============================================================================
Vm_entropy_bins = np.arange(nif.V_reset-5, nif.V_th+1, .1)

    
pre_FR = 2.
post_FR = 3.
# post_FR = 4.
post_FR = 5.
# post_FR = 6.
# post_FR = 7.
# post_FR = 8.
#
pre_FR = 5.
post_FR = 10.
#
pre_FR = 10.
post_FR = 5.
# # #
# pre_FR = 4.
# post_FR = 6.
# post_FR = 8.
# post_FR = 10.
# # post_FR = 12.

# pre_FR = 4.
# post_FR = 2.
means = np.arange(220, 320+1, 40.)
# means = np.asarray([-2.5, 35])
# means = np.arange(260, 320+1, 120.)
# means = np.arange(240, 290+1, 10.)
# means = np.append(means, 320.
seeds = np.arange(200, dtype=int) #40
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
@functimer
def main():
    control, params = load_config()
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        #===============================================================================
        # SIMULATION
        #===============================================================================
        pre_means = np.zeros(len(means))
        post_means = np.zeros(len(means))
        pre_stds = np.zeros(len(means))
        post_stds = np.zeros(len(means))
        # for delta in (mean_tag, ):
        for delta in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                pre_means[m] = mean
                pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=params.dt).root, 2)
                pre_stds[m] = pre_std

                if delta == mean_tag:
                    post_means[m] = round(siegert.find_parameter(pre_std, target_FR=post_FR, given_parameter=std_tag, dt=params.dt).root, 2) # ie delta mean
                    post_stds[m] = pre_std
                elif delta == std_tag:
                    post_means[m] = mean
                    post_stds[m] = round(siegert.find_parameter(mean, target_FR=post_FR, dt=params.dt).root, 2) # ie delta std
                elif delta == mean_std_tag:
                    post_std = round(siegert.find_parameter(mean, target_FR=post_FR, dt=params.dt).root, 2)
                    post_stds[m] = pre_std + (post_std - pre_std) / 2
                    post_mean  = round(siegert.find_parameter(pre_std, target_FR=post_FR, dt=params.dt, given_parameter=std_tag).root, 2)
                    post_means[m] = mean + (post_mean - mean) / 2
                else:
                    raise ValueError("No valid delta chosen")

            # Run simulations
            for pre_mean, post_mean, pre_std, post_std in zip(pre_means, post_means, pre_stds, post_stds):
                logger.warning(f"Tag: {delta}; Mean: {pre_mean}")
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
            # Add entropy (if not already there)
            # if not hfile.has_entropy(run_id) or not hfile.has_Vdistribution(run_id):
            #     Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
            #     time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
            #
            #     # Pre
            #     mask = np.logical_and(time >= params.warmup, time < params.warmup+params.duration_pre)
            #     dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_entropy_bins, density=True)
            #     pre_entropy = entropy(dist, nan_policy="raise")
            #
            #     hfile.add_entropy(run_id, pre_entropy)               
            #
            #     # Post        
            #     buffer = 0.2 * params.duration_post
            #     mask_post = np.logical_and(time >= params.warmup+params.duration_pre+buffer, time < params.warmup+params.duration_pre+params.duration_post)
            #     dist_post, _ = np.histogram(Vm[:, mask_post].ravel(), bins=Vm_entropy_bins, density=True)
            #
            #     hfile.add_Vdistribution(run_id, dist, dist_post)
            #

                
            if not hfile.has_spikes_by_sender(run_id):
                spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
                senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
                spikes_by_sender = get_spikes_by_sender(spike_times, senders, params.N)
                hfile.add_spikes_by_sender(run_id, spikes_by_sender)
        hfile.flush()
    
    logger.info("Finished...")           

# @functimer
def simulate(params:object, control:object, pre_mean:float, pre_std:float, post_mean:float, post_std:float, seed:None) -> tuple:
    """
    History:
        - v0.1: Initial implementation.
        - v0.1a: Remove dt as parameter (use params.dt instead).
    """
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    # seed = rnd.randint(0, 2**32-1)
    logger.info(f"Seed: {seed}")
    nest.SetKernelStatus({
        "resolution": params.dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 4,
    })

    logger.info("Create Network...")
    neurons = nif.create_LIF(params.N)
    voltmeter = None
    # voltmeter = nif.create_voltmeter(params.warmup)
    spike_detector = nif.create_spike_detector()
    nif.measure_neuron(neurons, voltmeter, spike_detector)

    logger.info("Stimulate Neurons...")
    generator = nest.Create(Generator.noise_generator, 1, params={
        "mean": pre_mean, "std": pre_std, "dt": params.dt,
    })
    nif.connect_generator_with_neuron(generator, neurons)


    with nest.RunManager():
        logger.info("Run simulation...")
        nest.Run(params.warmup)
        nest.Run(params.duration_pre)
    
        logger.info("Change generator settings...")
        generator.mean = post_mean
        generator.std = post_std
        
        logger.info("Stimulate after changing the input...")    
        if control.brief_stimulus:
            for _ in range(params.stim_reps):
                generator.mean = post_mean
                generator.std = post_std
                logger.info("Stimulate after changing the input...")
                nest.Run(params.stim_duration)
                
                logger.info("Return to pre generator-settings...")
                generator.mean = pre_mean
                generator.std = pre_std
                
                logger.info("Simulate break...")
                nest.Run(params.break_duration)
            logger.info("Simulate remaining duration...")
            nest.Run(params.duration_post - params.stim_reps*(params.stim_duration+params.break_duration))
        else:
            nest.Run(params.duration_post)
            

    logger.info("Collect Spikes...")
    senders, spike_times = nif.collect_spikes(spike_detector).values()
    mask = np.logical_and(spike_times >= params.warmup, spike_times < params.duration_pre+params.warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_pre)}")
    mask = np.logical_and(spike_times >= params.duration_pre+params.warmup + 0.25*params.duration_post, spike_times < params.duration_pre+params.duration_post+params.warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_post)}")

    if voltmeter is None:
        return senders, spike_times, None, None

    logger.info("Get free Vm...")
    time, Vm = nif.collect_Vm(voltmeter)
    return senders, spike_times, time, Vm





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

