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
__version__ = '0.1c'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import nest
nest.set_verbosity("M_WARNING")
nest.print_time = True

import numpy as np
import matplotlib.pyplot as plt


from constants import mean_tag, std_tag, mean_std_tag

import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, exc_tag, inh_tag

from lib import siegert
from lib.analysis import get_transient
from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm
from lib.rnn import RNN
from lib.util import pairwise, save_figure, functimer, h5path


#===============================================================================
# CONSTANTS
#===============================================================================
from config import load_config
control, params = load_config(is_network=True)

Imean_ext   = 260.
means = [260., ]
# means = [260., 300.,]
means = np.arange(220, 320+1, 40.)

tags = (mean_tag, std_tag, mean_std_tag)
# tags = (mean_tag, )

pre_FR  = 5
post_FR = 10
FR_I    = pre_FR

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    """
    History:
        - v0.1a: Refactor with params class.
        - v0.1b: Filter by stimulation parameters.
        - v0.1c: Update with the RNN class.
    """
    #===============================================================================
    # EXPERIMENT  - Simulation with delta for the E population
    #===============================================================================
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"
    
    rnn = RNN(params, pre_FR, post_FR, FR_I, drive_Imean=Imean_ext)

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        pre_Emeans = np.zeros(len(means))
        post_Emeans = np.zeros(len(means))
        
        pre_Estds = np.zeros(len(means))
        post_Estds = np.zeros(len(means))
        
        pre_Imeans = np.zeros(len(means))
        post_Imeans = np.zeros(len(means))
        
        pre_Istds = np.zeros(len(means))
        post_Istds = np.zeros(len(means))
        for delta in tags:
            for m, mean in enumerate(means):
                rnn.set_up_network(mean, delta)
                
                pre_Emeans[m] = rnn.pre_drive_Emean
                pre_Estds[m]  = rnn.pre_drive_Estd
                
                post_Emeans[m] = rnn.post_drive_Emean
                post_Estds[m]  = rnn.post_drive_Estd
                
                pre_Imeans[m] = rnn.pre_drive_Imean
                pre_Istds[m]  = rnn.pre_drive_Istd
                
                post_Imeans[m] = rnn.post_drive_Imean
                post_Istds[m]  = rnn.post_drive_Istd
                
            # SIMULATE
            for m, mean in enumerate(means):
                for seed in np.arange(params.seeds):
                    if not control.force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                           pre_mean=pre_Emeans[m], post_mean=post_Emeans[m],
                                                           pre_std=pre_Estds[m], post_std=post_Estds[m], seed=seed,
                                                           stim_duration=params.stim_duration, break_duration=params.break_duration, stim_reps=params.stim_reps)
                    ):
                        logger.info("Skip simulation...")
                        continue
                    
                    logger.info("Run simulation...")
                    (Esenders, Espike_times), (Isenders, Ispike_times), _ = simulate(params, control,
                                                                 pre_Emeans[m], pre_Estds[m], post_Emeans[m], post_Estds[m],
                                                                 pre_Imeans[m], pre_Istds[m], post_Imeans[m], post_Istds[m],
                                                                 seed=seed,
                                                                 )
                        
                    logger.info("Save simulation...")
                    stim_keys = ["stim_duration", "break_duration", "stim_reps"]
                    stim_kwargs = {key: getattr(params, key) for key in stim_keys}
        
                    run_id = hfile.add_run(pre_FR, post_FR, pre_Emeans[m], post_Emeans[m], pre_Estds[m], post_Estds[m], seed=seed, **stim_kwargs)
                    hfile.add_data_to_run(run_id, Esenders, Espike_times, subgroup=exc_tag)
                    hfile.add_data_to_run(run_id, Isenders, Ispike_times, subgroup=inh_tag)
                    hfile.flush()
        logger.info("Simulation finished...")
        
        #===============================================================================
        # POST-PROCESSING
        #===============================================================================
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        run_ids = rows[id_tag] # id_tag is the tag for all runs
        for run_id in run_ids:
            for pop in (exc_tag, inh_tag):
                if not hfile.has_spikes_by_sender(run_id, subgroup=pop):
                    logger.info(f"Processing (run id: {run_id})...")
                    target_group = hfile.get_node(h5path(hfile.data._v_pathname, f"run{run_id}", pop))
                    spike_times = target_group.spikes.read()
                    senders = target_group.senders.read()
                    N = params.N if pop == exc_tag else (params.N // 4)
                    spikes_by_sender = get_spikes_by_sender(spike_times, senders, N)
                    hfile.add_spikes_by_sender(run_id, spikes_by_sender, subgroup=pop)
            logger.info(f"Processing (run id: {run_id}) finished...")
        hfile.flush()
        logger.info("Processing finished...")
    
    
#===============================================================================
# METHODS
#===============================================================================
@functimer
def simulate(params:object, control:object,
        Emean_pre:float, Estd_pre:float, Emean_post:float, Estd_post:float, 
        Imean_pre:float, Istd_pre:float, Imean_post:float, Istd_post:float,
        seed:int=None) -> tuple:
    """
    History:
        - v0.1: Initial implementation.
        - v0.1a: Add params and control parameter. Remove dt param (is in params).
    """
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    # nest.SetKernelStatus({'print_time': True})
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    logger.info(f"Update seed: {seed}")
    nest.SetKernelStatus({
        "resolution": params.dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 4,
    })

    logger.info("Create Network...")
    Eneurons = nif.create_LIF(params.N)
    Ineurons = nif.create_LIF(params.N // params.gamma)
    
    logger.info("Connect Network...")
    # Connect(pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False)¶
    nest.Connect(Eneurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': params.C_EE}, syn_spec={"weight":           params.J})
    nest.Connect(Ineurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': params.C_EI}, syn_spec={"weight": -params.g*params.J})
    nest.Connect(Eneurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': params.C_IE}, syn_spec={"weight":           params.J})
    nest.Connect(Ineurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': params.C_II}, syn_spec={"weight": -params.g*params.J})


    Evoltmeter = None
    # Evoltmeter = nif.create_voltmeter(warmup)
    Espike_detector = nif.create_spike_detector()
    nif.measure_neuron(Eneurons, Evoltmeter, Espike_detector)
    
    Ivoltmeter = None
    # Ivoltmeter = nif.create_voltmeter(warmup)
    Ispike_detector = nif.create_spike_detector()
    nif.measure_neuron(Ineurons, Ivoltmeter, Ispike_detector)

    logger.info("Stimulate I-Neurons...")
    Igenerator = nest.Create(Generator.noise_generator, 1, params={
        "mean": Imean_pre, "std": Istd_pre, "dt": params.dt,
    })
    nif.connect_generator_with_neuron(Igenerator, Ineurons)
    
    logger.info("Stimulate E-Neurons...")
    Egenerator = nest.Create(Generator.noise_generator, 1, params={
        "mean": Emean_pre, "std": Estd_pre, "dt": params.dt,
    })
    nif.connect_generator_with_neuron(Egenerator, Eneurons)

    logger.info("Run simulation...")
    nest.Simulate(params.warmup)
    nest.Simulate(params.duration_pre)

    logger.info("Change generator settings...")
    Egenerator.mean = Emean_post
    Egenerator.std  = Estd_post
    Igenerator.mean = Imean_post
    Igenerator.std  = Istd_post
    
    if not control.brief_stimulus:
        logger.info("Stimulate post change...")
        nest.Simulate(params.duration_post)
    else:
        for _ in range(params.stim_reps):
            logger.info("Stimulate brief stimulus...")
            Egenerator.mean = Emean_post
            Egenerator.std  = Estd_post
            Igenerator.mean = Imean_post
            Igenerator.std  = Istd_post
            nest.Simulate(params.stim_duration)
            
            logger.info("Stimulate break time...")
            Egenerator.mean = Emean_pre
            Egenerator.std  = Estd_pre
            Igenerator.mean = Imean_pre
            Igenerator.std  = Istd_pre
            nest.Simulate(params.break_duration)
            
        logger.info("Stimulate remaining duration...")
        nest.Simulate(params.duration_post - params.stim_reps*(params.stim_duration+params.break_duration))
        
        

    logger.info("Collect exc. Spikes...")
    Esenders, Espike_times = nif.collect_spikes(Espike_detector).values()
    mask = np.logical_and(Espike_times >= params.warmup, Espike_times < params.duration_pre+params.warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_pre)}")
    mask = np.logical_and(Espike_times >= params.duration_pre+params.warmup, Espike_times < params.duration_pre+params.duration_post+params.warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N, params.duration_post)}")
    
    logger.info("Collect inh. Spikes...")
    Isenders, Ispike_times = nif.collect_spikes(Ispike_detector).values()
    mask = np.logical_and(Ispike_times >= params.warmup, Ispike_times < params.duration_pre+params.warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N // 4, params.duration_pre)}")
    mask = np.logical_and(Ispike_times >= params.duration_pre+params.warmup, Ispike_times < params.duration_pre+params.duration_post+params.warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), params.N // 4, params.duration_post)}")

    if Evoltmeter is None:
        return (Esenders, Espike_times), (Isenders, Ispike_times), None

    logger.info("Get free Vm")
    _,    EVm = nif.collect_Vm(Evoltmeter)
    time, IVm = nif.collect_Vm(Ivoltmeter)
    return (Esenders, Espike_times), (Isenders, Ispike_times), (time, EVm, IVm)


#===============================================================================
if __name__ == '__main__':
    from lib.util import play_beep
    play_beep(repeat=1)
    main()
    play_beep(pause = 0.5)
    import time
    time.sleep(1)
    plt.show()
