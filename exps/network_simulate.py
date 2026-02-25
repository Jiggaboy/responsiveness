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


import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender, exc_tag, inh_tag

from lib import siegert
from lib.util import pairwise, save_figure, functimer, h5path
from lib.analysis import get_transient
from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm
from constants import mean_tag, std_tag, mean_std_tag


#===============================================================================
# CONSTANTS
#===============================================================================

from config import load_config
control, params = load_config(is_network=True)

Imean_ext   = 280.
means = [260., ]

hist_binwidth = 2.5 #ms

pre_FR = 5 # for E and I
post_FR = 10
FR_I = pre_FR

seeds = np.arange(50, dtype=int)
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    """
    History:
        - v0.1a: Refactor with params class.
    """
    #===============================================================================
    # EXPERIMENT  - Simulation with delta for the E population
    #===============================================================================
    with ResponseHdf5(params.network_filename, "a", metadata=params.metadata) as hfile:
        pre_Emeans = np.zeros(len(means))
        post_Emeans = np.zeros(len(means))
        pre_Estds = np.zeros(len(means))
        post_Estds = np.zeros(len(means))
        pre_Imeans = np.zeros(len(means))
        post_Imeans = np.zeros(len(means))
        pre_Istds = np.zeros(len(means))
        post_Istds = np.zeros(len(means))
        for delta in (mean_tag, std_tag, mean_std_tag):
            for m, mean in enumerate(means):
                ### EXCITATION
                ## PRE
                EEmean_pre, EEvar_pre = siegert.get_drive_moments(nif.tau,           params.J, params.C_EE, pre_FR)  # these are the moments of the free Vm
                EImean_pre, EIvar_pre = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_EI, FR_I)    # free Vm
                Emean_int_pre = from_free_Vm_to_generator(EEmean_pre + EImean_pre)              # given in pA
                Evar_int_pre  = from_free_Vm_to_generator(var_V=(EEvar_pre + EIvar_pre), dt=params.dt) # given in pA
    
                # Total fluctuation level with network recurrency
                Estd_pre_tmp = siegert.find_parameter(mean+Emean_int_pre, target_FR=pre_FR, dt=params.dt).root # Estd_tmp in Generator space/pA
    
                Estd_pre  = np.sqrt(Estd_pre_tmp**2 - Evar_int_pre) # recurrent network compensated
                pre_Emeans[m] = mean
                pre_Estds[m] = Estd_pre
    
                ## POST
                # Recurrent network effects
                EEmean_post, EEvar_post = siegert.get_drive_moments(nif.tau,           params.J, params.C_EE, post_FR) # these are the moments of the free Vm
                EImean_post, EIvar_post = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_EI, FR_I)
                Emean_int_post = from_free_Vm_to_generator(EEmean_post + EImean_post)              # given in pA
                Evar_int_post  = from_free_Vm_to_generator(var_V=(EEvar_post + EIvar_post), dt=params.dt) # given in pA
    
                if delta == mean_tag:
                    # Delta mean (Factor sqrt(2) required, cf Tsodyks 1991)
                    Emean_post_tmp = siegert.find_parameter(np.sqrt(Estd_pre**2 + Evar_int_post), target_FR=post_FR, given_parameter="std", dt=params.dt).root # Estd_tmp in Generator space

                    post_Emeans[m] = Emean_post_tmp - Emean_int_post
                    post_Estds[m] = Estd_pre
                elif delta == std_tag:
                    # Delta std - keeping the same ext. drive; update with new internal network effects
                    Estd_post_tmp = siegert.find_parameter(mean+Emean_int_post, target_FR=post_FR, dt=params.dt).root # Estd_tmp in Generator space
                    Estd_post  = np.sqrt(Estd_post_tmp**2 - Evar_int_post) # Remove updated internal network effects

                    post_Emeans[m] = mean
                    post_Estds[m] = Estd_post
                elif delta == mean_std_tag:                    
                    # Delta std - keeping the same ext. drive; update with new internal network effects
                    Estd_post_tmp = siegert.find_parameter(mean+Emean_int_post, target_FR=post_FR, dt=params.dt).root # Estd_tmp in Generator space
                    Estd_post  = np.sqrt(Estd_post_tmp**2 - Evar_int_post) # Remove updated internal network effects
                    
                    # Set the std post change to the pre + delta/2
                    post_Estds[m] = pre_Estds[m] + (Estd_post - pre_Estds[m]) / 2
                    
                    # Update the post mean accordingly
                    Emean_post_tmp = siegert.find_parameter(np.sqrt(post_Estds[m]**2 + Evar_int_post), target_FR=post_FR, given_parameter="std", dt=params.dt).root # Estd_tmp in Generator space
                    post_Emeans[m] = Emean_post_tmp - Emean_int_post
                else:
                    raise ValueError("No valid delta chosen")


                
                ### INHIBITION
                ## PRE
                IEmean_pre, IEvar_pre = siegert.get_drive_moments(nif.tau,           params.J, params.C_IE, pre_FR)
                IImean_pre, IIvar_pre = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_II, FR_I)
                Imean_int_pre = from_free_Vm_to_generator(IEmean_pre + IImean_pre)              # given in pA
                Ivar_int_pre  = from_free_Vm_to_generator(var_V=(IEvar_pre + IIvar_pre), dt=params.dt) # given in pA
                
                Istd_pre_tmp = siegert.find_parameter(Imean_ext+Imean_int_pre, target_FR=FR_I, dt=params.dt).root
                pre_Imeans[m] = Imean_ext
                pre_Istds[m]  = np.sqrt(Istd_pre_tmp**2 - Ivar_int_pre) # recurrent network compensated
    
    
                ## POST
                # Recurrent networks effects
                
                IEmean_post, IEvar_post = siegert.get_drive_moments(nif.tau,           params.J, params.C_IE, post_FR)   # these are the moments of the free Vm
                IImean_post, IIvar_post = siegert.get_drive_moments(nif.tau, -params.g*params.J, params.C_II, FR_I)      # free Vm
                Imean_int_post = from_free_Vm_to_generator(IEmean_post + IImean_post)               # given in pA
                Ivar_int_post  = from_free_Vm_to_generator(var_V=(IEvar_post + IIvar_post), dt=params.dt)  # given in pA
                
                # # Delta std - Compensation for the increased exc. FR
                # Istd_post_tmp = siegert.find_parameter(Imean_ext + Imean_int_post, target_FR=FR_I, dt=dt).root
                # post_Imeans[m] = Imean_ext
                # post_Istds[m]  = np.sqrt(Istd_post_tmp**2 - Ivar_int_post)
                
                # Delta mean - Compensation for the increased exc. FR
                Imean_post_tmp = siegert.find_parameter(np.sqrt(pre_Istds[m]**2 + Ivar_int_post), target_FR=FR_I, given_parameter="std", dt=params.dt).root
                post_Imeans[m] = Imean_post_tmp - Imean_int_post
                post_Istds[m]  = pre_Istds[m]
                
            # SIMULATE
            for m, mean in enumerate(means):
                for seed in seeds:
                    if not control.force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                           pre_mean=pre_Emeans[m], post_mean=post_Emeans[m],
                                                           pre_std=pre_Estds[m], post_std=post_Estds[m], seed=seed)):
                        logger.info("Skip simulation...")
                        continue
                    
                    logger.info("Run simulation...")
                    (Esenders, Espike_times), (Isenders, Ispike_times), _ = simulate(params, control,
                                                                 pre_Emeans[m], pre_Estds[m], post_Emeans[m], post_Estds[m],
                                                                 pre_Imeans[m], pre_Istds[m], post_Imeans[m], post_Istds[m],
                                                                 )
                        
                    logger.info("Save simulation...")
                    run_id = hfile.add_run(pre_FR, post_FR, pre_Emeans[m], post_Emeans[m], pre_Estds[m], post_Estds[m], seed=seed)
                    hfile.add_data_to_run(run_id, Esenders, Espike_times, subgroup=exc_tag)
                    hfile.add_data_to_run(run_id, Isenders, Ispike_times, subgroup=inh_tag)
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
        logger.info("Processing finished...")
    return    
    t_bins = np.arange(0., params.duration_pre+params.duration_post+hist_binwidth, float(hist_binwidth)) + params.warmup
        
    plt.figure()
    plt.xlabel("time [ms]")
    plt.ylabel("FR [Hz]")
    
    mask = Espike_times >= t_bins[-1]
    spikecounts, _ = np.histogram(Espike_times[~mask], bins=t_bins)
    FR = spikecounts / params.N / (hist_binwidth*1e-3)
    print("Exc:", FR.mean())
                    
    plt.plot(t_bins[:-1], FR, label="exc")
    
    
    mask = Ispike_times >= t_bins[-1]
    spikecounts, _ = np.histogram(Ispike_times[~mask], bins=t_bins)
    FR = spikecounts / (params.N // 4) / (hist_binwidth*1e-3)
    print("Inh:", FR.mean())
    plt.plot(t_bins[:-1], FR, label="inh")
    
    plt.legend()
    
    
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
    Ineurons = nif.create_LIF(params.N // 4)
    
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
    logger.info("Stimulate after changing the input...")
    nest.Simulate(params.duration_post)

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
    main()
    plt.show()
