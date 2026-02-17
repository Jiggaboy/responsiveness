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
from lib.util import pairwise, save_figure, functimer
from lib.analysis import get_transient
from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm


#===============================================================================
# CONSTANTS
#===============================================================================
NE              = 10000
NI              = 2500
dt              = .02
warmup          = 100.
duration_pre    = 300.
duration_post   = 600.
filename        = "nettest.hdf5"

J = syn_weight = 0.01
g = 8.
# Target-Source notation
# Indegree definition
C_EE = 100
C_EI = 200
C_IE = 200
C_II = 100

Imean_ext   = 280.
Emean_ext   = 260.


hist_binwidth = 2.5 #ms

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    pre_FR = 5 # for E and I
    post_FR = 10
    FR_I = pre_FR
    
    
    #===============================================================================
    # EXPERIMENT 1 - Converting between generator space (pA) and free Vm
    #===============================================================================
    #
    # # Network effects
    # EEmean_pre, EEvar_pre = siegert.get_drive_moments(nif.tau, J, C_EE, pre_FR)     # these are the moments of the free Vm
    # EImean_pre, EIvar_pre = siegert.get_drive_moments(nif.tau, -g*J, C_EI, FR_I)    # free Vm
    # Emean_int_pre = from_free_Vm_to_generator(EEmean_pre + EImean_pre)              # given in pA
    # Evar_int_pre  = from_free_Vm_to_generator(var_V=(EEvar_pre + EIvar_pre), dt=dt) # given in pA
    #
    # # find_parameter is in generator space
    # Estd_tmp = siegert.find_parameter(Emean_ext+Emean_int_pre, target_FR=pre_FR, dt=dt).root # Estd_tmp in Generator space
    # print("Mean (External Generator space):", Emean_ext)
    # print("Mean (Internal Generator space):", Emean_int_pre)
    # print("Mean (Total Generator space):", Emean_ext+Emean_int_pre)
    # print("Mean (Total induced free Vm):", from_generator_to_free_Vm(Emean_ext+Emean_int_pre))
    # print("Mean (internal free Vm):", from_generator_to_free_Vm(EEmean_pre + EImean_pre))
    # print("Mean (external free Vm):", from_generator_to_free_Vm(Emean_ext)) 
    # print()
    # print("Var (External Generator space):", Estd_tmp**2-Evar_int_pre)
    # print("Var (Internal Generator space):", Evar_int_pre)
    # print("Var (Total Generator space):", Estd_tmp**2)
    # print("Var (Total induced free Vm):", from_generator_to_free_Vm(var_pA=Estd_tmp**2, dt=dt))
    # print("Var (internal free Vm):", from_generator_to_free_Vm(var_pA=Evar_int_pre, dt=dt))
    # print("Var (external free Vm):", from_generator_to_free_Vm(var_pA=Estd_tmp**2-Evar_int_pre, dt=dt))
    #
    #=============================================================================== TEST - START
    # EEmean_pre, EEvar_pre = siegert.get_drive_moments(nif.tau, J, C_EE, pre_FR)     # these are the moments of the free Vm
    # EImean_pre, EIvar_pre = siegert.get_drive_moments(nif.tau, -g*J, C_EI, FR_I)    # these are the moments of the free Vm
    # Evar_int_pre = (EEvar_pre + EIvar_pre)                                          # these are the moments of the free Vm
    #
    # print("New method:")
    # print("Mean (External Generator space):", Emean_ext)
    # print("Mean (Internal Generator space):", from_free_Vm_to_generator(EEmean_pre+EImean_pre))
    # print("Mean (Total Generator space):", Emean_ext + from_free_Vm_to_generator(EEmean_pre+EImean_pre))
    # print("Mean (Total induced free Vm):", from_generator_to_free_Vm(Emean_ext) + (EEmean_pre+EImean_pre))
    # print("Mean (internal free Vm):", from_generator_to_free_Vm(EEmean_pre + EImean_pre))
    # print("Mean (external free Vm):", from_generator_to_free_Vm(Emean_ext))
    #
    # Estd_tmp = siegert.find_parameter_new(from_generator_to_free_Vm(Emean_ext) + (EEmean_pre + EImean_pre), target_FR=pre_FR, dt=dt).root # Estd_tmp in free Vm space
    # print()
    # print("Var (External Generator space):", from_free_Vm_to_generator(var_V=Estd_tmp**2/2-Evar_int_pre, dt=dt))
    # print("Var (Internal Generator space):", from_free_Vm_to_generator(var_V=Evar_int_pre, dt=dt))
    # print("Var (Total Generator space):", from_free_Vm_to_generator(var_V=Estd_tmp**2/2, dt=dt))
    # print("Var (Total induced free Vm):", Estd_tmp**2/2)
    # print("Var (internal free Vm):", Evar_int_pre)
    # print("Var (external free Vm):", Estd_tmp**2/2-Evar_int_pre) 
    #=============================================================================== TEST - END
    
    #===============================================================================
    # EXPERIMENT 2 - Simulation with delta mean for the E population
    #===============================================================================
    ## Pre
    # Recurrent network effects
    EEmean_pre, EEvar_pre = siegert.get_drive_moments(nif.tau,    J, C_EE, pre_FR)  # these are the moments of the free Vm
    EImean_pre, EIvar_pre = siegert.get_drive_moments(nif.tau, -g*J, C_EI, FR_I)    # free Vm
    Emean_int_pre = from_free_Vm_to_generator(EEmean_pre + EImean_pre)              # given in pA
    Evar_int_pre  = from_free_Vm_to_generator(var_V=(EEvar_pre + EIvar_pre), dt=dt) # given in pA
    
    # Total fluctuation level with network recurrency
    Estd_pre_tmp = siegert.find_parameter(Emean_ext+Emean_int_pre, target_FR=pre_FR, dt=dt).root # Estd_tmp in Generator space/pA
    
    Emean_pre = Emean_ext
    Estd_pre  = np.sqrt(Estd_pre_tmp**2 - Evar_int_pre) # recurrent network compensated
    
    
    ## Post
    # Recurrent network effects
    EEmean_post, EEvar_post = siegert.get_drive_moments(nif.tau,    J, C_EE, post_FR) # these are the moments of the free Vm
    EImean_post, EIvar_post = siegert.get_drive_moments(nif.tau, -g*J, C_EI, FR_I)
    Emean_int_post = from_free_Vm_to_generator(EEmean_post + EImean_post)              # given in pA
    Evar_int_post  = from_free_Vm_to_generator(var_V=(EEvar_post + EIvar_post), dt=dt) # given in pA
    
    
    # Delta std - keeping the same ext. drive; update with new internal network effects
    Estd_post_tmp = siegert.find_parameter(Emean_ext+Emean_int_post, target_FR=post_FR, dt=dt).root # Estd_tmp in Generator space
    Emean_post = Emean_ext
    Estd_post  = np.sqrt(Estd_post_tmp**2 - Evar_int_post) # Remove updated internal network effects

    print("Delta std")
    print(f"Ext. Pre to post (mean): {Emean_pre} to {Emean_post}")
    print(f"Int. Pre to post (mean): {Emean_int_pre} to {Emean_int_post}")
    print(f"Ext. Pre to post (std): {Estd_pre} to {Estd_post}")
    print(f"Int. Pre to post (std): {np.sqrt(Evar_int_pre)} to {np.sqrt(Evar_int_post)}")
    
    # Delta mean (Factor sqrt(2) required, cf Tsodyks 1991)
    Emean_post_tmp = siegert.find_parameter(np.sqrt(Estd_pre**2 + Evar_int_post), target_FR=post_FR, given_parameter="std", dt=dt).root # Estd_tmp in Generator space
    Emean_post = Emean_post_tmp - Emean_int_post
    Estd_post  = Estd_pre
    
    print("Delta mean")
    print(f"Ext. Pre to post (mean): {Emean_pre} to {Emean_post}")
    print(f"Int. Pre to post (mean): {Emean_int_pre} to {Emean_int_post}")
    print(f"Ext. Pre to post (std): {Estd_pre} to {Estd_post}")
    print(f"Int. Pre to post (std): {Evar_int_pre} to {Evar_int_post}")
        

    IEmean_pre, IEvar_pre = siegert.get_drive_moments(nif.tau,    J, C_IE, pre_FR)
    IImean_pre, IIvar_pre = siegert.get_drive_moments(nif.tau, -g*J, C_II, FR_I)
    Imean_int_pre = from_free_Vm_to_generator(IEmean_pre + IImean_pre)              # given in pA
    Ivar_int_pre  = from_free_Vm_to_generator(var_V=(IEvar_pre + IIvar_pre), dt=dt) # given in pA
    
    Istd_pre_tmp = siegert.find_parameter(Imean_ext+Imean_int_pre, target_FR=FR_I, dt=dt).root
    Imean_pre = Imean_ext
    Istd_pre  = np.sqrt(Istd_pre_tmp**2 - Ivar_int_pre) # recurrent network compensated
    
    # No change (external current is only applied to the exc. population, only network effects are observed here)
    Imean_post = Imean_pre
    Istd_post  = Istd_pre
    
    ## Post
    # Recurrent networks effects
    
    IEmean_post, IEvar_post = siegert.get_drive_moments(nif.tau,    J, C_IE, post_FR)   # these are the moments of the free Vm
    IImean_post, IIvar_post = siegert.get_drive_moments(nif.tau, -g*J, C_II, FR_I)      # free Vm
    Imean_int_post = from_free_Vm_to_generator(IEmean_post + IImean_post)               # given in pA
    Ivar_int_post  = from_free_Vm_to_generator(var_V=(IEvar_post + IIvar_post), dt=dt)  # given in pA
    
    # Delta std - Compensation for the increased exc. FR
    Istd_post_tmp = siegert.find_parameter(Imean_ext + Imean_int_post, target_FR=FR_I, dt=dt).root
    Imean_post = Imean_ext
    Istd_post  = np.sqrt(Istd_post_tmp**2 - Ivar_int_post)
    
    
    print("Delta std (inh)")
    print(f"Ext. Pre to post (mean): {Imean_pre} to {Imean_post}")
    print(f"Int. Pre to post (mean): {Imean_int_pre} to {Imean_int_post}")
    print(f"Ext. Pre to post (std): {Istd_pre} to {Istd_post}")
    print(f"Int. Pre to post (std): {Ivar_int_pre} to {Ivar_int_post}")
    
    # Delta mean - Compensation for the increased exc. FR
    Imean_post_tmp = siegert.find_parameter(np.sqrt(Istd_pre**2 + Ivar_int_post), target_FR=FR_I, given_parameter="std", dt=dt).root
    Imean_post = Imean_post_tmp - Imean_int_post
    Istd_post  = Istd_pre
    
    
    print("Delta mean (inh)")
    print(f"Ext. Pre to post (mean): {Imean_pre} to {Imean_post}")
    print(f"Int. Pre to post (mean): {Imean_int_pre} to {Imean_int_post}")
    print(f"Ext. Pre to post (std): {Istd_pre} to {Istd_post}")
    print(f"Int. Pre to post (std): {Ivar_int_pre} to {Ivar_int_post}")
    

    (Esenders, Espike_times), (Isenders, Ispike_times), _ = simulate(Emean_pre, Estd_pre, Emean_post, Estd_post,
                                                                     Imean_pre, Istd_pre, Imean_post, Istd_post,
                                                                     dt=dt)
    
    t_bins = np.arange(0., duration_pre+duration_post+hist_binwidth, float(hist_binwidth)) + warmup
        
    plt.figure()
    plt.xlabel("time [ms]")
    plt.ylabel("FR [Hz]")
    
    mask = Espike_times >= t_bins[-1]
    spikecounts, _ = np.histogram(Espike_times[~mask], bins=t_bins)
    FR = spikecounts / NE / (hist_binwidth*1e-3)
    print("Exc:", FR.mean())
                    
    plt.plot(t_bins[:-1], FR, label="exc")
    
    
    mask = Ispike_times >= t_bins[-1]
    spikecounts, _ = np.histogram(Ispike_times[~mask], bins=t_bins)
    FR = spikecounts / NI / (hist_binwidth*1e-3)
    print("Inh:", FR.mean())
    plt.plot(t_bins[:-1], FR, label="inh")
    
    plt.legend()
    
    
#===============================================================================
# METHODS
#===============================================================================
@functimer
def simulate(Emean_pre:float, Estd_pre:float, Emean_post:float, Estd_post:float, 
             Imean_pre:float, Istd_pre:float, Imean_post:float, Istd_post:float, 
             dt:float, seed:int=None) -> tuple:
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    # nest.SetKernelStatus({'print_time': True})
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    logger.info(f"Update seed: {seed}")
    nest.SetKernelStatus({
        "resolution": dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 4,
    })

    logger.info("Create Network...")
    Eneurons = nif.create_LIF(NE)
    Ineurons = nif.create_LIF(NI)
    
    logger.info("Connect Network...")
    # Connect(pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False)¶
    nest.Connect(Eneurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': C_EE}, syn_spec={"weight": syn_weight})
    nest.Connect(Ineurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': C_EI}, syn_spec={"weight": -g*syn_weight})
    nest.Connect(Eneurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': C_IE}, syn_spec={"weight": syn_weight})
    nest.Connect(Ineurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': C_II}, syn_spec={"weight": -g*syn_weight})


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
        "mean": Imean_pre, "std": Istd_pre, "dt": dt,
    })
    nif.connect_generator_with_neuron(Igenerator, Ineurons)
    
    logger.info("Stimulate E-Neurons...")
    Egenerator = nest.Create(Generator.noise_generator, 1, params={
        "mean": Emean_pre, "std": Estd_pre, "dt": dt,
    })
    nif.connect_generator_with_neuron(Egenerator, Eneurons)

    logger.info("Run simulation...")
    nest.Simulate(warmup)
    nest.Simulate(duration_pre)

    logger.info("Change generator settings...")
    Egenerator.mean = Emean_post
    Egenerator.std  = Estd_post
    Igenerator.mean = Imean_post
    Igenerator.std  = Istd_post
    logger.info("Stimulate after changing the input...")
    nest.Simulate(duration_post)

    logger.info("Collect exc. Spikes...")
    Esenders, Espike_times = nif.collect_spikes(Espike_detector).values()
    mask = np.logical_and(Espike_times >= warmup, Espike_times < duration_pre+warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), NE, duration_pre)}")
    mask = np.logical_and(Espike_times >= duration_pre+warmup, Espike_times < duration_pre+duration_post+warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), NE, duration_post)}")
    
    logger.info("Collect inh. Spikes...")
    Isenders, Ispike_times = nif.collect_spikes(Ispike_detector).values()
    mask = np.logical_and(Ispike_times >= warmup, Ispike_times < duration_pre+warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), NI, duration_pre)}")
    mask = np.logical_and(Ispike_times >= duration_pre+warmup, Ispike_times < duration_pre+duration_post+warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), NI, duration_post)}")

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
