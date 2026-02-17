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
from lib.util import pairwise, save_figure
from lib.analysis import get_transient



#===============================================================================
# CONSTANTS
#===============================================================================
test = True
test = False
force= True
force= False

weak_link = True
# weak_link = False

if test:
    NE              = 1000
    NI              = 250
    dt              = .05
    warmup          = 100.
    duration_pre    = 200.
    duration_post   = 300.
    filename        = "nettest.hdf5"
else:
    NE              = 10000
    NI              = 2500
    dt              = 0.1
    warmup          = 100.
    duration_pre    = 400.
    duration_post   = 1000.
    if weak_link:
        syn_weight = 0.2
        filename = "network_weak_link.hdf5"
    else:
        syn_weight = 0.5
        filename = "network_strong_link.hdf5"


g = 8.

Imean   = 280.
Istd    = round(siegert.find_parameter(Imean, target_FR=5., dt=dt).root, 2)
    
hist_binwidth = 2.5 #ms
metadata = {"NE": NE, "NI": NI, "dt": dt, 
            "warmup": warmup, "duration_pre": duration_pre, "duration_post": duration_post, 
            "syn_weight": syn_weight, "g": g, 
            "Imean": Imean, "Istd": Istd}

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    pre_FR = 5
    post_FR = 10
    
    means = np.arange(240, 300+1, 20.)
    seeds = np.arange(30)
    
    
    with ResponseHdf5(filename, "a", metadata=metadata) as hfile:
        #===============================================================================
        # SIMULATION
        #===============================================================================
        
        for delta in ("mean", "std"):
            for mean in means:
                pre_mean = mean 
                pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=dt).root, 2)
    
                if delta == "mean":
                    post_mean = round(siegert.find_parameter(pre_std, target_FR=post_FR, dt=dt, given_parameter="std").root, 2) # ie delta mean
                    post_std = pre_std
                elif delta == "std":
                    post_mean = mean
                    post_std = round(siegert.find_parameter(mean, target_FR=post_FR, dt=dt).root, 2) # ie delta std
                else:
                    raise ValueError("No valid delta chosen")

                for seed in seeds:
                    if not force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                           pre_mean=pre_mean, post_mean=post_mean,
                                                           pre_std=pre_std, post_std=post_std, seed=seed)):
                        logger.info("Skip simulation...")
                        continue
                    logger.info("Run simulation...")
                    (Esenders, Espike_times), (Isenders, Ispike_times), Vm = simulate(pre_mean, pre_std, post_mean, post_std, dt, seed=seed)
                    if Vm is not None:
                        time, EVm, IVm = Vm
                    
                    logger.info("Save simulation...")
                    run_id = hfile.add_run(pre_FR, post_FR, pre_mean, post_mean, pre_std, post_std, seed=seed)
                    
                    hfile.add_data_to_run(run_id, Esenders, Espike_times, subgroup=exc_tag)
                    hfile.add_data_to_run(run_id, Isenders, Ispike_times, subgroup=inh_tag)
        
        #===============================================================================
        # POST-PROCESSING
        #===============================================================================
        # rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        # run_ids = rows[id_tag]
        # for run_id in run_ids:
            # Add entropy (if not already there)
            # if not hfile.has_entropy(run_id) or not hfile.has_Vdistribution(run_id):
            #     Vm_bins = np.arange(nif.V_reset-5, nif.V_th+1, .1)
            #     Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
            #     time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
            #
            #     mask = np.logical_and(time >= warmup, time < warmup+duration_pre)
            #     dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins, density=True)
            #     pre_entropy = entropy(dist, nan_policy="raise")
            #
            #     hfile.add_entropy(run_id, pre_entropy)               
            #
            #     buffer = 0.2 * duration_post
            #     mask_post = np.logical_and(time >= warmup+duration_pre+buffer, time < warmup+duration_pre+duration_post)
            #     dist_post, _ = np.histogram(Vm[:, mask_post].ravel(), bins=Vm_bins, density=True)
            #
            #
            #     hfile.add_Vdistribution(run_id, dist, dist_post)
                
                
            # if not hfile.has_spikes_by_sender(run_id):
            #     spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
            #     senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
            #     spikes_by_sender = get_spikes_by_sender(spike_times, senders, N)
            #     hfile.add_spikes_by_sender(run_id, spikes_by_sender)
        all_delay_estimates = pd.DataFrame(columns=["delay", "tag", "mean"]).astype({
            "delay": "float",
            "tag": "string",
            "mean": "float",
        })
   
        B = 10                   # No of bootstapping
        samplesize = 10          # 25
        # DELAY - time bins
        t_bins = np.arange(0., duration_pre+duration_post+hist_binwidth, float(hist_binwidth)) + warmup
        
        for fixed in ("mean", "std"):
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                rows_filtered = rows[rows[f"pre_{fixed}"] == rows[f"post_{fixed}"]]  
                run_ids = rows_filtered[id_tag]
        
                figname = f"Spikecount with fixed {fixed} (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR})"
                fig, ax1 = plt.subplots(num=figname)
                plt.xlabel("time [ms]")
                plt.ylabel("FR [Hz]")
                plt.axvline(warmup+duration_pre, color="red", zorder=10, ls="--")
                plt.xlim(warmup+duration_pre - 10, warmup+duration_pre + 125)
                plt.xticks(list(plt.xticks()[0]) + [warmup+duration_pre, ], list(plt.xticks()[0]) + [r"$t_\Delta$", ])                
                
                Edelay_estimates = np.zeros(B)
                Idelay_estimates = np.zeros(B)
                Epopulation_FR   = np.zeros((B, t_bins.size-1))
                Ipopulation_FR   = np.zeros((B, t_bins.size-1))
                for b in range(B):
                    np.random.shuffle(run_ids)
                    samples = run_ids[:samplesize]
                    # DELAY 
                    Espikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins, subgroup=exc_tag)
                    index = (t_bins >= warmup+duration_pre).argmax() # Gets first value that is larger than duration_pre
                    
                    SEM, delay = get_transient(Espikecounts_all_runs.mean(axis=0)[index:])
                    Edelay_estimates[b]  = delay * hist_binwidth

                    Ispikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins, subgroup=inh_tag)
                    SEM, delay = get_transient(Ispikecounts_all_runs.mean(axis=0)[index:])
                    Idelay_estimates[b] = delay * hist_binwidth
                    
                    Epopulation_FR[b] = Espikecounts_all_runs.mean(axis=0) / NE / (hist_binwidth*1e-3)
                    Ipopulation_FR[b] = Ispikecounts_all_runs.mean(axis=0) / NI / (hist_binwidth*1e-3)
                
                ax2 = ax1.twinx()
                ax2.hist(Edelay_estimates + warmup + duration_pre, bins=t_bins, color="tab:blue", density=True, zorder=-4, label="delay (exc)")
                ax2.hist(Idelay_estimates + warmup + duration_pre, bins=t_bins, color="tab:orange", density=True, zorder=-4, label="delay (inh)")
                
                ax1.plot(t_bins[:-1] + hist_binwidth/2, Epopulation_FR.mean(axis=0), color="tab:blue", label="FR (exc)")
                ax1.fill_between(t_bins[:-1] + hist_binwidth/2, 
                                  Epopulation_FR.mean(axis=0) + Epopulation_FR.std(axis=0), 
                                  Epopulation_FR.mean(axis=0) - Epopulation_FR.std(axis=0),
                                  color="tab:blue", alpha=0.25)
                    
                ax1.plot(t_bins[:-1] + hist_binwidth/2, Ipopulation_FR.mean(axis=0), color="tab:orange", label="FR (inh)")
                ax1.fill_between(t_bins[:-1] + hist_binwidth/2, 
                                  Ipopulation_FR.mean(axis=0) + Ipopulation_FR.std(axis=0), 
                                  Ipopulation_FR.mean(axis=0) - Ipopulation_FR.std(axis=0),
                                  color="tab:orange", alpha=0.25)
                # ax2.set_yticks(np.linspace(0, 0.5, 3))
                
                ax1.set_ylim(bottom=0)
        
                save_figure(figname, fig)
        
                new_rows = pd.DataFrame({
                    "delay": Edelay_estimates,
                    "tag": [fixed] * len(Edelay_estimates),
                    "mean": [mean] * len(Edelay_estimates)
                })
                all_delay_estimates = pd.concat([all_delay_estimates, new_rows], ignore_index=True)
            
                ax2.set_ylabel("Density of delays")
                ax2.legend(loc="lower right")
                ax1.legend()
                ax1.set_zorder(ax2.get_zorder()+1)
                ax1.patch.set_visible(False)
                
        ##### FIGURE -  TRANSIENT ESTIMATES ######################################
        figname = f"Delay estimates (FR: {pre_FR} to {post_FR})"
        fig = plt.figure(figname)
        plt.xticks(ticks=np.arange(len(means)), labels=means)
        sns.violinplot(all_delay_estimates, x="mean", y="delay", hue="tag", cut=0, density_norm="width", common_norm=True)
        # sns.violinplot(data=all_delay_estimates, cut=0, color=color)
        # sns.stripplot(data=all_delay_estimates, color="black", size=4, jitter=True, alpha=0.5)
        plt.xlabel(r"mean drive $\mu_{pre}$")
        plt.ylabel("delay [ms]")
        plt.ylim(0, 80)
        handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
                   mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
        plt.legend(handles=handles)
        # save_figure(figname, fig)
            
#===============================================================================
# METHODS
#===============================================================================


def simulate(pre_mean:float, pre_std:float, post_mean:float, post_std:float, dt:float, seed:None) -> tuple:
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    # nest.SetKernelStatus({'print_time': True})
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    logger.info(f"Update seed: {seed}")
    nest.SetKernelStatus({
        "resolution": dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 2,
    })

    logger.info("Create Network...")
    Eneurons = nif.create_LIF(NE)
    Ineurons = nif.create_LIF(NI)
    
    
    logger.info("Connect Network...")
    # Connect(pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False)¶
    # nest.Connect(Eneurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': 300}, syn_spec={"weight": syn_weight})
    nest.Connect(Ineurons, Eneurons, conn_spec={'rule': 'fixed_indegree', 'indegree': 250}, syn_spec={"weight": -g*syn_weight})
    nest.Connect(Eneurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': 1000}, syn_spec={"weight": syn_weight})
    # nest.Connect(Ineurons, Ineurons, conn_spec={'rule': 'fixed_indegree', 'indegree': 250}, syn_spec={"weight": -g*syn_weight})


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
        "mean": Imean, "std": Istd, "dt": dt,
    })
    nif.connect_generator_with_neuron(Igenerator, Ineurons)
    
    logger.info("Stimulate E-Neurons...")
    Egenerator = nest.Create(Generator.noise_generator, 1, params={
        "mean": pre_mean, "std": pre_std, "dt": dt,
    })
    nif.connect_generator_with_neuron(Egenerator, Eneurons)

    logger.info("Run simulation...")
    nest.Simulate(warmup)
    nest.Simulate(duration_pre)

    logger.info("Change generator settings...")
    Egenerator.mean = post_mean
    Egenerator.std  = post_std
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
    time, EVm = nif.collect_Vm(Evoltmeter)
    time, IVm = nif.collect_Vm(Ivoltmeter)
    return (Esenders, Espike_times), (Isenders, Ispike_times), (time, EVm, IVm)



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
