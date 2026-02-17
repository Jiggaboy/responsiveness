#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


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


import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
import seaborn as sns


import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

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

double_step = True
double_step = False
double_change = True
# double_change = False

if test:
    N               = 100
    dt              = .05
    warmup          = 100.
    duration_pre    = 200.
    duration_post   = 300.
    filename        = "test_data.hdf5"
else:
    N               = 2500
    dt              = 0.1
    warmup          = 100.
    duration_pre    = 400.
    duration_post   = 1000.
    delta_step      = nif.tau
    if double_step and not double_change:
        filename        = "double_step.hdf5"
    elif double_change and not double_step:
        filename        = "double_change.hdf5"
    elif double_change and double_step:
        filename        = "double_step_change.hdf5"
    else:
        raise ValueError("Invalid arguments")
    
    
hist_binwidth = 2.5 #ms
metadata = {"N": N, "dt": dt, "warmup": warmup, "duration_pre": duration_pre, "duration_post": duration_post, }

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
    

def main():
    pre_FR = 2.
    post_FR = 4.
    # pre_FR = 5.
    # post_FR = 10.
    # pre_FR = 4.
    # # post_FR = 6.
    # post_FR = 12.
    # pre_FR = 4.
    # post_FR = 2.
    means = np.arange(240, 320+1, 20.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)
    # rnd = np.random.RandomState()
    # seed = rnd.randint(0, 2**32-1)
    seeds = np.arange(60, dtype=int)

    
    with ResponseHdf5(filename, "a", metadata=metadata) as hfile:
        #===============================================================================
        # SIMULATION
        #===============================================================================
        pre_means = np.zeros(len(means))
        post_means = np.zeros(len(means))
        pre_stds = np.zeros(len(means))
        post_stds = np.zeros(len(means))
        if not double_change:
            for delta in ("mean", "std"):
                for m, mean in enumerate(means):
                    pre_means[m] = mean
                    pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=dt).root, 2)
                    pre_stds[m] = pre_std
                    
                    if delta == "mean":
                        post_means[m] = round(siegert.find_parameter(pre_std, target_FR=post_FR, dt=dt, given_parameter="std").root, 2) # ie delta mean
                        post_stds[m] = pre_std
                    elif delta == "std":
                        post_means[m] = mean
                        post_stds[m] = round(siegert.find_parameter(mean, target_FR=post_FR, dt=dt).root, 2) # ie delta std
                    else:
                        raise ValueError("No valid delta chosen")
                        
                for pre_mean, post_mean, pre_std, post_std in zip(pre_means, post_means, pre_stds, post_stds):
                    for seed in seeds:
                        if not force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                               pre_mean=pre_mean, post_mean=post_mean,
                                                               pre_std=pre_std, post_std=post_std, seed=seed)):
                            logger.info("Skip simulation...")
                            continue
                        logger.info("Run simulation...")
                        senders, spike_times, time, Vm = simulate(pre_mean, pre_std, post_mean, post_std, dt, seed=seed)
                        
                        logger.info("Save simulation...")
                        run_id = hfile.add_run(pre_FR, post_FR, pre_mean, post_mean, pre_std, post_std, seed=seed)
                        hfile.add_data_to_run(run_id, senders, spike_times, time, Vm)
        elif double_change:
            for m, mean in enumerate(means):
                pre_means[m] = mean
                pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=dt).root, 2)
                pre_stds[m] = pre_std
                post_std = round(siegert.find_parameter(mean, target_FR=post_FR, dt=dt).root, 2)
                post_stds[m] = pre_std + (post_std - pre_std) / 2
                post_mean  = round(siegert.find_parameter(pre_std, target_FR=post_FR, dt=dt, given_parameter="std").root, 2)
                post_means[m] = mean + (post_mean - mean) / 2
        else:
            raise ValueError("No parameter defined")  
                 
        for pre_mean, post_mean, pre_std, post_std in zip(pre_means, post_means, pre_stds, post_stds):
            for seed in seeds:
                if not force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                       pre_mean=pre_mean, post_mean=post_mean,
                                                       pre_std=pre_std, post_std=post_std, seed=seed)):
                    logger.info("Skip simulation...")
                    continue
                logger.info("Run simulation...")
                senders, spike_times, time, Vm = simulate(pre_mean, pre_std, post_mean, post_std, dt, seed=seed)
                
                logger.info("Save simulation...")
                run_id = hfile.add_run(pre_FR, post_FR, pre_mean, post_mean, pre_std, post_std, seed=seed)
                hfile.add_data_to_run(run_id, senders, spike_times, time, Vm)

                       
        #===============================================================================
        # POST-PROCESSING
        #===============================================================================
        rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR)
        run_ids = rows[id_tag]
        for run_id in run_ids:
            # Add entropy (if not already there)
            if not hfile.has_entropy(run_id) or not hfile.has_Vdistribution(run_id):
                Vm_bins = np.arange(nif.V_reset-5, nif.V_th+1, 1)
                Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
                time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
                
                mask = np.logical_and(time >= warmup, time < warmup+duration_pre)
                dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins)
                pre_entropy = entropy(dist, nan_policy="raise")
                
                hfile.add_entropy(run_id, pre_entropy)               
                
                logger.info("Update dist")
                buffer = 0.2 * duration_post
                mask_post = np.logical_and(time >= warmup+duration_pre+buffer, time < warmup+duration_pre++duration_post)
                dist_post, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins)
                hfile.add_Vdistribution(run_id, dist, dist_post)
                
                
            if not hfile.has_spikes_by_sender(run_id):
                spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
                senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
                spikes_by_sender = get_spikes_by_sender(spike_times, senders, N)
                hfile.add_spikes_by_sender(run_id, spikes_by_sender)

        #===============================================================================
        # MORE METHODS
        #===============================================================================
        # rows_filtered = rows[rows[f"pre_mean"] == rows[f"post_mean"]]  # delta std
        # rows_filtered = rows[rows["pre_std"] == rows["post_std"]]    # delta mean
    
        all_delay_estimates = pd.DataFrame(columns=["delay", "tag", "mean"]).astype({
            "delay": "float",
            "tag": "string",
            "mean": "float",
        })
                
        B = 100 # No of bootstapping
        samplesize = 30          # 25
        # DELAY - time bins
        t_bins = np.arange(0., duration_pre+duration_post+hist_binwidth, float(hist_binwidth)) + warmup
        
        iterator = ("mean", "std") if not double_change else ("None", )
        for tag in iterator:
            all_pre_entropy_estimates = np.zeros((len(means), B))
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                if double_change:
                    rows_filtered = rows
                else:
                    rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
                run_ids = rows_filtered[id_tag]
                
                figname = f"Double-Step with fixed {tag} (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR})"
                fig, ax1 = plt.subplots(num=figname)
                plt.xlabel("time [ms]")
                plt.ylabel("FR [Hz]")
                plt.axvline(warmup+duration_pre, color="red", zorder=10, ls="--")
                plt.xlim(warmup+duration_pre - 10, warmup+duration_pre + 125)
                if double_step:
                    plt.axvline(warmup+duration_pre+delta_step, color="red", zorder=10, ls="--")
                    plt.xticks(list(plt.xticks()[0]) + [warmup+duration_pre, warmup+duration_pre+delta_step], list(plt.xticks()[0]) + [r"$t_\Delta$", r"$t_\Delta'$"])  
                else:
                    plt.xticks(list(plt.xticks()[0]) + [warmup+duration_pre, ], list(plt.xticks()[0]) + [r"$t_\Delta$", ])                

                entropies_pre   = np.zeros(B)
                delay_estimates = np.zeros(B)
                population_FR   = np.zeros((B, t_bins.size-1))
                for b in range(B):
                    np.random.shuffle(run_ids)
                    samples = run_ids[:samplesize]
                
                    # DELAY 
                    spikecounts_all_runs = load_and_merge_spikes(hfile, samples, t_bins)
                    index = (t_bins >= warmup+duration_pre).argmax() # Gets first value that is larger than duration_pre
                    
                    SEM, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                    delay_estimates[b] = delay * hist_binwidth
                    
                    # ENTROPY
                    pre_entropies  = hfile.read_rows(samples)["pre_entropy"]
                    entropies_pre[b] = pre_entropies.mean()
                    
                    fig = plt.figure(figname)
                    FR_all_runs = spikecounts_all_runs.mean(axis=0) / N / (hist_binwidth*1e-3)
                    population_FR[b] = FR_all_runs
                    # Individual runs
                    # plt.plot(t_bins[:-1] + hist_binwidth/2, FR_all_runs, color="blue", alpha=.6) 
                    # Individual runs as histogram  
                    # plt.bar(t_bins[:-1] + hist_binwidth/2, spikecounts_all_runs.mean(axis=0), width=hist_binwidth, color="blue", edgecolor="black")   
                    
                    # plt.axvline(delay * hist_binwidth + warmup + duration_pre + np.random.normal(), c="k", lw=1, ls="--")
                plt.plot(t_bins[:-1] + hist_binwidth/2, population_FR.mean(axis=0), color="tab:blue")
                plt.fill_between(t_bins[:-1] + hist_binwidth/2, 
                                  population_FR.mean(axis=0) + population_FR.std(axis=0), 
                                  population_FR.mean(axis=0) - population_FR.std(axis=0),
                                  color="tab:blue", alpha=0.25)
                ax2 = ax1.twinx()
                ax2.hist(delay_estimates + warmup + duration_pre, bins=t_bins, color="tab:orange", density=True, zorder=-4)
                ax2.set_ylabel("Density of delays")
                # ax2.set_yticks(np.linspace(0, 0.5, 3))
                
                # DELAY 
                spikecounts_all_runs = load_and_merge_spikes(hfile, run_ids, t_bins)
                index = (t_bins >= warmup+duration_pre).argmax() # Gets first value that is larger than duration_pre
                
                _, delay = get_transient(spikecounts_all_runs.mean(axis=0)[index:])
                ax2.axvline(delay * hist_binwidth + warmup + duration_pre, c="k", lw=2, ls="--", zorder=15)
                
                ax1.set_ylim(bottom=0)
                
                save_figure(figname, fig)
                
                new_rows = pd.DataFrame({
                    "delay": delay_estimates,
                    "tag": [tag] * len(delay_estimates),
                    "mean": [mean] * len(delay_estimates)
                })
                all_delay_estimates = pd.concat([all_delay_estimates, new_rows], ignore_index=True)
                
                all_pre_entropy_estimates[m]  = entropies_pre
            all_pre_entropy_estimates = all_pre_entropy_estimates.T
                

def simulate(pre_mean:float, pre_std:float, post_mean:float, post_std:float, dt:float, seed:None) -> tuple:
    logger.info("Reset Nest kernel...")
    nest.ResetKernel()
    # nest.SetKernelStatus({'print_time': True})
    rnd = np.random.RandomState()
    seed = seed if seed is not None else rnd.randint(0, 2**32-1)
    print("SEED:", seed)
    nest.SetKernelStatus({
        "resolution": dt,
        "rng_seed": int(seed+1),
        "local_num_threads": 2,
    })

    logger.info("Create Network...")
    neurons = nif.create_LIF(N)
    voltmeter = nif.create_voltmeter(warmup)
    spike_detector = nif.create_spike_detector()
    nif.measure_neuron(neurons, voltmeter, spike_detector)

    logger.info("Stimulate Neurons...")
    generator = nest.Create(Generator.noise_generator, 1, params={
        "mean": pre_mean, "std": pre_std, "dt": dt,
    })
    nif.connect_generator_with_neuron(generator, neurons)

    logger.info("Run simulation...")
    nest.Simulate(warmup)
    nest.Simulate(duration_pre)

    logger.info("Change generator settings...")
    generator.mean = post_mean
    generator.std = post_std
    
    logger.info("Stimulate after changing the input...")    
    if double_step:
        logger.info("Stimulate after changing the input...")
        nest.Simulate(delta_step)
        logger.info("Change generator settings...")
        generator.mean = pre_mean
        generator.std = pre_std
        nest.Simulate(duration_post-delta_step)
    else:
        nest.Simulate(duration_post)
        

    logger.info("Collect Spikes...")
    senders, spike_times = nif.collect_spikes(spike_detector).values()
    mask = np.logical_and(spike_times >= warmup, spike_times < duration_pre+warmup)
    logger.info(f"FR pre: {nif.FR_from_spikecount(np.count_nonzero(mask), N, duration_pre)}")
    mask = np.logical_and(spike_times >= duration_pre+warmup, spike_times < duration_pre+duration_post+warmup)
    logger.info(f"FR post: {nif.FR_from_spikecount(np.count_nonzero(mask), N, duration_post)}")

    # return senders, spike_times, None, None

    logger.info("Get free Vm")
    time, Vm = nif.collect_Vm(voltmeter)
    return senders, spike_times, time, Vm



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
