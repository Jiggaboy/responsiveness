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

from functools import partial
import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import pandas as pd
from scipy.stats import entropy
from scipy.optimize import root_scalar
import seaborn as sns


import lib.nest_interface as nif
from lib.nest_interface import Generator
from lib.responsehdf5 import ResponseHdf5, id_tag

from lib import siegert
from lib.util import pairwise, save_figure


#===============================================================================
# CONSTANTS
#===============================================================================
test = True
test = False
force= True
force= False

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
    filename        = "sim_data.hdf5"
    # duration_post   = 500.
    # filename        = "shortsim_sim_data.hdf5"
    
    
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
    pre_FR = 4.
    post_FR = 2.
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
        
        for delta in ("mean", "std"):
            for mean in means:
                pre_mean = mean 
                pre_std = round(find_parameter(mean, target_FR=pre_FR).root, 2)
                
                
                if delta == "mean":
                    post_mean = round(find_parameter(pre_std, target_FR=post_FR, given_parameter="std").root, 2) # ie delta mean
                    post_std = pre_std
                elif delta == "std":
                    post_mean = mean
                    post_std = round(find_parameter(mean, target_FR=post_FR).root, 2) # ie delta std
                else:
                    raise ValueError("No valid delta chosen")
                    
                
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
            if not hfile.has_entropy(run_id):
                Vm_bins = np.arange(nif.V_reset-5, nif.V_th+1, 1)
                Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
                time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
                
                mask = np.logical_and(time >= warmup, time < warmup+duration_pre)
                dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins)
                pre_entropy = entropy(dist, nan_policy="raise")
                
                hfile.add_entropy(run_id, pre_entropy)
                
                
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
        
        for tag in ("mean", "std"):
            all_pre_entropy_estimates = np.zeros((len(means), B))
            for m, mean in enumerate(means):
                logger.info(f"Run mean {mean} ({m+1} of {len(means)})...")
                rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
                rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
                run_ids = rows_filtered[id_tag]
                
                figname = f"Spikecount with fixed {tag} (mean: {mean}; pre_FR: {pre_FR}; post_FR: {post_FR})"
                fig, ax1 = plt.subplots(num=figname)
                plt.xlabel("time [ms]")
                plt.ylabel("FR [Hz]")
                plt.axvline(warmup+duration_pre, color="red", zorder=10, ls="--")
                plt.xlim(warmup+duration_pre - 10, warmup+duration_pre + 125)
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
                
            figname_preentropy = f"preentropy (FR: {pre_FR} to {post_FR})"
            fig = plt.figure(figname_preentropy)
            color = "tab:blue" if tag == "mean" else "tab:orange"
            params = {
                "levels": 4, 
                "fill": False, 
                "color": color,
            }
            for m, mean in enumerate(means):
                delays_tmp = all_delay_estimates.delay[(all_delay_estimates["mean"]==mean) & (all_delay_estimates["tag"]==tag)]
                sns.kdeplot(x=all_pre_entropy_estimates[:, m], y=delays_tmp, **params)
                # sns.kdeplot(x=all_pre_entropy_estimates[:, m], y=all_delay_estimates[:, m], **params, label= f"is fixed")
            handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
                       mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
            plt.xlabel("entropy [nats]")
            plt.ylabel("delay [ms]")
            plt.ylim(0, 80)
            plt.legend(handles=handles)
            
            
        
        ##### FIGURE -  TRANSIENT ESTIMATES ######################################
        # figname = f"Delay estimates (FR: {pre_FR} to {post_FR})"
        # fig = plt.figure(figname)
        # plt.xticks(ticks=np.arange(len(means)), labels=means)
        # sns.violinplot(all_delay_estimates, x="mean", y="delay", hue="tag", cut=0, density_norm="width", common_norm=True)
        # # sns.violinplot(data=all_delay_estimates, cut=0, color=color)
        # # sns.stripplot(data=all_delay_estimates, color="black", size=4, jitter=True, alpha=0.5)
        # plt.xlabel(r"mean drive $\mu_{pre}$")
        # plt.ylabel("delay [ms]")
        # plt.ylim(0, 80)
        # handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
        #            mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
        # plt.legend(handles=handles)
        # save_figure(figname, fig)
            


                
        ##### FIGURE -  TIME TO FIRST SPIKE ######################################
        # df = pd.DataFrame(columns=["firstspike", "tag", "mean"])
        #
        # for tag in ("mean", "std"):
        #     for m, mean in enumerate(means):
        #         rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
        #         rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
        #         run_ids = rows_filtered[id_tag]
        #
        #         first_spike = []
        #         for run_id in run_ids:
        #             spikes_by_sender = hfile.get_node(hfile.data, f"run{run_id}").spikes_by_sender.read()
        #             for spikes in spikes_by_sender:
        #                 spikes_tmp = spikes[spikes >= warmup+duration_pre]
        #                 if len(spikes_tmp) > 0:
        #                     first_spike.append(spikes_tmp[0])
        #         new_rows = pd.DataFrame({
        #             "firstspike": np.asarray(first_spike)-warmup-duration_pre,
        #             "tag": [tag] * len(first_spike),
        #             "mean": [mean] * len(first_spike)
        #         })
        #         df = pd.concat([df, new_rows], ignore_index=True)
        #
        # figname_spike = f"Spike to first spike (FR: {pre_FR} to {post_FR})"
        # fig = plt.figure(figname_spike)
        # sns.violinplot(df, x="mean", y="firstspike", hue="tag")
        # plt.xlabel(r"mean drive $\mu_{pre}$")
        # plt.ylabel("delay [ms]")
        # # plt.xticks(plt.xticks()[0], plt.xticks()[1])
        # handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
        #            mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
        # plt.legend(handles=handles)
        # save_figure(figname_spike, fig)


                
def load_and_merge_spikes(hfile:object, run_ids:np.ndarray, t_bins:np.ndarray) -> np.ndarray:  
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
        spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
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
    voltmeter, spike_detector = nif.create_detectors(warmup)
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
    nest.Simulate(duration_post)

    logger.info("Collect Spikes...")
    senders, spike_times = nif.collect_spikes(spike_detector).values()
    mask = np.logical_and(spike_times >= warmup, spike_times < duration_pre+warmup)
    logger.info(f"FR pre: {FR_from_spikecount(np.count_nonzero(mask), N, duration_pre)}")
    mask = np.logical_and(spike_times >= duration_pre+warmup, spike_times < duration_pre+duration_post+warmup)
    logger.info(f"FR post: {FR_from_spikecount(np.count_nonzero(mask), N, duration_post)}")

    # return senders, spike_times, None, None

    logger.info("Get free Vm")
    time, Vm = nif.collect_Vm(voltmeter)
    return senders, spike_times, time, Vm


def FR_from_spikecount(spikecount:int, N:int, time:float):
    return spikecount / N / (time*1e-3)


def find_parameter(value:float, target_FR:float, given_parameter:str="mean", low:float=1., high:float=5000.):
    if given_parameter not in ("mean", "std"):
        raise ValueError
    if given_parameter == "mean":
        search = "std"
    else:
        search = "mean"
    kwargs = {given_parameter: value}
    
    loaded = partial(FR_from_siegert, **kwargs)
    def objective(param):
        return loaded(**{search: param}) - target_FR
    return root_scalar(objective, bracket=[low, high])


def FR_from_siegert(mean:float, std:float, dt:float=dt)->float:
    """
    Requires nest_interface for default values of the neurons.

    Parameters
    ----------
    mean : float
        Value of the Noise Generator (nest).
    std : float
        Value of the Noise Generator (nest).
    dt : float
        Time step of the Noise Generator, default is 10*dt of the simulation (nest: https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html).

    Returns
    -------
    float
        Predicted FR.

    """
    mean_tmp, var_tmp = siegert.potential_from_moments(mean, std**2, tau_ms=nif.tau, dt=dt)
    return siegert.siegert(mean_tmp, np.sqrt(var_tmp), tau_ref=nif.t_ref*1e-3, tau_m=nif.tau*1e-3, threshold=nif.V_th*1e-3)

#===============================================================================
# STATISTICAL METHODS
#===============================================================================

def get_transient(data:np.ndarray, ddof:int=1, min_samples:int=10) -> int:
    """
    Discards the initial {d} samples and calculates the SEM for the {data}.

    Parameters
    ----------
    data : np.ndarray
        1d-array.
    ddof : int, optional
        Degree of freedom for calculation of std. The default is 1.
    min_samples : int, optional
        Minimum number of samples to calculate the SEM from. Default is 10.

    Returns
    -------
    SEM: np.ndarray: SEM for the data with {d} discarded samples
    d: int: the index of the lowest SEM

    """
    l = data.size
    prefactor = np.zeros(l - min_samples)
    std = np.zeros(l - min_samples)
    SEM = np.zeros(l - min_samples)
    SEMsq = np.zeros(l - min_samples)
    SEMsqrt = np.zeros(l - min_samples)
    for d in np.arange(l - min_samples):
        SEMsqrt[d] = np.sqrt(1 / (l - d)) * data[d:].std(ddof=ddof)
        prefactor[d] = 1 / (l - d)
        std[d] = data[d:].std(ddof=ddof)
        SEMsq[d] = (1 / (l - d))**2 * data[d:].std(ddof=ddof)
        SEM[d] = 1 / (l - d) * data[d:].std(ddof=ddof)
    # plt.figure("SEM")
    # plt.plot((SEM-SEM.min())/(SEM.max()-SEM.min()), label="SEM")
    # plt.plot((SEMsqrt-SEMsqrt.min())/(SEMsqrt.max()-SEMsqrt.min()), label="sqrt(SEM)")
    # plt.plot((SEMsq-SEMsq.min())/(SEMsq.max()-SEMsq.min()), label="SEM**2")
    # plt.legend()
    # plt.figure("std")
    # plt.plot(std)
    # plt.figure("1/...")
    # plt.plot(prefactor, label="pref")
    # plt.plot(np.diff(prefactor) / prefactor[:-1], label="diff(pref)")
    # plt.plot(np.sqrt(prefactor), label="sqrt(pref)")
    # plt.plot(np.diff(np.sqrt(prefactor)) / np.sqrt(prefactor)[:-1], label="diff(sqrt(pref))")
    
    
    return SEMsqrt, np.argmin(SEM)

#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
