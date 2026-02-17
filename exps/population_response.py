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

from lib.conversion import from_free_Vm_to_generator, from_generator_to_free_Vm

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
    duration_post   = 500.
    filename        = "shortsim_sim_data.hdf5"
    
    
hist_binwidth = 2.5 #ms
metadata = {"N": N, "dt": dt, "warmup": warmup, "duration_pre": duration_pre, "duration_post": duration_post, }

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    pre_FR = 2.
    post_FR = 4.
    pre_FR = 5.
    post_FR = 10.
    # pre_FR = 4.
    # # post_FR = 6.
    # post_FR = 12.
    # pre_FR = 4.
    # post_FR = 2.
    means = np.arange(240, 320+1, 30.)
    # means = np.arange(240, 290+1, 10.)
    # means = np.append(means, 320.)
    # rnd = np.random.RandomState()
    # seed = rnd.randint(0, 2**32-1)
    seeds = np.arange(20, dtype=int)

    
    with ResponseHdf5(filename, "a", metadata=metadata) as hfile:
        #===============================================================================
        # SIMULATION
        #===============================================================================
        for delta in ("mean", "std"):
            for mean in means:
                # Set the parameter pre change
                pre_mean = mean 
                pre_std = round(siegert.find_parameter(mean, target_FR=pre_FR, dt=dt).root, 2)
                
                # Set the parameter post change (based on what is changed)
                if delta == "mean":
                    post_mean = round(siegert.find_parameter(pre_std, target_FR=post_FR, given_parameter="std", dt=dt).root, 2) # ie delta mean
                    post_std = pre_std
                elif delta == "std":
                    post_mean = mean
                    post_std = round(siegert.find_parameter(mean, target_FR=post_FR, dt=dt).root, 2) # ie delta std
                else:
                    raise ValueError("No valid delta chosen")
                
                    
                mean_tmp, var_tmp = from_generator_to_free_Vm(pre_mean, 2*pre_std**2, dt)
                print("Predicted FR:", siegert.siegert(mean_tmp, np.sqrt(var_tmp), tau_ref=nif.t_ref*1e-3, tau_m=nif.tau*1e-3, threshold=nif.V_th*1e-3))
                
                for seed in seeds:
                    # Check if the entry was made already
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
        run_ids = rows[id_tag] # id_tag is the tag for all runs
        for run_id in run_ids:
            # Add entropy (if not already there)
            if not hfile.has_entropy(run_id) or not hfile.has_Vdistribution(run_id):
                # TODO: MAGIC NUMBERS
                Vm_bins = np.arange(nif.V_reset-5, nif.V_th+1, .1)
                Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
                time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
                
                # Pre
                mask = np.logical_and(time >= warmup, time < warmup+duration_pre)
                dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins, density=True)
                pre_entropy = entropy(dist, nan_policy="raise")
                
                hfile.add_entropy(run_id, pre_entropy)               
        
                # Post        
                buffer = 0.2 * duration_post
                mask_post = np.logical_and(time >= warmup+duration_pre+buffer, time < warmup+duration_pre+duration_post)
                dist_post, _ = np.histogram(Vm[:, mask_post].ravel(), bins=Vm_bins, density=True)
        
                hfile.add_Vdistribution(run_id, dist, dist_post)
                
                
            if not hfile.has_spikes_by_sender(run_id):
                spike_times = hfile.get_node(hfile.data, f"run{run_id}").spikes.read()
                senders = hfile.get_node(hfile.data, f"run{run_id}").senders.read()
                spikes_by_sender = get_spikes_by_sender(spike_times, senders, N)
                hfile.add_spikes_by_sender(run_id, spikes_by_sender)

        #===============================================================================
        # MORE METHODS
        #===============================================================================    
        all_delay_estimates = pd.DataFrame(columns=["delay", "tag", "mean"]).astype({
            "delay": "float",
            "tag": "string",
            "mean": "float",
        })
                
        B = 25 # No of bootstapping
        samplesize = 10          # 25
        # DELAY - time bins
        t_bins = np.arange(0., duration_pre+duration_post+hist_binwidth, float(hist_binwidth)) + warmup
        
        ##### ALL ANALYSES ######################################
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
                double_step = False
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
                    samples = run_ids[:samplesize] # Bootstrapping
        
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
        save_figure(figname, fig)
        


                
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
        
        

                
        ##### FIGURE -  EMD before and after ######################################
        # from scipy.stats import wasserstein_distance
        # df = pd.DataFrame(columns=["emd", "tag", "mean"])
        #
        # for tag in ("mean", "std"):
        #     for m, mean in enumerate(means):
        #         rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
        #         rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
        #         run_ids = rows_filtered[id_tag]
        #
        #         # INSERT HERE
        #         emds = np.zeros(len(run_ids))
        #         for r, run_id in enumerate(run_ids):
        #             Vdist_pre  = hfile.get_node(hfile.data, f"run{run_id}").dist_pre.read()
        #             Vdist_post = hfile.get_node(hfile.data, f"run{run_id}").dist_post.read()
        #
        #             emd = wasserstein_distance(Vdist_pre, Vdist_post)
        #             emds[r] = emd
        #
        #         new_rows = pd.DataFrame({
        #             "emd": emds,
        #             "tag": [tag] * len(emds),
        #             "mean": [mean] * len(emds),
        #         })
        #         df = pd.concat([df, new_rows], ignore_index=True)
        #
        # figname_spike = f"Earth-Mover-Distance between p(V) pre and post (FR: {pre_FR} to {post_FR})"
        # fig = plt.figure(figname_spike)
        # sns.violinplot(df, x="mean", y="emd", hue="tag")
        # plt.xlabel(r"mean drive $\mu_{pre}$")
        # plt.ylabel("emd [au?]")
        # # plt.xticks(plt.xticks()[0], plt.xticks()[1])
        # handles = [mpatches.Patch(facecolor="tab:blue", label=r"$\Delta \, \sigma$"),
        #            mpatches.Patch(facecolor="tab:orange", label=r"$\Delta \, \mu$")]
        # plt.legend(handles=handles)
        # save_figure(figname_spike, fig)


        # Vm_bins = np.arange(nif.V_reset-5, nif.V_th+1, .1)
        # data = {"mean": None, "std": None}
        # for tag in ("mean", "std"):
        #     dist_pres  = np.zeros((len(means), len(run_ids), Vm_bins.size-1))
        #     dist_posts = np.zeros((len(means), len(run_ids), Vm_bins.size-1))
        #     for m, mean in enumerate(means):
        #         rows = hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR, pre_mean=mean)
        #         rows_filtered = rows[rows[f"pre_{tag}"] == rows[f"post_{tag}"]]  
        #         run_ids = rows_filtered[id_tag]
        #
        #         for r, run_id in enumerate(run_ids):
        #             Vm = hfile.get_node(hfile.data, f"run{run_id}").Vm.read()
        #             time = hfile.get_node(hfile.data, f"run{run_id}").time.read()
        #
        #             mask = np.logical_and(time >= warmup, time < warmup+duration_pre)
        #             dist, _ = np.histogram(Vm[:, mask].ravel(), bins=Vm_bins, density=True)            
        #
        #             buffer = 0.2 * duration_post
        #             mask_post = np.logical_and(time >= warmup+duration_pre+buffer, time < warmup+duration_pre++duration_post)
        #             dist_post, _ = np.histogram(Vm[:, mask_post].ravel(), bins=Vm_bins, density=True)
        #
        #             dist_pres[m, r] = dist
        #             dist_posts[m, r] = dist_post
        #     data[tag] = {"pre": dist_pres.mean(axis=1), "post": dist_posts.mean(axis=1)}
        #
        # from matplotlib.widgets import Slider
        # # Create the figure and the line that we will manipulate
        # fig, ax = plt.subplots(num="tag is mean (fixed)")
        # mline_pre, = ax.plot(Vm_bins[1:] + 0.5, data["mean"]["pre"][0], lw=2)
        # mline_post, = ax.plot(Vm_bins[1:] + 0.5, data["mean"]["post"][0], lw=2)
        # fig.subplots_adjust(bottom=0.25)
        #
        # maxslider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        # slider = Slider(
        #     ax=maxslider,
        #     label="Means index",
        #     valmin=0,
        #     valmax=len(means)-1,
        #     valinit=0,
        #     valstep=1,
        # )
        #
        # def mupdate(val):
        #     mline_pre.set_ydata(data["mean"]["pre"][int(val)])
        #     mline_post.set_ydata(data["mean"]["post"][int(val)])
        #     fig.canvas.draw_idle()
        # slider.on_changed(mupdate)
        #
        # fig, ax = plt.subplots(num="tag is std (fixed)")
        # sline_pre, = ax.plot(Vm_bins[1:] + 0.5, data["std"]["pre"][0], lw=2)
        # sline_post, = ax.plot(Vm_bins[1:] + 0.5, data["std"]["post"][0], lw=2)
        # fig.subplots_adjust(bottom=0.25)
        #
        # saxslider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        # sslider = Slider(
        #     ax=saxslider,
        #     label="Std index",
        #     valmin=0,
        #     valmax=len(means)-1,
        #     valinit=0,
        #     valstep=1,
        # )
        #
        # def supdate(val):
        #     sline_pre.set_ydata(data["std"]["pre"][int(val)])
        #     sline_post.set_ydata(data["std"]["post"][int(val)])
        #     fig.canvas.draw_idle()
        # sslider.on_changed(supdate)
        # plt.show()

               
        



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





#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
    
    plt.show()
    quit()
