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
if __name__ == '__main__':
    logger.setLevel("INFO")

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
from config import load_config, NetworkParams
import lib.nest_interface as nif
from lib.responsehdf5 import ResponseHdf5, id_tag, load_and_merge_spikes, get_spikes_by_sender

from lib.rnn import RNN
from lib.util import functimer

from exps.simulate import simulate

#===============================================================================
# CONSTANTS
#===============================================================================
pre_FR = 5.
post_FR = 10.

# Set to the same values as in network_simulate.py
Imean_ext = 260.
FR_I = pre_FR

means = [260., ]
means = np.arange(220, 320+1, 40.)

tags = (mean_tag, std_tag, mean_std_tag)
# tags = (mean_tag, )

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
@functimer
def main():
    control, params = load_config(no_stim=True)
    networkparams = NetworkParams(control)
    base_filename, suffix = params.filename.rsplit(".", maxsplit=1)
    tmp_filename = base_filename + f"_control_{networkparams.J}" + f"_{float(pre_FR)}_{float(post_FR)}" + f".{suffix}"

    rnn = RNN(networkparams, pre_FR, post_FR, FR_I, drive_Imean=Imean_ext)

    with ResponseHdf5(tmp_filename, "a", metadata=params.metadata) as hfile:
        
        for delta in tags:
            for mean in means:
                rnn.set_up_network(mean, delta)
                pre_mean, pre_std = rnn.pre_Esetpoint
                post_mean, post_std = rnn.post_Esetpoint
                
                # Run simulations
                for seed in np.arange(params.seeds):
                    if not control.force and len(hfile.filter_rows(hfile.run, pre_FR=pre_FR, post_FR=post_FR,
                                                           pre_mean=pre_mean, post_mean=post_mean,
                                                           pre_std=pre_std, post_std=post_std, seed=seed,
                                                           stim_duration=params.stim_duration, break_duration=params.break_duration, stim_reps=params.stim_reps)
                    ):
                        logger.info("Skip simulation...")
                        continue
                    logger.info(f"Run simulation ({seed}/ {params.seeds})...")
                    senders, spike_times, time, Vm = simulate(params, control, pre_mean, pre_std, post_mean, post_std, seed=seed)
                    
                    logger.info("Save simulation...")
                    run_id = hfile.add_run(pre_FR, post_FR, pre_mean, post_mean, pre_std, post_std, seed=seed, stim_duration=params.stim_duration, break_duration=params.break_duration, stim_reps=params.stim_reps)
                    hfile.add_data_to_run(run_id, senders, spike_times, time, Vm)
                    hfile.flush()
                logger.info(f"Simulation finished ({delta}; {mean})...")

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

