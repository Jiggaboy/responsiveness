#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Used in the scripts population response and synaptic_correlation.

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import nest

#===============================================================================
# NEURON PARAMETER
#===============================================================================
neuron_model = "iaf_psc_delta_ps"
tau = 15.           # ms
t_ref = 2.          # ms
E_L = 0.            # mV
V_reset = 0.        # mV
V_th = 20.          # mV
capacitance = 250.  # pF

#===============================================================================
# RECORDER PARAMETER
#===============================================================================

voltmeter_interval = 1.  #ms

class Recorder:
    voltmeter = "voltmeter"
    multimeter = "multimeter"
    spikemeter = "spike_recorder"


class Generator:
    noise_generator = "noise_generator"
    step_current_generator = "step_current_generator"
    poisson_generator = "poisson_generator"
    sinusoidal_poisson_generator = "sinusoidal_poisson_generator"


#===============================================================================
# METHODS
#===============================================================================

def create_LIF(N):
    logger.info(f"Neuron model: {neuron_model}")
    logger.info(f"Membrane time constant (tau): {tau}")
    neuron_params = {
          "tau_m": float(tau),
          "E_L": float(E_L),
          "V_reset": float(V_reset),
          "V_th": float(V_th),
          "t_ref": float(t_ref),
          "V_m": np.random.normal(loc=(V_th-V_reset) / 2, scale=(V_th-V_reset) / 4, size=N)
      }
    return nest.Create(neuron_model, N, params=neuron_params)


def create_voltmeter(start:float) -> object:
    Vm_params = {"interval": voltmeter_interval, "record_from": ["V_m"], "start":start}
    return nest.Create(Recorder.voltmeter, 1, params=Vm_params)
    
    
    
def create_spike_detector() -> object:
    return nest.Create(Recorder.spikemeter, 1)


def measure_neuron(neuron, voltmeter:list=None, spike_detector:list=None, start:float=0.):
    """Connects the measuring devices with the neuron (population)."""
    if voltmeter is not None:
        for v in voltmeter:
            v.start = start
            nest.Connect(v, neuron, conn_spec={"rule": "all_to_all"})
    if spike_detector is not None:
        for s in spike_detector:
            s.start = start
            nest.Connect(neuron, s, conn_spec={"rule": "all_to_all"})


def connect_generator_with_neuron(generator, neuron, synapse:dict=None)->None:
    synapse = synapse or {}
    nest.Connect(generator, neuron, conn_spec={"rule": "all_to_all"}, syn_spec=synapse)


def collect_spikes(spike_detector):
    return nest.GetStatus(spike_detector, "events")[0]


def collect_spike_times(spike_detector):
    return collect_spikes(spike_detector)["times"]


def collect_spikes_per_neuron(spike_detector):
    senders = collect_spikes(spike_detector)["senders"]
    neuron_ids = set(senders)
    times = collect_spike_times(spike_detector)
    spikes_per_neuron = []
    for neuron in neuron_ids:
        spikes_per_neuron.append(times[neuron == senders])
    return spikes_per_neuron


### Collect membrane potentials
def collect_voltmeter(voltmeter):
    return voltmeter.get("events")


def collect_Vm(voltmeter):
    voltmeter = collect_voltmeter(voltmeter)

    by_sender = defaultdict(list)
    senders = set(voltmeter["senders"])
    no_neurons = len(senders)
    for sender, vm in zip(voltmeter["senders"], voltmeter["V_m"]):
        by_sender[sender].append(vm)

    time = voltmeter["times"][voltmeter["senders"] == list(senders)[0]]
    Vm = np.zeros(shape=(no_neurons, time.size))
    for s, (sender, vm) in enumerate(by_sender.items()):
        Vm[s] = vm

    return time, Vm

#===============================================================================
# METHODS - UTIL SPIKE/VM
#===============================================================================

def FR_from_spikecount(spikecount:int, N:int, time:float):
    return spikecount / N / (time*1e-3)
