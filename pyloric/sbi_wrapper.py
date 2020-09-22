import numpy as np
from parameters import ParameterSet
import os

import pyximport
import numpy
# setup_args needed on my MacOS
pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

from sbi_simulator import sim_time
from sbi_simulator_energyScape import sim_time_energyscape
from sbi_simulator_general import sim_time_general
from sbi_summstats import PrinzStats


t_burnin = 1000
t_window = 10000
noise_fact = 0.001
tmax = t_burnin + t_window
dt = 0.025
seed = 0

dirname = os.path.dirname(__file__)
neumodels = ParameterSet(dirname+'/models.prm')
setups_dict = ParameterSet(dirname+'/setups.prm')


def get_time():
    return np.arange(0, tmax, dt)


def wrapper(params):

    full_data = simulate(params)
    ss = stats(full_data)

    return ss


def simulate(params, seed=None):
    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    t = np.arange(0, tmax, dt)

    membrane_params = params[0:-7]
    membrane_params = np.float64(np.reshape(membrane_params, (3, 8)))
    synaptic_params = np.exp(params[-7:])
    conns = build_conns(-synaptic_params)

    I = rng.normal(scale=noise_fact, size=(3, len(t)))

    # calling the solver --> HH.HH()
    data = sim_time(
        dt,
        t,
        I,
        membrane_params,  # membrane conductances
        conns,  # synaptic conductances (always variable)
        temp=283,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    full_data = {
        'data': data['Vs'],
        'tmax': tmax,
        'dt': dt,
        'I': I,
        'energy': data['energy'],
    }

    return full_data



def simulate_energyscape(params, seed=None):
    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    t = np.arange(0, tmax, dt)

    membrane_params = params[0:-7]
    membrane_params = np.float64(np.reshape(membrane_params, (3, 8)))
    synaptic_params = np.exp(params[-7:])
    conns = build_conns(-synaptic_params)

    I = rng.normal(scale=noise_fact, size=(3, len(t)))

    # calling the solver --> HH.HH()
    data = sim_time_energyscape(
        dt,
        t,
        I,
        membrane_params,  # membrane conductances
        conns,  # synaptic conductances (always variable)
        temp=283,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    full_data = {
        'data': data['Vs'],
        'tmax': tmax,
        'dt': dt,
        'I': I,
        'energy': data['energy'],
        'all_energies': data['all_energies']
    }

    return full_data


def simulate_general(params, hyperparams, seed):
    """
    Runs the STG model with a subset of all parameters.

    The parameters are specified in params. The setup file hyperparams defines which
    parameters these values should be assigned to. The remaining parameters are filled
    with default values.
    """

    t = np.arange(0, tmax, dt)

    neurons = create_neurons(hyperparams.neurons)
    proctolin = np.asarray(hyperparams.proctolin)

    # define lists to loop over to assemble the parameters
    param_classes = [hyperparams.Q10_gbar_syn,
                     hyperparams.Q10_tau_syn,
                     hyperparams.Q10_gbar_mem,
                     hyperparams.Q10_tau_m,
                     hyperparams.Q10_tau_h,
                     hyperparams.Q10_tau_CaBuff]
    class_defaults = [hyperparams.Q10_gbar_syn_default,
                      hyperparams.Q10_tau_syn_default,
                      hyperparams.Q10_gbar_mem_default,
                      hyperparams.Q10_tau_m_default,
                      hyperparams.Q10_tau_h_default,
                      hyperparams.Q10_tau_CaBuff_default]

    param_classes = np.flip(param_classes)
    class_defaults = np.flip(class_defaults)

    # loop over lists
    split_parameters = []
    for pclass, classdefault in zip(param_classes, class_defaults):
        if np.any(pclass):
            split_parameters.append(params[-np.sum(pclass):])
            params = params[:-np.sum(pclass)]
        else:
            split_parameters.append(classdefault)
    split_parameters = np.flip(split_parameters)

    # split_parameters = class_defaults

    # extend the parameter values for synapses, gbar and tau
    # split_parameters[0][0] = gbar q10 for glutamate synapse
    # split_parameters[0][1] = gbar q10 for cholinergic synapse
    split_parameters[0] = [split_parameters[0][0], split_parameters[0][1],
                           split_parameters[0][0],
                           split_parameters[0][1], split_parameters[0][0],
                           split_parameters[0][0],
                           split_parameters[0][0]]  # gbar of synapses
    split_parameters[1] = [split_parameters[1][0], split_parameters[1][1],
                           split_parameters[1][0],
                           split_parameters[1][1], split_parameters[1][0],
                           split_parameters[1][0],
                           split_parameters[1][0]]  # tau of synapses

    # extend the parameter values for tau_m and tau_h
    if isinstance(hyperparams.Q10_tau_m, bool): split_parameters[3] = np.tile(
        [split_parameters[3]], 8).flatten()
    if isinstance(hyperparams.Q10_tau_h, bool): split_parameters[4] = np.tile(
        [split_parameters[4]], 4).flatten()

    # get the conductance params
    conductance_params = params  # membrane and synapse gbar

    assert conductance_params.ndim == 1, 'params.ndim must be 1'


    membrane_params = conductance_params[0:-7]
    synaptic_params = np.exp(conductance_params[-7:])
    conns = build_conns(-synaptic_params)

    # build the used membrane conductances as parameters. Rest as fixed values.
    current_num = 0
    membrane_conds = []
    for neuron_num in range(3):  # three neurons
        membrane_cond = []
        for cond_num in range(8):  # 8 membrane conductances per neuron
            if hyperparams.use_membrane[neuron_num][cond_num]:
                membrane_cond.append(membrane_params[current_num])
                current_num += 1
            else:
                membrane_cond.append(neurons[neuron_num][cond_num])
        if hyperparams.use_proctolin:
            membrane_cond.append(membrane_params[current_num])
            current_num += 1
        else:
            membrane_cond.append(proctolin[neuron_num])  # proctolin is made part of the membrane conds here.
        membrane_conds.append(np.asarray(membrane_cond))

    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    I = rng.normal(scale=hyperparams.noise_fact, size=(3, len(t)))

    if isinstance(split_parameters[5], float):
        split_parameters[5] = [split_parameters[5]]

    # calling the solver --> HH.HH()
    data = sim_time_general(dt,
                            t,
                            I,
                            membrane_conds,  # membrane conductances
                            conns,  # synaptic conductances (always variable)
                            g_q10_conns_gbar=split_parameters[0],
                            g_q10_conns_tau=split_parameters[1],  # Q10 synaptic tau
                            g_q10_memb_gbar=split_parameters[2],
                            g_q10_memb_tau_m=split_parameters[3],
                            g_q10_memb_tau_h=split_parameters[4],
                            g_q10_memb_tau_CaBuff=split_parameters[5],
                            temp=hyperparams.model_params.temp,
                            save_all_energy_currents=False,
                            verbose=False,
                            start_val_input=0.5
                            )

    return {'data': data['Vs'],
            'params': conductance_params,
            'tmax': tmax,
            'dt': dt,
            'I': I,
            'energy': data['energy']}


def stats(full_data):
    stats_object = PrinzStats(
        t_on=t_burnin,
        t_off=t_burnin + t_window,
        include_pyloric_ness=True,
        include_plateaus=True,
        seed=seed,
        energy=True
    )

    ss = stats_object.calc([full_data])[0]
    return ss


def build_conns(params):

    # Reversal voltages and dissipation time constants for the synapses, taken from
    # Prinz 2004, p. 1351
    Esglut = -70            # mV
    kminusglut = 40         # ms

    Eschol = -80            # mV
    kminuschol = 100        # ms

    return np.asarray([
        [1, 0, params[0], Esglut, kminusglut],
        [1, 0, params[1], Eschol, kminuschol],
        [2, 0, params[2], Esglut, kminusglut],
        [2, 0, params[3], Eschol, kminuschol],
        [0, 1, params[4], Esglut, kminusglut],
        [2, 1, params[5], Esglut, kminusglut],
        [1, 2, params[6], Esglut, kminusglut]
    ])

def create_neurons(neuron_list):
    ret = []
    for n in neuron_list:
        neuron = np.asarray(neumodels[n[0]][n[1]]) * n[2]
        ret.append(neuron)
    return ret

def load_setup(name):
    return setups_dict[name]
