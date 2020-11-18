import numpy as np
import os
from typing import Optional, Dict, Union
import pyximport
import numpy
import pandas as pd

# setup_args needed on my MacOS
pyximport.install(
    setup_args={"include_dirs": numpy.get_include()},
    reload_support=True,
    language_level=3,
)

from pyloric.simulator import sim_time
from pyloric.summary_statistics import PrinzStats
from pyloric.utils import (
    build_conns,
    create_neurons,
    membrane_conductances_replaced_with_defaults,
    synapses_replaced_with_defaults,
)


def simulate(
    circuit_parameters: Union[np.array, pd.DataFrame],
    dt: float = 0.025,
    t_max: int = 11000,
    temperature: int = 283,
    noise_std: float = 0.001,
    track_energy: bool = False,
    track_currents: bool = False,
    seed: Optional[int] = None,
    customization: Dict = {},
    defaults: Dict = {},
):
    r"""
    Runs the STG model with a subset of all parameters.

    Args:
        circuit_parameters: Parameters of the circuit model. This should be a pandas
            DataFrame sampled from the prior.
        dt: Step size in milliseconds.
        t_max: Overall runtime of the simulation in milliseconds.
        temperature: Temperature in Kelvin that the simulation is run at.
        noise_std: Standard deviation of the noise added at every time step. Will
            **not** be rescaled with the step-size.
        track_energy: Whether to keep track of and return the energy consumption at any
            step during the simulation. The output dictionary will have the additional
            entry `energy`.
        track_currents: Tracks the conductance values of all channels (also synapses).
            The currents can easily be computed from the conductance values by
            $I = g \cdot (V-E)$. For the calcium channels, the reversal potential of
            the calcium channels is also saved. The output dictionary will have
            additional entries 'membrane_conds', 'synaptic_conds', 'reversal_calcium'.
        seed: Possible seed for the simulation.
        customization:  If you want to exclude some of the `circuit_parameters` and use
            constant default values for them, you have to set these entries to `False`
            in the `use_membrane` key in the `customization` dictionary. If you want
            to include $Q_{10}$ values, you have to set them in the same dictionary and
            append the values of the $Q_{10}$s to the `circuit_parameters`.
        defaults: For all parameters specified as `False` in `customization`, this
            dictionary allows to set the default value, i.e. the value that is used for
            it.
    """

    setup_dict = {
        "membrane_gbar": [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ],
        "Q10_gbar_syn": [False, False],  # first for glutamate, second for choline
        "Q10_tau_syn": [False, False],  # first for glutamate, second for choline
        "Q10_gbar_mem": [False, False, False, False, False, False, False, False],
        "Q10_tau_m": False,
        "Q10_tau_h": False,
        "Q10_tau_CaBuff": False,
    }
    setup_dict.update(customization)

    defaults_dict = {
        "membrane_gbar": [
            ["PM", "PM_4", 0.628e-3],
            ["LP", "LP_3", 0.628e-3],
            ["PY", "PY_4", 0.628e-3],
        ],
        "Q10_gbar_mem": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        "Q10_gbar_syn": [1.5, 1.5],
        "Q10_tau_m": [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7],
        "Q10_tau_h": [2.8, 2.8, 2.8, 2.8],
        "Q10_tau_CaBuff": 1.7,
        "Q10_tau_syn": [1.7, 1.7],
    }
    defaults_dict.update(defaults)

    t = np.arange(0, t_max, dt)

    membrane_conductances = membrane_conductances_replaced_with_defaults(
        circuit_parameters, defaults_dict
    )
    synaptic_conductances = synapses_replaced_with_defaults(
        circuit_parameters, defaults_dict
    )
    # q10_values_as_pd = ... # todo

    if isinstance(circuit_parameters, pd.DataFrame):
        circuit_parameters = circuit_parameters.to_numpy()

    # define lists to loop over to assemble the parameters
    param_classes = [
        setup_dict["Q10_gbar_mem"],
        setup_dict["Q10_gbar_syn"],
        setup_dict["Q10_tau_m"],
        setup_dict["Q10_tau_h"],
        setup_dict["Q10_tau_CaBuff"],
        setup_dict["Q10_tau_syn"],
    ]
    class_defaults = [
        defaults_dict["Q10_gbar_mem"],
        defaults_dict["Q10_gbar_syn"],
        defaults_dict["Q10_tau_m"],
        defaults_dict["Q10_tau_h"],
        defaults_dict["Q10_tau_CaBuff"],
        defaults_dict["Q10_tau_syn"],
    ]
    param_classes.reverse()
    class_defaults.reverse()

    # loop over lists
    split_parameters = []
    for pclass, classdefault in zip(param_classes, class_defaults):
        if np.any(pclass):
            split_parameters.append(circuit_parameters[-np.sum(pclass) :])
            circuit_parameters = circuit_parameters[: -np.sum(pclass)]
        else:
            split_parameters.append(classdefault)
    split_parameters.reverse()

    # extend the parameter values for synapses, gbar and tau
    # split_parameters[0][0] = gbar q10 for glutamate synapse
    # split_parameters[0][1] = gbar q10 for cholinergic synapse
    split_parameters[0] = [
        split_parameters[0][0],
        split_parameters[0][1],
        split_parameters[0][0],
        split_parameters[0][1],
        split_parameters[0][0],
        split_parameters[0][0],
        split_parameters[0][0],
    ]  # gbar of synapses
    split_parameters[1] = [
        split_parameters[1][0],
        split_parameters[1][1],
        split_parameters[1][0],
        split_parameters[1][1],
        split_parameters[1][0],
        split_parameters[1][0],
        split_parameters[1][0],
    ]  # tau of synapses

    # extend the parameter values for tau_m and tau_h
    if isinstance(setup_dict["Q10_tau_m"], bool):
        split_parameters[3] = np.tile([split_parameters[3]], 8).flatten()
    if isinstance(setup_dict["Q10_tau_h"], bool):
        split_parameters[4] = np.tile([split_parameters[4]], 4).flatten()

    # get the conductance params
    conductance_params = circuit_parameters  # membrane and synapse gbar

    assert conductance_params.ndim == 1, "params.ndim must be 1"

    membrane_conductances = conductance_params[0:-7]
    synaptic_conductances = np.exp(conductance_params[-7:])
    conns = build_conns(-synaptic_conductances)

    # build the used membrane conductances as parameters. Rest as fixed values.
    current_num = 0
    membrane_conds = []
    for neuron_num in range(3):  # three neurons
        membrane_cond = []
        for cond_num in range(8):  # 8 membrane conductances per neuron
            if setup_dict["membrane_gbar"][neuron_num][cond_num]:
                membrane_cond.append(membrane_conductances[current_num])
                current_num += 1
            else:
                membrane_cond.append(neurons[neuron_num][cond_num])
        membrane_conds.append(np.asarray(membrane_cond))

    if isinstance(split_parameters[5], float):
        split_parameters[5] = [split_parameters[5]]

    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    I = rng.normal(scale=noise_std, size=(3, len(t)))

    # print("g_q10_conns_gbar", split_parameters[0])
    # print("g_q10_conns_tau", split_parameters[1])
    # print("g_q10_memb_gbar", split_parameters[2])
    # print("g_q10_memb_tau_m", split_parameters[3])
    # print("g_q10_memb_tau_h", split_parameters[4])
    # print("g_q10_memb_tau_CaBuff", split_parameters[5])

    data = sim_time(
        dt,
        t,
        I,
        membrane_conds,  # membrane conductances
        conns,  # synaptic conductances (always variable)
        g_q10_conns_gbar=split_parameters[0],
        g_q10_conns_tau=split_parameters[1],
        g_q10_memb_gbar=split_parameters[2],
        g_q10_memb_tau_m=split_parameters[3],
        g_q10_memb_tau_h=split_parameters[4],
        g_q10_memb_tau_CaBuff=split_parameters[5],
        temp=temperature,
        num_energy_timesteps=len(t) if track_energy else 0,
        num_energyscape_timesteps=len(t) if track_currents else 0,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    results_dict = {"voltage": data["Vs"], "dt": dt, "t_max": t_max}
    if track_energy:
        results_dict.update({"energy": data["energy"]})
    if track_currents:
        results_dict.update({"membrane_conds": data["membrane_conds"]})
        results_dict.update({"synaptic_conds": data["synaptic_conds"]})
        results_dict.update({"reversal_calcium": data["reversal_calcium"]})

    return results_dict


def stats(
    simulation_outputs: Dict, stats_customization: Dict = {}, t_burn_in=1000
) -> pd.DataFrame:
    """
    Return summary statistics of the voltage trace.

    Args:
        simulation_outputs: Dictionary returned by `simulate()`. Contains (at least)
            the voltage traces, the time step, and the duration of the simulation.
        stats_customization: Allows to add summary statistics. Possible keys are:
            `plateau_durations`: Maximum duration of voltage plateaus above -30mV in
                each model neuron.
            `pyloric_like`: bool indicating whether the rhythm was pyloric-like (see
                Prinz 2004 for a definition).
            `pyloric`: bool indicating whether the rhythm was pyloric (see
                Prinz 2004 for a definition).
            `num_bursts`: Integer indicating the number of bursts in each neuron.
            `num_spikes`: Integer indicating the number of spikes in each neuron.
            `voltage_moments`: The first four moments (mean, std, skew, kurtosis) of
                the voltage trace of each neuron.
            `energies`: The energy (in microJoule) consumed by each neuron.
            `energies_per_spike`: The average energy per spike (in microJoule) in each
                neuron.
        t_burn_in: The time (in milliseconds) that should be excluded from the
            computation of summary statistics. The idea is that the rhythm should first
            reach a steady state and only then should one compute the summary
            statistics.

    Returns:
        Summary statistics.
    """

    setups = {
        "cycle_period": True,
        "burst_durations": True,
        "duty_cycles": True,
        "start_phases": True,
        "starts_to_starts": True,
        "ends_to_starts": True,
        "phase_gaps": True,
        "plateau_durations": False,
        "pyloric_like": False,
        "pyloric": False,
        "num_bursts": False,
        "num_spikes": False,
        "spike_times": False,
        "spike_heights": False,
        "rebound_times": False,
        "voltage_means": False,
        "voltage_stds": False,
        "voltage_skews": False,
        "voltage_kurtoses": False,
        "energies": False,
        "energies_per_burst": False,
        "energies_per_spike": False,
    }
    setups.update(stats_customization)

    stats_object = PrinzStats(
        setup=setups,
        t_on=t_burn_in,
        t_off=simulation_outputs["t_max"],
        dt=simulation_outputs["dt"],
    )

    ss = stats_object.calc_dict(simulation_outputs)
    return ss
