from pyloric import create_prior, simulate, stats
import pytest
import torch
import numpy as np


def test_simulator():
    prior = create_prior()
    torch.manual_seed(0)
    p = prior.sample((1,))
    sim_out = simulate(p.loc[0], seed=0)
    ss = stats(sim_out)
    ground_truth = np.asarray(
        [1275.5107, 180.5469, 433.2531, 279.2781, 0.1415, 0.3397, 0.2190,]
    )
    assert np.all(np.abs(ss.to_numpy()[0, :7] - ground_truth) < 1e-3)


@pytest.mark.parametrize("temperature", (283, 299))
def test_temperature(temperature):
    prior = create_prior(
        customization={
            "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
            "Q10_gbar_syn": [True, True],
            "Q10_tau_m": [True],
            "Q10_tau_h": [True],
            "Q10_tau_CaBuff": [True],
            "Q10_tau_syn": [True, True],
        }
    )
    torch.manual_seed(12)
    p = prior.sample((1,))
    sim_out = simulate(p.loc[0], seed=0, temperature=temperature)
    ss = stats(sim_out)
    if temperature == 283:
        ground_truth = np.asarray([701.5107, 93.3317])
    else:
        ground_truth = np.asarray([467.2786, 137.2682])

    assert np.all(np.abs(ss.to_numpy()[0, :2] - ground_truth) < 1e-3)


def test_advanced_summstats():
    prior = create_prior()
    torch.manual_seed(2)
    p = prior.sample((1,))
    sim_out = simulate(p.loc[0], seed=1, track_energy=True)
    ss = stats(
        sim_out,
        stats_customization={
            "cycle_period": True,
            "burst_durations": True,
            "duty_cycles": True,
            "start_phases": True,
            "starts_to_starts": True,
            "ends_to_starts": True,
            "phase_gaps": True,
            "plateau_durations": True,
            "voltage_means": True,
            "voltage_stds": True,
            "voltage_skews": True,
            "voltage_kurtoses": True,
            "num_bursts": True,
            "num_spikes": True,
            "spike_times": True,
            "spike_heights": True,
            "rebound_times": True,
            "energies": True,
            "energies_per_burst": True,
            "energies_per_spike": True,
            "pyloric_like": True,
        },
    )

    print(ss.to_numpy()[0, :7])

    gt = [810.3958, 75.4423, 73.3750, 170.7275, 0.0931, 0.0905, 0.2107]
    assert np.all(np.abs(ss.to_numpy()[0, :7] - gt) < 1e-3)

    plateau_durations = np.asarray([2.5000, 2.5000, 2.5000])
    voltage_means = np.asarray([-55.7357, -48.8904, -54.3601])
    voltage_stds = np.asarray([10.9472, 2.3825, 12.9266])
    voltage_skews = np.asarray([5.8171, 8.4293, 3.8297])
    voltage_kurtoses = np.asarray([45.7317, 277.2516, 25.0692])
    num_bursts = np.asarray([13, 1, 10])
    num_spikes = np.asarray([44, 2, 55])
    energies = np.asarray([22793.9701, 470.1559, 31964.3796])
    energies_per_burst = np.asarray([1645.7645, 206.8974, 3091.3413])
    energies_per_spike = np.asarray([486.2486, 103.4487, 562.0621])

    assert np.all(
        np.abs(ss["plateau_durations"].to_numpy()[0] - plateau_durations) < 1e-3
    )
    assert np.all(np.abs(ss["voltage_means"].to_numpy()[0] - voltage_means) < 1e-3)
    assert np.all(np.abs(ss["voltage_stds"].to_numpy()[0] - voltage_stds) < 1e-3)
    assert np.all(np.abs(ss["voltage_skews"].to_numpy()[0] - voltage_skews) < 1e-3)
    assert np.all(
        np.abs(ss["voltage_kurtoses"].to_numpy()[0] - voltage_kurtoses) < 1e-3
    )
    assert np.all(np.abs(ss["num_bursts"].to_numpy()[0] - num_bursts) < 1e-3)
    assert np.all(np.abs(ss["num_spikes"].to_numpy()[0] - num_spikes) < 1e-3)
    assert np.all(np.abs(ss["energies"].to_numpy()[0] - energies) < 1e-3)
    assert np.all(
        np.abs(ss["energies_per_burst"].to_numpy()[0] - energies_per_burst) < 1e-3
    )
    assert np.all(
        np.abs(ss["energies_per_spike"].to_numpy()[0] - energies_per_spike) < 1e-3
    )
    assert not ss["pyloric_like"].to_numpy()[0]
