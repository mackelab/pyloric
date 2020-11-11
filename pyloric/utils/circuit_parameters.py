import numpy as np


def create_neurons(neuron_list):
    prinz_neurons = {
        "LP": {
            "LP_0": [100, 0, 8, 40, 5, 75, 0.05, 0.02],  # this3 g_CaS g_A g_Kd
            "LP_1": [100, 0, 6, 30, 5, 50, 0.05, 0.02],  # this2 # KCa, H      # this3
            "LP_2": [100, 0, 10, 50, 5, 100, 0.0, 0.03],
            "LP_3": [100, 0, 4, 20, 0, 25, 0.05, 0.03],
            "LP_4": [100, 0, 6, 30, 0, 50, 0.03, 0.02],  # this2
        },
        "PY": {
            "PY_1": [200, 7.5, 0, 50, 0, 75, 0.05, 0.0],
            # this3  # this3 g_Na, g_CaT, g_CaS
            "PY_0": [100, 2.5, 2, 50, 0, 125, 0.05, 0.01],  # this3
            "PY_3": [400, 2.5, 2, 50, 0, 75, 0.05, 0.0],
            # this3        # this3 g_leak, g_Kd, g_Na
            "PY_5": [500, 2.5, 2, 40, 0, 125, 0.0, 0.02],  # this2 # g_H, g_leak
            "PY_2": [200, 10.0, 0, 50, 0, 100, 0.03, 0.0],  # this3 # CaT Kd H
            "PY_4": [500, 2.5, 2, 40, 0, 125, 0.01, 0.03],  # this2
        },
        "PM": {
            "PM_0": [400, 2.5, 6, 50, 10, 100, 0.01, 0.0],  # this2  g_Na, KCa
            "PM_3": [200, 5.0, 4, 40, 5, 125, 0.01, 0.0],  # this3 CaT, g_A, g_Kd
            "PM_4": [300, 2.5, 2, 10, 5, 125, 0.01, 0.0],
            "PM_1": [100, 2.5, 6, 50, 5, 100, 0.01, 0.0],  # this2
            "PM_2": [200, 2.5, 4, 50, 5, 50, 0.01, 0.0],  # this3
        },
    }
    # Note (PM_0 or PM_1) / (LP_2) / (PY_0) is figure 5a in Prinz 2004.
    # Note (PM_4)         / (LP_3) / (PY_4) is figure 5b in Prinz 2004.

    ret = []
    for n in neuron_list:
        neuron = np.asarray(prinz_neurons[n[0]][n[1]]) * n[2]
        ret.append(neuron)
    return ret


def build_conns(params):

    # Reversal voltages and dissipation time constants for the synapses, taken from
    # Prinz 2004, p. 1351
    Esglut = -70  # mV
    kminusglut = 40  # ms

    Eschol = -80  # mV
    kminuschol = 100  # ms

    return np.asarray(
        [
            [1, 0, params[0], Esglut, kminusglut],
            [1, 0, params[1], Eschol, kminuschol],
            [2, 0, params[2], Esglut, kminusglut],
            [2, 0, params[3], Eschol, kminuschol],
            [0, 1, params[4], Esglut, kminusglut],
            [2, 1, params[5], Esglut, kminusglut],
            [1, 2, params[6], Esglut, kminusglut],
        ]
    )
