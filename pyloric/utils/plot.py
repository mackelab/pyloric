import matplotlib.pyplot as plt
import numpy as np


def show_traces(traces, figsize=(6, 3)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    global_min = np.min(traces)
    global_max = np.max(traces)
    neuron_labels = ["AB/PD", "LP", "PY"]
    for i, ax in enumerate(axes):
        ax.plot(traces[i])
        ax.set_ylim([global_min, global_max])
        if i < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Timesteps")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(neuron_labels[i])
    return fig, axes
