import numpy as np
from sbi.utils import BoxUniform
import torch


def create_prior(log=True):
    """
    Create prior over 8 membrane conductances per neuron as well as over 7 synapses.
    :param log:
    :return:
    """

    # maximal membrane conductances
    # rows:    LP, PY, PD
    # columns: the eight membrane conductances
    # contains the minimal values that were used by Prinz et al.
    low_val = 0.0
    membrane_cond_mins = np.asarray([[100, 2.5, 2, 10, 5, 50, 0.01, low_val],  # PM
                                     [100, low_val, 4, 20, low_val, 25, low_val, 0.02],  # LP
                                     [100, 2.5, low_val, 40, low_val, 75, low_val, low_val]]) * 0.628e-3  # PY

    # contains the maximal values that were used by Prinz et al.
    membrane_cond_maxs = np.asarray([[400, 5.0, 6, 50, 10, 125, 0.01, low_val],  # PM
                                     [100, low_val, 10, 50, 5, 100, 0.05, 0.03],  # LP
                                     [500, 10, 2, 50, low_val, 125, 0.05, 0.03]]) * 0.628e-3  # PY

    ranges = np.asarray([100, 2.5, 2, 10, 5, 25, 0.01, 0.01]) * 0.628e-3
    membrane_cond_mins = (membrane_cond_mins - ranges).flatten()
    membrane_cond_maxs = (membrane_cond_maxs + ranges).flatten()
    membrane_cond_mins[membrane_cond_mins < 0.0] = 0.0

    # synapses
    syn_dim_mins = np.ones(7) * 1e-8
    syn_dim_maxs = np.ones(7) * 1e-3
    syn_dim_maxs[0] *= 10.0

    if log:
        syn_dim_mins = np.log(syn_dim_mins)
        syn_dim_maxs = np.log(syn_dim_maxs)

    membrane_and_sny_mins = np.concatenate((membrane_cond_mins, syn_dim_mins,))
    membrane_and_sny_maxs = np.concatenate((membrane_cond_maxs, syn_dim_maxs,))

    tt_membrane_and_sny_mins = torch.tensor(membrane_and_sny_mins)
    tt_membrane_and_sny_maxs = torch.tensor(membrane_and_sny_maxs)

    return BoxUniform(tt_membrane_and_sny_mins, tt_membrane_and_sny_maxs)


# creates a prior for inference
def create_prior_general(hyperparams, log=False):

    ### prior bounds for Q10 values ###

    # synapses gbar
    low_Val_syn_gbar = 1.0
    highVal_syn_gbar = 2.0

    # synapses tau
    low_Val_syn_tau = 1.0
    highVal_syn_tau = 4.0

    # membrane gbar
    low_Val_mem_gbar = 1.0
    highVal_mem_gbar = 2.0

    # tau m
    low_Val_tau_m = 1.0
    highVal_tau_m = 4.0

    # tau h
    low_Val_tau_h = 1.0
    highVal_tau_h = 4.0

    # CaBuff
    low_Val_CaBuff = 1.0
    highVal_CaBuff = 4.0

    assert hyperparams.comp_neurons is None, "Is you are using a novel prior, you can not use comp_neurons"

    # maximal membrane conductances
    # rows:    LP, PY, PD
    # columns: the eight membrane conductances
    # contains the minimal values that were used by Prinz et al.
    low_val = 0.0
    membrane_cond_mins = np.asarray([[100, 2.5, 2, 10, 5, 50, 0.01, low_val],  # PM
                                     [100, low_val, 4, 20, low_val, 25, low_val, 0.02],  # LP
                                     [100, 2.5, low_val, 40, low_val, 75, low_val, low_val]]) * 0.628e-3  # PY

    # contains the maximal values that were used by Prinz et al.
    membrane_cond_maxs = np.asarray([[400, 5.0, 6, 50, 10, 125, 0.01, low_val],  # PM
                                     [100, low_val, 10, 50, 5, 100, 0.05, 0.03],  # LP
                                     [500, 10, 2, 50, low_val, 125, 0.05, 0.03]]) * 0.628e-3  # PY

    ranges = np.asarray([100, 2.5, 2, 10, 5, 25, 0.01, 0.01]) * 0.628e-3
    membrane_cond_mins = membrane_cond_mins - ranges
    membrane_cond_maxs = membrane_cond_maxs + ranges
    membrane_cond_mins[membrane_cond_mins<0.0] = 0.0
    use_membrane = np.asarray(hyperparams.use_membrane)
    membrane_used_mins = membrane_cond_mins[use_membrane == True].flatten()
    membrane_used_maxs = membrane_cond_maxs[use_membrane == True].flatten()

    # proctolin
    proctolin_gbar_mins = [0.0, 0.0, 0.0]
    proctolin_gbar_maxs = np.asarray([6.0, 8.0, 0.0]) * 1e-6
    use_proctolin = hyperparams.use_proctolin

    # synapses
    syn_dim_mins = np.ones_like(hyperparams.true_params) * hyperparams.syn_min  # syn_min is the start of uniform interval
    syn_dim_maxs = np.ones_like(hyperparams.true_params) * hyperparams.syn_max  # syn_max is the end of uniform interval
    syn_dim_maxs[0] *= 10.0

    if log:
        syn_dim_mins = np.log(syn_dim_mins)
        syn_dim_maxs = np.log(syn_dim_maxs)

    # q10 values for synapses # both, maximal conds and tau
    gbar_q10_syn_mins = np.asarray([low_Val_syn_gbar, low_Val_syn_gbar])
    gbar_q10_syn_maxs = np.asarray([highVal_syn_gbar, highVal_syn_gbar])
    tau_q10_syn_mins  = np.asarray([low_Val_syn_tau,  low_Val_syn_tau])
    tau_q10_syn_maxs  = np.asarray([highVal_syn_tau,  highVal_syn_tau])
    use_gbar_syn = np.asarray(hyperparams.Q10_gbar_syn)
    use_tau_syn  = np.asarray(hyperparams.Q10_tau_syn)
    gbar_q10_syn_used_mins = gbar_q10_syn_mins[use_gbar_syn].flatten()
    gbar_q10_syn_used_maxs = gbar_q10_syn_maxs[use_gbar_syn].flatten()
    tau_q10_syn_used_mins  = tau_q10_syn_mins[use_tau_syn].flatten()
    tau_q10_syn_used_maxs  = tau_q10_syn_maxs[use_tau_syn].flatten()

    # q10 values for maximal membrane conductances
    gbar_q10_mem_mins = low_Val_mem_gbar * np.ones(8)
    gbar_q10_mem_maxs = highVal_mem_gbar * np.ones(8)
    use_gbar_mem = np.asarray(hyperparams.Q10_gbar_mem)
    gbar_q10_mem_used_mins = gbar_q10_mem_mins[use_gbar_mem].flatten()
    gbar_q10_mem_used_maxs = gbar_q10_mem_maxs[use_gbar_mem].flatten()

    # tau_m
    if isinstance(hyperparams.Q10_tau_m, bool):
        tau_m_q10_mem_mins = np.asarray([low_Val_tau_m])
        tau_m_q10_mem_maxs = np.asarray([highVal_tau_m])
    else:
        tau_m_q10_mem_mins = low_Val_tau_m * np.ones(8)
        tau_m_q10_mem_maxs = highVal_tau_m * np.ones(8)
    use_tau_m_mem = np.asarray(hyperparams.Q10_tau_m)
    tau_m_q10_mem_used_mins = tau_m_q10_mem_mins[use_tau_m_mem].flatten()
    tau_m_q10_mem_used_maxs = tau_m_q10_mem_maxs[use_tau_m_mem].flatten()

    # tau_h
    if isinstance(hyperparams.Q10_tau_h, bool):
        tau_h_q10_mem_mins = np.asarray([low_Val_tau_h])
        tau_h_q10_mem_maxs = np.asarray([highVal_tau_h])
    else:
        tau_h_q10_mem_mins = low_Val_tau_h * np.ones(4)
        tau_h_q10_mem_maxs = highVal_tau_h * np.ones(4)
    use_tau_h_mem = np.asarray(hyperparams.Q10_tau_h)
    tau_h_q10_mem_used_mins = tau_h_q10_mem_mins[use_tau_h_mem].flatten()
    tau_h_q10_mem_used_maxs = tau_h_q10_mem_maxs[use_tau_h_mem].flatten()

    # tau_CaBuff
    tau_CaBuff_q10_mem_mins = np.asarray([low_Val_CaBuff])
    tau_CaBuff_q10_mem_maxs = np.asarray([highVal_CaBuff])
    use_tau_CaBuff_mem = np.asarray(hyperparams.Q10_tau_CaBuff)
    tau_CaBuff_q10_mem_used_mins = tau_CaBuff_q10_mem_mins[use_tau_CaBuff_mem].flatten()
    tau_CaBuff_q10_mem_used_maxs = tau_CaBuff_q10_mem_maxs[use_tau_CaBuff_mem].flatten()

    # assemble prior bounds
    if use_proctolin:
        membrane_and_sny_mins = np.concatenate((membrane_used_mins, proctolin_gbar_mins, syn_dim_mins, gbar_q10_syn_used_mins, tau_q10_syn_used_mins, gbar_q10_mem_used_mins, tau_m_q10_mem_used_mins, tau_h_q10_mem_used_mins, tau_CaBuff_q10_mem_used_mins))
        membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, proctolin_gbar_maxs, syn_dim_maxs, gbar_q10_syn_used_maxs, tau_q10_syn_used_maxs, gbar_q10_mem_used_maxs, tau_m_q10_mem_used_maxs, tau_h_q10_mem_used_maxs, tau_CaBuff_q10_mem_used_maxs))
    else:
        membrane_and_sny_mins = np.concatenate((membrane_used_mins, syn_dim_mins, gbar_q10_syn_used_mins, tau_q10_syn_used_mins, gbar_q10_mem_used_mins, tau_m_q10_mem_used_mins, tau_h_q10_mem_used_mins, tau_CaBuff_q10_mem_used_mins))
        membrane_and_sny_maxs = np.concatenate((membrane_used_maxs, syn_dim_maxs, gbar_q10_syn_used_maxs, tau_q10_syn_used_maxs, gbar_q10_mem_used_maxs, tau_m_q10_mem_used_maxs, tau_h_q10_mem_used_maxs, tau_CaBuff_q10_mem_used_maxs))

    tt_membrane_and_sny_mins = torch.tensor(membrane_and_sny_mins)
    tt_membrane_and_sny_maxs = torch.tensor(membrane_and_sny_maxs)

    return BoxUniform(tt_membrane_and_sny_mins, tt_membrane_and_sny_maxs)
