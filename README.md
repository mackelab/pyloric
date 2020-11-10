# pyloric
Simulator of the pyloric network in the stomatogastric ganglion (STG) in cython.

The model was proposed in Prinz, Bucher, Marder, Nature Neuroscience, 2004.

### Installation 
```
pip install .
```

### Usage
```
from pyloric import simulate, create_prior
prior = create_prior()
parameter_set = prior.sample((1,))
simulation_output = simulate(parameter_set)
```

### Flexible simulator
For having more flexibility with the simulator (e.g. add Q10 values, different temperature, proctolin (warninig: untested!), or only a subset of membrane conductances), we also provide a more flexible interface. Use it as follows:
```
from parameters import ParameterSet
from pyloric import create_prior_general, simulate_general
setups_dict = ParameterSet('path_to_pyloric/pyloric/pyloric/setups.prm')
hyperparams = setups_dict['collect_samples_15deg_energy_ssRanges']

general_prior = create_prior_general(hyperparams)
parameter_set = general_prior.sample((1,))
simulation_outputs = simulate_general(parameter_set, hyperparams)
```

### Parallelization
One can easily parallelize the code, e.g. with joblib:
```
from joblib import Parallel, delayed

def simulator(params_set):
    out_target = simulate_general(
        deepcopy(params_set[:-1].astype(np.float64)),
        hyperparams_11,
        seed=int(params_set[-1]),
    )
    return stats(out_target)

simulation_outputs = Parallel(n_jobs=24)(
    delayed(simulator)(batch)
    for batch in params_with_seeds
)
```

### License
MIT
