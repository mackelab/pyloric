# pyloric
Simulator of the pyloric network in the stomatogastric ganglion (STG) in cython.

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
from pyloric import create_prior_general, simulate_general
setups_dict = ParameterSet('path_to_pyloric/pyloric/pyloric/setups.prm')
hyperparams = setups_dict['collect_samples_15deg_energy_ssRanges']

general_prior = create_prior_general(hyperparams)
parameter_set = general_prior.sample((1,))
simulation_outputs = simulate_general(parameter_set, hyperparams)
```

### License
MIT
