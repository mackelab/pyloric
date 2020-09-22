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

### License
MIT
