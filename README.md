# pyloric
Simulator of the pyloric network in the stomatogastric ganglion (STG) in cython.

The model was proposed in Prinz, Bucher, Marder, Nature Neuroscience, 2004.

### Installation 
```
git clone https://github.com/mackelab/pyloric.git
cd pyloric
pip install .
```

### Usage
```
from pyloric import simulate, create_prior, stats
prior = create_prior()
parameter_set = prior.sample((1,))
simulation_output = simulate(parameter_set.loc[0])
summary_statistics = stats(simulation_output)
```


### License
MIT
